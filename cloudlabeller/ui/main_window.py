# CloudLabeller — photogrammetric reconstruction and bidirectional 2D <-> 3D
# point-cloud labelling with U-Net label propagation.
# Copyright (C) 2026 Ítalo Gomes Gonçalves
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program. If not, see <https://www.gnu.org/licenses/>.
#
# Commercial licensing: this program is also available under a separate
# commercial license from the author — see README.md.

"""Main window: dockable shell tying the 3D viewer, image view and panels
together through the shared :class:`EventBus`."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QThreadPool, QTimer
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
)

from cloudlabeller.config import AppConfig
from cloudlabeller.core.events import EventBus
from cloudlabeller.core.project import Project
from cloudlabeller.ui.dataset_panel import DatasetPanel
from cloudlabeller.ui.image_view import ImageView
from cloudlabeller.ui.label_panel import LabelPanel
from cloudlabeller.ui.log_panel import LogPanel
from cloudlabeller.ui.view_panel import ViewPanel
from cloudlabeller.ui.viewer3d import Viewer3D


class MainWindow(QMainWindow):
    """The application shell: builds the docks, menus and status bar, owns
    the open :class:`~cloudlabeller.core.project.Project`, and orchestrates
    every long-running job (reconstruction subprocesses via ProcessJob,
    in-thread transfers/exports via Job) — all cross-pane communication
    flows through the :class:`~cloudlabeller.core.events.EventBus`."""

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config = config
        self.bus = EventBus()
        self.project: Project | None = None
        self._pipeline: dict | None = None     # active full-pipeline run state

        self.setWindowTitle("CloudLabeller")
        self.resize(1600, 1000)

        # Central + docks ----------------------------------------------------
        self.viewer3d = Viewer3D(self.bus, config)
        self.setCentralWidget(self.viewer3d)

        self.image_view = ImageView(self.bus)
        self.label_panel = LabelPanel(self.bus)
        self.dataset_panel = DatasetPanel(self.bus)
        self.view_panel = ViewPanel(self.bus)
        self.log_panel = LogPanel()

        self._docks: list[QDockWidget] = []    # feeds the Panes menu
        self._add_dock("Image", self.image_view, Qt.RightDockWidgetArea)
        # Left column, top→bottom: Labels, View, Dataset.
        self._add_dock("Labels", self.label_panel, Qt.LeftDockWidgetArea)
        self._add_dock("View", self.view_panel, Qt.LeftDockWidgetArea)
        self._add_dock("Dataset", self.dataset_panel, Qt.LeftDockWidgetArea)
        self._add_dock("Log", self.log_panel, Qt.BottomDockWidgetArea)

        # View-control panel drives the 3D viewer.
        self.view_panel.representation_changed.connect(self.viewer3d.set_representation)
        self.view_panel.cameras_toggled.connect(self.viewer3d.set_cameras_visible)
        self.view_panel.frustum_scale_changed.connect(self.viewer3d.set_frustum_scale)
        self.view_panel.label_blend_changed.connect(self.viewer3d.set_label_blend)
        self.view_panel.lasso_toggled.connect(self.viewer3d.set_lasso_mode)
        self.view_panel.apply_lasso.connect(self.viewer3d.apply_lasso_label)
        self.view_panel.delete_selection.connect(self._delete_lasso_points)
        self.viewer3d.selection_changed.connect(self.view_panel.set_selection_count)
        self.image_view.propagate_requested.connect(self._propagate_image)

        self._build_menus()
        self._build_statusbar()
        self._connect_jobs()
        self._setup_autosave()

    # -- autosave ----------------------------------------------------------
    def _setup_autosave(self) -> None:
        """Persist labels automatically: on a timer when dirty, and on close."""
        self._dirty = False
        for sig in (self.bus.cloud_labels_changed, self.bus.schema_changed):
            sig.connect(self._mark_dirty)
        # A label edit marks the heavy labels dirty AND persists the manifest now
        # (so the frustum-dot statuses survive a reload immediately, not only on
        # the 60s autosave / clean close).
        self.bus.image_labels_changed.connect(self._on_label_edit)

        self._autosave = QTimer(self)
        self._autosave.setInterval(60_000)          # 60 s
        self._autosave.timeout.connect(self._autosave_if_dirty)
        self._autosave.start()

    def _mark_dirty(self, *_args) -> None:
        self._dirty = True

    def _on_label_edit(self, *_args) -> None:
        self._dirty = True
        if self.project:
            try:
                self.project.save_manifest()   # persist dot statuses immediately (cheap)
            except Exception:
                pass

    def _autosave_if_dirty(self) -> None:
        if self.project and self._dirty:
            self._save()

    def _save(self) -> None:
        if not self.project:
            return
        try:
            self.project.save()
            self._dirty = False
        except Exception as exc:  # never let a save error crash the UI
            self.statusBar().showMessage(f"Auto-save failed: {exc}", 5000)

    def _running_child_jobs(self) -> list[str]:
        """Human-readable names of subprocess jobs still running."""
        from PySide6.QtCore import QProcess

        labels = {"_sfm_job": "Reconstruction (SfM)", "_mvs_job": "Dense MVS",
                  "_mesh_job": "Mesh build", "_train_job": "U-Net training",
                  "_predict_job": "Prediction"}
        running = []
        for attr, label in labels.items():
            proc = getattr(getattr(self, attr, None), "proc", None)
            if proc is not None and proc.state() != QProcess.NotRunning:
                running.append(label)
        return running

    def closeEvent(self, event) -> None:
        # Closing the app KILLS child processes. An overnight MVS run once died
        # silently this way — never again without an explicit yes.
        running = self._running_child_jobs()
        if running:
            answer = QMessageBox.question(
                self, "Jobs still running",
                "Quitting will KILL these running jobs:\n  • "
                + "\n  • ".join(running)
                + "\n\n(COLMAP dense stereo can resume finished depth maps; "
                "other stages restart from scratch.)\n\nQuit anyway?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                event.ignore()
                return
        self._pipeline_end("aborted (app closed)")   # releases keep-awake
        self._save()
        from PySide6.QtCore import QSettings

        settings = QSettings("CloudLabeller", "CloudLabeller")
        settings.setValue("window/geometry", self.saveGeometry())
        settings.setValue("window/state", self.saveState())
        super().closeEvent(event)

    def restore_layout(self) -> None:
        """Restore window size + dock layout from the previous session.
        Called after construction (docks must exist before restoreState)."""
        from PySide6.QtCore import QSettings

        settings = QSettings("CloudLabeller", "CloudLabeller")
        geometry = settings.value("window/geometry")
        state = settings.value("window/state")
        if geometry is not None:
            self.restoreGeometry(geometry)
        if state is not None:
            self.restoreState(state)

    # -- construction helpers ---------------------------------------------
    def _add_dock(self, title: str, widget, area: Qt.DockWidgetArea) -> None:
        dock = QDockWidget(title, self)
        dock.setObjectName(f"dock_{title}")   # required for saveState/restoreState
        dock.setWidget(widget)
        self.addDockWidget(area, dock)
        self._docks.append(dock)              # its toggle action → Panes menu

    def _build_menus(self) -> None:
        m_file = self.menuBar().addMenu("&File")
        m_file.addAction("New Project…", self._new_project)
        m_file.addAction("Open Project…", self._open_project_dialog)
        m_file.addSeparator()
        m_file.addAction("Add Images…", self._add_images)
        m_file.addAction("Project Info…", self._show_project_info)
        m_file.addSeparator()
        m_file.addAction("Export Point Cloud…", self._export_cloud)
        m_file.addAction("Export Mesh…", self._export_mesh)
        # No "Save": the project auto-saves (after reconstruction, on a timer,
        # and on close); reconstruction products are written to the project folder.
        m_file.addSeparator()
        m_file.addAction("Quit", self.close)

        m_photo = self.menuBar().addMenu("&Photogrammetry")
        m_photo.addAction("Run Full Pipeline…", self._run_pipeline)
        m_photo.addSeparator()
        m_photo.addAction("Run SfM…", self._run_sfm)
        m_photo.addAction("Run Dense (MVS)…", self._run_mvs)
        m_photo.addAction("Create Mesh", self._create_mesh)
        m_photo.addSeparator()
        m_photo.addAction("Clean Sparse Cloud…", self._clean_sparse_cloud)
        m_photo.addSeparator()
        m_photo.addAction("Download COLMAP…", self._download_colmap)

        # Georeferencing grows its own menu (Tier 2 — GCPs — lands here too).
        m_georef = self.menuBar().addMenu("&Georeferencing")
        m_georef.addAction("Align to EXIF GPS…", self._georeference)
        m_georef.addAction("Reproject to CRS…", self._reproject)

        m_transfer = self.menuBar().addMenu("&Transfer")
        m_transfer.addAction("Images → Cloud", self._images_to_cloud)
        m_transfer.addAction("Cloud → Images", self._cloud_to_images)

        m_ml = self.menuBar().addMenu("&Model")
        m_ml.addAction("Model Settings…", self._configure_model)
        m_ml.addAction("Train U-Net…", self._train)
        m_ml.addAction("Predict All Images…", self._predict_all)

        m_edit = self.menuBar().addMenu("&Edit")
        m_edit.addAction("Undo", self._undo, "Ctrl+Z")
        m_edit.addAction("Redo", self._redo, "Ctrl+Y")

        # A closed pane is otherwise unrecoverable — Qt's toggle actions stay
        # checked/unchecked in sync with each dock's visibility.
        m_panes = self.menuBar().addMenu("&Panes")
        for dock in self._docks:
            m_panes.addAction(dock.toggleViewAction())

        m_help = self.menuBar().addMenu("&Help")
        m_help.addAction("About CloudLabeller…", self._show_about)

    def _build_statusbar(self) -> None:
        # Permanent (always-visible) project summary on the right; transient
        # messages (showMessage) use the left area.
        self.status_crs = QLabel("")            # current coordinate frame
        self.statusBar().addPermanentWidget(self.status_crs)
        self.status_info = QLabel("No project open")
        self.status_info.setToolTip("Click for the full project summary "
                                    "(origin, counts, models…)")
        self.status_info.setCursor(Qt.PointingHandCursor)
        self.status_info.installEventFilter(self)   # click -> Project Info
        self.statusBar().addPermanentWidget(self.status_info)
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(220)
        self.progress.hide()
        self.statusBar().addPermanentWidget(self.progress)
        # Stop button for cancellable jobs (e.g. training): shown by
        # _start_transfer_job(cancellable=True), wired to Job.cancel().
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setMaximumWidth(60)
        self.btn_stop.hide()
        self.statusBar().addPermanentWidget(self.btn_stop)
        self._active_job = None
        self._colmap_dl_job = None
        self._georef_after_sfm = False

        for sig in (self.bus.project_opened, self.bus.images_changed,
                    self.bus.cloud_changed, self.bus.schema_changed):
            sig.connect(self._refresh_project_status)
        self.bus.job_finished.connect(self._refresh_project_status)

    def _refresh_project_status(self, *_args) -> None:
        p = self.project
        if not p:
            self.status_info.setText("No project open")
            self.status_crs.setText("")
            return
        from cloudlabeller.photogrammetry.crs import frame_labels

        short, full = frame_labels(p.settings.get("georeferenced"))
        self.status_crs.setText(short + "  ·")
        self.status_crs.setToolTip(full)
        ds = p.dataset
        n_pts = ds.cloud.n_points if ds.cloud is not None else 0
        status_txt = {
            "none": "not reconstructed",
            "current": "SfM up to date",
            "outdated": "⚠ SfM outdated — re-run",
        }[p.reconstruction_status()]
        self.status_info.setText(
            f"{Path(p.root).name}   ·   {len(ds.images)} images   ·   "
            f"{len(ds.solved_images())} cameras   ·   {n_pts:,} pts   ·   {status_txt}"
        )

    def _connect_jobs(self) -> None:
        self.bus.job_started.connect(lambda desc: (self.progress.show(),
                                                    self.statusBar().showMessage(desc)))
        self.bus.job_progress.connect(
            lambda _desc, f: self.progress.setValue(int(f * 100)))
        self.bus.job_finished.connect(
            lambda desc, ok: (self.progress.hide(),
                              self.statusBar().showMessage(
                                  f"{desc}: {'done' if ok else 'failed'}", 5000)))

    # -- actions (thin; real work delegated to controllers/workers) -------
    def _new_project(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose an empty project folder")
        if path:
            self._set_project(Project.create(path))

    def _open_project_dialog(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Open project folder")
        if path:
            self.open_project(path)

    def open_project(self, path: str | Path) -> None:
        try:
            self._set_project(Project.open(path))
        except Exception as exc:
            QMessageBox.critical(self, "Open failed", str(exc))

    def adopt_project(self, project: Project) -> None:
        """Adopt an already-loaded project (e.g. from the welcome dialog)."""
        self._set_project(project)

    def _set_project(self, project: Project) -> None:
        self._save()                       # persist the current project before switching
        self.project = project
        self._dirty = False
        self.config.add_recent(str(project.root))
        self.config.save()
        self.bus.project_opened.emit(project)

    # -- full pipeline (SfM → dense MVS → mesh, unattended) ------------------
    def _run_pipeline(self) -> None:
        """Chain SfM → dense MVS → mesh with all options collected upfront —
        made for overnight runs. Stages hand over via the project folder; a
        failed stage stops the chain and keeps the earlier products."""
        if not self.project:
            QMessageBox.information(self, "No project", "Create or open a project first.")
            return
        from cloudlabeller.core.images import list_image_files
        from cloudlabeller.photogrammetry.mvs import find_colmap_binary

        n_images = len(list_image_files(self.project.images_dir))
        if n_images == 0:
            QMessageBox.information(
                self, "No images",
                "The project's image store is empty.\n"
                "Add images first via File → Add Images…")
            return
        colmap_binary = find_colmap_binary(self.config.colmap_binary)
        if colmap_binary is None:
            answer = QMessageBox.question(
                self, "No CUDA COLMAP",
                "The dense-MVS stage needs the CUDA COLMAP executable, which "
                "was not found — the pipeline would stop after SfM.\n\n"
                "Download it now (one-time, official COLMAP release)? "
                "Start the pipeline again once it finishes.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if answer == QMessageBox.Yes:
                self._download_colmap()
            return

        from cloudlabeller.ui.run_pipeline_dialog import RunPipelineDialog

        dialog = RunPipelineDialog(n_images,
                                   source_resolution=self.project.source_resolution(),
                                   gpu_available=True,   # colmap_binary checked above
                                   gps_found=self._gps_image_hint(),
                                   parent=self)
        if dialog.exec() != RunPipelineDialog.Accepted:
            return
        self._georef_after_sfm = dialog.georeference()

        import time

        from cloudlabeller.workers.resources import keep_system_awake

        self._pipeline = {"mvs_px": dialog.mvs_pixels(), "colmap": colmap_binary,
                          "mvs_quality": dialog.mvs_quality(),
                          "mesh": dialog.mesh_options(), "t0": time.monotonic()}
        keep_system_awake(True)                   # no sleeping mid-pipeline
        self.log_panel.append("=== Full pipeline started: SfM → dense MVS → mesh "
                              "(the machine is kept awake) ===")
        self._launch_sfm(dialog.sfm_options())

    def _pipeline_stage_started(self, job) -> None:
        """While a pipeline runs, the Stop button cancels its current stage
        (which aborts the chain; finished stages are kept)."""
        if self._pipeline is not None:
            self._begin_cancellable(job, "Pipeline")

    def _pipeline_end(self, message: str) -> None:
        """Close out a pipeline run (complete or aborted). No-op otherwise."""
        if self._pipeline is None:
            return
        import time

        from cloudlabeller.workers.resources import keep_system_awake

        minutes = (time.monotonic() - self._pipeline["t0"]) / 60.0
        self._pipeline = None
        keep_system_awake(False)
        if self._active_job is not None:
            self._end_cancellable(self._active_job)
        self.log_panel.append(f"=== Pipeline {message} — {minutes:.1f} min total ===")
        self.statusBar().showMessage(f"Pipeline {message} ({minutes:.1f} min)", 0)

    def _run_sfm(self) -> None:
        """Run reconstruction in a child process (images → sparse cloud + cameras).

        COLMAP runs out-of-process so its native threads / GPU OpenGL context
        can't collide with the viewer's VTK context, and a native crash fails the
        job instead of killing the app.
        """
        if not self.project:
            QMessageBox.information(self, "No project", "Create or open a project first.")
            return

        from cloudlabeller.core.images import list_image_files
        from cloudlabeller.ui.run_sfm_dialog import RunSfmDialog
        from cloudlabeller.workers.process_job import ProcessJob

        n_images = len(list_image_files(self.project.images_dir))
        if n_images == 0:
            QMessageBox.information(
                self, "No images",
                "The project's image store is empty.\n"
                "Add images first via File → Add Images…")
            return

        from cloudlabeller.photogrammetry.mvs import find_colmap_binary

        gpu_ok = find_colmap_binary(self.config.colmap_binary) is not None
        dialog = RunSfmDialog(n_images,
                              source_resolution=self.project.source_resolution(),
                              gpu_available=gpu_ok,
                              gps_found=self._gps_image_hint(), parent=self)
        if dialog.exec() != RunSfmDialog.Accepted:
            return
        self._georef_after_sfm = dialog.georeference()
        self._launch_sfm(dialog.options())

    def _gps_image_hint(self) -> int:
        """How many of the store's first images have GPS EXIF (capped scan —
        enough to decide whether auto-georeferencing can be offered)."""
        from cloudlabeller.core.images import list_image_files
        from cloudlabeller.photogrammetry.georef import count_gps_images

        return count_gps_images(list_image_files(self.project.images_dir))

    def _launch_sfm(self, options) -> None:
        """Start the SfM child process (also the pipeline's first stage)."""
        from cloudlabeller.workers.process_job import ProcessJob

        desc = "Reconstruction"
        store = str(self.project.images_dir)
        workspace = str(self.project.reconstruction_dir)
        log_path = self.project.reconstruction_dir / "sfm.log"
        # SfM always runs on the project's own image store (populated via
        # File → Add Images…); no ingestion happens here.
        args = [store, workspace, *options.to_cli_args()]
        from cloudlabeller.photogrammetry.mvs import find_colmap_binary

        binary = find_colmap_binary(self.config.colmap_binary)
        if binary:
            args += ["--colmap-binary", binary]    # GPU SIFT/matching with --gpu
        self._sfm_job = ProcessJob(
            "cloudlabeller.photogrammetry.run_cli", args,
            log_path=log_path,
        )
        self._sfm_job.progress.connect(lambda f, m="": self.bus.job_progress.emit(desc, f))
        self._sfm_job.log_line.connect(self.log_panel.append)
        self._sfm_job.finished.connect(self._on_sfm_done)
        self._sfm_job.failed.connect(lambda e: self._on_sfm_failed(desc, e))
        self.log_panel.append(f"--- Reconstruction started (log: {log_path}) ---")
        self.bus.job_started.emit(desc)
        self._pipeline_stage_started(self._sfm_job)
        self._sfm_job.start()

    def _on_sfm_done(self) -> None:
        """Rebuild the dataset from what the child process wrote, refresh views."""
        try:
            self.project.load_products()          # cloud + images from the store
        except Exception as exc:
            self._on_sfm_failed("Reconstruction", f"could not load result: {exc}")
            return
        if self.project.dataset.cloud is None:
            self._on_sfm_failed("Reconstruction", "no point cloud was produced")
            return
        self.project.mark_reconstruction_current()
        # A fresh solve is a new (local) frame: any previous alignment and its
        # backup no longer correspond to this model.
        if self.project.settings.pop("georeferenced", None) is not None:
            self.log_panel.append("New solve — previous georeferencing cleared.")
        backup = self.project.reconstruction_dir / "sparse_prealigned"
        if backup.exists():
            import shutil

            shutil.rmtree(backup, ignore_errors=True)
        self._save()                              # persist manifest + labels
        self.bus.job_finished.emit("Reconstruction", True)
        self.bus.cloud_changed.emit()
        self.bus.images_changed.emit()
        n = self.project.dataset.cloud.n_points
        ncam = len(self.project.dataset.solved_images())
        self.statusBar().showMessage(f"Reconstruction: {n} points, {ncam} cameras", 8000)
        # Chain: auto-georeference first (its handlers resume the pipeline;
        # failure falls back to the local frame), then dense MVS.
        if self._georef_after_sfm:
            self._georef_after_sfm = False
            self.log_panel.append("--- Auto-georeferencing to EXIF GPS ---")
            self._start_georef_job(auto=True)
            return
        if self._pipeline is not None:
            self.log_panel.append("=== Pipeline: SfM complete — starting dense MVS ===")
            self._launch_mvs(self._pipeline["mvs_px"], self._pipeline["colmap"],
                             self._pipeline.get("mvs_quality", "standard"))

    def _on_sfm_failed(self, desc: str, error: str) -> None:
        self.bus.job_finished.emit(desc, False)
        if self._pipeline is not None:
            headline = (error.splitlines() or ["unknown error"])[0]
            self._pipeline_end(f"aborted at {desc}: {headline}")
            self.log_panel.append(f"{desc} FAILED: {error}")
            return                                # no modal at 3 a.m.
        QMessageBox.critical(self, f"{desc} failed", error)

    def _clean_sparse_cloud(self) -> None:
        """Filter stray sparse points by SfM confidence (track length and
        reprojection error), read from the saved COLMAP model — no re-run."""
        if not self.project or self.project.dataset.cloud is None:
            QMessageBox.information(self, "No sparse cloud", "Run SfM first.")
            return
        import numpy as np
        from PySide6.QtWidgets import QApplication

        from cloudlabeller.photogrammetry.extract import (
            load_best_model,
            sparse_point_stats,
        )

        cloud = self.project.dataset.cloud
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            rec = load_best_model(self.project.reconstruction_dir / "sparse")
            views, errors = sparse_point_stats(rec, cloud.xyz)
        except Exception as exc:
            QMessageBox.critical(self, "Clean sparse cloud",
                                 f"Could not load the COLMAP model:\n{exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()

        matched = float((views > 0).mean())
        if matched < 0.9:
            QMessageBox.warning(
                self, "Model mismatch",
                f"Only {matched:.0%} of the cloud's points were found in the "
                "saved COLMAP model — it does not correspond to the current "
                "cloud. Re-run SfM before cleaning.")
            return

        from cloudlabeller.ui.clean_cloud_dialog import CleanCloudDialog

        # Non-modal: the 3D pane stays live, tinting the points that the
        # current thresholds would remove as the user adjusts them.
        self.view_panel.select_representation("sparse")   # preview needs sparse view
        dialog = CleanCloudDialog(views, errors, parent=self)
        self._clean_dialog = dialog                       # keep alive (non-modal)
        dialog.setModal(False)
        dialog.mask_changed.connect(self.viewer3d.set_exclusion_preview)
        dialog.accepted.connect(lambda: self._apply_sparse_clean(dialog.mask()))
        dialog.finished.connect(
            lambda *_: (self.viewer3d.set_exclusion_preview(None),
                        setattr(self, "_clean_dialog", None)))
        dialog.show()
        self.viewer3d.set_exclusion_preview(dialog.mask())

    def _delete_lasso_points(self) -> None:
        """Delete the lasso-selected points from the sparse cloud (View pane)."""
        import numpy as np

        cloud = self.project.dataset.cloud if self.project else None
        selection = self.viewer3d.lasso_selection()
        if cloud is None or selection is None or not selection.size:
            return
        if selection.max() >= cloud.n_points:
            return                                 # stale selection: ignore
        answer = QMessageBox.question(
            self, "Delete points",
            f"Delete {selection.size:,} selected point(s) from the sparse "
            "cloud?\nThis cannot be undone.")
        if answer != QMessageBox.Yes:
            return
        keep = np.ones(cloud.n_points, dtype=bool)
        keep[selection] = False
        self.viewer3d.clear_lasso_selection()
        self._apply_sparse_clean(keep)

    def _apply_sparse_clean(self, keep) -> None:
        """Drop the sparse points where ``keep`` is False — persist the
        filtered cloud, let the labels follow, keep the camera in place."""
        import numpy as np

        from cloudlabeller.core.dataset import PointCloud
        from cloudlabeller.photogrammetry.pipeline import CLOUD_FILE

        cloud = self.project.dataset.cloud if self.project else None
        if cloud is None or len(keep) != cloud.n_points:
            QMessageBox.warning(self, "Clean sparse cloud",
                                "The cloud changed while the dialog was open — "
                                "nothing was removed. Reopen the dialog.")
            return
        n_before = cloud.n_points
        new_cloud = PointCloud(
            xyz=cloud.xyz[keep],
            rgb=cloud.rgb[keep] if cloud.rgb is not None else None)
        np.savez(self.project.reconstruction_dir / CLOUD_FILE,
                 xyz=new_cloud.xyz,
                 rgb=new_cloud.rgb if new_cloud.rgb is not None
                 else np.empty((0, 3), np.uint8))
        self.project.dataset.cloud = new_cloud
        self.project.labels.filter_cloud(keep)     # labels follow; history cleared
        self._save()
        self.viewer3d.preserve_view_once()         # in-place edit: keep the camera
        self.bus.cloud_changed.emit()
        msg = (f"Sparse cloud cleaned: removed {n_before - new_cloud.n_points:,} "
               f"of {n_before:,} points")
        self.log_panel.append(msg)
        self.statusBar().showMessage(msg, 8000)

    def _georeference(self) -> None:
        """Align the whole model to the images' EXIF GPS (metric ENU frame) —
        native RANSAC + Umeyama fit, transforming every product in place."""
        if not self.project or self.project.dataset.cloud is None:
            QMessageBox.information(self, "No reconstruction", "Run SfM first.")
            return
        if self.project.settings.get("georeferenced"):
            QMessageBox.information(
                self, "Already georeferenced",
                "This project is already aligned to GPS — aligning twice "
                "would double-transform the model.")
            return
        answer = QMessageBox.question(
            self, "Georeference model",
            "Align the model to the images' EXIF GPS?\n\n"
            "• Coordinates become a local East-North-Up frame: metres, true "
            "north, origin at the site.\n"
            "• The sparse/dense clouds, mesh, cameras and labels are all "
            "transformed consistently.\n\n"
            "Do this AFTER the reconstruction stages are final: an in-progress "
            "dense reconstruction (partial depth maps) from the unaligned "
            "model cannot be resumed afterwards.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:
            return
        self._start_georef_job(auto=False)

    def _start_georef_job(self, auto: bool) -> None:
        """Launch the georeferencing subprocess. ``auto`` = chained after SfM:
        failures are logged and the pipeline continues in the local frame."""
        from cloudlabeller.workers.process_job import ProcessJob

        desc = "Georeference"
        job = ProcessJob(
            "cloudlabeller.photogrammetry.georef_cli",
            [str(self.project.root)],
            log_path=self.project.reconstruction_dir / "georef.log")
        self._georef_job = job
        job.progress.connect(lambda f, m="": self.bus.job_progress.emit(desc, f))
        job.log_line.connect(self.log_panel.append)
        job.finished.connect(lambda: self._on_georef_done(job))
        if auto:
            job.failed.connect(self._on_georef_auto_failed)
        else:
            job.failed.connect(lambda e: self._on_sfm_failed(desc, e))
        self.log_panel.append("--- Georeferencing started ---")
        self.bus.job_started.emit(desc)
        self._pipeline_stage_started(job)
        job.start()

    def _on_georef_done(self, job) -> None:
        """Persist the new ENU frame, refresh every view, report the fit —
        and continue the pipeline (dense MVS) when one is running."""
        out = self._job_result(job) or {}
        self.project.load_products()               # reload transformed products
        self.project.settings["georeferenced"] = {
            "frame": "ENU", "origin_lla": out.get("origin_lla"),
            "origin_convention": out.get("origin_convention"),
            "scale_m_per_unit": out.get("scale_m_per_unit"),
            "n_gps": out.get("n_gps"),
        }
        self._save()
        self.bus.job_finished.emit("Georeference", True)
        self.bus.cloud_changed.emit()
        self.bus.mesh_changed.emit()
        self.bus.images_changed.emit()             # camera coords in Dataset pane
        origin = out.get("origin_lla") or [0, 0, 0]
        msg = (f"Georeferenced to ENU: 1 unit = 1 m now "
               f"(previous scale ×{out.get('scale_m_per_unit', 0):.4f}); "
               f"origin ≈ {origin[0]:.6f}°, {origin[1]:.6f}°, {origin[2]:.1f} m; "
               f"{out.get('n_inliers', '?')}/{out.get('n_gps', 0)} GPS images "
               f"agree within {out.get('fit_rms_m', 0):.1f} m rms")
        self.log_panel.append(msg)
        self.statusBar().showMessage(msg, 10000)
        if self._pipeline is not None:
            self.log_panel.append("=== Pipeline: georeferencing done — "
                                  "starting dense MVS ===")
            self._launch_mvs(self._pipeline["mvs_px"], self._pipeline["colmap"],
                             self._pipeline.get("mvs_quality", "standard"))

    def _resolved_enu_origin(self):
        """The project's exact ENU origin (healing legacy projects that
        stored the mean GPS — persisted once recovered)."""
        from cloudlabeller.photogrammetry.crs import (
            ORIGIN_CONVENTION_FIRST_GPS,
            resolve_enu_origin,
        )

        geo = self.project.settings["georeferenced"]
        origin = resolve_enu_origin(geo, self.project.reconstruction_dir,
                                    self.project.images_dir)
        if origin.exact and geo.get("origin_convention") != ORIGIN_CONVENTION_FIRST_GPS:
            geo["origin_lla"] = list(origin.lla)
            geo["origin_convention"] = ORIGIN_CONVENTION_FIRST_GPS
            self.project.save_manifest()
        return origin

    def _reproject(self) -> None:
        """Move the whole project (clouds, mesh, cameras, COLMAP model) into
        a chosen projected CRS — stored minus a km offset for precision."""
        if not self.project or self.project.dataset.cloud is None:
            QMessageBox.information(self, "No reconstruction", "Run SfM first.")
            return
        geo = self.project.settings.get("georeferenced")
        if not geo:
            QMessageBox.information(
                self, "Not georeferenced",
                "Reprojection needs a georeferenced model — run "
                "Georeferencing → Align to EXIF GPS… first.")
            return
        from cloudlabeller.ui.reproject_dialog import ReprojectDialog

        origin = self._resolved_enu_origin()
        dlg = ReprojectDialog(origin.lla, geo, parent=self)
        if not dlg.exec():
            return

        from cloudlabeller.workers.process_job import ProcessJob

        desc = "Reproject"
        args = [str(self.project.root), "--epsg", str(dlg.epsg())]
        if dlg.orthometric():
            args.append("--orthometric")
        job = ProcessJob("cloudlabeller.photogrammetry.reproject_cli", args,
                         log_path=self.project.reconstruction_dir / "reproject.log")
        self._georef_job = job
        job.progress.connect(lambda f, m="": self.bus.job_progress.emit(desc, f))
        job.log_line.connect(self.log_panel.append)
        job.finished.connect(lambda: self._on_reproject_done(job))
        job.failed.connect(lambda e: self._on_sfm_failed(desc, e))
        self.log_panel.append("--- Reprojection started ---")
        self.bus.job_started.emit(desc)
        job.start()

    def _on_reproject_done(self, job) -> None:
        """Record the new CRS frame (epsg/name/offset) in the project
        settings and refresh everything that displays coordinates."""
        out = self._job_result(job) or {}
        self.project.load_products()               # reload transformed products
        self.project.settings["georeferenced"]["crs"] = {
            "epsg": out.get("epsg"), "name": out.get("name"),
            "orthometric": out.get("orthometric", False),
            "offset": out.get("offset"),
        }
        self._save()
        self.bus.job_finished.emit("Reproject", True)
        self.bus.cloud_changed.emit()
        self.bus.mesh_changed.emit()
        self.bus.images_changed.emit()             # camera coords in Dataset pane
        off = out.get("offset") or [0, 0, 0]
        msg = (f"Reprojected to EPSG:{out.get('epsg')} — {out.get('name')} "
               f"(stored offset {off[0]:.0f}, {off[1]:.0f}, {off[2]:.0f}; "
               f"projection-vs-similarity rms {out.get('fit_rms_m', 0):.3f} m)")
        self.log_panel.append(msg)
        self.statusBar().showMessage(msg, 10000)

    def _on_georef_auto_failed(self, error: str) -> None:
        """Auto-georeferencing is best-effort: keep the local frame and keep
        going (Photogrammetry → Georeference remains available)."""
        headline = (error.splitlines() or ["unknown error"])[0]
        self.bus.job_finished.emit("Georeference", False)
        if "cancelled" in error.lower():           # user stop = abort, not skip
            self.log_panel.append("Georeferencing cancelled.")
            self._pipeline_end("aborted at Georeference")
            return
        self.log_panel.append(f"Auto-georeferencing failed — continuing in "
                              f"the local frame: {headline}")
        if self._pipeline is not None:
            self.log_panel.append("=== Pipeline: starting dense MVS ===")
            self._launch_mvs(self._pipeline["mvs_px"], self._pipeline["colmap"],
                             self._pipeline.get("mvs_quality", "standard"))

    def _ask_conflict_policy(self, source_dir: str) -> str | None:
        """If the source has names already in the store, ask how to resolve.

        Returns "overwrite", "skip", or None (cancel). When there are no
        conflicts the policy is irrelevant, so we return "skip".
        """
        from cloudlabeller.core.images import find_conflicts

        conflicts = find_conflicts(source_dir, self.project.images_dir)
        if not conflicts:
            return "skip"
        sample = ", ".join(conflicts[:6]) + ("…" if len(conflicts) > 6 else "")
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle("Duplicate image names")
        box.setText(f"{len(conflicts)} image name(s) already exist in this project.")
        box.setInformativeText(
            f"e.g. {sample}\n\nOverwrite them with the source versions, "
            f"or keep the existing ones?")
        overwrite = box.addButton("Overwrite", QMessageBox.YesRole)
        keep = box.addButton("Keep existing", QMessageBox.NoRole)
        box.addButton(QMessageBox.Cancel)
        box.exec()
        clicked = box.clickedButton()
        if clicked is overwrite:
            return "overwrite"
        if clicked is keep:
            return "skip"
        return None

    def _add_images(self) -> None:
        """Copy a folder of images into the project store (off-thread)."""
        if not self.project:
            QMessageBox.information(self, "No project", "Create or open a project first.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Select a folder of images to add")
        if not folder:
            return
        policy = self._ask_conflict_policy(folder)
        if policy is None:
            return
        from cloudlabeller.workers.process_job import ProcessJob

        desc = "Add images"
        store = str(self.project.images_dir)
        workspace = str(self.project.reconstruction_dir)
        self._ingest_job = ProcessJob(
            "cloudlabeller.photogrammetry.run_cli",
            [store, workspace, "--ingest-from", folder, "--ingest-only",
             "--on-conflict", policy],
            log_path=self.project.reconstruction_dir / "ingest.log",
        )
        self._ingest_job.log_line.connect(self.log_panel.append)
        self._ingest_job.progress.connect(lambda f, m="": self.bus.job_progress.emit(desc, f))
        self._ingest_job.finished.connect(self._on_ingest_done)
        self._ingest_job.failed.connect(lambda e: self._on_sfm_failed(desc, e))
        self.bus.job_started.emit(desc)
        self._ingest_job.start()

    def _on_ingest_done(self) -> None:
        # If the store actually changed and a reconstruction exists, it's now stale.
        changed = "changed=1" in self._ingest_job.result_text
        if changed and self.project.dataset.cloud is not None:
            self.project.mark_reconstruction_outdated()
        self.project.load_products()              # rescan store into the dataset
        self._save()
        self.bus.job_finished.emit("Add images", True)
        self.bus.images_changed.emit()
        n = len(self.project.dataset.images)
        tail = (" — SfM is now outdated, re-run to register new images."
                if changed and self.project.reconstruction_status() == "outdated" else "")
        self.statusBar().showMessage(f"Image store now has {n} images.{tail}", 9000)

    # -- COLMAP download ------------------------------------------------------
    def _download_colmap(self) -> None:
        """Fetch the official COLMAP bundle into ``~/.cloudlabeller`` (the
        location ``find_colmap_binary`` searches). COLMAP is not shipped with
        the app — its official bundle links GPL components — so this one-time
        download is how the GPU engine (SfM SIFT, dense MVS, Delaunay
        meshing) becomes available."""
        from cloudlabeller.photogrammetry.colmap_fetch import (
            COLMAP_VERSION,
            DOWNLOAD_MB,
            cuda_capable,
            default_install_dir,
        )
        from cloudlabeller.photogrammetry.mvs import find_colmap_binary

        if self._colmap_dl_job is not None:
            QMessageBox.information(self, "Already downloading",
                                    "A COLMAP download is already running — "
                                    "see the progress bar below.")
            return
        existing = find_colmap_binary(self.config.colmap_binary)
        has_gpu = cuda_capable()
        text = (f"Download the official COLMAP {COLMAP_VERSION} release from "
                f"GitHub into<br><code>{default_install_dir()}</code>?<br><br>")
        if existing:
            text += (f"COLMAP is currently found at<br><code>{existing}</code>"
                     "<br>— the download will take precedence if it is the "
                     "same location, otherwise the existing copy keeps being "
                     "used first.<br><br>")
        if has_gpu:
            text += ("An NVIDIA GPU was detected — the <b>CUDA</b> version is "
                     "recommended: it enables dense MVS and ~10× faster SIFT.")
        else:
            text += ("<b>No NVIDIA GPU detected.</b> The CUDA version needs "
                     "one. The CPU-only version still provides Delaunay "
                     "meshing, but not dense MVS.")
        box = QMessageBox(QMessageBox.Question, "Download COLMAP", text,
                          QMessageBox.NoButton, self)
        b_cuda = box.addButton(f"CUDA (~{DOWNLOAD_MB['cuda']} MB)",
                               QMessageBox.AcceptRole)
        b_cpu = box.addButton(f"CPU-only (~{DOWNLOAD_MB['nocuda']} MB)",
                              QMessageBox.AcceptRole)
        box.addButton(QMessageBox.Cancel)
        box.setDefaultButton(b_cuda if has_gpu else b_cpu)
        box.exec()
        if box.clickedButton() is b_cuda:
            variant = "cuda"
        elif box.clickedButton() is b_cpu:
            variant = "nocuda"
        else:
            return

        from PySide6.QtCore import QThreadPool

        from cloudlabeller.photogrammetry.colmap_fetch import install_colmap
        from cloudlabeller.workers.job import Job

        desc = "Download COLMAP"
        job = Job(install_colmap, variant=variant)
        self._colmap_dl_job = job
        job.signals.progress.connect(self._on_colmap_dl_progress)
        job.signals.finished.connect(lambda path: self._on_colmap_dl_done(job, path))
        job.signals.failed.connect(lambda e: self._on_colmap_dl_failed(job, e))
        self._begin_cancellable(job, desc)
        self.log_panel.append(f"--- COLMAP download started ({variant}, "
                              f"~{DOWNLOAD_MB[variant]} MB) ---")
        self.bus.job_started.emit(desc)
        QThreadPool.globalInstance().start(job)

    def _on_colmap_dl_progress(self, frac: float, msg: str = "") -> None:
        self.bus.job_progress.emit("Download COLMAP", frac)
        if msg:
            self.statusBar().showMessage(msg, 4000)

    def _on_colmap_dl_done(self, job, path) -> None:
        self._colmap_dl_job = None
        self._end_cancellable(job)
        self.bus.job_finished.emit("Download COLMAP", True)
        self.log_panel.append(f"COLMAP installed: {path}")
        self.statusBar().showMessage(
            "COLMAP installed — the GPU/MVS options are now available.", 9000)

    def _on_colmap_dl_failed(self, job, error: str) -> None:
        self._colmap_dl_job = None
        self._end_cancellable(job)
        self.bus.job_finished.emit("Download COLMAP", False)
        self.log_panel.append(f"COLMAP download failed: {error}")
        if "cancelled" not in error.lower():   # user cancel needs no modal
            QMessageBox.warning(
                self, "Download COLMAP",
                f"The download failed:\n{error}\n\n"
                "Check the internet connection and try again via "
                "Photogrammetry → Download COLMAP…")

    def _run_mvs(self) -> None:
        """Run dense MVS on the project's SfM model (out-of-process)."""
        if not self.project:
            QMessageBox.information(self, "No project", "Create or open a project first.")
            return
        if self.project.dataset.cloud is None:
            QMessageBox.information(self, "No reconstruction",
                                    "Run SfM first — dense MVS needs a sparse model.")
            return

        from cloudlabeller.photogrammetry.mvs import find_colmap_binary
        from cloudlabeller.ui.run_mvs_dialog import RunMvsDialog
        from cloudlabeller.workers.process_job import ProcessJob

        colmap_binary = find_colmap_binary(self.config.colmap_binary)
        if colmap_binary is None:
            answer = QMessageBox.question(
                self, "No CUDA COLMAP",
                "Dense MVS needs the CUDA COLMAP executable, which was not "
                "found.\n\nDownload it now (one-time, official COLMAP "
                "release)?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if answer == QMessageBox.Yes:
                self._download_colmap()
                return
        dlg = RunMvsDialog(colmap_binary=colmap_binary,
                           source_resolution=self.project.source_resolution(),
                           parent=self)
        if dlg.exec() != RunMvsDialog.Accepted:
            return
        self._launch_mvs(dlg.max_image_size(), colmap_binary, dlg.quality())

    def _launch_mvs(self, max_image_size: int, colmap_binary: str | None,
                    quality: str = "standard") -> None:
        """Start the dense-MVS child process (also the pipeline's second stage)."""
        from cloudlabeller.workers.process_job import ProcessJob

        desc = "Dense MVS"
        workspace = str(self.project.reconstruction_dir)
        store = str(self.project.images_dir)
        args = [workspace, store, "--max-image-size", str(max_image_size),
                "--quality", quality]
        if colmap_binary:
            args += ["--colmap-binary", colmap_binary]
        self._mvs_job = ProcessJob(
            "cloudlabeller.photogrammetry.mvs_cli", args,
            log_path=self.project.reconstruction_dir / "mvs.log",
        )
        self._mvs_job.progress.connect(lambda f, m="": self.bus.job_progress.emit(desc, f))
        self._mvs_job.log_line.connect(self.log_panel.append)
        self._mvs_job.finished.connect(self._on_mvs_done)
        self._mvs_job.failed.connect(lambda e: self._on_sfm_failed(desc, e))
        self.log_panel.append("--- Dense MVS started ---")
        self.bus.job_started.emit(desc)
        self._pipeline_stage_started(self._mvs_job)
        self._mvs_job.start()

    def _on_mvs_done(self) -> None:
        self.project.load_products()              # picks up dense.npz
        self.bus.job_finished.emit("Dense MVS", True)
        self.bus.cloud_changed.emit()             # enables the Dense view
        ds = self.project.dataset
        n = ds.dense_cloud.n_points if ds.dense_cloud is not None else 0
        self.statusBar().showMessage(f"Dense cloud: {n:,} points", 8000)
        # The mesh is a product of the dense cloud: rebuild it automatically.
        self._create_mesh(auto=True)

    def _create_mesh(self, auto: bool = False) -> None:
        """(Re)build the triangulated mesh from the dense cloud, out-of-process.

        Runs automatically whenever the dense cloud changes; also available via
        Photogrammetry → Create Mesh. ``auto`` softens missing-prerequisite
        handling to log lines (no modal dialogs mid-pipeline).
        """
        if not self.project:
            if not auto:
                QMessageBox.information(self, "No project",
                                        "Create or open a project first.")
            return
        if self.project.dataset.dense_cloud is None \
                or not self.project.dense_cloud_path.exists():
            if auto:
                self.log_panel.append("Mesh: skipped (no dense cloud on disk)")
                self._pipeline_end("complete (mesh skipped: no dense cloud)")
            else:
                QMessageBox.information(self, "No dense cloud",
                                        "Run Dense (MVS) first — meshing "
                                        "needs the dense cloud.")
            return

        from cloudlabeller.photogrammetry.mvs import find_colmap_binary
        from cloudlabeller.workers.process_job import ProcessJob

        dense_ws = self.project.reconstruction_dir / "dense"
        if auto:
            # Post-MVS rebuild / pipeline stage: the pipeline dialog's choice,
            # else the default — Poisson matched to the cloud's density.
            opts = (self._pipeline or {}).get("mesh") or {
                "method": "poisson", "depth": None,
                "trim": 10.0, "point_weight": 1.0}
        else:
            from cloudlabeller.ui.mesh_dialog import CreateMeshDialog

            dialog = CreateMeshDialog(self.project.dataset.dense_cloud,
                                      workspace=dense_ws, parent=self)
            if dialog.exec() != CreateMeshDialog.Accepted:
                return
            opts = dialog.options()

        desc = "Create mesh"
        args = [str(self.project.dense_cloud_path), str(self.project.mesh_path),
                "--method", opts["method"]]
        if opts["method"] == "delaunay":
            args += ["--workspace", str(dense_ws)]
        else:
            if opts.get("depth") is not None:
                args += ["--depth", str(int(opts["depth"]))]
            args += ["--trim", str(opts.get("trim", 10.0)),
                     "--point-weight", str(opts.get("point_weight", 1.0))]
        colmap_binary = find_colmap_binary(self.config.colmap_binary)
        if colmap_binary:
            args += ["--colmap-binary", colmap_binary]
        self._mesh_job = ProcessJob(
            "cloudlabeller.photogrammetry.mesh_cli", args,
            log_path=self.project.reconstruction_dir / "mesh.log",
        )
        self._mesh_job.progress.connect(lambda f, m="": self.bus.job_progress.emit(desc, f))
        self._mesh_job.log_line.connect(self.log_panel.append)
        self._mesh_job.finished.connect(self._on_mesh_done)
        if auto:
            self._mesh_job.failed.connect(
                lambda e: (self.bus.job_finished.emit(desc, False),
                           self.log_panel.append(f"{desc} FAILED: {e}"),
                           self._pipeline_end(f"aborted at {desc}")))
        else:
            self._mesh_job.failed.connect(lambda e: self._on_sfm_failed(desc, e))
        self.log_panel.append("--- Mesh build started ---")
        self.bus.job_started.emit(desc)
        self._pipeline_stage_started(self._mesh_job)
        self._mesh_job.start()

    def _on_mesh_done(self) -> None:
        self.project.load_products()              # picks up products/mesh.npz
        self.bus.job_finished.emit("Create mesh", True)
        self.bus.mesh_changed.emit()              # enables the Mesh view
        mesh = self.project.dataset.mesh
        if mesh is not None:
            self.statusBar().showMessage(
                f"Mesh: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces", 8000)
        self._sync_mesh_labels()                  # labels follow the dense cloud
        self._pipeline_end("complete")

    def _sync_mesh_labels(self) -> None:
        """Recompute the mesh's per-vertex labels from the dense cloud (nearest
        point), off-thread. Called whenever the mesh is (re)built or the cloud
        labels change via a transfer."""
        if not self.project:
            return
        mesh = self.project.dataset.mesh
        dense = self.project.dataset.dense_cloud
        if mesh is None or dense is None:
            return
        import numpy as np

        labels = self.project.labels.dense_cloud_labels
        if labels is None or len(labels) != dense.n_points or not (labels != -1).any():
            # Nothing labelled on the dense cloud: the whole mesh is unlabelled.
            self.project.labels.mesh_labels = np.full(len(mesh.vertices), -1, np.int32)
            self.bus.cloud_labels_changed.emit()
            return

        from cloudlabeller.transfer.mesh_cloud import cloud_to_mesh
        from cloudlabeller.workers.job import Job

        snapshot = labels.copy()                  # stable copy for the worker
        nn_cache = self.project.mesh_nn_cache_path

        def work(progress=None):
            # NN indices are cached: repeat syncs are a single indexing op.
            return cloud_to_mesh(dense, snapshot, mesh, cache_path=nn_cache)

        job = Job(work)
        job.signals.finished.connect(self._on_mesh_labels_done)
        job.signals.failed.connect(
            lambda e: self.log_panel.append(f"Cloud → mesh labels FAILED: {e}"))
        self.log_panel.append(
            f"Transferring labels to the mesh ({len(mesh.vertices):,} vertices)…")
        QThreadPool.globalInstance().start(job)

    def _on_mesh_labels_done(self, mesh_labels) -> None:
        self.project.labels.mesh_labels = mesh_labels
        self._save()                              # persists labels/mesh_labels.npy
        self.bus.cloud_labels_changed.emit()      # mesh view recolours (same slider)
        n = int((mesh_labels != -1).sum())
        self.log_panel.append(f"Mesh labels updated: {n:,} labelled vertices")

    # -- export ------------------------------------------------------------
    def _ask_export_crs(self):
        """Export coordinate spec: a dict for :meth:`_build_export_transform`,
        or None if the user cancelled the dialog.

        No dialog is shown when there is nothing to ask: unreferenced
        projects export the local frame; reprojected projects are already in
        their CRS (the export just adds the stored offset back)."""
        geo = self.project.settings.get("georeferenced")
        if not geo:
            return {"kind": "local"}
        crs_info = geo.get("crs")
        if crs_info:
            return {"kind": "offset", "epsg": crs_info["epsg"],
                    "orthometric": crs_info.get("orthometric", False),
                    "offset": crs_info.get("offset") or [0.0, 0.0, 0.0]}
        from cloudlabeller.ui.export_crs_dialog import ExportCrsDialog

        origin = self._resolved_enu_origin()
        dlg = ExportCrsDialog(origin.lla, geo, parent=self)
        if not dlg.exec():
            return None
        choice = dlg.choice()
        if choice.mode == "projected":
            geo.update(export_mode="projected", export_epsg=choice.epsg,
                       export_orthometric=choice.orthometric)
            self.project.save_manifest()
            return {"kind": "enu", "epsg": choice.epsg,
                    "orthometric": choice.orthometric, "origin": origin.lla}
        geo["export_mode"] = "local"
        self.project.save_manifest()
        return {"kind": "local"}

    @staticmethod
    def _build_export_transform(spec, progress):
        """(transform, crs) for an export work fn; (None, None) = local frame.
        Runs on the worker thread — may download the geoid grid once."""
        if spec["kind"] == "local":
            return None, None
        import numpy as np
        import pyproj

        from cloudlabeller.photogrammetry.crs import (
            _compound_crs,
            crs_display,
            download_geoid_grids,
            enu_to_crs_transform,
            geoid_ready,
        )

        epsg, orthometric = spec["epsg"], spec["orthometric"]
        if spec["kind"] == "offset":
            # Stored coordinates are already in the CRS, minus the offset.
            offset = np.asarray(spec["offset"], np.float64)
            crs = (_compound_crs(epsg) if orthometric
                   else pyproj.CRS.from_epsg(int(epsg)))
            transform = lambda xyz: np.asarray(xyz, np.float64) + offset  # noqa: E731
        else:
            if orthometric and not geoid_ready(epsg):
                progress(0.01, "Downloading the EGM96 geoid grid "
                               "(~3 MB, one-time)…")
                download_geoid_grids(epsg)
            transform, crs = enu_to_crs_transform(spec["origin"], epsg,
                                                  orthometric)
        progress(0.02, f"Coordinates: {crs_display(epsg)}"
                       + (" + EGM96 heights" if orthometric else ""))
        return transform, crs

    def _export_cloud(self) -> None:
        """Export the dense cloud (or sparse, if no dense yet) with labels."""
        if not self.project:
            QMessageBox.information(self, "No project", "Create or open a project first.")
            return
        ds, lb = self.project.dataset, self.project.labels
        if ds.dense_cloud is not None:
            cloud, labels, kind = ds.dense_cloud, lb.dense_cloud_labels, "dense"
        elif ds.cloud is not None:
            cloud, labels, kind = ds.cloud, lb.cloud_labels, "sparse"
        else:
            QMessageBox.information(self, "No cloud", "Run SfM first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, f"Export {kind} cloud ({cloud.n_points:,} points)", "",
            "LAS point cloud (*.las);;CSV (*.csv);;PLY (*.ply)")
        if not path:
            return
        if not path.lower().endswith((".las", ".csv", ".ply")):
            path += ".las"
        spec = self._ask_export_crs()
        if spec is None:
            return
        labels_copy = None if labels is None else labels.copy()   # thread snapshot

        def work(progress=None):
            progress = progress or (lambda f, m="": None)
            from cloudlabeller.io.export import export_cloud

            transform, crs = self._build_export_transform(spec, progress)
            progress(0.0, f"Exporting {kind} cloud ({cloud.n_points:,} points) "
                          f"to {path}")
            export_cloud(cloud, labels_copy, path, progress,
                         transform=transform, crs=crs)
            return path

        self._start_transfer_job("Export cloud", work, self._on_export_done)

    def _export_mesh(self) -> None:
        if not self.project or self.project.dataset.mesh is None:
            QMessageBox.information(self, "No mesh",
                                    "Create the mesh first (Photogrammetry → Create Mesh).")
            return
        mesh = self.project.dataset.mesh
        path, _ = QFileDialog.getSaveFileName(
            self, f"Export mesh ({len(mesh.faces):,} faces)", "",
            "PLY mesh (*.ply);;OBJ (*.obj);;STL (*.stl)")
        if not path:
            return
        if not path.lower().endswith((".ply", ".obj", ".stl")):
            path += ".ply"
        spec = self._ask_export_crs()
        if spec is None:
            return

        def work(progress=None):
            progress = progress or (lambda f, m="": None)
            from cloudlabeller.io.export import export_mesh

            transform, crs = self._build_export_transform(spec, progress)
            progress(0.0, f"Exporting mesh to {path}")
            export_mesh(mesh, path, progress, transform=transform, crs=crs)
            return path

        self._start_transfer_job("Export mesh", work, self._on_export_done)

    def _on_export_done(self, path) -> None:
        self.bus.job_finished.emit("Export", True)
        msg = f"Exported: {path}"
        self.log_panel.append(msg)
        self.statusBar().showMessage(msg, 8000)

    # -- transfer job plumbing (in-thread; logs steps to the Log pane) -----
    def _start_transfer_job(self, desc, work, on_done, cancellable: bool = False) -> None:
        """Run ``work`` on the thread pool with standard progress/log/error
        plumbing — used by transfers, exports and other in-process jobs."""
        from cloudlabeller.workers.job import Job

        job = Job(work)
        job.signals.progress.connect(lambda f, m="": self._on_transfer_progress(desc, f, m))
        job.signals.finished.connect(lambda result: (self._end_cancellable(job),
                                                     on_done(result)))
        job.signals.failed.connect(lambda e: (self._end_cancellable(job),
                                              self._on_transfer_failed(desc, e)))
        if cancellable:
            self._begin_cancellable(job, desc)
        self.log_panel.append(f"--- {desc} started ---")
        self.bus.job_started.emit(desc)
        QThreadPool.globalInstance().start(job)

    def _begin_cancellable(self, job, desc: str, on_stop=None) -> None:
        """Show the status-bar Stop button. ``on_stop`` defaults to the job's
        cancel flag; training passes a stop-file toucher instead so the child
        process winds down gracefully (finishes the epoch, still saves)."""
        self._active_job = job
        stop_action = on_stop or job.cancel
        if getattr(self, "_stop_slot", None) is not None:
            self.btn_stop.clicked.disconnect(self._stop_slot)

        def request_stop() -> None:
            stop_action()
            self.btn_stop.setEnabled(False)         # one shot; job winds down
            self.log_panel.append(f"{desc}: stop requested — finishing up…")

        self._stop_slot = request_stop
        self.btn_stop.clicked.connect(request_stop)
        self.btn_stop.setToolTip(f"Stop {desc} (finishes the current step, "
                                 "results are kept)")
        self.btn_stop.setEnabled(True)
        self.btn_stop.show()

    def _end_cancellable(self, job) -> None:
        if self._active_job is job:
            self._active_job = None
            self.btn_stop.hide()

    def _on_transfer_progress(self, desc, frac, msg) -> None:
        self.bus.job_progress.emit(desc, frac)
        if msg:
            self.log_panel.append(msg)

    def _on_transfer_failed(self, desc, error) -> None:
        self.bus.job_finished.emit(desc, False)
        self.log_panel.append(f"{desc} FAILED: {error}")
        QMessageBox.critical(self, f"{desc} failed", error)

    def _images_to_cloud(self) -> None:
        """Project image labels onto the sparse cloud AND the dense cloud (if any),
        in one command (majority vote)."""
        if not self.project or self.project.dataset.cloud is None:
            QMessageBox.information(self, "No cloud", "Run SfM and label some images first.")
            return
        from cloudlabeller.transfer import images_to_cloud
        from cloudlabeller.transfer.project_to_cloud import MaskSource

        project = self.project
        dataset = project.dataset
        labels = project.labels
        sparse, dense = dataset.cloud, dataset.dense_cloud
        masks = dict(labels.image_masks)                       # snapshot for the thread
        n_classes = len(project.schema.classes)
        n_labelled = sum(1 for m in masks.values() if (m != -1).any())
        vis = project.visibility_index()                       # cached visibility

        # Offer to include the auto-labelled (yellow) images: U-Net predictions
        # carry new information onto the cloud; cloud projections just re-affirm
        # current cloud labels in the vote.
        auto_names: list[str] = [
            r.name for r in dataset.solved_images()
            if r.camera is not None and r.name not in masks
            and labels.status_of(r.name) in ("auto", "ml")]
        if auto_names:
            n_ml = sum(1 for n in auto_names if labels.status_of(n) == "ml")
            answer = QMessageBox.question(
                self, "Include auto-labelled images?",
                f"Besides the {n_labelled} user-labelled image(s), "
                f"{len(auto_names)} image(s) are auto-labelled "
                f"({n_ml} by the U-Net, {len(auto_names) - n_ml} by cloud "
                "projection).\n\nInclude their labels in the transfer?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if answer != QMessageBox.Yes:
                auto_names = []
        if not auto_names and n_labelled == 0:
            QMessageBox.information(self, "No labelled images",
                                    "Draw some polygons (or predict labels) first.")
            return
        # Auto masks are rendered lazily inside the transfer (one at a time) —
        # materialising them all up-front is ~84 MB per image. User-drawn masks
        # are HARD data: their votes outrank auto/ML ones wherever they reach.
        hard_names = set(masks)
        # Cloud-projection ("auto") images all render the SAME cloud labels —
        # only their coverage matters, so a covering subset (overlap graph)
        # casts the same votes at a fraction of the renders. U-Net ("ml")
        # predictions are independent evidence: all of them keep voting.
        pure_auto = [n for n in auto_names if labels.status_of(n) == "auto"]
        ml_names = [n for n in auto_names if labels.status_of(n) == "ml"]

        def work(progress=None):
            progress = progress or (lambda f, m="": None)
            selected = auto_names
            if pure_auto:
                adj = project.image_adjacency()
                if adj is not None:
                    from cloudlabeller.photogrammetry.adjacency import select_covering

                    covering = select_covering(pure_auto, adj)
                    if len(covering) < len(pure_auto):
                        progress(0.0, f"  {len(covering)} of {len(pure_auto)} "
                                      "cloud-projection images cover the same "
                                      "surface (overlap graph) — skipping the rest")
                    selected = ml_names + covering
            mask_source = MaskSource(masks, selected, project.render_auto_mask)
            progress(0.0, f"Transferring labels from "
                          f"{n_labelled + len(selected)} images to the cloud")
            out = {}
            progress(0.02, f"Labelling sparse cloud ({sparse.n_points:,} points)…")
            out["sparse"] = images_to_cloud(
                sparse, dataset, mask_source, n_classes,
                lambda f, m="": progress(0.02 + 0.46 * f, ""), visibility=vis,
                priority_names=hard_names)
            if dense is not None:
                progress(0.5, f"Labelling dense cloud ({dense.n_points:,} points)…")
                out["dense"] = images_to_cloud(
                    dense, dataset, mask_source, n_classes,
                    lambda f, m="": progress(0.5 + 0.48 * f, ""), visibility=vis,
                    priority_names=hard_names)
            progress(1.0, "Images → cloud complete")
            return out

        self._start_transfer_job("Images → cloud", work, self._on_images_to_cloud_done)

    def _on_images_to_cloud_done(self, out) -> None:
        """Fold the projected labels into the cloud's label arrays.

        MERGE, not replace: points without image votes keep whatever they
        had (e.g. lasso-assigned labels) — a wholesale replace here used to
        wipe lasso work.
        """
        labels = self.project.labels
        ns = labels.merge_cloud_labels(out["sparse"])
        msg = f"Updated {ns:,} sparse points"
        if "dense" in out:
            nd = labels.merge_cloud_labels(out["dense"], dense=True)
            msg += f" + {nd:,} dense points"
        msg += " from images (other labels kept)"
        self._save()
        self.bus.job_finished.emit("Images → cloud", True)
        self.bus.cloud_labels_changed.emit()
        self.log_panel.append(msg)
        self.statusBar().showMessage(msg, 8000)
        self._sync_mesh_labels()                  # keep the mesh in step

    def _cloud_to_images(self) -> None:
        """Project cloud labels onto images (auto-labelling). Uses the dense cloud
        when it's labelled (solid masks), else the sparse cloud. User-drawn images
        are left untouched."""
        if not self.project or self.project.dataset.cloud is None:
            QMessageBox.information(self, "No cloud", "Run SfM first.")
            return
        labels = self.project.labels
        dense, dense_lab = self.project.dataset.dense_cloud, labels.dense_cloud_labels
        sparse, sparse_lab = self.project.dataset.cloud, labels.cloud_labels
        if dense is not None and dense_lab is not None and (dense_lab != -1).any():
            src_cloud, src_labels, src_name = dense, dense_lab, "dense"
        elif sparse_lab is not None and (sparse_lab != -1).any():
            src_cloud, src_labels, src_name = sparse, sparse_lab, "sparse"
        else:
            QMessageBox.information(
                self, "Cloud not labelled",
                "Label the cloud first (draw on images, then Transfer → Images → Cloud).")
            return

        from cloudlabeller.transfer import cloud_to_image

        targets = [(r.name, r.camera) for r in self.project.dataset.solved_images()
                   if r.camera is not None and labels.status_of(r.name) != "user"]
        vis = self.project.visibility_index()                  # cached visibility

        def work(progress=None):
            progress = progress or (lambda f, m="": None)
            progress(0.0, f"Projecting {src_name} cloud onto {len(targets)} images")
            # Auto masks are NOT stored (they are re-rendered on demand from the
            # cloud); we only count labelled pixels to decide the image status.
            counts = {}
            for i, (name, cam) in enumerate(targets, 1):
                mask = cloud_to_image(src_cloud, src_labels, cam,
                                      name=name, visibility=vis)
                counts[name] = int((mask != -1).sum())
                progress(i / max(len(targets), 1),
                         f"  {name}: {counts[name]:,} px ({i}/{len(targets)})")
            return counts

        self._start_transfer_job("Cloud → images", work, self._on_cloud_to_images_done)

    def _on_cloud_to_images_done(self, counts, desc: str = "Cloud → images",
                                 min_px: int = 1) -> None:
        """Mark images auto-labelled by their received-pixel count. No masks are
        materialised — the Image pane renders auto overlays from the cloud."""
        n = 0
        for name, n_px in counts.items():
            if n_px >= min_px:
                self.project.labels.mark_auto_labeled(name)      # -> yellow dot
                n += 1
        try:
            self.project.save_manifest()                         # statuses only: cheap
        except Exception:
            pass
        self.bus.job_finished.emit(desc, True)
        for name in counts:
            self.bus.image_labels_changed.emit(name)             # refresh dots + overlay
        msg = f"Auto-labelled {n} images from the cloud"
        self.log_panel.append(msg)
        self.statusBar().showMessage(msg, 8000)

    # Ignore neighbours that receive fewer labelled pixels than this — a handful
    # of stray points shouldn't flip an image's status to "auto".
    PROPAGATE_MIN_PX = 500

    def _propagate_image(self, name: str) -> None:
        """Spread one image's labels, via the 3D cloud, to the images that see
        the same area. The projected labels are merged into the stored cloud
        labels (only where this image labelled; nothing is erased) — they are
        what the neighbours' auto overlays render from."""
        if not self.project:
            return
        labels = self.project.labels
        mask = labels.image_masks.get(name)
        if mask is None or not (mask != -1).any():
            QMessageBox.information(self, "No labels",
                                    f"{name} has no labels to propagate — draw "
                                    "some polygons first.")
            return
        ds = self.project.dataset
        source = ds.image_by_name(name)
        if source is None or source.camera is None:
            QMessageBox.information(self, "Not registered",
                                    f"{name} has no solved camera (re-run SfM).")
            return
        cloud = ds.dense_cloud if ds.dense_cloud is not None else ds.cloud
        if cloud is None:
            QMessageBox.information(self, "No cloud", "Run SfM first.")
            return

        from cloudlabeller.core.dataset import Dataset
        from cloudlabeller.transfer import cloud_to_image, images_to_cloud

        kind = "dense" if cloud is ds.dense_cloud else "sparse"
        n_classes = len(self.project.schema.classes)
        targets = [(r.name, r.camera) for r in ds.solved_images()
                   if r.camera is not None and r.name != name
                   and labels.status_of(r.name) != "user"]
        desc = f"Propagate {name}"
        vis = self.project.visibility_index()                  # cached visibility

        def work(progress=None):
            progress = progress or (lambda f, m="": None)
            # Restrict the candidates to images that actually see surface in
            # common with the source (COLMAP covisibility; may be derived and
            # cached on first use for older projects). None = unknown -> all.
            progress(0.0, "Finding images overlapping " + name + "…")
            overlap = self.project.overlapping_images(name)
            cand = (targets if overlap is None
                    else [(nm, cam) for nm, cam in targets if nm in overlap])
            skipped = len(targets) - len(cand)
            progress(0.02, f"Projecting {name} onto the {kind} cloud"
                           f" ({cloud.n_points:,} points)…")
            temp = images_to_cloud(cloud, Dataset(images=[source]), {name: mask},
                                   n_classes, visibility=vis)
            n_pts = int((temp != -1).sum())
            progress(0.15, f"  {n_pts:,} cloud points labelled; projecting onto "
                           f"{len(cand)} overlapping images"
                           + (f" ({skipped} non-overlapping skipped)…"
                              if skipped else "…"))
            counts = {}
            for i, (nm, cam) in enumerate(cand, 1):
                m = cloud_to_image(cloud, temp, cam, name=nm, visibility=vis)
                counts[nm] = int((m != -1).sum())
                progress(0.15 + 0.85 * i / max(len(cand), 1),
                         f"  {nm}: {counts[nm]:,} px ({i}/{len(cand)})")
            return {"labels": temp, "counts": counts, "kind": kind}

        self._start_transfer_job(desc, work,
                                 lambda result: self._on_propagate_done(result, desc))

    def _on_propagate_done(self, result, desc: str) -> None:
        labels = self.project.labels
        n = labels.merge_cloud_labels(result["labels"],
                                      dense=(result["kind"] == "dense"))
        if n:                                      # merge (undoable), erase nothing
            self.bus.cloud_labels_changed.emit()
        self._on_cloud_to_images_done(result["counts"], desc=desc,
                                      min_px=self.PROPAGATE_MIN_PX)
        self._save()                               # cloud labels changed (npy is small)
        if n:
            self._sync_mesh_labels()               # propagate merges into the cloud too
    def _configure_model(self) -> None:
        """Edit and persist the U-Net model settings (:class:`ModelSpec`)."""
        if not self.project:
            QMessageBox.information(self, "No project", "Create or open a project first.")
            return
        from cloudlabeller.ui.model_spec_dialog import ModelSpecDialog

        dialog = ModelSpecDialog(
            spec=self.project.model_spec(),
            source_resolution=self.project.source_resolution(),
            parent=self,
        )
        if dialog.exec() != ModelSpecDialog.Accepted:
            return
        self.project.set_model_spec(dialog.spec())
        self.project.save_manifest()
        self.statusBar().showMessage("Model settings saved.", 4000)

    # -- U-Net training / prediction ----------------------------------------
    def _train(self) -> None:
        """Train the U-Net on the labelled images and save it under a name."""
        if not self.project:
            QMessageBox.information(self, "No project", "Create or open a project first.")
            return
        project = self.project
        if not project.schema.classes:
            QMessageBox.information(self, "No classes", "Define some label classes first.")
            return
        source_res = project.source_resolution()
        if source_res is None:
            QMessageBox.information(self, "No images", "Add images to the project first.")
            return
        spec = project.model_spec()

        labels = project.labels
        user_names = [n for n, m in labels.image_masks.items()
                      if labels.status_of(n) == "user" and (m != -1).any()]
        # Cloud-projected images can supplement training (rendered on demand);
        # 'ml'-status images are excluded — never train on the model's own output.
        auto_names = [r.name for r in project.dataset.solved_images()
                      if r.camera is not None and labels.status_of(r.name) == "auto"]

        from cloudlabeller.ml.model_store import list_models
        from cloudlabeller.ui.train_dialog import TrainDialog

        dialog = TrainDialog(len(user_names), len(auto_names),
                             spec.training_size(*source_res),
                             manifests=list_models(project.ml_dir),
                             n_classes=len(project.schema.classes), parent=self)
        if dialog.exec() != TrainDialog.Accepted:
            return
        opts = dialog.options()
        if not opts["resume"]:
            # Fresh models are built from the current Model Settings; resumed
            # ones carry their own architecture, so no validation needed there.
            error = spec.validate_size(*source_res)
            if error:
                QMessageBox.warning(self, "Model settings invalid",
                                    error + "\n\nFix it in Model → Model Settings…")
                return

        # TensorFlow runs in a CHILD PROCESS (like COLMAP): in-process TF on a
        # QThread deadlocked the GIL against Qt's event loop. The child reads
        # the project from disk, so persist labels/statuses first.
        self._save()
        from cloudlabeller.workers.process_job import ProcessJob

        desc = "Train U-Net"
        stop_file = project.ml_dir / "train.stop"
        stop_file.unlink(missing_ok=True)
        cli_args = [str(project.root),
                    "--name", opts["name"],
                    "--epochs", str(opts["epochs"]),
                    "--batch-size", str(opts["batch_size"]),
                    "--val-fraction", str(opts["val_fraction"]),
                    "--augment-rolls", str(opts["augment_rolls"]),
                    "--stop-file", str(stop_file)]
        if opts["include_auto"]:
            cli_args.append("--include-auto")
        if opts["resume"]:
            cli_args.append("--resume")

        log_path = project.ml_dir / "train.log"
        job = ProcessJob("cloudlabeller.ml.train_cli", cli_args, log_path=log_path)
        self._train_job = job
        job.log_line.connect(self.log_panel.append)
        job.progress.connect(lambda f, m="": self.bus.job_progress.emit(desc, f))
        job.finished.connect(lambda: self._on_train_done(job, stop_file))
        job.failed.connect(lambda e: self._on_ml_failed(desc, e, job, stop_file))
        # Graceful stop: the child polls the stop file each epoch and still
        # saves the model (a hard kill would lose the weights).
        self._begin_cancellable(job, desc,
                                on_stop=lambda: stop_file.touch(exist_ok=True))
        self.log_panel.append(f"--- {desc} started (log: {log_path}) ---")
        self.bus.job_started.emit(desc)
        job.start()

    def _on_train_done(self, job, stop_file) -> None:
        """Clean up the stop-file plumbing and report the training summary."""
        self._end_cancellable(job)
        stop_file.unlink(missing_ok=True)
        self.bus.job_finished.emit("Train U-Net", True)
        out = self._job_result(job)
        if not out:
            self.log_panel.append("Training finished (no summary reported).")
            return
        summary = ", ".join(f"{k}={v:.4f}" for k, v in out["metrics"].items())
        verb = ("resumed and trained for another" if out.get("resumed")
                else "trained for")
        msg = (f"Model '{out['name']}' {verb} {out['epochs_run']} epoch(s) on "
               f"{out['n_train']} samples ({out['n_val']} validation) — {summary}")
        self.log_panel.append(msg)
        self.statusBar().showMessage(f"Model '{out['name']}' saved.", 8000)

    def _on_ml_failed(self, desc, error, job, stop_file=None) -> None:
        self._end_cancellable(job)
        if stop_file is not None:
            stop_file.unlink(missing_ok=True)
        self._on_transfer_failed(desc, error)

    @staticmethod
    def _job_result(job) -> dict | None:
        """Parse a ProcessJob's ``RESULT <json>`` line (or None)."""
        import json

        text = job.result_text.partition(" ")[2].strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def _predict_all(self) -> None:
        """Predict masks for the red/yellow images with a saved model. Green
        (user-labelled) images are never overwritten."""
        if not self.project:
            QMessageBox.information(self, "No project", "Create or open a project first.")
            return
        project = self.project
        from cloudlabeller.ml.model_store import list_models
        from cloudlabeller.ui.load_model_dialog import LoadModelDialog

        manifests = list_models(project.ml_dir)
        if not manifests:
            QMessageBox.information(self, "No models",
                                    "No saved models yet — run Model → Train U-Net… first.")
            return
        dialog = LoadModelDialog(manifests, title="Predict All Images",
                                 ml_dir=project.ml_dir, parent=self)
        if dialog.exec() != LoadModelDialog.Accepted or not dialog.selected_name():
            return
        model_name = dialog.selected_name()
        manifest = next(m for m in manifests if m["name"] == model_name)
        if manifest["n_classes"] != len(project.schema.classes):
            QMessageBox.warning(
                self, "Class mismatch",
                f"Model '{model_name}' was trained with {manifest['n_classes']} "
                f"classes but the project now has {len(project.schema.classes)} — "
                "its predictions would map to the wrong labels.\n"
                "Retrain the model with the current schema.")
            return

        labels = project.labels
        targets = [r.name for r in project.dataset.images
                   if labels.status_of(r.name) != "user"]    # red + yellow only
        if not targets:
            QMessageBox.information(self, "Nothing to predict",
                                    "All images are user-labelled (green).")
            return

        # TensorFlow runs in a child process (see _train). The child reads
        # image statuses from the manifest on disk, so persist them first.
        self.project.save_manifest()
        from cloudlabeller.workers.process_job import ProcessJob

        desc = "Predict images"
        log_path = project.ml_dir / "predict.log"
        job = ProcessJob("cloudlabeller.ml.predict_cli",
                         [str(project.root), "--model", model_name],
                         log_path=log_path)
        self._predict_job = job
        job.log_line.connect(self.log_panel.append)
        job.progress.connect(lambda f, m="": self.bus.job_progress.emit(desc, f))
        job.finished.connect(lambda: self._on_predict_done(job))
        job.failed.connect(lambda e: self._on_ml_failed(desc, e, job))
        self.log_panel.append(f"--- {desc} started (log: {log_path}) ---")
        self.bus.job_started.emit(desc)
        job.start()

    def _on_predict_done(self, job) -> None:
        """Mark the predicted images as auto-labelled (yellow dots) and
        refresh their overlays; user-drawn masks were preserved by the CLI."""
        out = self._job_result(job) or {}
        predicted_names = out.get("predicted", [])
        labels = self.project.labels
        for name in predicted_names:
            labels.mark_ml_labeled(name)                      # -> yellow dot
        try:
            self.project.save_manifest()                      # statuses only: cheap
        except Exception:
            pass
        self.bus.job_finished.emit("Predict images", True)
        for name in predicted_names:
            self.bus.image_labels_changed.emit(name)          # dots + overlay
        msg = f"U-Net labelled {len(predicted_names)} images (user-drawn kept)"
        self.log_panel.append(msg)
        self.statusBar().showMessage(msg, 8000)

    def _show_about(self) -> None:
        from cloudlabeller.ui.about_dialog import AboutDialog

        AboutDialog(self).exec()

    def _show_project_info(self) -> None:
        if not self.project:
            QMessageBox.information(self, "No project", "Create or open a project first.")
            return
        from cloudlabeller.ui.project_info_dialog import ProjectInfoDialog

        ProjectInfoDialog(self.project, parent=self).exec()

    def eventFilter(self, obj, event) -> bool:
        from PySide6.QtCore import QEvent

        if (obj is self.status_info and event.type() == QEvent.MouseButtonRelease
                and self.project is not None):
            self._show_project_info()
            return True
        return super().eventFilter(obj, event)

    def _undo(self) -> None:
        if self.project:
            self._refresh_after_edit(self.project.labels.undo())

    def _redo(self) -> None:
        if self.project:
            self._refresh_after_edit(self.project.labels.redo())

    def _refresh_after_edit(self, changed) -> None:
        """Refresh the view affected by an undo/redo ((modality, key) or None)."""
        if changed is None:
            return
        from cloudlabeller.core.labels import Modality

        modality, key = changed
        if modality is Modality.IMAGE and key:
            self.bus.image_labels_changed.emit(key)
        else:
            self.bus.cloud_labels_changed.emit()
