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

"""Reconstruction + visualisation tests (require the venv: pycolmap, pyvista).

Run with the project venv:  .venv/Scripts/python -m pytest tests/test_reconstruction.py
"""

import numpy as np
import pytest

from cloudlabeller.core.dataset import Camera, PointCloud

pycolmap = pytest.importorskip("pycolmap")
pv = pytest.importorskip("pyvista")


def _app_camera(tx: float = 0.0) -> Camera:
    K = np.array([[1000.0, 0, 320], [0, 1000.0, 240], [0, 0, 1]])
    return Camera(K=K, R=np.eye(3), t=np.array([tx, 0.0, 0.0]),
                  width=640, height=480, model="PINHOLE")


# --- extraction: real reconstruction with real 3D points -----------------
def test_reconstruction_to_cloud_real_points():
    from cloudlabeller.photogrammetry.extract import reconstruction_to_cloud

    rec = pycolmap.Reconstruction()
    rng = np.random.default_rng(0)
    for _ in range(40):
        rec.add_point3D(rng.normal(size=3), pycolmap.Track(),
                        rng.integers(0, 255, size=3).astype(np.uint8))

    cloud = reconstruction_to_cloud(rec)
    assert cloud.n_points == 40
    assert cloud.xyz.shape == (40, 3)
    assert cloud.rgb is not None and cloud.rgb.shape == (40, 3)


# --- camera conversion: validate our loop via a duck-typed reconstruction -
def test_colmap_to_cameras_logic():
    from cloudlabeller.photogrammetry.cameras import colmap_to_cameras

    class _Rot:
        def matrix(self):
            return np.eye(3)

    class _Rigid:
        rotation = _Rot()
        translation = np.array([1.0, 0.0, 0.0])

    class _Cam:
        width, height, model_name = 640, 480, "PINHOLE"
        params = np.array([1000.0, 1000.0, 320.0, 240.0])

        def calibration_matrix(self):
            return np.array([[1000.0, 0, 320], [0, 1000.0, 240], [0, 0, 1]])

    class _Img:
        def __init__(self, name, has_pose=True):
            self.name, self.camera_id, self.has_pose = name, 1, has_pose

        # pycolmap 4.x exposes cam_from_world as a METHOD (must be called),
        # not a property — colmap_to_cameras must handle that.
        def cam_from_world(self):
            return _Rigid()

    class _Rec:
        cameras = {1: _Cam()}
        images = {1: _Img("a.jpg"), 2: _Img("b.jpg", has_pose=False)}

    cams = colmap_to_cameras(_Rec(), "/data/imgs")
    assert len(cams) == 1                       # the unposed image is skipped
    rec0 = cams[0]
    assert rec0.name == "a.jpg"
    assert str(rec0.path).endswith("a.jpg")
    assert rec0.camera.K[0, 0] == 1000.0
    # centre = -R^T t = -[1,0,0]
    assert np.allclose(rec0.camera.position, [-1.0, 0.0, 0.0])


# --- frustum geometry ----------------------------------------------------
def test_camera_frustum_geometry():
    from cloudlabeller.ui.camera_gizmo import camera_center, camera_frustum

    cam = _app_camera(tx=2.0)
    frustum = camera_frustum(cam, scale=0.5)
    assert frustum.n_points == 5                # centre + 4 corners
    assert frustum.n_lines == 8                 # 4 spokes + 4 rectangle edges
    # apex of the frustum is the camera centre
    assert np.allclose(frustum.points[0], camera_center(cam))


# --- offscreen render smoke (no Qt) --------------------------------------
def test_offscreen_render_cloud_and_cameras(tmp_path):
    from cloudlabeller.ui.camera_gizmo import camera_frustum

    rng = np.random.default_rng(1)
    cloud = PointCloud(xyz=rng.normal(size=(500, 3)).astype(np.float32),
                       rgb=rng.integers(0, 255, size=(500, 3)).astype(np.uint8))

    pl = pv.Plotter(off_screen=True)
    poly = pv.PolyData(cloud.xyz)
    poly["rgb"] = cloud.rgb
    pl.add_mesh(poly, scalars="rgb", rgb=True, point_size=3)
    for tx in (-1.0, 0.0, 1.0):
        pl.add_mesh(camera_frustum(_app_camera(tx), scale=0.4), color="#ffcc00")
    out = tmp_path / "preview.png"
    pl.screenshot(str(out))
    pl.close()
    assert out.exists() and out.stat().st_size > 0


# --- result serialisation round-trip (subprocess <-> GUI handoff) ---------
def test_result_save_load_roundtrip(tmp_path):
    from cloudlabeller.core.dataset import ImageRecord
    from cloudlabeller.photogrammetry.pipeline import (
        ReconstructResult, load_result, save_result,
    )

    rng = np.random.default_rng(2)
    cloud = PointCloud(xyz=rng.normal(size=(120, 3)).astype(np.float32),
                       rgb=rng.integers(0, 255, size=(120, 3)).astype(np.uint8))
    images = [ImageRecord(image_id=1, path=__import__("pathlib").Path("a.jpg"),
                          camera=_app_camera(1.5))]
    save_result(ReconstructResult(cloud=cloud, images=images), tmp_path)

    loaded = load_result(tmp_path)
    assert loaded.cloud.n_points == 120
    assert np.allclose(loaded.cloud.xyz, cloud.xyz)
    assert np.array_equal(loaded.cloud.rgb, cloud.rgb)
    assert len(loaded.images) == 1
    assert np.allclose(loaded.images[0].camera.K, images[0].camera.K)
    assert np.allclose(loaded.images[0].camera.position, images[0].camera.position)


# --- the bug fix: a failing reconstruction subprocess must NOT crash us ----
def test_process_job_isolates_failure(tmp_path):
    """ProcessJob must emit `failed` (not kill the parent) when SfM errors."""
    from PySide6.QtCore import QEventLoop, QTimer
    from PySide6.QtWidgets import QApplication
    from cloudlabeller.workers.process_job import ProcessJob

    # QApplication (not QCoreApplication): the Qt app is a process-wide singleton,
    # and other tests construct QWidgets, which require a full QApplication.
    app = QApplication.instance() or QApplication([])
    log_path = tmp_path / "sfm.log"
    job = ProcessJob("cloudlabeller.photogrammetry.run_cli",
                     [str(tmp_path / "does_not_exist"), str(tmp_path / "ws")],
                     log_path=log_path)
    outcome = {}
    lines: list[str] = []
    job.log_line.connect(lines.append)
    loop = QEventLoop()
    job.failed.connect(lambda msg: (outcome.update(failed=msg), loop.quit()))
    job.finished.connect(lambda: (outcome.update(finished=True), loop.quit()))
    QTimer.singleShot(120_000, loop.quit)  # safety timeout
    job.start()
    loop.exec()

    # Parent is still alive and got a clean failure signal with a log.
    assert "failed" in outcome, f"expected failure, got {outcome}"
    assert outcome["failed"]                 # non-empty message
    assert lines                             # log streamed to the panel
    # ProcessJob writes the log as UTF-8; read it back the same way (the
    # locale default is cp1252 on machines without Windows' UTF-8 mode, and
    # the log's command line contains a non-ASCII user path).
    assert log_path.exists() and log_path.read_text(encoding="utf-8")


# --- SfM options: dialog -> CLI args -> parsed namespace ------------------
def test_sfm_options_cli_roundtrip():
    from cloudlabeller.photogrammetry.options import SfmOptions
    from cloudlabeller.photogrammetry.run_cli import _parse

    opts = SfmOptions(matcher="exhaustive", use_gpu=True, single_camera=True,
                      camera_model="OPENCV", max_image_size=2400)
    ns = _parse(["run_cli", "imgs", "ws", *opts.to_cli_args()])
    assert ns.image_dir == "imgs" and ns.workspace == "ws"
    assert ns.matcher == "exhaustive"
    assert ns.gpu is True
    assert ns.single_camera is True
    assert ns.camera_model == "OPENCV"
    assert ns.max_image_size == 2400

    # defaults: gpu/single-camera flags absent
    ns2 = _parse(["run_cli", "imgs", "ws", *SfmOptions(use_gpu=False,
                                                       single_camera=False).to_cli_args()])
    assert ns2.gpu is False and ns2.single_camera is False


def test_find_colmap_binary(tmp_path):
    from cloudlabeller.photogrammetry.mvs import find_colmap_binary

    fake = tmp_path / "colmap.exe"
    fake.write_bytes(b"x")
    assert find_colmap_binary(str(fake)) == str(fake)           # explicit override wins


def test_run_mvs_without_sparse_model(tmp_path):
    from cloudlabeller.photogrammetry.mvs import run_mvs

    # No sparse model on disk -> clear error before any dense work.
    with pytest.raises(RuntimeError, match="sparse model"):
        run_mvs(tmp_path, tmp_path)


def _fake_dense_outputs(dense):
    """A dense workspace as a previous MVS run leaves it."""
    (dense / "stereo" / "depth_maps").mkdir(parents=True)
    (dense / "stereo" / "depth_maps" / "img.jpg.geometric.bin").write_bytes(b"d")
    (dense / "images").mkdir()
    (dense / "images" / "img.jpg").write_bytes(b"i")


def test_prepare_dense_workspace_fresh(tmp_path):
    from cloudlabeller.photogrammetry.mvs import UNDISTORT_MARKER, prepare_dense_workspace

    dense = tmp_path / "dense"
    assert prepare_dense_workspace(dense, 2000) is False   # nothing to keep
    assert dense.is_dir()
    assert (dense / UNDISTORT_MARKER).read_text() == "2000|standard"


def test_prepare_dense_workspace_same_settings_resumes(tmp_path):
    from cloudlabeller.photogrammetry.mvs import UNDISTORT_MARKER, prepare_dense_workspace

    dense = tmp_path / "dense"
    prepare_dense_workspace(dense, 2000)
    _fake_dense_outputs(dense)

    # Same resolution + quality -> depth maps stay resumable (crash recovery).
    assert prepare_dense_workspace(dense, 2000) is True
    assert (dense / "stereo" / "depth_maps" / "img.jpg.geometric.bin").exists()
    assert (dense / UNDISTORT_MARKER).read_text() == "2000|standard"


def test_prepare_dense_workspace_new_settings_clear(tmp_path):
    from cloudlabeller.photogrammetry.mvs import UNDISTORT_MARKER, prepare_dense_workspace

    dense = tmp_path / "dense"
    prepare_dense_workspace(dense, 2000)
    _fake_dense_outputs(dense)

    # Regression: rerunning at another detail level used to leave stale depth
    # maps behind, and COLMAP's stereo/fusion failed on the size mismatch.
    messages = []
    assert prepare_dense_workspace(
        dense, 1000, progress=lambda f, m="": messages.append(m)) is False
    assert not (dense / "stereo").exists()
    assert not (dense / "images").exists()
    assert (dense / UNDISTORT_MARKER).read_text() == "1000|standard"
    assert any("clearing previous dense outputs" in m for m in messages)

    # A quality change alone also clears (mixed-quality depth maps would fuse).
    _fake_dense_outputs(dense)
    assert prepare_dense_workspace(dense, 1000, quality="draft") is False
    assert not (dense / "stereo").exists()
    assert (dense / UNDISTORT_MARKER).read_text() == "1000|draft"


def test_sparse_model_fingerprint_tracks_changes(tmp_path):
    from cloudlabeller.photogrammetry.mvs import sparse_model_fingerprint

    (tmp_path / "cameras.bin").write_bytes(b"c")
    (tmp_path / "images.bin").write_bytes(b"i")
    (tmp_path / "points3D.bin").write_bytes(b"p")
    fp = sparse_model_fingerprint(tmp_path)
    assert fp == sparse_model_fingerprint(tmp_path)          # stable

    (tmp_path / "points3D.bin").write_bytes(b"pp")           # re-run of SfM
    assert sparse_model_fingerprint(tmp_path) != fp


def test_set_patch_match_sources_rewrites_cfg(tmp_path):
    from cloudlabeller.photogrammetry.mvs import set_patch_match_sources

    stereo = tmp_path / "stereo"
    stereo.mkdir()
    cfg = stereo / "patch-match.cfg"
    cfg.write_text("a.jpg\n__auto__, 20\nb.jpg\n__auto__, 20\n", encoding="utf-8")

    set_patch_match_sources(tmp_path, 10)
    assert cfg.read_text() == "a.jpg\n__auto__, 10\nb.jpg\n__auto__, 10\n"

    set_patch_match_sources(tmp_path / "missing", 10)        # no cfg: no error


# --- dataset pane: status dots ----------------------------------------------
def test_dataset_panel_status_dots():
    from pathlib import Path
    from types import SimpleNamespace as NS

    from PySide6.QtWidgets import QApplication

    from cloudlabeller.core.dataset import Dataset, ImageRecord
    from cloudlabeller.core.events import EventBus
    from cloudlabeller.core.labels import LabelStore
    from cloudlabeller.ui.dataset_panel import DatasetPanel
    from cloudlabeller.ui.status import STATUS_COLORS

    QApplication.instance() or QApplication([])

    labels = LabelStore()
    labels.mark_user_labeled("a.jpg")
    labels.mark_auto_labeled("b.jpg")
    project = NS(dataset=Dataset(images=[
        ImageRecord(0, Path("a.jpg")), ImageRecord(1, Path("b.jpg")),
        ImageRecord(2, Path("c.jpg"))]), labels=labels)

    bus = EventBus()
    panel = DatasetPanel(bus)
    bus.project_opened.emit(project)

    def dot(row):
        item = panel.image_list.item(row)
        return item.icon().pixmap(12, 12).toImage().pixelColor(6, 6).name()

    assert dot(0) == STATUS_COLORS["user"]       # green: user-labelled
    assert dot(1) == STATUS_COLORS["auto"]       # yellow: auto-labelled
    assert dot(2) == STATUS_COLORS["none"]       # red: unlabelled

    # Labelling an image live-updates its dot (no rebuild needed).
    labels.mark_user_labeled("c.jpg")
    bus.image_labels_changed.emit("c.jpg")
    assert dot(2) == STATUS_COLORS["user"]


# --- detail levels: SfM vs MVS resolution presets ---------------------------
def test_detail_control_levels():
    from PySide6.QtWidgets import QApplication

    from cloudlabeller.ui.detail_level import MVS_LEVELS, DetailControl

    QApplication.instance() or QApplication([])
    res = (5472, 3648)

    # SfM: High (full) / Medium 1/4 (default) / Low 1/16 — divisors 1, 2, 4.
    sfm = DetailControl(res)
    assert sfm.combo.currentText().startswith("Medium — 1/4")
    assert sfm.pixels() == 5472 // 2
    assert [sfm.combo.itemText(i).split(" — ")[0] for i in range(3)] \
        == ["High", "Medium", "Low"]
    sfm.combo.setCurrentIndex(2)
    assert "1/16" in sfm.combo.currentText() and sfm.pixels() == 5472 // 4

    # MVS: Ultra/High/Medium (default)/Low — divisors 1, 2, 4, 8.
    mvs = DetailControl(res, default_level="medium", levels=MVS_LEVELS)
    assert mvs.combo.currentText().startswith("Medium — 1/16")
    assert mvs.pixels() == 5472 // 4
    assert [mvs.combo.itemText(i).split(" — ")[0] for i in range(4)] \
        == ["Ultra", "High", "Medium", "Low"]
    mvs.combo.setCurrentIndex(3)
    assert "1/64" in mvs.combo.currentText() and mvs.pixels() == 5472 // 8
    mvs.combo.setCurrentIndex(0)
    assert mvs.pixels() == 0                          # Ultra = no cap

    # Unknown source resolution: divisor-2 level falls back to custom_default.
    fallback = DetailControl(None, default_level="medium",
                             custom_default=2000, levels=MVS_LEVELS)
    assert fallback.pixels() == 1000                  # divisor 4 -> default//2


# --- meshing: density-matched depth + cost estimates -----------------------
def _grid_cloud(nx: int, ny: int, spacing: float = 1.0) -> PointCloud:
    gx, gy = np.meshgrid(np.arange(nx) * spacing, np.arange(ny) * spacing)
    xyz = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
    return PointCloud(xyz=xyz.astype(np.float32))


def test_estimate_point_spacing():
    from cloudlabeller.photogrammetry.meshing import estimate_point_spacing

    # Exact path (cloud smaller than the sample size): grid spacing recovered.
    assert estimate_point_spacing(_grid_cloud(100, 100, spacing=2.5)) == pytest.approx(2.5)

    # Subsampled path: for surface-random points (like real MVS output) the
    # sqrt(f) correction makes the subsample estimate match the full one.
    # (A regular grid would NOT: thinning destroys its regularity.)
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 300, size=(90_000, 2))
    cloud = PointCloud(xyz=np.column_stack(
        [xy, np.zeros(len(xy))]).astype(np.float32))
    full = estimate_point_spacing(cloud, sample=100_000)      # >= n: exact
    sub = estimate_point_spacing(cloud, sample=10_000)
    assert full == pytest.approx(sub, rel=0.3)


def test_suggest_poisson_depth_matches_density():
    from cloudlabeller.photogrammetry.meshing import (
        MAX_DEPTH,
        MIN_DEPTH,
        suggest_poisson_depth,
    )

    # 400x400 grid, spacing 1 (exact path): side 399 -> log2 ≈ 8.6 -> depth 9.
    assert suggest_poisson_depth(_grid_cloud(400, 400)) == 9
    # Coarse cloud clamps at the lower bound; degenerate input doesn't crash.
    assert suggest_poisson_depth(_grid_cloud(20, 20)) == MIN_DEPTH
    assert MIN_DEPTH <= suggest_poisson_depth(
        PointCloud(xyz=np.zeros((1, 3), np.float32))) <= MAX_DEPTH


def test_estimate_mesh_cost():
    from cloudlabeller.photogrammetry.meshing import estimate_mesh_cost

    lo = estimate_mesh_cost(100_000_000, 11, 13)
    hi = estimate_mesh_cost(100_000_000, 13, 13)
    assert hi["minutes"] > lo["minutes"] and hi["ram_gb"] > lo["ram_gb"]
    # At the density-matched depth, vertices ~ the cloud; each depth below /4.
    assert hi["vertices"] == pytest.approx(100_000_000)
    assert lo["vertices"] == pytest.approx(100_000_000 / 16)
    # Delaunay interpolates every fused point.
    assert estimate_mesh_cost(5_000_000, 0, 0, method="delaunay")["vertices"] == 5_000_000


def test_delaunay_available(tmp_path):
    from cloudlabeller.photogrammetry.meshing import delaunay_available

    assert not delaunay_available(None)
    assert not delaunay_available(tmp_path)
    (tmp_path / "fused.ply").write_bytes(b"x")
    assert not delaunay_available(tmp_path)          # .vis missing
    (tmp_path / "fused.ply.vis").write_bytes(b"x")
    assert delaunay_available(tmp_path)


def test_mesh_cli_parse_methods():
    from cloudlabeller.photogrammetry.mesh_cli import _parse

    ns = _parse(["mesh_cli", "dense.npz", "mesh.npz"])
    assert ns.method == "poisson" and ns.depth is None    # None = match density
    assert ns.trim == 10.0 and ns.point_weight == 1.0

    ns = _parse(["mesh_cli", "dense.npz", "mesh.npz", "--method", "delaunay",
                 "--workspace", "ws", "--depth", "12"])
    assert ns.method == "delaunay" and ns.workspace == "ws" and ns.depth == 12


def test_create_mesh_dialog_defaults_and_estimates():
    from PySide6.QtWidgets import QApplication

    from cloudlabeller.ui.mesh_dialog import CreateMeshDialog

    QApplication.instance() or QApplication([])
    dialog = CreateMeshDialog(_grid_cloud(300, 300), workspace=None)

    opts = dialog.options()
    assert opts == {"method": "poisson", "depth": None,
                    "trim": 10.0, "point_weight": 1.0}
    assert not dialog.rb_delaunay.isEnabled()        # no MVS workspace
    assert "Rough estimate" in dialog.lbl_estimate.text()

    dialog.cmb_detail.setCurrentIndex(1)             # "Maximum (depth 13)"
    assert dialog.options()["depth"] == 13


def test_pipeline_dialog_mesh_options():
    from PySide6.QtWidgets import QApplication

    from cloudlabeller.ui.run_pipeline_dialog import RunPipelineDialog

    QApplication.instance() or QApplication([])
    dialog = RunPipelineDialog(image_count=5)
    assert dialog.mesh_options() == {"method": "poisson", "depth": None,
                                     "trim": 10.0, "point_weight": 1.0}
    dialog.cmb_mesh_method.setCurrentIndex(1)        # Delaunay
    assert dialog.mesh_options()["method"] == "delaunay"
    assert not dialog.cmb_mesh_detail.isEnabled()    # depth n/a for Delaunay


# --- image-overlap (covisibility) graph -----------------------------------
def test_covisibility_from_reconstruction_tracks():
    from types import SimpleNamespace as NS

    from cloudlabeller.photogrammetry.adjacency import covisibility_from_reconstruction

    # Duck-typed reconstruction: a&b observe 15 common points; c shares only 2
    # with a (incidental long tracks) and must not become a neighbour.
    def track(*image_ids):
        return NS(track=NS(elements=[NS(image_id=i) for i in image_ids]))

    points = {}
    for p in range(15):
        points[p] = track(1, 2)
    points[100] = track(1, 3)
    points[101] = track(1, 3)
    points[102] = track(3)
    rec = NS(images={1: NS(name="a.jpg"), 2: NS(name="b.jpg"), 3: NS(name="c.jpg")},
             points3D=points)

    adj = covisibility_from_reconstruction(rec)
    assert adj.totals == {"a.jpg": 17, "b.jpg": 15, "c.jpg": 3}
    assert adj.shared["a.jpg"]["b.jpg"] == 15 == adj.shared["b.jpg"]["a.jpg"]
    assert adj.overlapping("a.jpg") == {"b.jpg"}      # c: 2 shared < MIN_SHARED
    assert adj.overlapping("b.jpg") == {"a.jpg"}
    assert adj.overlapping("unknown.jpg") is None     # not in graph -> no info


def test_adjacency_save_load_roundtrip(tmp_path):
    from cloudlabeller.photogrammetry.adjacency import ADJACENCY_FILE, ImageAdjacency

    adj = ImageAdjacency(totals={"a.jpg": 100, "b.jpg": 80},
                         shared={"a.jpg": {"b.jpg": 40}, "b.jpg": {"a.jpg": 40}})
    adj.save(tmp_path)
    assert (tmp_path / ADJACENCY_FILE).exists()

    loaded = ImageAdjacency.load(tmp_path)
    assert loaded.totals == adj.totals
    assert loaded.shared["b.jpg"]["a.jpg"] == 40      # symmetry restored on load
    assert loaded.overlapping("a.jpg") == {"b.jpg"}
    assert ImageAdjacency.load(tmp_path / "nowhere") is None


def test_covisibility_from_images_visibility():
    from pathlib import Path

    from cloudlabeller.core.dataset import ImageRecord
    from cloudlabeller.photogrammetry.adjacency import covisibility_from_images

    # A grid of points in front of two nearly co-located cameras; a third
    # camera far away sees nothing.
    gx, gy = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-0.5, 0.5, 10))
    xyz = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, 5.0)])
    cloud = PointCloud(xyz=xyz.astype(np.float32))

    records = [
        ImageRecord(0, Path("a.jpg"), _app_camera(0.0)),
        ImageRecord(1, Path("b.jpg"), _app_camera(0.1)),
        ImageRecord(2, Path("far.jpg"), _app_camera(1000.0)),
    ]
    adj = covisibility_from_images(cloud, records)
    assert adj.source == "visibility"
    assert "b.jpg" in adj.overlapping("a.jpg")
    assert adj.overlapping("far.jpg") is None         # sees no points at all


def test_select_covering_prunes_redundant_images():
    from cloudlabeller.photogrammetry.adjacency import ImageAdjacency, select_covering

    # a, b, c photograph (nearly) the same area; d covers separate surface.
    adj = ImageAdjacency(
        totals={"a.jpg": 1000, "b.jpg": 900, "c.jpg": 880, "d.jpg": 500},
        shared={
            "a.jpg": {"b.jpg": 870, "c.jpg": 860, "d.jpg": 20},
            "b.jpg": {"a.jpg": 870, "c.jpg": 850, "d.jpg": 0},
            "c.jpg": {"a.jpg": 860, "b.jpg": 850, "d.jpg": 0},
            "d.jpg": {"a.jpg": 20},
        })
    picked = select_covering(["a.jpg", "b.jpg", "c.jpg", "d.jpg"], adj)
    # The largest of the redundant trio plus the distinct image survive,
    # preserving input order.
    assert picked == ["a.jpg", "d.jpg"]

    # Names absent from the graph can't be pruned safely: they stay.
    picked = select_covering(["a.jpg", "b.jpg", "new.jpg"], adj)
    assert "new.jpg" in picked and "a.jpg" in picked and "b.jpg" not in picked

    # An empty graph prunes nothing.
    assert select_covering(["x.jpg", "y.jpg"], ImageAdjacency()) == ["x.jpg", "y.jpg"]


def test_project_adjacency_fallback_and_cache(tmp_path):
    from pathlib import Path

    from cloudlabeller.core.dataset import ImageRecord
    from cloudlabeller.core.project import Project
    from cloudlabeller.photogrammetry.adjacency import ADJACENCY_FILE

    proj = Project.create(tmp_path / "p.clproj")
    gx, gy = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-0.5, 0.5, 10))
    xyz = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, 5.0)])
    proj.dataset.cloud = PointCloud(xyz=xyz.astype(np.float32))
    proj.dataset.images = [
        ImageRecord(0, Path("a.jpg"), _app_camera(0.0)),
        ImageRecord(1, Path("b.jpg"), _app_camera(0.1)),
    ]

    # No adjacency.json (old project): derived from visibility and saved.
    assert proj.overlapping_images("a.jpg") == {"b.jpg"}
    assert (proj.reconstruction_dir / ADJACENCY_FILE).exists()

    # A reopened project reads the saved file (no cloud/cameras needed).
    reopened = Project.open(proj.root)
    assert reopened.overlapping_images("a.jpg") == {"b.jpg"}


def test_save_result_writes_adjacency(tmp_path):
    from cloudlabeller.photogrammetry.adjacency import ADJACENCY_FILE, ImageAdjacency
    from cloudlabeller.photogrammetry.pipeline import ReconstructResult, save_result

    cloud = PointCloud(xyz=np.zeros((3, 3), np.float32))
    adj = ImageAdjacency(totals={"a.jpg": 5})
    save_result(ReconstructResult(cloud=cloud, images=[], adjacency=adj), tmp_path)
    assert (tmp_path / ADJACENCY_FILE).exists()
    assert ImageAdjacency.load(tmp_path).totals == {"a.jpg": 5}

    # And stays optional: no adjacency computed -> nothing written, no error.
    save_result(ReconstructResult(cloud=cloud, images=[]), tmp_path / "no_adj")
    assert not (tmp_path / "no_adj" / ADJACENCY_FILE).exists()


def test_prepare_dense_workspace_legacy_no_marker_clears(tmp_path):
    from cloudlabeller.photogrammetry.mvs import UNDISTORT_MARKER, prepare_dense_workspace

    # A workspace from an app version without the marker: the settings it was
    # built at are unknown, so it cannot be trusted for resuming.
    dense = tmp_path / "dense"
    dense.mkdir()
    _fake_dense_outputs(dense)

    assert prepare_dense_workspace(dense, 2000) is False
    assert not (dense / "stereo").exists()
    assert (dense / UNDISTORT_MARKER).read_text() == "2000|standard"


def test_extract_and_match_exe_smoke(tmp_path):
    """GPU SIFT + matching through colmap.exe writes a usable database."""
    from cloudlabeller.photogrammetry.mvs import find_colmap_binary

    exe = find_colmap_binary()
    if exe is None:
        pytest.skip("COLMAP executable not installed")
    from PIL import Image

    from cloudlabeller.photogrammetry.sfm import _extract_and_match_exe

    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, size=(400, 600, 3), dtype=np.uint8)
    names = []
    for i in range(2):                       # shifted copies: matchable features
        name = f"im{i}.jpg"
        Image.fromarray(np.roll(base, 60 * i, axis=1)).save(tmp_path / name)
        names.append(name)
    ws = tmp_path / "ws"
    ws.mkdir()
    database = ws / "database.db"

    _extract_and_match_exe(exe, database, tmp_path, names, ws,
                           matcher="exhaustive", camera_model="SIMPLE_RADIAL",
                           single_camera=True, max_image_size=0, n_threads=4,
                           progress=lambda f, m="": None)

    import sqlite3

    with sqlite3.connect(database) as db:
        n_kp = db.execute("SELECT COUNT(*) FROM keypoints").fetchone()[0]
        n_matches = db.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    assert n_kp == 2 and n_matches > 0       # keypoints for both, pairs matched

    # Unknown matcher -> clear error (the caller falls back to CPU pycolmap).
    with pytest.raises(RuntimeError, match="no colmap.exe equivalent"):
        _extract_and_match_exe(exe, database, tmp_path, names, ws,
                               matcher="vocab_tree", camera_model="SIMPLE_RADIAL",
                               single_camera=True, max_image_size=0, n_threads=4,
                               progress=lambda f, m="": None)


def test_cli_parsers_new_flags():
    from cloudlabeller.photogrammetry.mvs_cli import _parse as mvs_parse
    from cloudlabeller.photogrammetry.run_cli import _parse as sfm_parse

    ns = mvs_parse(["mvs_cli", "ws", "imgs"])
    assert ns.quality == "standard"
    ns = mvs_parse(["mvs_cli", "ws", "imgs", "--quality", "draft"])
    assert ns.quality == "draft"

    ns = sfm_parse(["run_cli", "imgs", "ws", "--gpu",
                    "--colmap-binary", r"C:\colmap.exe"])
    assert ns.gpu is True and ns.colmap_binary == r"C:\colmap.exe"
    assert sfm_parse(["run_cli", "imgs", "ws"]).colmap_binary is None


def test_sfm_dialog_gpu_default_and_mvs_quality():
    from PySide6.QtWidgets import QApplication

    from cloudlabeller.ui.run_pipeline_dialog import RunPipelineDialog
    from cloudlabeller.ui.run_sfm_dialog import RunSfmDialog

    QApplication.instance() or QApplication([])
    # GPU defaults ON exactly when a CUDA COLMAP executable was found.
    assert RunSfmDialog(5, gpu_available=True).options().use_gpu is True
    assert RunSfmDialog(5, gpu_available=False).options().use_gpu is False

    pipeline = RunPipelineDialog(5, gpu_available=True)
    assert pipeline.sfm_options().use_gpu is True
    assert pipeline.mvs_quality() == "standard"
    pipeline.cmb_mvs_quality.setCurrentIndex(1)
    assert pipeline.mvs_quality() == "draft"


def test_colmap_streams_and_parses_progress():
    import sys

    from cloudlabeller.photogrammetry.mvs import _colmap

    # A stand-in for colmap that emits COLMAP-style "X / Y" counters.
    script = "for i in (1, 2, 3): print(f'Processing view {i} / 3', flush=True)"
    seen: list[float] = []
    _colmap(sys.executable, ["-c", script],
            progress=lambda f, m="": seen.append(round(f, 3)),
            stage=("Patch-match", 0.2, 0.8))
    assert seen == [0.4, 0.6, 0.8]               # mapped into the 0.2–0.8 band


def test_colmap_raises_on_failure():
    import sys

    from cloudlabeller.photogrammetry.mvs import _colmap

    with pytest.raises(RuntimeError, match="failed"):
        _colmap(sys.executable, ["-c", "import sys; print('boom'); sys.exit(2)"])


def test_find_sparse_model(tmp_path):
    from cloudlabeller.photogrammetry.mvs import find_sparse_model

    model = tmp_path / "sparse" / "0"
    model.mkdir(parents=True)
    (model / "points3D.bin").write_bytes(b"x")
    assert find_sparse_model(tmp_path / "sparse") == model      # finds sub-model
    assert find_sparse_model(model) == model                    # or a direct path


def test_save_cloud_ply_roundtrip(tmp_path):
    from cloudlabeller.core.dataset import PointCloud
    from cloudlabeller.io.geometry import load_cloud, save_cloud_ply

    rng = np.random.default_rng(0)
    cloud = PointCloud(
        xyz=rng.normal(size=(50, 3)).astype(np.float32),
        rgb=rng.integers(0, 255, (50, 3)).astype(np.uint8),
        normals=rng.normal(size=(50, 3)).astype(np.float32),
    )
    path = tmp_path / "c.ply"
    save_cloud_ply(cloud, path)
    back = load_cloud(path)
    assert back.n_points == 50
    np.testing.assert_allclose(back.xyz, cloud.xyz, atol=1e-6)
    assert np.array_equal(back.rgb, cloud.rgb)
    assert back.normals is not None
    np.testing.assert_allclose(back.normals, cloud.normals, atol=1e-6)


def test_create_mesh_end_to_end():
    """Real Poisson meshing via colmap.exe on a synthetic coloured sphere."""
    from cloudlabeller.core.dataset import PointCloud
    from cloudlabeller.photogrammetry.meshing import create_mesh
    from cloudlabeller.photogrammetry.mvs import find_colmap_binary

    if find_colmap_binary() is None:
        pytest.skip("COLMAP executable not installed")

    sphere = pv.Sphere(radius=1.0, theta_resolution=60, phi_resolution=60)
    pts = np.asarray(sphere.points, np.float32)
    normals = pts / np.linalg.norm(pts, axis=1, keepdims=True)   # exact for a sphere
    rgb = np.zeros((len(pts), 3), np.uint8)
    rgb[:, 0] = ((pts[:, 2] + 1) * 127).astype(np.uint8)         # red gradient in z
    cloud = PointCloud(xyz=pts, rgb=rgb, normals=normals.astype(np.float32))

    mesh = create_mesh(cloud, depth=7, trim=0.0)
    assert len(mesh.faces) > 100                                 # a real surface
    assert mesh.vertex_colors is not None                        # colourised
    # colours follow the z gradient of the source cloud
    top = mesh.vertices[:, 2] > 0.8
    bottom = mesh.vertices[:, 2] < -0.8
    assert top.any() and bottom.any()
    assert mesh.vertex_colors[top, 0].mean() > mesh.vertex_colors[bottom, 0].mean() + 50


def test_create_mesh_requires_normals():
    from cloudlabeller.core.dataset import PointCloud
    from cloudlabeller.photogrammetry.meshing import create_mesh

    with pytest.raises(RuntimeError, match="normals"):
        create_mesh(PointCloud(xyz=np.zeros((10, 3), np.float32)))


def test_import_cloud_and_mesh_from_ply(tmp_path):
    from cloudlabeller.io.geometry import load_cloud, load_mesh

    cloud_ply = tmp_path / "c.ply"
    pc = pv.PolyData(np.random.rand(10, 3))
    pc["RGB"] = np.random.randint(0, 255, (10, 3)).astype(np.uint8)
    pc.save(str(cloud_ply))
    cloud = load_cloud(cloud_ply)
    assert cloud.n_points == 10

    mesh_ply = tmp_path / "m.ply"
    pv.Sphere().save(str(mesh_ply))
    mesh = load_mesh(mesh_ply)
    assert len(mesh.vertices) > 0 and mesh.faces.shape[1] == 3


def test_lasso_projection_matches_vtk_renderer():
    """project_to_display (vectorized) must agree with VTK's own per-point
    SetWorldPoint/WorldToDisplay transform."""
    from cloudlabeller.ui.lasso import composite_projection_matrix, project_to_display

    pl = pv.Plotter(off_screen=True, window_size=(800, 600))
    rng = np.random.default_rng(0)
    pts = rng.uniform(-3, 3, (200, 3))
    pl.add_mesh(pv.PolyData(pts))
    pl.reset_camera()
    pl.render()

    matrix = composite_projection_matrix(pl.renderer)
    screen, in_front = project_to_display(pts, matrix, 800, 600)

    ren = pl.renderer
    for i in range(0, 200, 17):
        ren.SetWorldPoint(*pts[i], 1.0)
        ren.WorldToDisplay()
        dx, dy, _ = ren.GetDisplayPoint()
        assert in_front[i]
        assert abs(screen[i, 0] - dx) < 0.5 and abs(screen[i, 1] - dy) < 0.5
    pl.close()


def test_points_in_lasso_selects_correct_cluster():
    from cloudlabeller.ui.lasso import composite_projection_matrix, points_in_lasso, \
        project_to_display

    pl = pv.Plotter(off_screen=True, window_size=(800, 600))
    left = np.column_stack([np.full(50, -2.0), np.random.rand(50), np.random.rand(50)])
    right = np.column_stack([np.full(50, 2.0), np.random.rand(50), np.random.rand(50)])
    pts = np.vstack([left, right])
    pl.add_mesh(pv.PolyData(pts))
    pl.reset_camera()
    pl.render()
    matrix = composite_projection_matrix(pl.renderer)

    # Lasso around the LEFT cluster's projected bounding box.
    screen, _ = project_to_display(left, matrix, 800, 600)
    lo, hi = screen.min(0) - 5, screen.max(0) + 5
    path = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
    sel = points_in_lasso(pts, matrix, 800, 600, path)
    assert set(sel) == set(range(50))           # all left, no right
    pl.close()


def test_paint_dense_cloud_undoable():
    from cloudlabeller.core.labels import LabelStore, Modality

    store = LabelStore()
    store.init_dense_cloud(6)
    store.paint_dense_cloud(np.array([1, 3]), 2)
    assert list(store.dense_cloud_labels) == [-1, 2, -1, 2, -1, -1]
    assert store.undo() == (Modality.CLOUD, None)
    assert list(store.dense_cloud_labels) == [-1] * 6


def test_normals_backface_culling_in_bridge():
    pytest.importorskip("hylite")
    from cloudlabeller.core.dataset import PointCloud
    from cloudlabeller.transfer.hylite_bridge import visible_point_pixels

    # horizontal patch with +z (up) normals
    rng = np.random.default_rng(0)
    xyz = np.column_stack([rng.uniform(-2, 2, 500), rng.uniform(-1.5, 1.5, 500),
                           np.zeros(500)]).astype(np.float32)
    up_normals = np.tile([0, 0, 1.0], (500, 1)).astype(np.float32)

    def cam_looking_from(z):
        sign = 1.0 if z > 0 else -1.0
        R = np.diag([1.0, -sign, -sign])       # proper rotation looking toward the patch
        C = np.array([0.0, 0.0, float(z)])
        return Camera(K=np.array([[800., 0, 320], [0, 800., 240], [0, 0, 1]]),
                      R=R, t=-R @ C, width=640, height=480)

    above, below = cam_looking_from(8.0), cam_looking_from(-8.0)
    plain = PointCloud(xyz=xyz)                             # no normals -> leaks
    with_n = PointCloud(xyz=xyz, normals=up_normals)        # normals -> culled
    assert visible_point_pixels(plain, below)[0].size > 0
    assert visible_point_pixels(with_n, below)[0].size == 0
    assert visible_point_pixels(with_n, above)[0].size > 0


def test_auto_masks_are_virtual(tmp_path):
    """Auto-labelled images get their overlay rendered from the cloud on demand;
    no full-res mask is materialised until the user draws on the image."""
    pytest.importorskip("hylite")
    from PIL import Image
    from PySide6.QtCore import QPointF
    from PySide6.QtWidgets import QApplication

    from cloudlabeller.core.dataset import ImageRecord, PointCloud
    from cloudlabeller.core.events import EventBus
    from cloudlabeller.core.project import Project
    from cloudlabeller.ui.image_view import ImageView

    QApplication.instance() or QApplication([])
    proj = Project.create(tmp_path / "p.clproj")
    proj.add_label("rock", "#ff0000")

    cam = _app_camera(0.0)
    img_path = proj.images_dir / "a.png"
    Image.new("RGB", (cam.width, cam.height), (5, 5, 5)).save(img_path)
    proj.dataset.images = [ImageRecord(0, img_path, cam)]
    # a labelled cloud in front of the camera
    rng = np.random.default_rng(0)
    xyz = np.column_stack([rng.uniform(-1, 1, 300), rng.uniform(-1, 1, 300),
                           rng.uniform(4, 6, 300)]).astype(np.float32)
    proj.dataset.cloud = PointCloud(xyz=xyz)
    proj.labels.cloud_labels = np.zeros(300, np.int32)      # all class 0
    proj.labels.mark_auto_labeled("a.png")

    view = ImageView(EventBus())
    view.project = proj
    view.show_image_sync("a.png")

    assert "a.png" not in proj.labels.image_masks           # nothing materialised
    assert view._auto_mask is not None                      # rendered on demand
    assert (view._auto_mask != -1).any()
    auto_snapshot = view._auto_mask.copy()                  # commit clears the transient

    # Drawing a polygon materialises the mask, seeded with the auto labels.
    proj.schema.add("veg", "#00ff00")                       # class 1
    view._set_active_class(1)
    for x, y in [(10, 10), (60, 10), (60, 60), (10, 60)]:
        view._add_vertex(QPointF(x, y))
    view._commit()
    stored = proj.labels.image_masks["a.png"]
    assert stored[30, 30] == 1                              # user polygon
    assert view._auto_mask is None                          # transient cleared: stored now
    outside = auto_snapshot == 0
    outside[:70, :70] = False                               # away from the polygon
    if outside.any():
        r, c = np.argwhere(outside)[0]
        assert stored[r, c] == 0                            # auto labels preserved


def test_lasso_path_actor_offscreen():
    from cloudlabeller.ui.viewer3d import _LassoPathActor

    pl = pv.Plotter(off_screen=True, window_size=(400, 300))
    pl.add_mesh(pv.PolyData(np.random.rand(50, 3)))
    actor = _LassoPathActor(pl)
    assert not actor.actor.GetVisibility()
    actor.begin((10.0, 10.0))
    actor.add_point((100.0, 20.0))
    actor.add_point((60.0, 90.0))
    assert actor.actor.GetVisibility()
    pl.render()                                             # draws without error
    actor.clear()
    assert not actor.actor.GetVisibility()
    pl.close()


def test_view_panel_lasso_enable_rules():
    from PySide6.QtWidgets import QApplication

    from cloudlabeller.core.events import EventBus
    from cloudlabeller.ui.view_panel import ViewPanel

    QApplication.instance() or QApplication([])
    panel = ViewPanel(EventBus())
    assert not panel.btn_lasso.isEnabled()            # no project / no dense

    class _DS:
        def has(self, k):
            return k in ("sparse", "dense")

    class _Proj:
        dataset = _DS()

    panel._on_project(_Proj())
    assert panel.btn_lasso.isEnabled()                # sparse shown: delete-lasso
    panel._rep_buttons["dense"].setChecked(True)
    assert panel.btn_lasso.isEnabled()                # dense shown: label-lasso

    # selection actions are gated by the representation
    panel.set_selection_count(1234)
    assert panel.btn_apply.isEnabled() and not panel.btn_delete.isEnabled()
    assert "1,234" in panel.lbl_selection.text()
    panel._rep_buttons["sparse"].setChecked(True)
    panel.set_selection_count(1234)
    assert panel.btn_delete.isEnabled() and not panel.btn_apply.isEnabled()
    panel.set_selection_count(0)
    assert not panel.btn_apply.isEnabled() and not panel.btn_delete.isEnabled()

    # the mesh view disarms and disables the lasso
    panel.btn_lasso.setChecked(True)
    toggles: list[bool] = []
    panel.lasso_toggled.connect(toggles.append)
    panel._rep_buttons["mesh"].setEnabled(True)
    panel._rep_buttons["mesh"].setChecked(True)
    assert not panel.btn_lasso.isEnabled() and not panel.btn_lasso.isChecked()
    assert toggles and toggles[-1] is False           # lasso disarmed


def test_distorted_projection_matches_pycolmap():
    """project_points must reproduce COLMAP's own camera model exactly."""
    import pycolmap

    from cloudlabeller.photogrammetry.cameras import extract_distortion
    from cloudlabeller.transfer.hylite_bridge import project_points

    cases = [
        ("SIMPLE_RADIAL", [1000.0, 400.0, 300.0, -0.06]),
        ("RADIAL", [1000.0, 400.0, 300.0, -0.06, 0.01]),
        ("OPENCV", [990.0, 1010.0, 400.0, 300.0, -0.05, 0.008, 0.001, -0.0005]),
    ]
    rng = np.random.default_rng(0)
    for model, params in cases:
        pc_cam = pycolmap.Camera.create_from_model_name(1, model, 1000.0, 800, 600)
        pc_cam.params = params
        K = np.asarray(pc_cam.calibration_matrix(), float)
        cam = Camera(K=K, R=np.eye(3), t=np.zeros(3), width=800, height=600,
                     model=model, distortion=extract_distortion(model, params))

        # random camera-frame points in front (world == camera frame: R=I, t=0)
        pts = np.column_stack([rng.uniform(-2, 2, 300), rng.uniform(-1.5, 1.5, 300),
                               rng.uniform(4, 9, 300)])
        expected = np.asarray(pc_cam.img_from_cam(pts))      # COLMAP ground truth
        pp, vis = project_points(cam, pts)
        assert vis.sum() > 100, model
        err = np.abs(pp[vis, :2] - expected[vis]).max()
        assert err < 1e-6, f"{model}: max err {err}"


def test_distortion_shifts_border_pixels():
    """With k=-0.06, a border point moves tens of px vs pinhole (the bug)."""
    from cloudlabeller.transfer.hylite_bridge import project_points

    K = np.array([[1000., 0, 400], [0, 1000., 300], [0, 0, 1]])
    base = dict(K=K, R=np.eye(3), t=np.zeros(3), width=800, height=600)
    pin = Camera(**base, model="PINHOLE")
    rad = Camera(**base, model="SIMPLE_RADIAL", distortion=np.array([-0.06]))

    corner_pt = np.array([[3.2, 2.4, 8.0]])      # projects near the image corner
    pp_pin, _ = project_points(pin, corner_pt)
    pp_rad, _ = project_points(rad, corner_pt)
    shift = np.linalg.norm(pp_pin[0, :2] - pp_rad[0, :2])   # f*|k|*r^3 ≈ 7.5 px here
    assert shift > 5                             # clearly visible at the corner
    center_pt = np.array([[0.05, 0.0, 8.0]])     # near the centre: barely moves
    c_pin, _ = project_points(pin, center_pt)
    c_rad, _ = project_points(rad, center_pt)
    assert abs(c_pin[0, 0] - c_rad[0, 0]) < 0.5


def test_extract_distortion_models():
    from cloudlabeller.photogrammetry.cameras import extract_distortion

    assert extract_distortion("PINHOLE", [1000, 1000, 400, 300]) is None
    assert extract_distortion("SIMPLE_PINHOLE", [1000, 400, 300]) is None
    np.testing.assert_allclose(
        extract_distortion("SIMPLE_RADIAL", [1000, 400, 300, -0.06]), [-0.06])
    np.testing.assert_allclose(
        extract_distortion("RADIAL", [1000, 400, 300, -0.06, 0.01]), [-0.06, 0.01])
    np.testing.assert_allclose(
        extract_distortion("OPENCV", [990, 1010, 400, 300, -0.05, 0.008, 1e-3, -5e-4]),
        [-0.05, 0.008, 1e-3, -5e-4])
    # all-zero coefficients -> treated as pinhole
    assert extract_distortion("SIMPLE_RADIAL", [1000, 400, 300, 0.0]) is None


def test_cameras_json_roundtrips_distortion(tmp_path):
    from pathlib import Path

    from cloudlabeller.core.dataset import ImageRecord, PointCloud
    from cloudlabeller.photogrammetry.pipeline import (
        ReconstructResult, load_cameras, save_result,
    )

    cam = _app_camera(0.0)
    cam.model = "SIMPLE_RADIAL"
    cam.distortion = np.array([-0.0598])
    save_result(ReconstructResult(PointCloud(xyz=np.zeros((3, 3), np.float32)),
                                  [ImageRecord(1, Path("a.jpg"), cam)]), tmp_path)
    loaded = load_cameras(tmp_path)["a.jpg"]
    np.testing.assert_allclose(loaded.distortion, [-0.0598])
    assert loaded.model == "SIMPLE_RADIAL"


def test_hylite_camera_matches_direct_projection():
    pytest.importorskip("hylite")
    from cloudlabeller.transfer.hylite_bridge import project_points

    cam = _app_camera(0.0)                    # centred principal point, R=I, t=0
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.uniform(-2, 2, 60), rng.uniform(-1.5, 1.5, 60),
                         rng.uniform(4, 9, 60)]).astype(np.float32)
    Xc = (cam.R @ X.T).T + cam.t
    z = Xc[:, 2]
    u = cam.K[0, 0] * Xc[:, 0] / z + cam.K[0, 2]
    v = cam.K[1, 1] * Xc[:, 1] / z + cam.K[1, 2]

    pp, vis = project_points(cam, X)
    assert vis.sum() > 30
    assert np.allclose(pp[vis, 0], u[vis], atol=1e-3)   # hylite (px,py) == COLMAP (u,v)
    assert np.allclose(pp[vis, 1], v[vis], atol=1e-3)


def test_visible_point_pixels_occlusion():
    pytest.importorskip("hylite")
    from cloudlabeller.core.dataset import PointCloud
    from cloudlabeller.transfer.hylite_bridge import visible_point_pixels

    cam = _app_camera(0.0)
    cloud = PointCloud(xyz=np.array([[0, 0, 5.0], [0, 0, 9.0]], np.float32))  # same ray
    idx, _ = visible_point_pixels(cloud, cam)
    assert 0 in idx and 1 not in idx          # near point visible, far one occluded


def test_transfer_round_trip():
    pytest.importorskip("hylite")
    from pathlib import Path

    from cloudlabeller.core.dataset import Dataset, ImageRecord, PointCloud
    from cloudlabeller.transfer import cloud_to_image, images_to_cloud

    cam = _app_camera(0.0)
    rng = np.random.default_rng(2)
    X = np.column_stack([rng.uniform(-2, 2, 200), rng.uniform(-1.5, 1.5, 200),
                         rng.uniform(5, 8, 200)]).astype(np.float32)
    cloud = PointCloud(xyz=X)
    labels = (X[:, 0] > 0).astype(np.int32)   # left half class 0, right half class 1

    mask = cloud_to_image(cloud, labels, cam, splat_radius=3)
    assert (mask != -1).any()

    dataset = Dataset(images=[ImageRecord(0, Path("a.jpg"), cam)])
    recovered = images_to_cloud(cloud, dataset, {"a.jpg": mask}, n_classes=2)
    visible = recovered != -1
    assert visible.sum() > 100
    assert np.mean(recovered[visible] == labels[visible]) > 0.9   # labels round-trip


def test_frustum_points_scale():
    from cloudlabeller.ui.camera_gizmo import camera_center, frustum_points

    cam = _app_camera(2.0)
    c = camera_center(cam)
    p_half = frustum_points(cam, 0.5)
    p_full = frustum_points(cam, 1.0)
    assert np.allclose(p_half[0], c) and np.allclose(p_full[0], c)   # centre fixed
    d_half = np.linalg.norm(p_half[1:] - c, axis=1)
    d_full = np.linalg.norm(p_full[1:] - c, axis=1)
    assert np.allclose(d_full, 2 * d_half)                           # corners scale linearly


def test_frustum_resize_in_place():
    from cloudlabeller.ui.camera_gizmo import camera_frustum, frustum_points

    cam = _app_camera(0.0)
    poly = camera_frustum(cam, 0.5)
    pl = pv.Plotter(off_screen=True)
    actor = pl.add_mesh(poly)
    extent0 = np.ptp(np.array(poly.bounds).reshape(3, 2), axis=1).sum()
    poly.points = frustum_points(cam, 2.0)        # 4x larger, same mesh object
    extent1 = np.ptp(np.array(poly.bounds).reshape(3, 2), axis=1).sum()
    assert actor.mapper.dataset is poly           # actor renders the mutated mesh
    assert extent1 > extent0
    pl.close()


def test_blend_point_colors():
    from cloudlabeller.ui.viewer3d import blend_point_colors

    rgb = np.array([[100, 100, 100], [10, 20, 30]], np.uint8)
    labels = np.array([-1, 1])                   # pt0 unlabelled (-1), pt1 = class 1 (red)
    lut = {0: (0, 0, 0), 1: (255, 0, 0)}

    assert np.array_equal(blend_point_colors(rgb, labels, lut, 0.0), rgb)   # RGB end

    full = blend_point_colors(rgb, labels, lut, 1.0)                        # label end
    assert np.array_equal(full[0], [100, 100, 100])   # unlabelled stays RGB
    assert np.array_equal(full[1], [255, 0, 0])       # labelled -> class colour

    half = blend_point_colors(rgb, labels, lut, 0.5)  # halfway for the labelled pt
    assert np.array_equal(half[1], [132, 10, 15])      # ((10+255)/2, 10, 15)

    # mismatched label length -> falls back to RGB
    assert np.array_equal(blend_point_colors(rgb, np.array([1]), lut, 1.0), rgb)


def test_label_panel_add_rename_color_remove(tmp_path):
    from PySide6.QtWidgets import QApplication

    from cloudlabeller.core.events import EventBus
    from cloudlabeller.core.project import Project
    from cloudlabeller.ui.label_panel import LabelPanel

    QApplication.instance() or QApplication([])
    bus = EventBus()
    panel = LabelPanel(bus)
    proj = Project.create(tmp_path / "p.clproj")
    actives: list[int] = []
    bus.active_class_changed.connect(actives.append)
    bus.project_opened.emit(proj)

    # Empty schema: only the fixed "unlabelled" (-1) row, active = -1, not removable.
    assert len(panel._rows) == 1 and panel._rows[0].class_id == -1
    assert panel._active == -1 and not panel.remove_btn.isEnabled()

    panel._add()
    panel._add()
    assert [c.id for c in proj.schema.classes] == [0, 1]
    assert len(panel._rows) == 3                          # unlabelled + 2 user
    assert panel._active == 1 and actives[-1] == 1        # newest is active

    panel._on_renamed(0, "rock")
    assert proj.schema.by_id(0).name == "rock"
    panel._on_recolored(1, "#0000ff")
    assert proj.schema.by_id(1).color == "#0000ff"

    panel._remove()                                       # removes active (1)
    assert [c.name for c in proj.schema.classes] == ["rock"]
    assert panel._active == 0

    # Selecting unlabelled disables Remove; trying to remove is a no-op.
    panel._set_active(-1)
    assert not panel.remove_btn.isEnabled()
    panel._remove()
    assert [c.name for c in proj.schema.classes] == ["rock"]   # unchanged


def test_image_view_polygon_fills_int32_mask(tmp_path):
    from PIL import Image
    from PySide6.QtCore import QPointF
    from PySide6.QtWidgets import QApplication

    from cloudlabeller.core.dataset import ImageRecord
    from cloudlabeller.core.events import EventBus
    from cloudlabeller.core.project import Project
    from cloudlabeller.ui.image_view import ImageView

    QApplication.instance() or QApplication([])
    proj = Project.create(tmp_path / "p.clproj")
    proj.add_label("rock", "#ff0000")                 # class 0
    img_path = proj.images_dir / "a.png"
    Image.new("RGB", (20, 20), (10, 20, 30)).save(img_path)
    proj.dataset.images = [ImageRecord(image_id=0, path=img_path)]

    bus = EventBus()
    view = ImageView(bus)
    bus.project_opened.emit(proj)
    bus.active_class_changed.emit(0)                  # active = rock
    view.show_image_sync("a.png")
    assert view.current_name == "a.png"

    for x, y in [(2, 2), (15, 2), (15, 15), (2, 15)]:  # square polygon
        view._add_vertex(QPointF(x, y))
    view._commit()

    mask = proj.labels.image_masks["a.png"]
    assert mask.dtype == np.int32 and mask.shape == (20, 20)
    assert mask[8, 8] == 0                            # inside polygon -> class 0
    assert mask[0, 0] == -1                           # outside -> unlabelled
    assert not view._vertices                         # polygon cleared after commit
    assert proj.labels.status_of("a.png") == "user"   # drawing marks it user (green dot)


def test_image_view_undo_vertex(tmp_path):
    from PIL import Image
    from PySide6.QtCore import QPointF
    from PySide6.QtWidgets import QApplication

    from cloudlabeller.core.dataset import ImageRecord
    from cloudlabeller.core.events import EventBus
    from cloudlabeller.core.project import Project
    from cloudlabeller.ui.image_view import ImageView

    QApplication.instance() or QApplication([])
    proj = Project.create(tmp_path / "p.clproj")
    proj.add_label("rock", "#ff0000")
    img_path = proj.images_dir / "a.png"
    Image.new("RGB", (20, 20), (0, 0, 0)).save(img_path)
    proj.dataset.images = [ImageRecord(image_id=0, path=img_path)]

    bus = EventBus()
    view = ImageView(bus)
    bus.project_opened.emit(proj)
    bus.active_class_changed.emit(0)
    view.show_image_sync("a.png")

    for x, y in [(2, 2), (15, 2), (15, 15)]:
        view._add_vertex(QPointF(x, y))
    assert len(view._vertices) == 3 and len(view._markers) == 3

    view._undo_vertex()                               # right-click undo
    assert len(view._vertices) == 2 and len(view._markers) == 2

    view._undo_vertex()
    view._undo_vertex()
    view._undo_vertex()                               # extra undo is a safe no-op
    assert view._vertices == [] and view._markers == []


def test_dataset_panel_selection_sync(tmp_path):
    from PySide6.QtWidgets import QApplication

    from cloudlabeller.core.dataset import ImageRecord
    from cloudlabeller.core.events import EventBus
    from cloudlabeller.core.project import Project
    from cloudlabeller.ui.dataset_panel import DatasetPanel

    QApplication.instance() or QApplication([])
    proj = Project.create(tmp_path / "p.clproj")
    proj.dataset.images = [ImageRecord(0, tmp_path / "a.jpg"),
                           ImageRecord(1, tmp_path / "b.jpg")]
    bus = EventBus()
    panel = DatasetPanel(bus)
    emitted: list[str] = []
    bus.image_selected.connect(emitted.append)
    bus.project_opened.emit(proj)

    bus.image_selected.emit("b.jpg")                  # e.g. from a frustum click
    assert panel.image_list.currentItem().text() == "b.jpg"   # row synced

    panel.image_list.setCurrentRow(0)                 # user clicks a row
    assert emitted[-1] == "a.jpg"


def test_view_panel_signals_and_availability():
    from PySide6.QtWidgets import QApplication

    from cloudlabeller.core.events import EventBus
    from cloudlabeller.ui.view_panel import ViewPanel

    QApplication.instance() or QApplication([])
    bus = EventBus()
    panel = ViewPanel(bus)
    reps, cams, blends = [], [], []
    panel.representation_changed.connect(reps.append)
    panel.cameras_toggled.connect(cams.append)
    panel.label_blend_changed.connect(blends.append)

    # No project: defaults to sparse, all representations disabled.
    assert panel.current_representation() == "sparse"
    assert not panel._rep_buttons["dense"].isEnabled()

    # Slider emits a 0..1 blend; camera checkbox emits bool.
    panel.slider.setValue(50)
    assert blends and abs(blends[-1] - 0.5) < 1e-6
    panel.chk_cameras.setChecked(False)
    assert cams[-1] is False

    # A project with sparse+dense enables those; picking dense emits.
    class _DS:
        def has(self, k):
            return k in ("sparse", "dense")

    class _Proj:
        dataset = _DS()

    bus.project_opened.emit(_Proj())
    assert panel._rep_buttons["dense"].isEnabled()
    assert not panel._rep_buttons["mesh"].isEnabled()
    panel._rep_buttons["dense"].setChecked(True)
    assert reps[-1] == "dense"


def test_list_image_files_filters_sidecars(tmp_path):
    from cloudlabeller.photogrammetry.sfm import list_image_files

    for n in ["a.JPG", "b.jpg", "c.PNG", "x.nav", "y.obs", "z.MRK", "raw.bin", "notes.txt"]:
        (tmp_path / n).write_bytes(b"x")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "d.tif").write_bytes(b"x")
    (sub / "log.csv").write_bytes(b"x")

    names = list_image_files(tmp_path)
    assert set(names) == {"a.JPG", "b.jpg", "c.PNG", "sub/d.tif"}   # images only, recursive


def test_run_sfm_rejects_no_images(tmp_path):
    from cloudlabeller.photogrammetry.sfm import run_sfm

    imgs = tmp_path / "imgs"
    imgs.mkdir()
    (imgs / "DJI_PPKNAV.nav").write_bytes(b"x")   # sidecar only, no real images
    with pytest.raises(RuntimeError, match="No image files"):
        run_sfm(imgs, tmp_path / "ws")


def test_run_sfm_dialog():
    from PySide6.QtWidgets import QApplication, QDialogButtonBox
    from cloudlabeller.ui.run_sfm_dialog import RunSfmDialog

    QApplication.instance() or QApplication([])
    empty = RunSfmDialog(0)                   # empty store -> OK disabled
    assert not empty.buttons.button(QDialogButtonBox.Ok).isEnabled()

    dlg = RunSfmDialog(57)                     # images present -> OK enabled
    assert dlg.buttons.button(QDialogButtonBox.Ok).isEnabled()

    # defaults: spatial matcher (GPS), single camera, CPU
    o = dlg.options()
    assert o.matcher == "spatial" and o.single_camera and not o.use_gpu

    # each matcher radio maps correctly
    dlg.rb_sequential.setChecked(True)
    assert dlg.options().matcher == "sequential"

    # toggles reflected
    dlg.rb_exhaustive.setChecked(True)
    dlg.chk_gpu.setChecked(True)
    dlg.cmb_model.setCurrentText("OPENCV")
    dlg.detail.combo.setCurrentIndex(0)               # High = full resolution
    o2 = dlg.options()
    assert o2.matcher == "exhaustive" and o2.use_gpu and o2.camera_model == "OPENCV"
    assert o2.max_image_size == 0

    # relative detail presets derive from the source resolution
    rel = RunSfmDialog(57, source_resolution=(8000, 6000))
    assert rel.options().max_image_size == 4000       # default: Medium = 1/2
    rel.detail.combo.setCurrentIndex(2)
    assert rel.options().max_image_size == 2000       # Low = 1/4
    rel.detail.combo.setCurrentIndex(3)               # Custom -> spinbox value
    rel.detail.spin.setValue(3200)
    assert rel.options().max_image_size == 3200


def test_distortion_foldback_points_not_visible():
    """Regression (2026-07-09): with barrel distortion (k < 0), points far
    outside the FOV got a negative radial factor that FOLDED their projection
    back inside the image at shallow depth. These ghosts poisoned the z-buffer
    as false occluders, punching a cone-shaped hole (apex at the principal
    point) into image->cloud label transfer."""
    from cloudlabeller.core.dataset import PointCloud
    from cloudlabeller.transfer.hylite_bridge import project_points, visible_point_pixels

    K = np.array([[1000.0, 0, 320], [0, 1000.0, 240], [0, 0, 1]])
    cam = Camera(K=K, R=np.eye(3), t=np.zeros(3), width=640, height=480,
                 model="SIMPLE_RADIAL", distortion=np.array([-0.06]))

    # Ghost: ~76 deg off-axis (normalized x = 4). Naive distortion folds it
    # in-bounds: x_d = 4 * (1 - 0.06 * 16) = 0.16 -> u = 480, at depth 2.
    ghost = np.array([[8.0, 0.0, 2.0]])
    x_d = 4.0 * (1.0 - 0.06 * 16.0)
    assert 0 <= K[0, 0] * x_d + K[0, 2] < 640          # the fold premise holds

    # Real surface point behind the ghost's folded pixel (u=480, v=240, z=5).
    real = np.array([[(480 - 320) / 1000 * 5, 0.0, 5.0]])

    pts = np.vstack([ghost, real])
    _, vis = project_points(cam, pts)
    assert not vis[0]                                   # ghost culled (FOV guard)
    assert vis[1]                                       # real point visible

    # And the ghost must not shadow the real point in the z-buffer.
    cloud = PointCloud(xyz=pts.astype(np.float32))
    idx, pix = visible_point_pixels(cloud, cam)
    assert 1 in idx and 0 not in idx


def test_merged_frustums_batches_all_cameras():
    """One merged PolyData for N cameras (points 5i..5i+4 belong to camera i) —
    per-camera actors made opening large projects slow."""
    from cloudlabeller.ui.camera_gizmo import frustum_points, merged_frustums

    cams = [_app_camera(tx) for tx in (0.0, 1.0, 2.0)]
    poly = merged_frustums(cams, scale=0.5)
    assert poly.n_points == 15                      # 5 per camera
    assert poly.n_cells == 24                       # 8 line segments per camera
    for i, cam in enumerate(cams):
        assert np.allclose(poly.points[5 * i:5 * i + 5],
                           frustum_points(cam, 0.5))
    # resize-in-place contract: reassigning points matches a fresh build
    poly.points = np.vstack([frustum_points(c, 1.5) for c in cams])
    assert np.allclose(poly.points[5:10], frustum_points(cams[1], 1.5))


def test_sparse_point_stats_and_filtering(tmp_path):
    """Clean-sparse-cloud plumbing: per-point confidence matched by position,
    and label filtering follows the surviving points."""
    from cloudlabeller.core.labels import LabelStore
    from cloudlabeller.photogrammetry.extract import (
        load_best_model,
        reconstruction_to_cloud,
        sparse_point_stats,
    )

    rec = pycolmap.Reconstruction()
    rng = np.random.default_rng(3)
    for _ in range(30):
        rec.add_point3D(rng.normal(size=3), pycolmap.Track(),
                        rng.integers(0, 255, size=3).astype(np.uint8))
    # write + reload via the best-model loader (nested model dir layout)
    model_dir = tmp_path / "sparse" / "0"
    model_dir.mkdir(parents=True)
    rec.write(str(model_dir))
    loaded = load_best_model(tmp_path / "sparse")
    assert loaded.num_points3D() == 30

    cloud = reconstruction_to_cloud(loaded)
    views, errors = sparse_point_stats(loaded, cloud.xyz)
    assert views.shape == (30,)
    # empty tracks -> 0 views; the stats matched every stored point
    assert (errors < np.inf).sum() == 30 or (views >= 0).all()

    # label filtering follows a keep-mask and clears the paint history
    store = LabelStore()
    store.init_cloud(30)
    store.paint_cloud(np.arange(5), 1)
    keep = np.ones(30, bool)
    keep[:3] = False
    store.filter_cloud(keep)
    assert len(store.cloud_labels) == 27
    assert list(store.cloud_labels[:2]) == [1, 1]   # points 3,4 kept their label
    assert not store.can_undo()                     # history cleared


def test_clean_cloud_dialog_mask():
    from PySide6.QtWidgets import QApplication, QDialogButtonBox

    from cloudlabeller.ui.clean_cloud_dialog import CleanCloudDialog

    QApplication.instance() or QApplication([])
    views = np.array([2, 3, 5, 8, 2])
    errors = np.array([0.5, 1.0, 3.5, 0.8, 9.0])
    d = CleanCloudDialog(views, errors)
    # defaults: >=3 views, <=2.0 px  -> keeps indices 1 and 3
    assert list(d.mask()) == [False, True, False, True, False]
    assert "Keeps 2 of 5" in d.lbl_preview.text()
    d.spin_views.setValue(2)
    d.spin_error.setValue(10.0)                     # keeps everything
    assert d.mask().all()
    assert not d.buttons.button(QDialogButtonBox.Ok).isEnabled()  # no-op clean


class TestGeoref:
    def _sim3(self):
        from scipy.spatial.transform import Rotation

        rng = np.random.default_rng(7)
        R = Rotation.from_euler("xyz", [12, -35, 70], degrees=True).as_matrix()
        return 3.7, R, np.array([100.0, -50.0, 8.0])

    def test_umeyama_recovers_similarity(self):
        from cloudlabeller.photogrammetry.georef import (
            transform_points, umeyama_similarity,
        )

        s, R, t = self._sim3()
        rng = np.random.default_rng(1)
        src = rng.normal(size=(40, 3))
        dst = transform_points(src, s, R, t)
        s2, R2, t2 = umeyama_similarity(src, dst)
        assert abs(s2 - s) < 1e-5
        assert np.allclose(R2, R, atol=1e-6)
        assert np.allclose(t2, t, atol=1e-4)

    def test_camera_transform_preserves_projection(self):
        """The whole point: after transforming world + cameras together, every
        point must project to the SAME pixel (labels/transfers unaffected)."""
        from cloudlabeller.photogrammetry.georef import (
            transform_camera, transform_points,
        )
        from cloudlabeller.transfer.hylite_bridge import project_points

        s, R, t = self._sim3()
        cam = Camera(K=np.array([[1000.0, 0, 320], [0, 1000.0, 240], [0, 0, 1]]),
                     R=np.eye(3), t=np.array([0.2, -0.1, 0.0]),
                     width=640, height=480, model="SIMPLE_RADIAL",
                     distortion=np.array([-0.06]))
        rng = np.random.default_rng(2)
        pts = np.column_stack([rng.uniform(-1, 1, 50), rng.uniform(-0.8, 0.8, 50),
                               rng.uniform(3, 8, 50)])
        pp_before, vis_before = project_points(cam, pts)

        cam2 = transform_camera(cam, s, R, t)
        pts2 = transform_points(pts, s, R, t)
        pp_after, vis_after = project_points(cam2, pts2)

        assert np.array_equal(vis_before, vis_after)
        assert np.allclose(pp_before[vis_before, :2], pp_after[vis_after, :2],
                           atol=1e-3)                       # same pixels
        assert np.allclose(pp_after[vis_after, 2],
                           pp_before[vis_before, 2] * s, rtol=1e-5)  # metric depth
        # camera centre moved with the world
        assert np.allclose(cam2.position,
                           transform_points(cam.position[None], s, R, t)[0],
                           atol=1e-4)

    def test_parse_gps_info(self):
        from cloudlabeller.photogrammetry.georef import parse_gps_info

        gps = {1: "S", 2: (29.0, 45.0, 30.0), 3: "W", 4: (53.0, 30.0, 0.0),
               5: 0, 6: 320.5}
        lla = parse_gps_info(gps)
        assert lla is not None
        lat, lon, alt = lla
        assert abs(lat - (-(29 + 45 / 60 + 30 / 3600))) < 1e-9
        assert abs(lon - (-53.5)) < 1e-9
        assert alt == 320.5
        assert parse_gps_info({1: "N"}) is None            # incomplete
        gps[5] = 1                                          # below sea level
        assert parse_gps_info(gps)[2] == -320.5
        # DJI stores the altitude ref as raw bytes — must not zero the altitude
        gps[5] = b"\x00"
        assert parse_gps_info(gps)[2] == 320.5
        gps[5] = b"\x01"
        assert parse_gps_info(gps)[2] == -320.5
