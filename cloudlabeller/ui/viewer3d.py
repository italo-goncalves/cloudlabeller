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

"""3D viewer: PyVista/VTK embedded in Qt.

Renders the selected representation (sparse cloud / dense cloud / mesh), the
camera frustums, and blends point colours between photographic RGB and label
colours. Geometry is rebuilt only when the data or representation changes;
colour and camera-visibility updates touch the existing actors so they stay
responsive.
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from cloudlabeller.config import AppConfig
from cloudlabeller.core.events import EventBus
from cloudlabeller.core.label_schema import UNLABELLED_ID
from cloudlabeller.ui.camera_gizmo import (
    camera_center,
    camera_frustum,
    frustum_points,
    merged_frustums,
)
from cloudlabeller.ui.lasso import composite_projection_matrix, points_in_lasso

SELECTION_COLOR = (255, 136, 0)   # orange tint for lasso-selected points
EXCLUDE_COLOR = (255, 0, 255)     # magenta: clean-preview points to be removed


class _LassoPathActor:
    """Draws the in-progress lasso as a VTK 2D actor in display coordinates.

    A Qt widget stacked over the VTK window cannot be composited above the
    native OpenGL surface (it hides the scene while shown), so the path is
    rendered *inside* the VTK scene instead.
    """

    def __init__(self, plotter) -> None:
        from vtkmodules.vtkCommonCore import vtkPoints
        from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
        from vtkmodules.vtkRenderingCore import (
            vtkActor2D, vtkCoordinate, vtkPolyDataMapper2D,
        )

        self._plotter = plotter
        self._points = vtkPoints()
        self._cells = vtkCellArray()
        self._poly = vtkPolyData()
        self._poly.SetPoints(self._points)
        self._poly.SetLines(self._cells)

        coord = vtkCoordinate()
        coord.SetCoordinateSystemToDisplay()
        mapper = vtkPolyDataMapper2D()
        mapper.SetInputData(self._poly)
        mapper.SetTransformCoordinate(coord)
        self.actor = vtkActor2D()
        self.actor.SetMapper(mapper)
        prop = self.actor.GetProperty()
        prop.SetColor(1.0, 0.53, 0.0)          # match SELECTION_COLOR
        prop.SetLineWidth(2.0)
        self.actor.SetVisibility(False)
        plotter.renderer.AddViewProp(self.actor)
        self._path: list[tuple[float, float]] = []

    def begin(self, xy: tuple[float, float]) -> None:
        self._path = [xy]
        self.actor.SetVisibility(True)
        self._sync()

    def add_point(self, xy: tuple[float, float]) -> None:
        self._path.append(xy)
        self._sync()

    def clear(self) -> None:
        self._path = []
        self.actor.SetVisibility(False)
        self._plotter.render()

    def _sync(self) -> None:
        """Rewrite the VTK polyline from the current path (closed loop)."""
        n = len(self._path)
        self._points.SetNumberOfPoints(n)
        for i, (x, y) in enumerate(self._path):
            self._points.SetPoint(i, x, y, 0.0)
        self._cells.Reset()
        if n >= 2:
            self._cells.InsertNextCell(n + 1)      # closed loop
            for i in range(n):
                self._cells.InsertCellPoint(i)
            self._cells.InsertCellPoint(0)
        self._points.Modified()
        self._poly.Modified()
        self._plotter.render()

# One camera's renderable pieces: the frustum/marker actors, the frustum mesh
# (mutated in place on resize) and the camera (to recompute its geometry).
# Per-camera bookkeeping inside the two merged actors: ``index`` addresses the
# camera's slice of the batched geometry (points 5i..5i+4, marker row i).
_Frustum = namedtuple("_Frustum", "index camera")


def dim_other_classes(colors, labels, class_id, keep=0.15):
    """Dim every point NOT of ``class_id`` to ``keep`` brightness (class
    isolation view). No-op when labels are missing/mismatched."""
    if labels is None or len(labels) != len(colors):
        return colors
    out = colors.copy()
    other = labels != class_id
    out[other] = (out[other] * keep).astype(np.uint8)
    return out


def blend_point_colors(rgb, labels, lut, blend, unlabelled_id=UNLABELLED_ID):
    """Blend per-point RGB toward label colours by ``blend`` in [0, 1].

    Unlabelled points keep their RGB regardless of ``blend``, so only labelled
    regions shift toward their class colour. Returns uint8 (N, 3).
    """
    rgb = rgb.astype(np.float32)
    if blend <= 0 or labels is None or len(labels) != len(rgb):
        return rgb.astype(np.uint8)
    target = rgb.copy()
    for class_id, color in lut.items():
        if class_id == unlabelled_id:
            continue
        target[labels == class_id] = color
    return ((1.0 - blend) * rgb + blend * target).astype(np.uint8)


class Viewer3D(QWidget):
    """The 3D pane: renders the active cloud or mesh with label colours,
    camera frustums with status dots, and hosts the selection/paint tools
    (see the module docstring for the rendering strategy)."""

    selection_changed = Signal(int)               # number of lasso-selected points

    def __init__(self, bus: EventBus, config: AppConfig) -> None:
        super().__init__()
        self.bus = bus
        self.config = config
        self.project = None

        self.plotter = QtInteractor(self)
        self.plotter.set_background(config.background_color)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plotter.interactor)

        self._geometry_actor = None
        self._camera_actors: list = []
        self._frustums: dict[str, _Frustum] = {}      # image name -> _Frustum
        self._frustum_poly: pv.PolyData | None = None  # merged wireframes
        self._marker_poly: pv.PolyData | None = None   # merged status dots
        self._hl_actors: list = []                     # highlight overlay actors
        self._hl_poly: pv.PolyData | None = None
        self._highlighted: str | None = None
        self._picking_enabled = False
        self._press_pos = None
        self._frustum_base_scale = config.default_brush_radius * 5
        self._frustum_scale_mult = 1.0
        self._poly: pv.PolyData | None = None      # active cloud, for in-place recolour
        self._representation = "sparse"
        self._label_blend = 0.0
        self._show_cameras = True
        self._active_class = -1
        self._isolated: int | None = None          # class isolation (dim others)
        self._exclusion_preview: np.ndarray | None = None  # clean-cloud keep mask
        self._preserve_view_once = False           # skip refit on next rebuild

        # Lasso state (dense-cloud-only labelling tool).
        self._lasso_mode = False                   # tool armed (left-drag draws)
        self._lasso_active = False                 # a drag is in progress
        self._lasso_path: list = []                # path in VTK display coords
        self._lasso_selection: np.ndarray | None = None   # dense-point indices
        self._overlay = _LassoPathActor(self.plotter)

        bus.project_opened.connect(self._on_project_opened)
        bus.cloud_changed.connect(self._rebuild)
        bus.mesh_changed.connect(self._rebuild)
        bus.images_changed.connect(self._build_cameras)
        bus.cloud_labels_changed.connect(self._update_colors)
        bus.schema_changed.connect(self._update_colors)   # recolour on palette edits
        bus.image_selected.connect(self.set_highlighted_image)
        bus.image_labels_changed.connect(self._refresh_marker)  # recolour status dot
        bus.active_class_changed.connect(lambda c: setattr(self, "_active_class", c))
        bus.class_isolation_changed.connect(self.set_isolated_class)

    # -- project wiring ----------------------------------------------------
    def _on_project_opened(self, project) -> None:
        self.project = project
        self._rebuild()

    # -- public controls (wired from ViewPanel) ---------------------------
    def set_representation(self, kind: str) -> None:
        """Switch what is rendered: "sparse", "dense" or "mesh". Only the
        geometry changes — frustums and the current view are kept."""
        self._representation = kind
        if kind == "mesh":                         # lasso works on clouds only
            self.set_lasso_mode(False)
        self._lasso_selection = None               # indices are per-cloud
        self.selection_changed.emit(0)
        self._rebuild_geometry()

    # -- lasso (dense cloud only) -------------------------------------------
    def set_lasso_mode(self, on: bool) -> None:
        """Arm/disarm the lasso. While armed, left-drag draws a lasso instead of
        rotating the camera, and the camera frustums are not clickable."""
        on = bool(on)
        if on == self._lasso_mode:
            return
        self._lasso_mode = on
        self._lasso_active = False
        self._lasso_path = []
        self._overlay.clear()
        if not on:
            self.clear_lasso_selection()
        self.plotter.interactor.setCursor(Qt.CrossCursor if on else Qt.ArrowCursor)

    def clear_lasso_selection(self) -> None:
        if self._lasso_selection is not None:
            self._lasso_selection = None
            self._update_colors()
        self.selection_changed.emit(0)

    def lasso_selection(self):
        """Indices selected by the lasso in the ACTIVE cloud (or None)."""
        return self._lasso_selection

    def apply_lasso_label(self) -> None:
        """Assign the active label to the lasso-selected dense-cloud points.
        (Labelling is dense-only; on the sparse cloud the lasso feeds deletion.)"""
        if (self.project is None or self._lasso_selection is None
                or self._representation != "dense"):
            return
        dense = self.project.dataset.dense_cloud
        if dense is None:
            return
        labels = self.project.labels
        if (labels.dense_cloud_labels is None
                or len(labels.dense_cloud_labels) != dense.n_points):
            labels.init_dense_cloud(dense.n_points)
        labels.paint_dense_cloud(self._lasso_selection, self._active_class)
        self._lasso_selection = None
        self.selection_changed.emit(0)
        self.bus.cloud_labels_changed.emit()       # recolour + autosave dirty flag

    def set_cameras_visible(self, visible: bool) -> None:
        """Toggle visibility on existing actors — no rebuild."""
        self._show_cameras = visible
        for actor in self._camera_actors + self._hl_actors:
            actor.SetVisibility(visible)
        self.plotter.render()

    def set_highlighted_image(self, name: str | None) -> None:
        """Move the highlight overlay (a single cyan frustum + enlarged dot)."""
        if name == self._highlighted:
            return
        self._highlighted = name
        self._remove_highlight()
        self._apply_highlight(name)
        self.plotter.render()

    def set_frustum_scale(self, multiplier: float) -> None:
        """Resize all frustums by mutating the merged points in place (no rebuild)."""
        self._frustum_scale_mult = multiplier
        scale = self._frustum_base_scale * multiplier
        if self._frustum_poly is not None and self._frustums:
            entries = sorted(self._frustums.values(), key=lambda e: e.index)
            self._frustum_poly.points = np.vstack(
                [frustum_points(e.camera, scale) for e in entries])
        if self._hl_poly is not None and self._highlighted in self._frustums:
            self._hl_poly.points = frustum_points(
                self._frustums[self._highlighted].camera, scale)
        self.plotter.render()

    def set_label_blend(self, t: float) -> None:
        self._label_blend = max(0.0, min(1.0, t))
        self._update_colors()

    def set_isolated_class(self, class_id) -> None:
        """Show only ``class_id``'s points at full brightness (None = off)."""
        self._isolated = class_id
        self._update_colors()

    def set_exclusion_preview(self, keep_mask) -> None:
        """Live preview for Clean Sparse Cloud: points about to be REMOVED are
        tinted magenta in the sparse view. ``None`` clears the preview."""
        self._exclusion_preview = (None if keep_mask is None
                                   else np.asarray(keep_mask, bool))
        self._update_colors()

    def view_from_camera(self, name: str) -> None:
        """Snap the 3D view to a photo's exact viewpoint (pose + field of view),
        for photo-vs-cloud comparison. Triggered by double-clicking a frustum."""
        entry = self._frustums.get(name)
        if entry is None:
            return
        cam = entry.camera
        position = cam.position
        forward = cam.R[2]                          # camera +z (view dir) in world
        up = -cam.R[1]                              # image up in world
        focal = position + forward * max(self._frustum_scale() * 10.0, 1e-3)
        self.plotter.camera_position = [position.tolist(), focal.tolist(),
                                        up.tolist()]
        fov = np.degrees(2.0 * np.arctan(cam.height / (2.0 * cam.K[1, 1])))
        self.plotter.camera.view_angle = float(fov)
        self.plotter.reset_camera_clipping_range()
        self.plotter.render()

    # -- data helpers ------------------------------------------------------
    def _active_cloud(self):
        """The cloud behind the current representation (None for mesh)."""
        ds = self.project.dataset if self.project else None
        if ds is None:
            return None
        if self._representation == "dense":
            return ds.dense_cloud
        if self._representation == "mesh":
            return None
        return ds.cloud

    def _point_colors(self, cloud) -> np.ndarray:
        """Per-point uint8 RGB, blended toward label colours by ``_label_blend``.

        Unlabelled points keep their RGB regardless of the blend, so only
        labelled regions shift toward their class colour.
        """
        n = cloud.n_points
        rgb = (cloud.rgb if cloud.rgb is not None
               else np.full((n, 3), 200, np.uint8))
        if self.project is None:
            return rgb
        labels = (self.project.labels.dense_cloud_labels
                  if self._representation == "dense"
                  else self.project.labels.cloud_labels)
        lut = self.project.schema.lookup_table()
        colors = blend_point_colors(rgb, labels, lut, self._label_blend)
        if self._isolated is not None:
            colors = dim_other_classes(colors, labels, self._isolated)
        # Clean-cloud preview: mark points that the filter would remove.
        keep = self._exclusion_preview
        if (self._representation == "sparse" and keep is not None
                and len(keep) == len(colors)):
            colors[~keep] = EXCLUDE_COLOR
        # Highlight the lasso selection (either cloud view) on top of everything.
        sel = self._lasso_selection
        if sel is not None and sel.size and sel.max() < len(colors):
            colors[sel] = SELECTION_COLOR
        return colors

    # -- rendering ---------------------------------------------------------
    def _rebuild_geometry(self) -> None:
        """Rebuild only the active geometry (cloud or mesh). Cameras untouched —
        switching representation must not reload the frustums."""
        if self._geometry_actor is not None:
            self.plotter.remove_actor(self._geometry_actor)
            self._geometry_actor = None
        self._poly = None
        if not self.project:
            self.plotter.render()
            return
        if self._representation == "mesh":
            self._build_mesh()
        else:
            self._build_cloud(self._active_cloud())
        self.plotter.render()

    def preserve_view_once(self) -> None:
        """Skip the home-view refit on the NEXT rebuild. In-place edits (point
        deletion / confidence cleaning) must not yank the camera away from
        where the user is looking."""
        self._preserve_view_once = True

    def _rebuild(self) -> None:
        """Full rebuild (data changed): geometry + cameras + refit the view."""
        self._rebuild_geometry()
        self._build_cameras()
        if self._preserve_view_once:
            self._preserve_view_once = False
        else:
            self._reset_view()
        self.plotter.render()

    def _scene_up(self) -> np.ndarray | None:
        """The scene's up direction.

        Georeferenced projects (ENU or a projected CRS) have +Z up by
        construction — use it, no guessing. Otherwise COLMAP's world axes are
        arbitrary and up is derived from the camera orientations: COLMAP's
        image y-axis points DOWN in the image, so world up ≈ -mean(R[1]).
        (The + sign used before was tuned on an unreferenced solve whose
        frame turned out to be flipped — georeferencing exposed it.)"""
        if self.project and self.project.settings.get("georeferenced"):
            return np.array([0.0, 0.0, 1.0])
        ups = [-e.camera.R[1] for e in self._frustums.values()]
        if not ups:
            return None
        up = np.mean(ups, axis=0)
        n = np.linalg.norm(up)
        return up / n if n > 1e-9 else None

    def _reset_view(self) -> None:
        """Home view: from the 'southeast' diagonal, elevated, looking down at
        the scene centre with the cameras' average up as vertical."""
        up = self._scene_up()
        if self._poly is None or not self._poly.n_points or up is None:
            self.plotter.reset_camera()
            return
        b = self._poly.bounds
        center = np.array([(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2])
        diag = float(np.linalg.norm([b[1] - b[0], b[3] - b[2], b[5] - b[4]]))
        ref = np.array([1.0, 0.0, 0.0])
        if abs(ref @ up) > 0.9:                      # up ~ x: pick another axis
            ref = np.array([0.0, 0.0, 1.0])
        east = np.cross(ref, up)
        east /= np.linalg.norm(east)
        south = np.cross(up, east)                   # in-plane, orthogonal to east
        se = (east + south) / np.sqrt(2.0)
        position = center + diag * (0.8 * se + 0.6 * up)
        self.plotter.camera_position = [position.tolist(), center.tolist(),
                                        up.tolist()]
        self.plotter.reset_camera_clipping_range()

    def _build_cloud(self, cloud) -> None:
        """Add the cloud as a point-rendered PolyData with blended colours."""
        if cloud is None:
            return
        poly = pv.PolyData(cloud.xyz)
        poly["colors"] = self._point_colors(cloud)
        self._poly = poly
        self._geometry_actor = self.plotter.add_mesh(
            poly, scalars="colors", rgb=True,
            render_points_as_spheres=False, point_size=2.0)

    def _build_mesh(self) -> None:
        """Add the triangle mesh with per-vertex colours (GPU-interpolated)."""
        mesh = self.project.dataset.mesh
        if mesh is None:
            return
        faces = np.hstack([np.full((len(mesh.faces), 1), 3), mesh.faces]).ravel()
        poly = pv.PolyData(mesh.vertices, faces)
        # Point-data RGB: the GPU interpolates the vertex colours across each
        # triangle, so a triangle's colour is the barycentric blend (average at
        # the centroid) of its 3 vertices — labelled vertices tint toward their
        # class, unlabelled ones keep RGB, and a triangle with no labelled
        # vertex stays pure RGB.
        poly["colors"] = self._mesh_colors(mesh)
        self._poly = poly
        self._geometry_actor = self.plotter.add_mesh(poly, scalars="colors", rgb=True)

    def _mesh_colors(self, mesh) -> np.ndarray:
        """Per-vertex RGB blended toward label colours by the shared slider value."""
        rgb = (mesh.vertex_colors if mesh.vertex_colors is not None
               else np.full((len(mesh.vertices), 3), 188, np.uint8))
        if self.project is None:
            return rgb
        colors = blend_point_colors(rgb, self.project.labels.mesh_labels,
                                    self.project.schema.lookup_table(),
                                    self._label_blend)
        if self._isolated is not None:
            colors = dim_other_classes(colors, self.project.labels.mesh_labels,
                                       self._isolated)
        return colors

    def _update_colors(self) -> None:
        """Recompute colours in place (blend / label edits) without rebuilding."""
        if self._poly is None:
            return
        if self._representation == "mesh":
            mesh = self.project.dataset.mesh if self.project else None
            if mesh is None:
                return
            self._poly["colors"] = self._mesh_colors(mesh)
        else:
            cloud = self._active_cloud()
            if cloud is None:
                return
            self._poly["colors"] = self._point_colors(cloud)
        self.plotter.render()

    def _marker_color(self, name: str) -> str:
        """Dot colour by label status (shared with the Dataset pane); the
        frustum wireframe shows the highlight."""
        from cloudlabeller.ui.status import STATUS_COLORS

        status = self.project.labels.status_of(name) if self.project else "none"
        return STATUS_COLORS.get(status, "#ff3333")

    def _marker_rgb(self, name: str) -> tuple[int, int, int]:
        h = self._marker_color(name).lstrip("#")
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

    def _refresh_marker(self, name: str) -> None:
        """Recolour one camera's status dot after its labels changed."""
        entry = self._frustums.get(name)
        if entry is not None and self._marker_poly is not None:
            colors = self._marker_poly["colors"]
            colors[entry.index] = self._marker_rgb(name)
            self._marker_poly["colors"] = colors
            if name == self._highlighted:              # keep the big dot in sync
                self._remove_highlight()
                self._apply_highlight(name)
            self.plotter.render()

    def _build_cameras(self) -> None:
        """Full rebuild of the camera actors (only on dataset/cloud change).

        All frustum wireframes render as ONE merged actor and all status dots
        as another (per-point RGB scalars) — two ``add_mesh`` calls total.
        Per-camera actors made opening large projects crawl: hundreds of VTK
        actors take seconds to add and slow down every subsequent render.
        """
        for actor in self._camera_actors:
            self.plotter.remove_actor(actor)
        self._camera_actors.clear()
        self._frustums.clear()
        self._remove_highlight()
        self._frustum_poly = self._marker_poly = None
        records = ([r for r in self.project.dataset.solved_images()
                    if r.camera is not None] if self.project else [])
        if not records:
            self.plotter.render()
            return

        self._frustum_base_scale = self._frustum_scale()
        scale = self._frustum_base_scale * self._frustum_scale_mult
        cams = [r.camera for r in records]
        self._frustums = {r.name: _Frustum(i, r.camera)
                          for i, r in enumerate(records)}

        self._frustum_poly = merged_frustums(cams, scale)
        frustum_actor = self.plotter.add_mesh(
            self._frustum_poly, color="#ffcc00", line_width=1.5)
        self._marker_poly = pv.PolyData(np.vstack([camera_center(c) for c in cams]))
        self._marker_poly["colors"] = np.array(
            [self._marker_rgb(r.name) for r in records], dtype=np.uint8)
        marker_actor = self.plotter.add_mesh(
            self._marker_poly, scalars="colors", rgb=True,
            render_points_as_spheres=True, point_size=8.0)

        self._camera_actors += [frustum_actor, marker_actor]
        for actor in self._camera_actors:
            actor.SetVisibility(self._show_cameras)
        self._apply_highlight(self._highlighted)
        self._enable_picking()
        self.plotter.render()

    # -- highlight overlay (one camera, drawn on top of the merged actors) ---
    def _remove_highlight(self) -> None:
        for actor in self._hl_actors:
            self.plotter.remove_actor(actor)
        self._hl_actors = []
        self._hl_poly = None

    def _apply_highlight(self, name: str | None) -> None:
        """Draw the selected image's frustum in cyan with a bigger marker."""
        entry = self._frustums.get(name)
        if entry is None:
            return
        scale = self._frustum_base_scale * self._frustum_scale_mult
        self._hl_poly = camera_frustum(entry.camera, scale=scale)
        self._hl_actors = [
            self.plotter.add_mesh(self._hl_poly, color="#00e5ff", line_width=3.0),
            self.plotter.add_mesh(pv.PolyData(camera_center(entry.camera)[None, :]),
                                  color=self._marker_color(name),
                                  render_points_as_spheres=True, point_size=12.0),
        ]
        for actor in self._hl_actors:
            actor.SetVisibility(self._show_cameras)

    def _enable_picking(self) -> None:
        """Left-click near a frustum to select its image.

        We install a Qt event filter and, on click, project every frustum's
        vertices to screen and pick the camera nearest the cursor. This is robust
        to thin-wireframe imprecision, clustered frustums at low zoom, and any
        small HiDPI coordinate error — far more forgiving than cell-picking lines.
        """
        if self._picking_enabled:
            return
        try:
            self.plotter.interactor.installEventFilter(self)
            self._picking_enabled = True
        except Exception:
            pass  # picking is optional; Dataset-pane selection still works

    def eventFilter(self, obj, event) -> bool:
        if obj is getattr(self.plotter, "interactor", None):
            if self._lasso_mode:
                if self._lasso_event(event):
                    return True                     # consumed: no camera rotation
            elif event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                self._press_pos = event.position()
            elif (event.type() == QEvent.MouseButtonRelease
                    and event.button() == Qt.LeftButton):
                self._pick_at(event.position())     # frustum picking (not in lasso mode)
            elif (event.type() == QEvent.MouseButtonDblClick
                    and event.button() == Qt.LeftButton):
                name = self._camera_at(event.position())
                if name:
                    self.view_from_camera(name)     # fly to the photo's viewpoint
                    return True
        return super().eventFilter(obj, event)

    def _to_display(self, pos) -> tuple[float, float]:
        """Qt widget coords (logical, top-left) -> VTK display (physical, bottom-left)."""
        scale = self.plotter.interactor.devicePixelRatioF() or 1.0
        height = self.plotter.render_window.GetSize()[1]
        return (pos.x() * scale, height - pos.y() * scale)

    def _lasso_event(self, event) -> bool:
        """Handle mouse events while the lasso is armed. Returns True if consumed.
        Middle/right buttons pass through so pan/zoom still work."""
        etype = event.type()
        if etype == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            xy = self._to_display(event.position())
            self._lasso_active = True
            self._lasso_path = [xy]
            self._overlay.begin(xy)
            return True
        if etype == QEvent.MouseMove and self._lasso_active:
            xy = self._to_display(event.position())
            self._lasso_path.append(xy)
            self._overlay.add_point(xy)
            return True
        if (etype == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton
                and self._lasso_active):
            self._lasso_active = False
            self._overlay.clear()
            self._finish_lasso()
            return True
        return False

    def _finish_lasso(self) -> None:
        """Select the ACTIVE cloud's points whose projection lies inside the
        path (dense: for labelling; sparse: for deleting stray points)."""
        path, self._lasso_path = self._lasso_path, []
        cloud = self._active_cloud() if self._representation != "mesh" else None
        if cloud is None or len(path) < 3:
            return
        width, height = self.plotter.render_window.GetSize()   # physical px
        matrix = composite_projection_matrix(self.plotter.renderer)
        selected = points_in_lasso(cloud.xyz, matrix, width, height, np.asarray(path))
        self._lasso_selection = selected if selected.size else None
        self._update_colors()
        self.selection_changed.emit(int(selected.size))

    def _pick_at(self, pos) -> None:
        if self._press_pos is not None:
            dx, dy = pos.x() - self._press_pos.x(), pos.y() - self._press_pos.y()
            if dx * dx + dy * dy > 9:
                return  # mouse moved -> rotation drag, not a click
        name = self._camera_at(pos)
        if name:
            self.bus.image_selected.emit(name)

    def _camera_at(self, pos) -> str | None:
        """The camera whose frustum is nearest ``pos`` (or None)."""
        if not self._frustums or not self._show_cameras:
            return None
        # Cursor in VTK display coords: physical pixels, bottom-left origin.
        widget = self.plotter.interactor
        scale = widget.devicePixelRatioF() or 1.0
        cx = pos.x() * scale
        cy = (widget.height() - pos.y()) * scale
        renderer = self.plotter.renderer
        try:
            win_h = self.plotter.render_window.GetSize()[1]
        except Exception:
            win_h = widget.height() * scale
        best_name, best_d2 = None, (0.12 * win_h) ** 2   # ignore clicks far from any camera

        points = self._frustum_poly.points if self._frustum_poly is not None else None
        if points is None:
            return None
        for name, entry in self._frustums.items():
            for p in points[entry.index * 5:(entry.index + 1) * 5]:  # 5 vertices
                renderer.SetWorldPoint(float(p[0]), float(p[1]), float(p[2]), 1.0)
                renderer.WorldToDisplay()
                dx, dy, dz = renderer.GetDisplayPoint()
                if dz <= 0.0 or dz >= 1.0:
                    continue                             # behind camera / clipped
                d2 = (dx - cx) ** 2 + (dy - cy) ** 2
                if d2 < best_d2:
                    best_d2, best_name = d2, name
        return best_name

    def _frustum_scale(self) -> float:
        if self._poly is not None and self._poly.n_points:
            b = self._poly.bounds
            diag = float(np.linalg.norm([b[1] - b[0], b[3] - b[2], b[5] - b[4]]))
            if diag > 0:
                return 0.05 * diag
        return self.config.default_brush_radius * 5
