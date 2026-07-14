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

"""Side panel of 3D view controls.

Emits plain Qt signals (representation / cameras / label blend) that the main
window wires to the :class:`Viewer3D`. Representation options enable themselves
only when the corresponding data (sparse / dense / mesh) is available.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from cloudlabeller.core.events import EventBus

_REPRESENTATIONS = [("sparse", "Sparse cloud"), ("dense", "Dense cloud"), ("mesh", "Mesh")]


class ViewPanel(QWidget):
    representation_changed = Signal(str)     # "sparse" | "dense" | "mesh"
    cameras_toggled = Signal(bool)
    frustum_scale_changed = Signal(float)    # multiplier on the auto base size
    label_blend_changed = Signal(float)      # 0.0 (RGB) .. 1.0 (label colours)
    lasso_toggled = Signal(bool)             # lasso tool armed/disarmed
    apply_lasso = Signal()                   # apply the active label to the selection
    delete_selection = Signal()              # delete selected sparse points

    def __init__(self, bus: EventBus) -> None:
        super().__init__()
        self.bus = bus
        self.project = None

        layout = QVBoxLayout(self)

        # -- representation -------------------------------------------------
        rep_box = QGroupBox("Representation")
        rep_layout = QVBoxLayout(rep_box)
        self._rep_group = QButtonGroup(self)
        self._rep_buttons: dict[str, QRadioButton] = {}
        for key, label in _REPRESENTATIONS:
            rb = QRadioButton(label)
            self._rep_buttons[key] = rb
            self._rep_group.addButton(rb)
            rep_layout.addWidget(rb)
        self._rep_buttons["sparse"].setChecked(True)
        self._rep_group.buttonToggled.connect(self._on_representation)
        layout.addWidget(rep_box)
        # (select_representation() switches programmatically, radio kept in sync)

        # -- cameras --------------------------------------------------------
        cam_box = QGroupBox("Cameras")
        cam_layout = QVBoxLayout(cam_box)
        self.chk_cameras = QCheckBox("Show cameras")
        self.chk_cameras.setChecked(True)
        self.chk_cameras.toggled.connect(self.cameras_toggled)
        cam_layout.addWidget(self.chk_cameras)

        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Size"))
        self.frustum_slider = QSlider(Qt.Horizontal)
        self.frustum_slider.setRange(2, 80)          # 0.02x .. 0.8x of the auto size
        self.frustum_slider.setValue(20)             # 0.2x
        self.frustum_slider.setToolTip("Camera frustum size")
        self.frustum_slider.valueChanged.connect(
            lambda v: self.frustum_scale_changed.emit(v / 100.0))
        size_row.addWidget(self.frustum_slider)
        cam_layout.addLayout(size_row)
        layout.addWidget(cam_box)

        # -- RGB <-> label colour blend ------------------------------------
        blend_box = QGroupBox("Colour")
        blend_layout = QVBoxLayout(blend_box)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(0)
        self.slider.setToolTip("Blend point colours from photographic RGB (left) "
                               "to label colours (right).")
        self.slider.valueChanged.connect(
            lambda v: self.label_blend_changed.emit(v / 100.0))
        ends = QLabel("RGB ←──────→ Labels")
        ends.setAlignment(Qt.AlignCenter)
        blend_layout.addWidget(self.slider)
        blend_layout.addWidget(ends)
        layout.addWidget(blend_box)

        # -- lasso: label on the dense cloud, delete on the sparse one --------
        lasso_box = QGroupBox("Lasso")
        lasso_layout = QVBoxLayout(lasso_box)
        self.btn_lasso = QPushButton("Lasso select")
        self.btn_lasso.setCheckable(True)
        self.btn_lasso.setToolTip("Left-drag draws a lasso on the visible cloud; "
                                  "middle/right still pan/zoom.\n"
                                  "Dense view: apply the active label. "
                                  "Sparse view: delete stray points.")
        self.lbl_selection = QLabel("No selection")
        self.btn_apply = QPushButton("Apply active label")
        self.btn_apply.setEnabled(False)
        self.btn_apply.setToolTip("Label the selected dense-cloud points "
                                  "(dense view only)")
        self.btn_delete = QPushButton("Delete points")
        self.btn_delete.setEnabled(False)
        self.btn_delete.setToolTip("Remove the selected sparse points from the "
                                   "cloud (sparse view only)")
        self.btn_lasso.toggled.connect(self.lasso_toggled)
        self.btn_apply.clicked.connect(self.apply_lasso)
        self.btn_delete.clicked.connect(self.delete_selection)
        lasso_layout.addWidget(self.btn_lasso)
        lasso_layout.addWidget(self.lbl_selection)
        lasso_layout.addWidget(self.btn_apply)
        lasso_layout.addWidget(self.btn_delete)
        layout.addWidget(lasso_box)

        layout.addStretch(1)

        bus.project_opened.connect(self._on_project)
        bus.cloud_changed.connect(self._refresh_availability)
        bus.mesh_changed.connect(self._refresh_availability)
        self._refresh_availability()

    # -- state -------------------------------------------------------------
    def _on_project(self, project) -> None:
        self.project = project
        self._refresh_availability()
        # Sync the viewer to the slider's current frustum size (the slider is the
        # single source of truth; the viewer's frustums are built on project open).
        self.frustum_scale_changed.emit(self.frustum_slider.value() / 100.0)

    def _on_representation(self, button, checked: bool) -> None:
        if checked:
            self.representation_changed.emit(self.current_representation())
            self._update_lasso_enabled()

    def _update_lasso_enabled(self) -> None:
        """The lasso works on either cloud view (not the mesh)."""
        cloud_active = any(
            self._rep_buttons[k].isChecked() and self._rep_buttons[k].isEnabled()
            for k in ("sparse", "dense"))
        if not cloud_active and self.btn_lasso.isChecked():
            self.btn_lasso.setChecked(False)       # emits lasso_toggled(False)
        self.btn_lasso.setEnabled(cloud_active)

    def set_selection_count(self, count: int) -> None:
        self.lbl_selection.setText(f"{count:,} points selected" if count else "No selection")
        rep = self.current_representation()
        self.btn_apply.setEnabled(count > 0 and rep == "dense")
        self.btn_delete.setEnabled(count > 0 and rep == "sparse")

    def select_representation(self, key: str) -> None:
        """Switch representation programmatically (radio UI stays in sync;
        the toggle signal fires as if the user clicked)."""
        rb = self._rep_buttons.get(key)
        if rb is not None and rb.isEnabled() and not rb.isChecked():
            rb.setChecked(True)

    def current_representation(self) -> str:
        for key, rb in self._rep_buttons.items():
            if rb.isChecked():
                return key
        return "sparse"

    def _refresh_availability(self) -> None:
        """Enable only representations whose data exists; fall back if needed."""
        ds = self.project.dataset if self.project else None
        for key, rb in self._rep_buttons.items():
            rb.setEnabled(bool(ds) and ds.has(key))
        # If the active representation is no longer available, snap to sparse.
        active = self.current_representation()
        if ds and not ds.has(active) and ds.has("sparse"):
            self._rep_buttons["sparse"].setChecked(True)
        self._update_lasso_enabled()
