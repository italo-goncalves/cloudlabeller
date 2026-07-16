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

"""Full-pipeline dialog: SfM → Dense MVS → mesh in one unattended run.

Collects every option upfront so the chain never stops to ask questions —
made for "start it in the evening, find everything done in the morning".
Returns the SfM options and the MVS detail setting.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QVBoxLayout,
)

from cloudlabeller.photogrammetry.options import CAMERA_MODELS, SfmOptions
from cloudlabeller.ui.detail_level import MVS_LEVELS, DetailControl


class RunPipelineDialog(QDialog):
    """All options for the unattended SfM → georef → MVS → mesh chain."""

    def __init__(self, image_count: int,
                 source_resolution: tuple[int, int] | None = None,
                 gpu_available: bool = False,
                 gps_found: int = 0,
                 parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Run Full Pipeline")
        self.setMinimumWidth(560)

        layout = QVBoxLayout(self)
        header = QLabel(
            f"Run the whole chain on <b>{image_count}</b> image(s): "
            "SfM → dense cloud → mesh, unattended.<br>"
            "The machine is kept awake until the pipeline ends; each stage "
            "logs to its own file. If a stage fails, the chain stops and "
            "earlier products are kept.")
        header.setWordWrap(True)
        layout.addWidget(header)

        # -- SfM options (mirrors the Run SfM dialog) ------------------------
        matcher_row = QHBoxLayout()
        matcher_row.addWidget(QLabel("Matcher:"))
        self.rb_spatial = QRadioButton("Spatial (GPS)")
        self.rb_sequential = QRadioButton("Sequential")
        self.rb_exhaustive = QRadioButton("Exhaustive")
        self.rb_spatial.setChecked(True)
        self._matcher_group = QButtonGroup(self)
        for rb in (self.rb_spatial, self.rb_sequential, self.rb_exhaustive):
            self._matcher_group.addButton(rb)
            matcher_row.addWidget(rb)
        matcher_row.addStretch(1)
        layout.addLayout(matcher_row)

        form = QFormLayout()
        self.chk_single = QCheckBox("Single shared camera (recommended for one camera/drone)")
        self.chk_single.setChecked(True)
        self.chk_gpu = QCheckBox("Use GPU for SIFT + matching (CUDA COLMAP)")
        self.chk_gpu.setChecked(gpu_available)   # pipeline requires the exe anyway
        self.cmb_model = QComboBox()
        self.cmb_model.addItems(CAMERA_MODELS)
        self.sfm_detail = DetailControl(source_resolution, default_level="medium",
                                        custom_default=3200)
        self.mvs_detail = DetailControl(source_resolution, default_level="medium",
                                        custom_default=2000, levels=MVS_LEVELS)
        self.chk_georef = QCheckBox(
            "Georeference to EXIF GPS after SfM (metres, true north)")
        if gps_found >= 3:
            self.chk_georef.setChecked(True)
            self.chk_georef.setToolTip(
                "Aligns the model to the images' GPS between SfM and MVS — "
                "the dense cloud and mesh are then built in the metric frame "
                "and exports are GIS-ready. Uncheck if the GPS is unreliable.")
        else:
            self.chk_georef.setChecked(False)
            self.chk_georef.setEnabled(False)
            self.chk_georef.setToolTip(
                "Needs GPS EXIF in at least 3 images — not detected in this "
                "image store.")
        form.addRow(self.chk_single)
        form.addRow(self.chk_gpu)
        form.addRow(self.chk_georef)
        form.addRow("Camera model:", self.cmb_model)
        form.addRow("SfM detail:", self.sfm_detail)
        form.addRow("Dense (MVS) detail:", self.mvs_detail)
        self.cmb_mvs_quality = QComboBox()
        self.cmb_mvs_quality.addItem("Standard — two-pass patch-match "
                                     "(recommended)", "standard")
        self.cmb_mvs_quality.addItem("Draft — ~3-4× faster stereo, noisier "
                                     "cloud", "draft")
        form.addRow("Dense (MVS) quality:", self.cmb_mvs_quality)

        # -- mesh stage -------------------------------------------------------
        self.cmb_mesh_method = QComboBox()
        self.cmb_mesh_method.addItem("Poisson — watertight, smooths noise "
                                     "(recommended)", "poisson")
        self.cmb_mesh_method.addItem("Delaunay — maximum detail, keeps the "
                                     "cloud's noise", "delaunay")
        form.addRow("Mesh method:", self.cmb_mesh_method)

        # Same depth choices as the Create Mesh dialog; the density-matched
        # depth is resolved once the dense cloud exists, and the meshing stage
        # logs its rough time/RAM estimate when it starts.
        from cloudlabeller.ui.mesh_dialog import DETAIL_CHOICES

        self.cmb_mesh_detail = QComboBox()
        self.cmb_mesh_detail.addItem("Match dense cloud density (recommended)", None)
        for label, depth in DETAIL_CHOICES[1:]:
            self.cmb_mesh_detail.addItem(label, depth)
        form.addRow("Mesh detail:", self.cmb_mesh_detail)
        self.cmb_mesh_method.currentIndexChanged.connect(
            lambda i: self.cmb_mesh_detail.setEnabled(
                self.cmb_mesh_method.currentData() == "poisson"))
        layout.addLayout(form)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText("Run Pipeline")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(image_count > 0)
        layout.addWidget(self.buttons)

    # -- results -------------------------------------------------------------
    def georeference(self) -> bool:
        return self.chk_georef.isChecked()

    def sfm_options(self) -> SfmOptions:
        if self.rb_exhaustive.isChecked():
            matcher = "exhaustive"
        elif self.rb_sequential.isChecked():
            matcher = "sequential"
        else:
            matcher = "spatial"
        return SfmOptions(
            matcher=matcher,
            use_gpu=self.chk_gpu.isChecked(),
            single_camera=self.chk_single.isChecked(),
            camera_model=self.cmb_model.currentText(),
            max_image_size=self.sfm_detail.pixels(),
        )

    def mvs_pixels(self) -> int:
        return self.mvs_detail.pixels()

    def mvs_quality(self) -> str:
        return self.cmb_mvs_quality.currentData()

    def mesh_options(self) -> dict:
        """Mesh-stage options in :meth:`CreateMeshDialog.options` format."""
        return {
            "method": self.cmb_mesh_method.currentData(),
            "depth": self.cmb_mesh_detail.currentData(),
            "trim": 10.0,
            "point_weight": 1.0,
        }
