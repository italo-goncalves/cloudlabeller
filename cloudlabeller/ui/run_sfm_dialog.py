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

"""Run-SfM options dialog.

Reconstruction always runs on the project's own image store (populate it with
File → Add Images…), so this dialog only collects reconstruction *options* —
no source folder. Returns an
:class:`~cloudlabeller.photogrammetry.options.SfmOptions`.
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
from cloudlabeller.ui.detail_level import DetailControl


class RunSfmDialog(QDialog):
    def __init__(self, image_count: int,
                 source_resolution: tuple[int, int] | None = None,
                 gpu_available: bool = False,
                 parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Run Structure-from-Motion")
        self.setMinimumWidth(520)
        self._image_count = image_count

        layout = QVBoxLayout(self)

        # -- what will be reconstructed ------------------------------------
        if image_count:
            header = QLabel(f"Reconstruct <b>{image_count}</b> image(s) from the "
                            f"project's image store.")
        else:
            header = QLabel("The project's image store is empty.<br>"
                            "Add images first via <b>File → Add Images…</b>.")
        header.setWordWrap(True)
        layout.addWidget(header)

        # -- matcher: radio buttons ----------------------------------------
        matcher_row = QHBoxLayout()
        matcher_row.addWidget(QLabel("Matcher:"))
        self.rb_spatial = QRadioButton("Spatial (GPS)")
        self.rb_sequential = QRadioButton("Sequential")
        self.rb_exhaustive = QRadioButton("Exhaustive")
        self.rb_spatial.setChecked(True)   # default: best for geotagged drone imagery
        self.rb_spatial.setToolTip("Match images by EXIF GPS proximity — ideal for "
                                   "geotagged drone surveys. Needs GPS in the photos.")
        self.rb_sequential.setToolTip("Match each image to its neighbours — fast; "
                                      "best when photos are in capture order.")
        self.rb_exhaustive.setToolTip("Match every image pair — slower but finds more "
                                      "connections (revisited areas / unordered sets).")
        self._matcher_group = QButtonGroup(self)
        for rb in (self.rb_spatial, self.rb_sequential, self.rb_exhaustive):
            self._matcher_group.addButton(rb)
            matcher_row.addWidget(rb)
        matcher_row.addStretch(1)
        layout.addLayout(matcher_row)

        # -- advanced options ---------------------------------------------
        form = QFormLayout()
        self.chk_single = QCheckBox("Single shared camera (recommended for one camera/drone)")
        self.chk_single.setChecked(True)
        self.chk_gpu = QCheckBox("Use GPU for SIFT + matching (CUDA COLMAP)")
        # Default ON when the CUDA executable exists: extraction + matching
        # dominate SfM wall time and run ~10x faster on the GPU. Falls back
        # to CPU automatically if the GPU path fails at runtime.
        self.chk_gpu.setChecked(gpu_available)
        if not gpu_available:
            self.chk_gpu.setToolTip("No CUDA COLMAP executable found — "
                                    "SIFT will run on the CPU. Get it via "
                                    "Photogrammetry → Download COLMAP…")
        self.cmb_model = QComboBox()
        self.cmb_model.addItems(CAMERA_MODELS)
        self.detail = DetailControl(source_resolution, default_level="medium",
                                    custom_default=3200)
        form.addRow(self.chk_single)
        form.addRow(self.chk_gpu)
        form.addRow("Camera model:", self.cmb_model)
        form.addRow("Detail:", self.detail)
        layout.addLayout(form)

        # -- OK / Cancel ---------------------------------------------------
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(image_count > 0)
        layout.addWidget(self.buttons)

    # -- results -----------------------------------------------------------
    def options(self) -> SfmOptions:
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
            max_image_size=self.detail.pixels(),
        )
