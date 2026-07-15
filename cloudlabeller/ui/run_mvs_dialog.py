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

"""Dense MVS options dialog (runs on the project's existing SfM model)."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
)

from cloudlabeller.ui.detail_level import MVS_LEVELS, DetailControl


class RunMvsDialog(QDialog):
    def __init__(self, colmap_binary: str | None = None,
                 source_resolution: tuple[int, int] | None = None,
                 parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Run Dense MVS")
        self.setMinimumWidth(460)
        layout = QVBoxLayout(self)

        if colmap_binary:
            engine = f"Using CUDA COLMAP:<br><code>{colmap_binary}</code>"
        else:
            engine = ("<b>No CUDA COLMAP found.</b> Dense MVS needs CUDA — get "
                      "it via Photogrammetry → Download COLMAP…")
        note = QLabel("Dense reconstruction (patch-match stereo) on the current "
                      f"SfM model.<br>{engine}")
        note.setWordWrap(True)
        layout.addWidget(note)

        form = QFormLayout()
        # Patch-match stereo is far more memory-hungry than SIFT, so the MVS
        # default is Medium (1/16 of the pixels, i.e. 1/4 of the side) — near
        # the old 2000 px default for typical drone imagery on a 12 GB GPU.
        self.detail = DetailControl(source_resolution, default_level="medium",
                                    custom_default=2000, levels=MVS_LEVELS)
        form.addRow("Detail:", self.detail)
        from PySide6.QtWidgets import QComboBox

        self.cmb_quality = QComboBox()
        self.cmb_quality.addItem("Standard — two-pass patch-match (recommended)",
                                 "standard")
        self.cmb_quality.addItem("Draft — ~3-4× faster stereo, noisier cloud",
                                 "draft")
        self.cmb_quality.setToolTip(
            "Draft caps each depth map at 10 source images, skips the "
            "geometric-consistency pass and lightens the per-pixel solve — "
            "good for a first look before an overnight Standard run.")
        form.addRow("Quality:", self.cmb_quality)
        note2 = QLabel("The triangulated mesh is rebuilt automatically after the "
                       "dense cloud completes.")
        note2.setWordWrap(True)
        form.addRow(note2)
        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def max_image_size(self) -> int:
        return self.detail.pixels()

    def quality(self) -> str:
        return self.cmb_quality.currentData()
