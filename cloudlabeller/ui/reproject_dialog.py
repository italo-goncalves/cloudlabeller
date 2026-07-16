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

"""Reproject-to-CRS dialog: move the whole project into a projected CRS.

Shows the current frame, a searchable EPSG picker (UTM zone pre-selected)
and the EGM96 heights option. The heavy lifting happens in
``reproject_cli`` (subprocess).
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QMessageBox,
    QVBoxLayout,
)

from cloudlabeller.photogrammetry.crs import frame_labels, suggest_projected_epsg
from cloudlabeller.ui.crs_picker import CrsPicker


class ReprojectDialog(QDialog):
    """Target-CRS choice for reprojection (see module docstring)."""

    def __init__(self, origin_lla, geo_settings: dict, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Reproject to CRS")
        self.setMinimumWidth(540)
        layout = QVBoxLayout(self)

        _, current = frame_labels(geo_settings)
        header = QLabel(
            f"Current frame: {current.splitlines()[0]}<br><br>"
            "Reprojection moves <b>everything</b> — clouds, mesh, cameras and "
            "the COLMAP model — into the chosen CRS. Coordinates are stored "
            "minus a km-rounded offset to preserve precision; exports write "
            "the full map coordinates automatically.")
        header.setWordWrap(True)
        layout.addWidget(header)

        crs_info = geo_settings.get("crs") or {}
        preselect = crs_info.get("epsg") or suggest_projected_epsg(origin_lla)
        self.cmb_crs = CrsPicker(preselect_epsg=preselect)
        layout.addWidget(self.cmb_crs)

        self.chk_geoid = QCheckBox("Sea-level heights (EGM96 geoid) — may "
                                   "download a ~3 MB grid once")
        self.chk_geoid.setChecked(bool(crs_info.get("orthometric")))
        layout.addWidget(self.chk_geoid)

        note = QLabel("Tip: reproject once, when the reconstruction is final "
                      "— repeated reprojection accumulates small rounding.")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray;")
        layout.addWidget(note)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Reproject")
        buttons.accepted.connect(self._accept_validated)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _accept_validated(self) -> None:
        if not self.cmb_crs.resolve_text():
            QMessageBox.warning(self, "Unknown CRS",
                                "Pick a projection from the list — type "
                                "to search it (name or EPSG code).")
            return
        self.accept()

    def epsg(self) -> int:
        return self.cmb_crs.current_epsg()

    def orthometric(self) -> bool:
        return self.chk_geoid.isChecked()
