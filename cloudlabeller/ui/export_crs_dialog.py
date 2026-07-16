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

"""Coordinate-system choice for exports from a georeferenced project.

Local ENU (the internal frame) or a projected CRS picked from a searchable
drop-down of every EPSG projected system, with the UTM zone covering the
site pre-selected. Optional EGM96 orthometric heights.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QMessageBox,
    QRadioButton,
    QVBoxLayout,
)

from cloudlabeller.photogrammetry.crs import geoid_ready, suggest_projected_epsg
from cloudlabeller.ui.crs_picker import CrsPicker


@dataclass(frozen=True)
class CrsChoice:
    mode: str                     # "local" | "projected"
    epsg: int | None = None
    orthometric: bool = False


class ExportCrsDialog(QDialog):
    """Pick the export coordinate system for a georeferenced project."""

    def __init__(self, origin_lla, geo_settings: dict | None = None,
                 parent=None) -> None:
        super().__init__(parent)
        geo = geo_settings or {}
        self.setWindowTitle("Export coordinate system")
        self.setMinimumWidth(520)
        layout = QVBoxLayout(self)

        note = QLabel("This project is georeferenced — choose the coordinates "
                      "to write:")
        note.setWordWrap(True)
        layout.addWidget(note)

        self.rb_local = QRadioButton(
            "Local frame — metres, East-North-Up, origin at the site")
        self.rb_proj = QRadioButton("Projected CRS (for GIS / geomodelling):")
        layout.addWidget(self.rb_local)
        layout.addWidget(self.rb_proj)

        self.cmb_crs = CrsPicker()
        layout.addWidget(self.cmb_crs)

        self.chk_geoid = QCheckBox("Sea-level heights (EGM96 geoid) instead "
                                   "of GPS ellipsoidal heights")
        layout.addWidget(self.chk_geoid)
        self._lbl_geoid = QLabel("")
        self._lbl_geoid.setWordWrap(True)
        self._lbl_geoid.setStyleSheet("color: gray;")
        layout.addWidget(self._lbl_geoid)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept_validated)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Pre-select: last-used CRS if any, else the UTM zone covering the
        # site in the preferred datum (e.g. SIRGAS 2000 in Brazil).
        epsg = geo.get("export_epsg") or suggest_projected_epsg(origin_lla)
        if not self.cmb_crs.select_epsg(int(epsg)):
            self.cmb_crs.select_epsg(suggest_projected_epsg(origin_lla))
        (self.rb_local if geo.get("export_mode") == "local"
         else self.rb_proj).setChecked(True)
        self.chk_geoid.setChecked(bool(geo.get("export_orthometric")))

        self.rb_local.toggled.connect(self._sync_enabled)
        self.cmb_crs.currentIndexChanged.connect(self._sync_geoid_note)
        self.chk_geoid.toggled.connect(self._sync_geoid_note)
        self._sync_enabled()

    # -- helpers -----------------------------------------------------------
    def _sync_enabled(self) -> None:
        proj = self.rb_proj.isChecked()
        self.cmb_crs.setEnabled(proj)
        self.chk_geoid.setEnabled(proj)
        self._sync_geoid_note()

    def _sync_geoid_note(self) -> None:
        if not (self.rb_proj.isChecked() and self.chk_geoid.isChecked()):
            self._lbl_geoid.setText("")
            return
        epsg = self.cmb_crs.currentData()
        if epsg is not None and not geoid_ready(int(epsg)):
            self._lbl_geoid.setText(
                "The EGM96 geoid grid (~3 MB) will be downloaded once from "
                "cdn.proj.org when the export starts.")
        else:
            self._lbl_geoid.setText("")

    def _accept_validated(self) -> None:
        if self.rb_local.isChecked():
            self.accept()
            return
        # The user may have typed free text — resolve it to a catalogue entry.
        if not self.cmb_crs.resolve_text():
            QMessageBox.warning(self, "Unknown CRS",
                                "Pick a projection from the list — type "
                                "to search it (name or EPSG code).")
            return
        self.accept()

    # -- result --------------------------------------------------------------
    def choice(self) -> CrsChoice:
        if self.rb_local.isChecked():
            return CrsChoice(mode="local")
        return CrsChoice(mode="projected",
                         epsg=int(self.cmb_crs.currentData()),
                         orthometric=self.chk_geoid.isChecked())
