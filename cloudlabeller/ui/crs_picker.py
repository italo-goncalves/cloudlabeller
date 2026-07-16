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

"""Searchable drop-down over every EPSG projected CRS (~5300 entries).

An editable combo whose completer matches anywhere in "EPSG:code — name",
case-insensitively. Shared by the export dialog and the reproject dialog
(and, later, GCP coordinate entry).
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QCompleter

from cloudlabeller.photogrammetry.crs import projected_crs_catalogue


class CrsPicker(QComboBox):
    def __init__(self, preselect_epsg: int | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setEditable(True)
        self.setInsertPolicy(QComboBox.NoInsert)
        for code, name in projected_crs_catalogue():
            self.addItem(f"EPSG:{code} — {name}", code)
        completer = self.completer()
        completer.setCompletionMode(QCompleter.PopupCompletion)
        completer.setFilterMode(Qt.MatchContains)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.setToolTip("Type to search — e.g. “SIRGAS 22S”, "
                        "“UTM zone 31N” or an EPSG code.")
        if preselect_epsg is not None:
            self.select_epsg(int(preselect_epsg))

    def select_epsg(self, epsg: int) -> bool:
        idx = self.findData(int(epsg))
        if idx >= 0:
            self.setCurrentIndex(idx)
        return idx >= 0

    def resolve_text(self) -> bool:
        """Normalise free-typed text to a catalogue entry (contains-match).
        Returns False when nothing matches — the caller should refuse."""
        if self.findText(self.currentText()) >= 0:
            return True
        match = self.findText(self.currentText(), Qt.MatchContains)
        if match < 0:
            return False
        self.setCurrentIndex(match)
        return True

    def current_epsg(self) -> int:
        return int(self.currentData())
