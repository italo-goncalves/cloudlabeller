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

"""Dataset browser: list images; selecting one shows it (Image pane) and
highlights its camera frustum. Stays in sync when a frustum is clicked in 3D.

Each row carries a status dot in the same colours as the 3D frustum dots:
green = user-labelled, yellow = auto-labelled (cloud projection / U-Net),
red = unlabelled.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QListWidget, QListWidgetItem, QVBoxLayout, QWidget

from cloudlabeller.core.events import EventBus
from cloudlabeller.ui.status import status_icon


class DatasetPanel(QWidget):
    def __init__(self, bus: EventBus) -> None:
        super().__init__()
        self.bus = bus
        self.project = None
        self._syncing = False          # guard against selection<->signal feedback loops

        self.image_list = QListWidget()
        layout = QVBoxLayout(self)
        layout.addWidget(self.image_list)

        self.image_list.currentTextChanged.connect(self._on_row_selected)
        bus.project_opened.connect(self._on_project_opened)
        bus.images_changed.connect(self._rebuild)
        bus.image_selected.connect(self._sync_selection)
        bus.image_labels_changed.connect(self._refresh_status)   # recolour the dot

    def _on_project_opened(self, project) -> None:
        self.project = project
        self._rebuild()

    def _status_of(self, name: str) -> str:
        return self.project.labels.status_of(name) if self.project else "none"

    def _rebuild(self) -> None:
        self._syncing = True
        self.image_list.clear()
        if self.project:
            for im in self.project.dataset.images:
                self.image_list.addItem(
                    QListWidgetItem(status_icon(self._status_of(im.name)), im.name))
        self._syncing = False

    def _refresh_status(self, name: str) -> None:
        """Recolour one row's dot after its labels changed."""
        for item in self.image_list.findItems(name, Qt.MatchExactly):
            item.setIcon(status_icon(self._status_of(name)))

    def _on_row_selected(self, name: str) -> None:
        if name and not self._syncing:
            self.bus.image_selected.emit(name)

    def _sync_selection(self, name: str) -> None:
        """Select the row for ``name`` without re-emitting image_selected."""
        items = self.image_list.findItems(name, Qt.MatchExactly)
        if not items:
            return
        self._syncing = True
        self.image_list.setCurrentItem(items[0])
        self._syncing = False
