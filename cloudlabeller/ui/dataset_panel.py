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

Each row carries a status dot in the same colours as the 3D frustum dots
(green = user-labelled, yellow = auto-labelled, red = unlabelled) and — once
the camera is solved — its X, Y, Z position in the project frame (plus the
stored CRS offset, so reprojected projects show true map coordinates). The
image name lives in Qt.UserRole; the display text is name + coordinates.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QListWidget, QListWidgetItem, QVBoxLayout, QWidget

from cloudlabeller.core.events import EventBus
from cloudlabeller.ui.status import status_icon


class DatasetPanel(QWidget):
    """The Dataset pane widget (see module docstring for the row format)."""

    def __init__(self, bus: EventBus) -> None:
        super().__init__()
        self.bus = bus
        self.project = None
        self._syncing = False          # guard against selection<->signal feedback loops

        self.image_list = QListWidget()
        layout = QVBoxLayout(self)
        layout.addWidget(self.image_list)

        self.image_list.currentItemChanged.connect(self._on_row_selected)
        bus.project_opened.connect(self._on_project_opened)
        bus.images_changed.connect(self._rebuild)
        bus.image_selected.connect(self._sync_selection)
        bus.image_labels_changed.connect(self._refresh_status)   # recolour the dot

    def _on_project_opened(self, project) -> None:
        self.project = project
        self._rebuild()

    def _status_of(self, name: str) -> str:
        return self.project.labels.status_of(name) if self.project else "none"

    def _frame_offset(self) -> np.ndarray:
        settings = getattr(self.project, "settings", None) or {}
        crs_info = (settings.get("georeferenced") or {}).get("crs") or {}
        return np.asarray(crs_info.get("offset") or (0.0, 0.0, 0.0), np.float64)

    def _row_text(self, im, offset: np.ndarray) -> str:
        """Image name, plus the camera position (frame + offset) when solved."""
        cam = im.camera
        if cam is None:
            return im.name
        c = -(cam.R.T @ np.asarray(cam.t, np.float64)) + offset
        return f"{im.name}    ({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f})"

    def _rebuild(self) -> None:
        """Repopulate the whole list (project opened / images or cameras changed)."""
        self._syncing = True
        self.image_list.clear()
        if self.project:
            offset = self._frame_offset()
            for im in self.project.dataset.images:
                item = QListWidgetItem(status_icon(self._status_of(im.name)),
                                       self._row_text(im, offset))
                item.setData(Qt.UserRole, im.name)
                self.image_list.addItem(item)
        self._syncing = False

    def _items_named(self, name: str) -> list[QListWidgetItem]:
        return [self.image_list.item(i) for i in range(self.image_list.count())
                if self.image_list.item(i).data(Qt.UserRole) == name]

    def _refresh_status(self, name: str) -> None:
        """Recolour one row's dot after its labels changed."""
        for item in self._items_named(name):
            item.setIcon(status_icon(self._status_of(name)))

    def _on_row_selected(self, current, _previous=None) -> None:
        if current is not None and not self._syncing:
            self.bus.image_selected.emit(current.data(Qt.UserRole))

    def _sync_selection(self, name: str) -> None:
        """Select the row for ``name`` without re-emitting image_selected."""
        items = self._items_named(name)
        if not items:
            return
        self._syncing = True
        self.image_list.setCurrentItem(items[0])
        self._syncing = False
