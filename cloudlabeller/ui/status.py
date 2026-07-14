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

"""Shared label-status colours and icons.

One convention across the UI: the 3D frustum dots and the Dataset pane dots
use the same colours — green = user-labelled (hard data), yellow =
auto-labelled (cloud projection or U-Net prediction), red = unlabelled.
Kept free of heavy imports so list panels can use it without pulling in VTK.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap

# 'ml' (U-Net prediction) is auto-labelled too — same yellow as 'auto'.
STATUS_COLORS = {"user": "#33cc33", "auto": "#ffd400", "ml": "#ffd400",
                 "none": "#ff3333"}

_icons: dict[str, QIcon] = {}


def status_icon(status: str) -> QIcon:
    """A small filled-circle icon in the status colour (cached per colour)."""
    color = STATUS_COLORS.get(status, STATUS_COLORS["none"])
    icon = _icons.get(color)
    if icon is None:
        pixmap = QPixmap(12, 12)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QColor(color))
        painter.setPen(QColor(color).darker(140))
        painter.drawEllipse(1, 1, 9, 9)
        painter.end()
        icon = _icons[color] = QIcon(pixmap)
    return icon
