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

"""About dialog: logo, author, version, and the AI-assistance disclaimer."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QLabel,
    QVBoxLayout,
)

from cloudlabeller import __version__

AUTHOR = "Ítalo Gomes Gonçalves"

DISCLAIMER = (
    "CloudLabeller is an <b>AI-assisted project</b>: most of its application "
    "code was written, tested and debugged by an AI coding assistant "
    "(Anthropic's Claude) working under the direction of the author, who "
    "defined the requirements, the scientific approach and the "
    "machine-learning methodology. As with any software — AI-assisted or "
    "otherwise — results should be independently validated before use in "
    "scientific or operational work."
)

CREDITS = ("Licensed under the GNU GPL v3 or later; commercial licenses "
           "are available from the author (see README.md).<br>"
           "Built on COLMAP / pycolmap, PySide6, PyVista, TensorFlow, "
           "hylite, laspy and trimesh. Open-source license texts are "
           "collected in THIRD_PARTY_LICENSES.txt, distributed with the app.")


class AboutDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("About CloudLabeller")
        self.setMinimumWidth(480)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        icon_path = Path(__file__).parent.parent / "assets" / "icon.ico"
        if icon_path.exists():
            logo = QLabel()
            logo.setPixmap(QIcon(str(icon_path)).pixmap(160, 160))
            logo.setAlignment(Qt.AlignCenter)
            layout.addWidget(logo)

        title = QLabel(f"<h2>CloudLabeller</h2>"
                       f"<p style='color: gray'>version {__version__}</p>")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        author = QLabel(f"Created by <b>{AUTHOR}</b>")
        author.setAlignment(Qt.AlignCenter)
        layout.addWidget(author)

        subtitle = QLabel("Photogrammetric reconstruction and 2D ↔ 3D labelling "
                          "with U-Net label propagation.")
        subtitle.setWordWrap(True)
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle)

        # -- AI-assistance disclaimer (the important part) --------------------
        disclaimer = QLabel(DISCLAIMER)
        disclaimer.setWordWrap(True)
        disclaimer.setFrameStyle(QFrame.StyledPanel)
        disclaimer.setStyleSheet(
            "QLabel { background: rgba(128, 128, 128, 0.12); "
            "border: 1px solid rgba(128, 128, 128, 0.4); "
            "border-radius: 6px; padding: 10px; }")
        layout.addWidget(disclaimer)

        credits = QLabel(CREDITS)
        credits.setWordWrap(True)
        credits.setStyleSheet("color: gray; font-size: 11px;")
        credits.setAlignment(Qt.AlignCenter)
        layout.addWidget(credits)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)
