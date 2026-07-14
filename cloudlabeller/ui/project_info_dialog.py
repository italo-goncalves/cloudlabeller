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

"""Project Info dialog — the project's key facts in one copyable place.

Opened from File → Project Info… or by clicking the status-bar summary.
Values (georeferenced origin, counts, spec) are selectable, and the Copy
button puts the whole summary on the clipboard as plain text.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
)


class ProjectInfoDialog(QDialog):
    def __init__(self, project, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Project Info")
        self.setMinimumWidth(520)
        self._rows = project.summary_info()

        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        for label, value in self._rows:
            value_lbl = QLabel(value)
            value_lbl.setWordWrap(True)
            value_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
            key_lbl = QLabel(f"<b>{label}:</b>")
            form.addRow(key_lbl, value_lbl)
        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        self.btn_copy = buttons.addButton("Copy", QDialogButtonBox.ActionRole)
        self.btn_copy.clicked.connect(self._copy)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _copy(self) -> None:
        text = "\n".join(f"{label}: {value}" for label, value in self._rows)
        QApplication.clipboard().setText(text)
        self.btn_copy.setText("Copied!")
