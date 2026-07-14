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

"""Startup welcome dialog — a project must be chosen before the main window.

Offers New / Open / Quit (plus quick access to recent projects). The folder
choosing and the actual project create/open happen inside the dialog, so a
cancelled file dialog or a failed open returns here instead of leaving the app
without a project. On accept, :attr:`project` holds the loaded project.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from cloudlabeller.core.project import MANIFEST, Project


class WelcomeDialog(QDialog):
    def __init__(self, recent_projects: list[str] | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Welcome to CloudLabeller")
        self.setMinimumWidth(460)
        self.project: Project | None = None

        layout = QVBoxLayout(self)
        header = QLabel("<h3>CloudLabeller</h3>"
                        "Create a new project or open an existing one to begin.")
        header.setWordWrap(True)
        layout.addWidget(header)

        # Recent projects (only ones still on disk), double-click to open.
        self._recent = [p for p in (recent_projects or [])
                        if (Path(p) / MANIFEST).exists()]
        if self._recent:
            layout.addWidget(QLabel("Recent projects:"))
            self.lst_recent = QListWidget()
            self.lst_recent.addItems(self._recent)
            self.lst_recent.itemDoubleClicked.connect(
                lambda item: self._load(item.text()))
            self.lst_recent.setToolTip("Double-click to open")
            layout.addWidget(self.lst_recent)

        buttons = QHBoxLayout()
        btn_new = QPushButton("New Project…")
        btn_open = QPushButton("Open Project…")
        btn_quit = QPushButton("Quit")
        btn_new.clicked.connect(self._new)
        btn_open.clicked.connect(self._open)
        btn_quit.clicked.connect(self.reject)
        for b in (btn_new, btn_open, btn_quit):
            buttons.addWidget(b)
        layout.addLayout(buttons)

    # -- actions -------------------------------------------------------------
    def _new(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Choose a folder for the new project")
        if not path:
            return                                    # back to the welcome dialog
        if (Path(path) / MANIFEST).exists():
            answer = QMessageBox.question(
                self, "Folder already has a project",
                "This folder already contains a CloudLabeller project.\n"
                "Open it instead? (Choosing No returns to the welcome screen "
                "so you can pick another folder.)")
            if answer == QMessageBox.Yes:
                self._load(path)
            return
        try:
            self.project = Project.create(path)
        except Exception as exc:
            QMessageBox.critical(self, "Could not create project", str(exc))
            return
        self.accept()

    def _open(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Open project folder")
        if path:
            self._load(path)

    def _load(self, path: str) -> None:
        if not (Path(path) / MANIFEST).exists():
            QMessageBox.warning(
                self, "Not a project",
                f"No CloudLabeller project found in:\n{path}\n\n"
                f"(A project folder contains a {MANIFEST} file.)")
            return
        try:
            self.project = Project.open(path)
        except Exception as exc:
            QMessageBox.critical(self, "Open failed", str(exc))
            return
        self.accept()
