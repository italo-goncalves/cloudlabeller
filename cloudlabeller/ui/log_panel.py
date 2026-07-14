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

"""A simple read-only log panel that shows runtime / subprocess output."""

from __future__ import annotations

import logging

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QPlainTextEdit, QPushButton, QVBoxLayout, QWidget


class LogPanel(QWidget):
    """Append-only text view. Fed by job ``log_line`` signals and, optionally, a
    logging handler so in-process log records show here too."""

    MAX_BLOCKS = 5000

    def __init__(self) -> None:
        super().__init__()
        self.view = QPlainTextEdit(readOnly=True)
        self.view.setMaximumBlockCount(self.MAX_BLOCKS)
        self.view.setFont(QFont("Consolas", 9))
        self.view.setLineWrapMode(QPlainTextEdit.NoWrap)
        clear = QPushButton("Clear")
        clear.clicked.connect(self.view.clear)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(self.view)
        layout.addWidget(clear)

    def append(self, line: str) -> None:
        self.view.appendPlainText(line)
        # Mirror into the file log (~/.cloudlabeller/logs) so job history
        # (transfers, training, predictions) survives the session — the panel
        # alone made past runs impossible to reconstruct.
        logging.getLogger("cloudlabeller.jobs").info(line)

    def as_logging_handler(self) -> logging.Handler:
        """A logging.Handler that mirrors records into this panel (thread-safe via
        Qt's queued signal would be needed off-thread; in-process use only)."""
        panel = self

        class _Handler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                # Straight to the view — panel.append would re-log the line
                # (via cloudlabeller.jobs) and loop it back through here.
                panel.view.appendPlainText(self.format(record))

        h = _Handler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s",
                                         "%H:%M:%S"))
        return h
