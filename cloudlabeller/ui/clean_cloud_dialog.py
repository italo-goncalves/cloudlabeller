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

"""Clean-sparse-cloud dialog: filter stray points by SfM confidence.

Two per-point measures from the saved COLMAP model drive the filter:
  * track length — how many images observed the point (stray points are
    usually minimum-track triangulations);
  * mean reprojection error in pixels (high = poorly triangulated).
The keep-count preview updates live as the thresholds change.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
)


class CleanCloudDialog(QDialog):
    #: keep-mask for the current thresholds; emitted on every change so the 3D
    #: pane can tint the points that would be removed (dialog is non-modal).
    mask_changed = Signal(object)

    def __init__(self, views: np.ndarray, errors: np.ndarray, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Clean Sparse Cloud")
        self.setMinimumWidth(460)
        self._views = views
        self._errors = errors

        layout = QVBoxLayout(self)
        header = QLabel(
            "Remove low-confidence points using the SfM statistics: how many "
            "images saw each point, and how well its reprojection fits them. "
            "Points to be removed show <b><font color='#ff00ff'>magenta</font></b> "
            "in the 3D view while you adjust. The dense cloud and labels on "
            "other products are not affected.")
        header.setWordWrap(True)
        layout.addWidget(header)

        form = QFormLayout()
        self.spin_views = QSpinBox()
        self.spin_views.setRange(2, 50)
        self.spin_views.setValue(3)
        self.spin_views.setToolTip("Keep points observed by at least this many "
                                   "images (stray points are usually 2-view).")
        self.spin_views.valueChanged.connect(self._preview)
        form.addRow("Min. images per point:", self.spin_views)

        self.spin_error = QDoubleSpinBox()
        self.spin_error.setRange(0.1, 20.0)
        self.spin_error.setSingleStep(0.25)
        self.spin_error.setDecimals(2)
        self.spin_error.setValue(2.0)
        self.spin_error.setSuffix(" px")
        self.spin_error.setToolTip("Keep points whose mean reprojection error "
                                   "is at most this many pixels.")
        self.spin_error.valueChanged.connect(self._preview)
        form.addRow("Max. reprojection error:", self.spin_error)
        layout.addLayout(form)

        self.lbl_preview = QLabel()
        self.lbl_preview.setStyleSheet("color: gray;")
        layout.addWidget(self.lbl_preview)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText("Clean")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self._preview()

    def mask(self) -> np.ndarray:
        """Boolean keep-mask for the current thresholds."""
        return ((self._views >= self.spin_views.value())
                & (self._errors <= self.spin_error.value()))

    def _preview(self) -> None:
        mask = self.mask()
        keep = int(mask.sum())
        total = len(self._views)
        removed = total - keep
        pct = 100.0 * keep / total if total else 0.0
        self.lbl_preview.setText(
            f"Keeps {keep:,} of {total:,} points ({pct:.1f}%) — "
            f"removes {removed:,}.")
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(0 < keep < total)
        self.mask_changed.emit(mask)
