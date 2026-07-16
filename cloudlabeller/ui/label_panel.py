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

"""Labels pane: manage the project's label classes.

Each row shows a colour swatch (click → colour picker), the class number, and the
name (double-click → edit inline). Add / Remove buttons create and delete classes;
deleting renumbers the rest (0..N-1) and remaps the stored labels. Selecting a row
makes it the active paint class.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QColorDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from cloudlabeller.core.events import EventBus
from cloudlabeller.core.label_schema import UNLABELLED_ID


def format_count(count: int, total: int) -> str:
    """Compact "count · percent" text, e.g. ``1.2M · 54%`` (empty when no data)."""
    if total <= 0:
        return ""
    if count >= 1_000_000:
        num = f"{count / 1_000_000:.1f}M"
    elif count >= 1_000:
        num = f"{count / 1_000:.1f}k"
    else:
        num = str(count)
    return f"{num} · {100.0 * count / total:.0f}%"


class _NameEdit(QLineEdit):
    """Read-only label text that becomes editable on double-click."""

    clicked = Signal()
    committed = Signal(str)

    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.setReadOnly(True)
        self.setFrame(False)
        self.setStyleSheet("background: transparent;")
        self.editingFinished.connect(self._commit)

    def mousePressEvent(self, event) -> None:
        if self.isReadOnly():
            self.clicked.emit()           # single click selects the row, no edit
        else:
            super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        self.setReadOnly(False)
        self.setFrame(True)
        self.selectAll()
        self.setFocus()

    def focusOutEvent(self, event) -> None:
        self._commit()
        super().focusOutEvent(event)

    def _commit(self) -> None:
        if not self.isReadOnly():
            self.setReadOnly(True)
            self.setFrame(False)
            self.committed.emit(self.text())


class LabelRow(QFrame):
    """One class in the list: colour swatch, editable name, hotkey hint.
    Clicking activates the class; double-click renames; the swatch recolours."""

    activated = Signal(int)
    renamed = Signal(int, str)
    recolored = Signal(int, str)

    def __init__(self, class_id: int, name: str, color: str, fixed: bool = False) -> None:
        super().__init__()
        self.class_id = class_id
        self._color = color
        self.fixed = fixed                 # the unlabelled row: no edit / no delete

        if fixed:
            self.swatch = QLabel()         # non-interactive, neutral swatch
            self.swatch.setFixedSize(18, 18)
            self.swatch.setStyleSheet(
                "background: #555; border: 1px solid #888; border-radius: 3px;")
        else:
            self.swatch = QToolButton()
            self.swatch.setFixedSize(18, 18)
            self.swatch.setCursor(Qt.PointingHandCursor)
            self.swatch.clicked.connect(self._pick_color)
            self._apply_swatch()

        self.num = QLabel(str(class_id))
        self.num.setFixedWidth(20)
        self.num.setAlignment(Qt.AlignCenter)

        if fixed:
            self.name = QLabel(name)
            self.name.setStyleSheet("color: #aaa; font-style: italic;")
        else:
            self.name = _NameEdit(name)
            self.name.clicked.connect(lambda: self.activated.emit(self.class_id))
            self.name.committed.connect(lambda t: self.renamed.emit(self.class_id, t))

        self.count = QLabel("")                    # point count · % of total
        self.count.setStyleSheet("color: #999; font-size: 11px;")
        self.count.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.addWidget(self.swatch)
        layout.addWidget(self.num)
        layout.addWidget(self.name, 1)
        layout.addWidget(self.count)

    def set_count(self, text: str) -> None:
        self.count.setText(text)

    def mousePressEvent(self, event) -> None:
        self.activated.emit(self.class_id)
        super().mousePressEvent(event)

    def set_active(self, active: bool) -> None:
        self.setStyleSheet("LabelRow { background: #3a6ea5; border-radius: 3px; }"
                           if active else "")

    def _apply_swatch(self) -> None:
        self.swatch.setStyleSheet(
            f"background: {self._color}; border: 1px solid #888; border-radius: 3px;")

    def _pick_color(self) -> None:
        chosen = QColorDialog.getColor(QColor(self._color), self, "Label colour")
        if chosen.isValid():
            self._color = chosen.name()
            self._apply_swatch()
            self.recolored.emit(self.class_id, self._color)


class LabelPanel(QWidget):
    """The Labels pane: the class list plus add/remove, kept in sync with
    the project's :class:`~cloudlabeller.core.label_schema.LabelSchema`."""

    def __init__(self, bus: EventBus) -> None:
        super().__init__()
        self.bus = bus
        self.project = None
        self._active: int | None = None
        self._rows: list[LabelRow] = []

        self._rows_container = QWidget()
        self._rows_layout = QVBoxLayout(self._rows_container)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(1)
        self._rows_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._rows_container)

        self.add_btn = QPushButton("Add label")
        self.remove_btn = QPushButton("Remove")
        self.isolate_btn = QPushButton("Isolate")
        self.isolate_btn.setCheckable(True)
        self.isolate_btn.setToolTip("Show only the active class in 3D "
                                    "(other points are dimmed)")
        self.add_btn.clicked.connect(self._add)
        self.remove_btn.clicked.connect(self._remove)
        self.isolate_btn.toggled.connect(self._emit_isolation)
        buttons = QHBoxLayout()
        buttons.addWidget(self.add_btn)
        buttons.addWidget(self.remove_btn)
        buttons.addWidget(self.isolate_btn)

        layout = QVBoxLayout(self)
        layout.addWidget(scroll, 1)
        layout.addLayout(buttons)

        bus.project_opened.connect(self._on_project_opened)
        bus.schema_changed.connect(self._rebuild)
        bus.cloud_labels_changed.connect(self._refresh_counts)
        bus.cloud_changed.connect(self._refresh_counts)
        bus.active_class_changed.connect(self._on_external_active)
        self._update_buttons()

    # -- state -------------------------------------------------------------
    def _on_project_opened(self, project) -> None:
        self.project = project
        # Default to the first user class if any, else the unlabelled row.
        self._active = 0 if project.schema.classes else UNLABELLED_ID
        self._rebuild()
        self.bus.active_class_changed.emit(self._active)

    def _rebuild(self) -> None:
        for row in self._rows:
            row.setParent(None)
        self._rows.clear()
        if not self.project:
            self._highlight()
            self._update_buttons()
            return

        # Fixed, non-deletable "unlabelled" (-1) row, always first.
        unlabelled = LabelRow(UNLABELLED_ID, "unlabelled", "#555", fixed=True)
        unlabelled.activated.connect(self._set_active)
        self._add_row_widget(unlabelled)

        for cls in self.project.schema.classes:
            row = LabelRow(cls.id, cls.name, cls.color)
            row.activated.connect(self._set_active)
            row.renamed.connect(self._on_renamed)
            row.recolored.connect(self._on_recolored)
            self._add_row_widget(row)

        self._highlight()
        self._update_buttons()
        self._refresh_counts()

    def _refresh_counts(self) -> None:
        """Point count + share of total per class (dense cloud first, else
        sparse) — makes class imbalance visible before training."""
        if not self._rows:
            return
        labels = None
        if self.project is not None:
            lb = self.project.labels
            labels = (lb.dense_cloud_labels if lb.dense_cloud_labels is not None
                      and len(lb.dense_cloud_labels) else lb.cloud_labels)
        if labels is None or not len(labels):
            for row in self._rows:
                row.set_count("")
            return
        total = int(len(labels))
        n_classes = max((r.class_id for r in self._rows), default=-1) + 1
        counts = np.bincount(np.clip(labels + 1, 0, n_classes),
                             minlength=n_classes + 1)
        for row in self._rows:
            row.set_count(format_count(int(counts[row.class_id + 1]), total))

    def _add_row_widget(self, row: LabelRow) -> None:
        self._rows_layout.insertWidget(self._rows_layout.count() - 1, row)
        self._rows.append(row)

    def _highlight(self) -> None:
        for row in self._rows:
            row.set_active(row.class_id == self._active)

    def _update_buttons(self) -> None:
        has_project = self.project is not None
        self.add_btn.setEnabled(has_project)
        # Only real user classes (0..N-1) can be removed; not "unlabelled".
        self.remove_btn.setEnabled(
            has_project and self._active is not None and self._active >= 0)

    # -- actions -----------------------------------------------------------
    def _set_active(self, class_id: int) -> None:
        self._active = class_id
        self._highlight()
        self._update_buttons()
        self.bus.active_class_changed.emit(class_id)
        self._emit_isolation()                     # isolation follows the active class

    def _on_external_active(self, class_id: int) -> None:
        """Sync when the active class changes elsewhere (e.g. number keys in
        the Image pane) — update the highlight without re-emitting."""
        if class_id == self._active:
            return
        self._active = class_id
        self._highlight()
        self._update_buttons()
        self._emit_isolation()

    def _emit_isolation(self, _checked=None) -> None:
        if self.isolate_btn.isChecked() and self._active is not None:
            self.bus.class_isolation_changed.emit(self._active)
        else:
            self.bus.class_isolation_changed.emit(None)

    def _add(self) -> None:
        if not self.project:
            return
        new = self.project.add_label()
        self._active = new.id
        self.bus.schema_changed.emit()       # triggers _rebuild
        self.bus.active_class_changed.emit(new.id)

    def _remove(self) -> None:
        if not self.project or self._active is None or self._active < 0:
            return                            # never delete the unlabelled row
        self.project.delete_label(self._active)
        remaining = len(self.project.schema.classes)
        self._active = min(self._active, remaining - 1) if remaining else UNLABELLED_ID
        self.bus.schema_changed.emit()
        self.bus.cloud_labels_changed.emit()  # labels were remapped
        self.bus.active_class_changed.emit(self._active)

    def _on_renamed(self, class_id: int, name: str) -> None:
        if self.project and name:
            self.project.rename_label(class_id, name)
            self.bus.schema_changed.emit()

    def _on_recolored(self, class_id: int, color: str) -> None:
        if self.project:
            self.project.set_label_color(class_id, color)
            self.bus.schema_changed.emit()
            self.bus.cloud_labels_changed.emit()  # refresh 3D blend colours
