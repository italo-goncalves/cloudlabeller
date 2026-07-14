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

"""Load-model dialog — pick a saved U-Net by name.

Lists every model under ``ml/models/`` with its architectural parameters
(from the manifest, no TensorFlow import). Returns the selected model name.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QLabel,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


class LoadModelDialog(QDialog):
    COLUMNS = ("Name", "Created", "Input size", "Classes",
               "Channels", "Blocks", "Dropout", "Filter", "Final metrics")

    def __init__(self, manifests: list[dict], title: str = "Select Model",
                 ml_dir=None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(760)
        self._manifests = list(manifests)
        self._ml_dir = ml_dir                      # enables the Delete button

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"{len(manifests)} saved model(s):"))

        self.table = QTableWidget(len(manifests), len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)

        for row, m in enumerate(manifests):
            for col, text in enumerate(self._cells(m)):
                self.table.setItem(row, col, QTableWidgetItem(text))
        if manifests:
            self.table.selectRow(0)
        self.table.doubleClicked.connect(lambda *_: self.accept())
        self.table.itemSelectionChanged.connect(self._update_ok)
        layout.addWidget(self.table)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        if ml_dir is not None:
            self.btn_delete = self.buttons.addButton(
                "Delete…", QDialogButtonBox.ActionRole)
            self.btn_delete.clicked.connect(self._delete_selected)
        layout.addWidget(self.buttons)
        self._update_ok()

    @staticmethod
    def _cells(m: dict) -> tuple:
        spec = m.get("spec", {})
        w, h = m.get("input_size", ("?", "?"))
        classes = m.get("class_names") or []
        metrics = m.get("metrics") or {}
        shown = [(k, metrics[k]) for k in ("accuracy", "loss", "val_accuracy",
                                           "val_loss") if k in metrics]
        return (
            m.get("name", "?"),
            str(m.get("created", ""))[:16].replace("T", " "),
            f"{w} × {h}",
            ", ".join(classes) if classes else str(m.get("n_classes", "?")),
            str(spec.get("channels", "?")),
            str(spec.get("blocks", "?")),
            str(spec.get("dropout", "?")),
            str(spec.get("filter_size", "?")),
            "  ".join(f"{k.replace('accuracy', 'acc')} {v:.3f}"
                      for k, v in shown) or "—",
        )

    def _delete_selected(self) -> None:
        name = self.selected_name()
        if name is None:
            return
        answer = QMessageBox.question(
            self, "Delete model",
            f"Delete the saved model '{name}'?\nThis cannot be undone.")
        if answer != QMessageBox.Yes:
            return
        from cloudlabeller.ml.model_store import delete_model

        delete_model(self._ml_dir, name)
        row = self.table.selectionModel().selectedRows()[0].row()
        self.table.removeRow(row)
        del self._manifests[row]
        self._update_ok()

    def _update_ok(self) -> None:
        has_sel = bool(self.table.selectionModel().selectedRows())
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(has_sel)
        if hasattr(self, "btn_delete"):
            self.btn_delete.setEnabled(has_sel)

    def selected_name(self) -> str | None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return None
        return self._manifests[rows[0].row()].get("name")
