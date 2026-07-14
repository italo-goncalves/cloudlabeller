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

"""Train-U-Net dialog — model name + training options.

The architecture itself comes from Model → Model Settings… (ModelSpec); this
dialog only collects what varies per run. Typing (or picking) the name of an
already-saved model offers a choice: overwrite it with a fresh model, or load
it and train it for more epochs (resume). OK is disabled until there is at
least one image to train on.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
)


class TrainDialog(QDialog):
    def __init__(self, n_user: int, n_auto: int,
                 training_size: tuple[int, int] | None,
                 manifests: list[dict] | None = None,
                 n_classes: int | None = None,
                 parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Train U-Net")
        self.setMinimumWidth(500)
        self._n_user = n_user
        self._n_auto = n_auto
        self._n_classes = n_classes
        self._manifests = {m["name"]: m for m in (manifests or [])}

        layout = QVBoxLayout(self)

        size_txt = (f"training at {training_size[0]} × {training_size[1]} px"
                    if training_size else "training size unknown")
        header = QLabel(
            f"<b>{n_user}</b> user-labelled and <b>{n_auto}</b> auto-labelled "
            f"image(s) available, {size_txt}.<br>"
            "Architecture comes from <b>Model → Model Settings…</b>")
        header.setWordWrap(True)
        layout.addWidget(header)

        form = QFormLayout()

        # Editable combo: type a new name, or pick a saved model to overwrite /
        # continue training.
        self.cmb_name = QComboBox()
        self.cmb_name.setEditable(True)
        self.cmb_name.addItems(sorted(self._manifests))
        self.cmb_name.setCurrentText("unet")
        self.cmb_name.setToolTip("The model is saved under this name "
                                 "(ml/models/<name>) and shown in the load dialog.")
        self.cmb_name.editTextChanged.connect(self._validate)
        form.addRow("Model name:", self.cmb_name)

        # Existing-model choice (visible only when the name matches one).
        self.rb_overwrite = QRadioButton("Start fresh (overwrite the saved model)")
        self.rb_resume = QRadioButton("Continue training the saved model")
        self.rb_overwrite.setChecked(True)
        self.rb_resume.setToolTip(
            "Loads the saved weights and trains for the epochs below. "
            "Architecture and training size come from the saved model — "
            "current Model Settings are ignored.")
        self.rb_overwrite.toggled.connect(self._validate)
        self.lbl_saved = QLabel()
        self.lbl_saved.setStyleSheet("color: gray;")
        self.lbl_saved.setWordWrap(True)
        form.addRow("", self.rb_overwrite)
        form.addRow("", self.rb_resume)
        form.addRow("", self.lbl_saved)

        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 10000)
        self.spin_epochs.setValue(100)
        self.lbl_epochs = QLabel("Epochs:")
        form.addRow(self.lbl_epochs, self.spin_epochs)

        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 64)
        self.spin_batch.setValue(2)
        self.spin_batch.setToolTip("Images per gradient step. Larger is faster "
                                   "but needs more memory.")
        form.addRow("Batch size:", self.spin_batch)

        self.spin_val = QDoubleSpinBox()
        self.spin_val.setRange(0.0, 0.5)
        self.spin_val.setSingleStep(0.05)
        self.spin_val.setDecimals(2)
        self.spin_val.setValue(0.2)
        self.spin_val.setToolTip("Fraction of images held out for validation "
                                 "(0 = train on everything).")
        form.addRow("Validation fraction:", self.spin_val)

        self.spin_rolls = QSpinBox()
        self.spin_rolls.setRange(0, 20)
        self.spin_rolls.setValue(5)
        self.spin_rolls.setToolTip(
            "Mirror/slide/gamma augmentation: each image becomes 2×N shifted "
            "copies with random gamma. 0 disables augmentation.")
        form.addRow("Augmentation rolls:", self.spin_rolls)

        self.chk_auto = QCheckBox(
            f"Include {n_auto} auto-labelled image(s) at half weight")
        self.chk_auto.setChecked(False)
        self.chk_auto.setEnabled(n_auto > 0)
        self.chk_auto.setToolTip(
            "Cloud-projected masks join the training set with sample weight "
            "0.5 (they are noisier than hand-drawn polygons). U-Net-predicted "
            "images are never used for training.")
        self.chk_auto.toggled.connect(self._validate)
        layout.addLayout(form)
        layout.addWidget(self.chk_auto)

        self.lbl_error = QLabel()
        self.lbl_error.setWordWrap(True)
        self.lbl_error.setStyleSheet("color: #c0392b;")
        self.lbl_error.hide()
        layout.addWidget(self.lbl_error)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self._validate()

    # -- state ---------------------------------------------------------------
    def _current_manifest(self) -> dict | None:
        from cloudlabeller.ml.model_store import sanitize_name

        return self._manifests.get(sanitize_name(self.cmb_name.currentText()))

    def _validate(self) -> None:
        manifest = self._current_manifest()
        exists = manifest is not None
        for widget in (self.rb_overwrite, self.rb_resume, self.lbl_saved):
            widget.setVisible(exists)
        resume = exists and self.rb_resume.isChecked()
        if exists:
            spec = manifest.get("spec", {})
            w, h = manifest.get("input_size", ("?", "?"))
            self.lbl_saved.setText(
                f"Saved model: {w} × {h} px, {spec.get('channels', '?')} ch × "
                f"{spec.get('blocks', '?')} blocks, "
                f"{manifest.get('n_classes', '?')} classes, "
                f"created {str(manifest.get('created', ''))[:16].replace('T', ' ')}")
        self.lbl_epochs.setText("Epochs (additional):" if resume else "Epochs:")

        n = self._n_user + (self._n_auto if self.chk_auto.isChecked() else 0)
        error = None
        if n == 0:
            error = ("No training data: draw polygons on some images, or "
                     "include auto-labelled images.")
        elif not self.cmb_name.currentText().strip():
            error = "Enter a model name."
        elif (resume and self._n_classes is not None
              and manifest.get("n_classes") != self._n_classes):
            error = (f"The saved model has {manifest.get('n_classes')} classes "
                     f"but the project now has {self._n_classes} — it cannot "
                     "be resumed. Start fresh instead.")
        self.lbl_error.setVisible(error is not None)
        if error:
            self.lbl_error.setText(error)
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(error is None)

    # -- results -------------------------------------------------------------
    def options(self) -> dict:
        return {
            "name": self.cmb_name.currentText().strip(),
            "epochs": self.spin_epochs.value(),
            "batch_size": self.spin_batch.value(),
            "val_fraction": self.spin_val.value(),
            "augment_rolls": self.spin_rolls.value(),
            "include_auto": self.chk_auto.isChecked(),
            "resume": self._current_manifest() is not None and self.rb_resume.isChecked(),
        }
