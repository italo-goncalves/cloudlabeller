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

"""Model specification dialog — collects U-Net hyper-parameters.

Returns a :class:`~cloudlabeller.ml.model_spec.ModelSpec`. Pass the project's
source resolution so the dialog can preview the resulting training size as the
divisor is changed.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
)

from cloudlabeller.ml.model_spec import ModelSpec


class ModelSpecDialog(QDialog):
    def __init__(
        self,
        spec: ModelSpec | None = None,
        source_resolution: tuple[int, int] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Model Settings")
        self.setMinimumWidth(460)
        self._source = source_resolution
        spec = spec or ModelSpec()

        layout = QVBoxLayout(self)

        header = QLabel(
            "U-Net architecture and the resolution used for training. Images are "
            "downscaled by dividing their resolution; smaller divisors keep more "
            "detail but cost more memory and time."
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        form = QFormLayout()

        # Training resolution: divisor applied to each image's resolution.
        self.spin_divisor = QSpinBox()
        self.spin_divisor.setRange(1, 100)
        self.spin_divisor.setValue(spec.resolution_divisor)
        self.spin_divisor.setToolTip(
            "Each image's width and height are divided by this factor before "
            "training (cv2.resize). 1 = full resolution."
        )
        self.spin_divisor.valueChanged.connect(self._update_preview)
        form.addRow("Resolution divisor:", self.spin_divisor)

        # Live preview of the resulting training size.
        self.lbl_preview = QLabel()
        self.lbl_preview.setStyleSheet("color: gray;")
        form.addRow("", self.lbl_preview)

        # Channels in the first U-Net layer.
        self.spin_channels = QSpinBox()
        self.spin_channels.setRange(1, 256)
        self.spin_channels.setValue(spec.channels)
        self.spin_channels.setToolTip("Feature channels in the first U-Net layer; "
                                      "doubles at each deeper block.")
        form.addRow("First-layer channels:", self.spin_channels)

        # Number of max-pooling layers (U-Net blocks / depth).
        self.spin_blocks = QSpinBox()
        self.spin_blocks.setRange(1, 8)
        self.spin_blocks.setValue(spec.blocks)
        self.spin_blocks.setToolTip("Number of down/up-sampling stages. Each "
                                    "halves the resolution, so it must divide the "
                                    "training size evenly enough.")
        self.spin_blocks.valueChanged.connect(self._update_preview)
        form.addRow("U-Net blocks (max-pooling):", self.spin_blocks)

        # Dropout fraction.
        self.spin_dropout = QDoubleSpinBox()
        self.spin_dropout.setRange(0.0, 0.9)
        self.spin_dropout.setSingleStep(0.05)
        self.spin_dropout.setDecimals(2)
        self.spin_dropout.setValue(spec.dropout)
        form.addRow("Dropout fraction:", self.spin_dropout)

        # Convolution filter size (odd kernel).
        self.spin_filter = QSpinBox()
        self.spin_filter.setRange(1, 11)
        self.spin_filter.setSingleStep(2)      # keep it odd
        self.spin_filter.setValue(spec.filter_size)
        self.spin_filter.setToolTip("Convolution kernel size (odd values).")
        form.addRow("Filter size:", self.spin_filter)

        layout.addLayout(form)

        # Inline validation message (shown instead of a popup; OK is disabled
        # while the size is incompatible with the network depth).
        self.lbl_error = QLabel()
        self.lbl_error.setWordWrap(True)
        self.lbl_error.setStyleSheet("color: #c0392b;")
        self.lbl_error.hide()
        layout.addWidget(self.lbl_error)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self._update_preview()

    def _update_preview(self) -> None:
        ok = self.buttons.button(QDialogButtonBox.Ok)
        if not self._source:
            self.lbl_preview.setText("Original resolution unknown "
                                     "(add images / run SfM first).")
            self.lbl_error.hide()
            ok.setEnabled(True)      # nothing to validate against yet
            return

        from cloudlabeller.ml.model_spec import snap_aspect

        w, h = self._source
        spec = self.spec()
        aw, ah = snap_aspect(w, h)
        k = spec.pow2_scale(w, h)
        tw, th = spec.training_size(w, h)
        self.lbl_preview.setText(
            f"Trains at {tw} × {th} px  (from {w} × {h}; "
            f"aspect {aw}:{ah} × 2^{k}).")

        error = spec.validate_size(w, h)
        if error:
            self.lbl_error.setText(error)
        self.lbl_error.setVisible(error is not None)
        ok.setEnabled(error is None)

    def spec(self) -> ModelSpec:
        return ModelSpec(
            resolution_divisor=self.spin_divisor.value(),
            channels=self.spin_channels.value(),
            blocks=self.spin_blocks.value(),
            dropout=self.spin_dropout.value(),
            filter_size=self.spin_filter.value(),
        )
