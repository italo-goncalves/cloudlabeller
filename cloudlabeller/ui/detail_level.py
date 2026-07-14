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

"""Relative level-of-detail control for reconstruction dialogs.

Translates named levels into COLMAP's ``max_image_size`` relative to the
project's ORIGINAL image resolution. Levels are (name, linear divisor of the
longest side); the label shows the AREA fraction — dividing the side by 2
quarters the pixels, so divisor 2 reads "1/4 resolution". A Custom entry
exposes the raw pixel cap for fine-tuning. When the source resolution is
unknown the presets fall back to sensible absolute sizes.
"""

from __future__ import annotations

from PySide6.QtWidgets import QComboBox, QHBoxLayout, QSpinBox, QWidget

# (name, linear divisor). Feature extraction is cheap enough to keep detail;
# patch-match stereo is far more memory-hungry, so MVS starts one level lower
# and reaches one level deeper.
SFM_LEVELS = [("High", 1), ("Medium", 2), ("Low", 4)]
MVS_LEVELS = [("Ultra", 1), ("High", 2), ("Medium", 4), ("Low", 8)]


class DetailControl(QWidget):
    """A combo of named levels + a pixel spinbox for Custom."""

    def __init__(self, source_resolution: tuple[int, int] | None = None,
                 default_level: str = "medium", custom_default: int = 3200,
                 levels: list[tuple[str, int]] | None = None,
                 parent=None) -> None:
        super().__init__(parent)
        levels = levels or SFM_LEVELS
        long_side = max(source_resolution) if source_resolution else None

        self.combo = QComboBox()
        self._pixels = []
        for name, div in levels:
            if div == 1:
                px = 0                              # 0 = no cap (full resolution)
                label = (f"{name} — full resolution ({long_side} px)"
                         if long_side else f"{name} — full resolution")
            else:
                # Unknown source: scale the fallback so divisor 2 lands on
                # custom_default (the historical Medium preset).
                px = (long_side // div) if long_side else (custom_default * 2 // div)
                label = f"{name} — 1/{div * div} resolution ({px} px)"
            self.combo.addItem(label)
            self._pixels.append(px)
        self.combo.addItem("Custom…")
        self.combo.setToolTip("Images are downscaled to this cap before "
                              "reconstruction. Higher detail = slower + more memory.")

        self.spin = QSpinBox()
        self.spin.setRange(0, 20000)
        self.spin.setSingleStep(200)
        self.spin.setValue(custom_default)
        self.spin.setSpecialValueText("Full resolution")   # shown when value == 0
        self.spin.setSuffix(" px")
        self.spin.hide()                                   # visible for Custom only

        names = [name.lower() for name, _ in levels]
        try:
            index = names.index(default_level.lower())
        except ValueError:
            index = min(1, len(levels) - 1)
        self.combo.setCurrentIndex(index)
        self.combo.currentIndexChanged.connect(
            lambda i: self.spin.setVisible(i >= len(self._pixels)))

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.combo, 1)
        row.addWidget(self.spin)

    def pixels(self) -> int:
        """COLMAP ``max_image_size`` (0 = full resolution)."""
        i = self.combo.currentIndex()
        return self._pixels[i] if i < len(self._pixels) else self.spin.value()
