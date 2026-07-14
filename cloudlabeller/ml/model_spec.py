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

"""Model specification — the U-Net hyper-parameters chosen by the user.

Kept deliberately free of TensorFlow so it can be imported by the UI and the
project layer without pulling in the (heavy) training stack. The values map
one-to-one onto ``ml.unet.build_model`` / ``u_net`` arguments.

Training-size rule: the U-Net halves the resolution at every max-pooling block
and doubles it back up, so both sides of the training image must be divisible
by ``2**blocks`` or the skip connections won't line up. Instead of resizing to
``resolution / divisor`` directly, the image's aspect ratio is snapped to a
small integer ratio (3:2, 4:3, 16:9, …) — stretching slightly if needed — and
the training size is ``(ratio_w, ratio_h) * 2**k``, with ``k`` chosen so the
pixel count best approximates the requested divisor.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from fractions import Fraction

# Largest term allowed in a snapped aspect ratio. 21 admits the common photo /
# video ratios (3:2, 4:3, 5:4, 16:9, 21:9…) while keeping the base tile small
# enough that multiplying by 2**k can approximate any sensible divisor.
_MAX_ASPECT_TERM = 21


def snap_aspect(width: int, height: int) -> tuple[int, int]:
    """Snap ``width:height`` to the closest small integer ratio.

    E.g. 5472x3648 -> (3, 2); 1920x1080 -> (16, 9); 4000x3000 -> (4, 3).
    A slightly-off ratio (cropped panoramas, odd sensors) lands on the nearest
    small ratio, accepting a little stretch at resize time.
    """
    frac = Fraction(int(width), int(height)).limit_denominator(_MAX_ASPECT_TERM)
    # limit_denominator only bounds the denominator; bound wide/tall ratios'
    # numerator too by working on the inverted fraction when needed.
    if frac.numerator > _MAX_ASPECT_TERM:
        inv = Fraction(int(height), int(width)).limit_denominator(_MAX_ASPECT_TERM)
        frac = 1 / inv
    return frac.numerator, frac.denominator


@dataclass
class ModelSpec:
    """User-facing training/model settings.

    ``resolution_divisor`` sets the target shrink factor: a divisor of 10 aims
    a 5472x3648 photo at ~547x365, which is then snapped to the nearest
    U-Net-compatible size (see module docstring) — 576x384 (= 3:2 x 2**7).
    Predictions are produced at that reduced size and later upscaled back to
    the image's original resolution.
    """

    resolution_divisor: int = 10
    channels: int = 16          # channels in the first U-Net layer
    blocks: int = 4             # number of max-pooling layers (U-Net depth)
    dropout: float = 0.1        # dropout fraction
    filter_size: int = 5        # convolution kernel size

    def pow2_scale(self, width: int, height: int) -> int:
        """The exponent ``k`` such that ``snap_aspect * 2**k`` best matches the
        pixel count of ``(width/divisor) x (height/divisor)``."""
        d = max(1, int(self.resolution_divisor))
        aw, ah = snap_aspect(width, height)
        target_area = (int(width) / d) * (int(height) / d)
        scale = math.sqrt(target_area / (aw * ah))   # ideal (fractional) 2**k
        return max(0, round(math.log2(scale))) if scale > 0 else 0

    def training_size(self, width: int, height: int) -> tuple[int, int]:
        """The U-Net-compatible (width, height) an image of ``width``x``height``
        is resized to for training/inference: snapped aspect ratio x ``2**k``."""
        aw, ah = snap_aspect(width, height)
        k = self.pow2_scale(width, height)
        return (aw * 2 ** k, ah * 2 ** k)

    def validate_size(self, width: int, height: int) -> str | None:
        """Return an error message if the training size is incompatible with
        the network depth (needs ``k >= blocks``), else None."""
        k = self.pow2_scale(width, height)
        if k < self.blocks:
            tw, th = self.training_size(width, height)
            return (f"Training size {tw} × {th} px only supports {k} halving(s) "
                    f"(2^{k}), but the network has {self.blocks} max-pooling "
                    f"blocks. Lower the resolution divisor or use fewer blocks.")
        return None

    def unet_kwargs(self) -> dict:
        """Keyword arguments for ``ml.unet.build_model`` / ``u_net``."""
        return {
            "channels": self.channels,
            "blocks": self.blocks,
            "dropout_prob": self.dropout,
            "filter_size": self.filter_size,
        }

    # -- persistence -------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "ModelSpec":
        data = data or {}
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})
