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

Kept free of TensorFlow so the Engine can import it. The network works on
**Tiles**: square pieces of ``tile_size`` photo pixels, cut on a grid whose
neighbours overlap by ``overlap`` of a Tile, and shrunk by ``shrink`` before
they reach the network. Photos of any size or shape give Tiles of one size,
so one model trains on Surveys taken with different cameras.

The U-Net halves the resolution at every max-pooling block and doubles it
back up, so the network's input, ``tile_size / shrink``, must be divisible
by ``2**blocks``.

``normals`` (experimental) adds the dense cloud's normals, drawn into each
photo in the camera's frame (``transfer.normal_map``), as three channels
beside RGB: on FIDA it halved the errors on hand-painted photos
(``docs/research/normals-as-features.md``). Such a model trains and
predicts only photos of Surveys with a dense Reconstruction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

SHRINKS = (1, 2, 4, 8)


@dataclass
class ModelSpec:
    """User-facing training/model settings."""

    tile_size: int = 1024       # Tile side, in photo pixels
    shrink: int = 2             # the Tile is shrunk by this before the network
    overlap: float = 0.35       # of a Tile, between neighbours on the grid
    min_labelled: float = 0.05  # a training Tile needs this much labelled
    channels: int = 16          # channels in the first U-Net layer
    blocks: int = 4             # number of max-pooling layers (U-Net depth)
    dropout: float = 0.1        # dropout fraction
    filter_size: int = 5        # convolution kernel size
    normals: bool = False       # experimental: the dense cloud's normals as 3 more channels

    @property
    def net_size(self) -> int:
        """The side of what the network sees: the Tile shrunk."""
        return int(self.tile_size) // max(1, int(self.shrink))

    @property
    def in_channels(self) -> int:
        """What the network takes per pixel: RGB, plus the normals when asked."""
        return 6 if self.normals else 3

    @property
    def buffer(self) -> int:
        """Photo pixels neighbouring Tiles share."""
        return int(self.tile_size * self.overlap)

    def validate(self) -> str | None:
        """A message when these settings can't build a network, else None."""
        if self.shrink not in SHRINKS:
            return f"The shrink factor must be one of {', '.join(map(str, SHRINKS))}."
        if self.tile_size < 32 or self.tile_size % self.shrink:
            return (f"The Tile size ({self.tile_size} px) must be at least 32 and a "
                    f"multiple of the shrink factor ({self.shrink}).")
        if not 0 <= self.overlap < 0.5:
            return "The overlap must be at least 0 and less than half a Tile."
        step = 2 ** int(self.blocks)
        if self.net_size % step:
            return (f"The network sees {self.net_size} × {self.net_size} px, which "
                    f"{self.blocks} max-pooling blocks can't halve evenly: it must be a "
                    f"multiple of {step}. Change the Tile size or the shrink factor.")
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


def default_tile_size(width: int, height: int) -> int:
    """The Tile size for photos of this size: the first power of two greater
    than a quarter of their longer side (2048 px for 5472 × 3648 drone
    photos), and at least 32."""
    quarter = max(int(width), int(height)) / 4
    size = 32
    while size <= quarter:
        size *= 2
    return size


def is_tiled(manifest: dict) -> bool:
    """Whether a saved model was trained on Tiles (0.4.0); older ones were
    trained on whole photos and must be retrained."""
    return "tile_size" in (manifest.get("spec") or {})
