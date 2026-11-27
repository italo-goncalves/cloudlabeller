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


"""Normal maps: the dense cloud's surface normals drawn into a photo.

The normals of the points a photo sees (its visibility, ``transfer.
visibility``) are splatted into their pixels as an auto mask's labels are,
and stored as uint8 RGB: ``round(127.5 * (n + 1))`` per component, so 128
means "no normal here" and a normal map doubles as a false-colour picture
(red = +x, green = +y, blue = +z).

Two frames:

- ``camera``: x right, y down, z towards the scene, what the U-Net takes as
  three channels beside RGB (an experimental option, see
  ``docs/research/normals-as-features.md``);
- ``world``: the Project frame, for showing a photo coloured as its cloud is.

Only a dense cloud has normals.
"""

from __future__ import annotations

import numpy as np

#: The encoded value of a missing normal component (0).
NO_NORMAL = 128


def encode_normals(normals: np.ndarray) -> np.ndarray:
    """Normals in [-1, 1] as uint8 0..255 (0 -> 128): a false-colour RGB."""
    n = np.asarray(normals, np.float32)
    return np.clip(np.rint(127.5 * (n + 1.0)), 0, 255).astype(np.uint8)


def unit(normals: np.ndarray) -> np.ndarray:
    n = np.asarray(normals, np.float64)
    return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)


def normal_map(normals: np.ndarray, pix: np.ndarray, width: int, height: int,
               rotation: np.ndarray | None = None, splat: int = 2) -> np.ndarray:
    """(height, width, 3) uint8 normal map.

    ``normals`` (K, 3) of the visible points and ``pix`` (K, 2) their
    (col, row) pixels, as ``VisibilityIndex.get`` gives them. ``rotation``
    (3, 3) turns them into another frame first (a camera's ``R`` for the
    camera frame). Each is splatted in a square of side ``2 * splat + 1``."""
    out = np.full((int(height), int(width), 3), NO_NORMAL, np.uint8)
    if len(pix) == 0:
        return out
    n = np.asarray(normals, np.float64)
    if rotation is not None:
        n = n @ np.asarray(rotation, np.float64).T
    enc = encode_normals(unit(n))
    cols, rows = pix[:, 0], pix[:, 1]
    for dy in range(-splat, splat + 1):
        for dx in range(-splat, splat + 1):
            out[np.clip(rows + dy, 0, height - 1), np.clip(cols + dx, 0, width - 1)] = enc
    return out
