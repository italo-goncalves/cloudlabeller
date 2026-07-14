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

"""Rasterisation helpers for 2D image labelling (pure NumPy + Pillow)."""

from __future__ import annotations

import numpy as np


def rasterize_polygon(points, height: int, width: int) -> np.ndarray:
    """Return a boolean (H, W) mask of the filled polygon.

    ``points`` is a sequence of (x, y) in pixel coordinates. Fewer than 3 points
    yields an all-False mask.
    """
    mask = np.zeros((height, width), dtype=bool)
    if len(points) < 3:
        return mask
    from PIL import Image, ImageDraw

    img = Image.new("L", (width, height), 0)
    ImageDraw.Draw(img).polygon([(float(x), float(y)) for x, y in points], fill=1)
    return np.asarray(img, dtype=bool)


def mask_from_file(path, expected_shape: tuple[int, int] | None = None) -> np.ndarray:
    """Load an external selection mask file as a boolean (H, W) array.

    * ``.jpg``/``.jpeg`` — selects the highest-intensity pixels (grayscale).
    * ``.png`` — selects the pixels with the highest alpha value.

    ``expected_shape`` is the (height, width) of the image being labelled;
    a mismatching file raises ``ValueError``.
    """
    from pathlib import Path

    from PIL import Image

    ext = Path(path).suffix.lower()
    if ext not in (".jpg", ".jpeg", ".png"):
        raise ValueError(f"Unsupported mask format {ext!r} — use .jpg "
                         "(intensity) or .png (alpha).")
    with Image.open(path) as im:
        if ext == ".png":
            if "A" not in im.getbands():
                raise ValueError("PNG mask has no alpha channel — export it "
                                 "with transparency, or use a JPG mask.")
            data = np.asarray(im.getchannel("A"))       # alpha
        else:
            data = np.asarray(im.convert("L"))          # intensity

    if expected_shape is not None and data.shape != tuple(expected_shape):
        raise ValueError(
            f"Mask size is {data.shape[1]} × {data.shape[0]} px but the image "
            f"is {expected_shape[1]} × {expected_shape[0]} px — the mask must "
            "match the image exactly.")
    return data == data.max()


def mask_to_indexed(mask: np.ndarray,
                    lut: dict[int, tuple[int, int, int]]):
    """(H, W) int label mask -> (uint8 index image, RGBA colour table).

    The colormap approach: instead of materialising a full RGBA overlay
    (179 MB / ~450 ms at 21 Mpx), emit a 1-byte-per-pixel index (45 MB, ~110 ms
    total with the Qt colour-table conversion). Row 0 of the table is fully
    transparent and represents unlabelled (-1) — the "NaN" of the palette;
    a trailing transparent sentinel row absorbs stray ids from older schemas.
    """
    n = max((cid for cid in lut if cid >= 0), default=-1) + 1
    table: list[tuple[int, int, int, int]] = [(0, 0, 0, 0)] * (n + 2)
    for class_id, color in lut.items():
        if 0 <= class_id < n:
            table[class_id + 1] = (color[0], color[1], color[2], 255)
    index = np.ascontiguousarray(np.clip(mask + 1, 0, n + 1).astype(np.uint8))
    return index, table


def mask_to_rgba(mask: np.ndarray, lut: dict[int, tuple[int, int, int]]) -> np.ndarray:
    """Colour an (H, W) int label mask into an (H, W, 4) uint8 RGBA overlay.

    Pixels whose label is in ``lut`` get that colour at full alpha; everything
    else (e.g. unlabelled -1) stays fully transparent.

    Implemented as a single palette lookup (one pass over the mask) — the
    per-class boolean-mask version cost ~1.1 s on a 21 Mpx mask; this is ~10x
    faster.
    """
    n = max((cid for cid in lut if cid >= 0), default=-1) + 1
    # Rows: 0 = unlabelled (-1), 1..n = classes, n+1 = out-of-range sentinel
    # (stray ids from an older schema stay transparent, as before).
    palette = np.zeros((n + 2, 4), dtype=np.uint8)
    for class_id, color in lut.items():
        if 0 <= class_id < n:
            palette[class_id + 1, :3] = color
            palette[class_id + 1, 3] = 255
    index = np.clip(mask + 1, 0, n + 1)
    return palette[index]
