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


def resample_label_mask(mask: np.ndarray, size: tuple[int, int],
                        min_label_fraction: float = 0.0) -> np.ndarray:
    """Resize an integer label mask to ``size`` = (width, height) without ever
    averaging class ids.

    Each labelled class (id >= 0) is resampled as a one-hot channel — bilinear
    when enlarging (smooth boundaries), area-averaged when shrinking (no
    moire) — and every target pixel takes the class with the largest coverage.
    A pixel becomes -1 (unlabelled) where labelled classes together cover less
    than ``min_label_fraction`` of it. The default 0 keeps a dense mask fully
    labelled while still leaving genuinely uncovered pixels unlabelled — so a
    mask that mixes labels with -1 keeps its -1 regions.

    Resizing the ids directly would blend them: bilinear turns a class-0 /
    class-2 border into a spurious class-1 seam, and INTER_AREA silently
    degenerates to nearest-neighbour when enlarging (no smoothing at all).
    Uses a running arg-max so memory stays at a few full-size buffers
    regardless of the class count.
    """
    import cv2

    mask = np.asarray(mask)
    width, height = int(size[0]), int(size[1])
    src_h, src_w = mask.shape
    labelled_ids = [int(c) for c in np.unique(mask) if c >= 0]
    if not labelled_ids:                              # nothing labelled
        return np.full((height, width), -1, dtype=mask.dtype)
    interp = (cv2.INTER_LINEAR if width * height >= src_w * src_h
              else cv2.INTER_AREA)
    best_cover = np.zeros((height, width), np.float32)
    best_id = np.full((height, width), labelled_ids[0], mask.dtype)
    total_cover = np.zeros((height, width), np.float32)
    for cid in labelled_ids:
        cover = cv2.resize((mask == cid).astype(np.float32), (width, height),
                           interpolation=interp)
        total_cover += cover
        winning = cover > best_cover
        best_cover[winning] = cover[winning]
        best_id[winning] = cid
    # A tiny floor (even when min_label_fraction is 0) keeps truly uncovered
    # pixels unlabelled — a dense mask still comes back fully labelled.
    threshold = max(min_label_fraction, 1e-6)
    return np.where(total_cover >= threshold, best_id, -1).astype(mask.dtype)


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
