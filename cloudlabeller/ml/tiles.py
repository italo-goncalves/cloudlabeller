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

"""Tiles: cutting photos into the network's pieces, and blending back.

Ported from ``unet_tools.project.SlicedSegmentationProject`` (same author,
GPL-3.0). A photo is padded so a grid of overlapping square Tiles covers it
exactly, each Tile is shrunk to the network's size, and predictions of
overlapping Tiles are blended with feathered weights: weights rise linearly
across the overlap and are 1 in the middle, so neighbours fade into each
other instead of leaving seams. Unlike ``unet_tools``, blending happens at
the network's resolution (a full-size probability stack of a 20 Mpx photo
is more than a gigabyte), training uses the same padded grid (the padding is
unlabelled, which the loss ignores), and the entropy of the blended
probabilities, scaled to [0, 1], is kept as the photo's Uncertainty.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from cloudlabeller.core.label_schema import UNLABELLED_ID


@dataclass(frozen=True)
class Grid:
    """Where the Tiles of a ``height`` × ``width`` photo go."""

    tile: int
    buffer: int
    rows: int
    cols: int

    @property
    def step(self) -> int:
        return self.tile - self.buffer

    @property
    def padded(self) -> tuple[int, int]:
        """(height, width) of the photo padded to the grid."""
        return (self.rows * self.step + self.buffer, self.cols * self.step + self.buffer)

    def origins(self):
        """(row, col, y, x) of each Tile's top-left corner, in photo pixels."""
        for i in range(self.rows):
            for j in range(self.cols):
                yield i, j, i * self.step, j * self.step


def grid_for(height: int, width: int, tile: int, overlap: float) -> Grid:
    buffer = int(tile * overlap)
    step = tile - buffer

    def n(size: int) -> int:
        return max(1, math.ceil((size - buffer) / step))
    return Grid(tile, buffer, n(height), n(width))


def pad(array: np.ndarray, grid: Grid, value) -> np.ndarray:
    """``array`` (H, W[, C]) padded at the bottom and right to the grid."""
    ph, pw = grid.padded
    h, w = array.shape[:2]
    widths = [(0, max(0, ph - h)), (0, max(0, pw - w))] + [(0, 0)] * (array.ndim - 2)
    return np.pad(array, widths, constant_values=value)


def feather(size: int, buffer: int) -> np.ndarray:
    """(size, size) blending weights: a linear ramp across the overlap at
    each edge, 1 inside (``unet_tools``' window weights)."""
    if buffer <= 0:
        return np.ones((size, size), np.float32)
    ramp = (np.arange(buffer) + 1) / (buffer + 1)
    line = np.ones(size)
    line[:buffer] = ramp
    line[size - buffer:] = np.minimum(line[size - buffer:], ramp[::-1])
    return ((line[:, None] + line[None, :]) / 2).astype(np.float32)


def shrink_image(tile: np.ndarray, shrink: int) -> np.ndarray:
    if shrink == 1:
        return tile
    import cv2

    n = tile.shape[0] // shrink
    return cv2.resize(tile, (n, n), interpolation=cv2.INTER_AREA)


def shrink_mask(tile: np.ndarray, shrink: int) -> np.ndarray:
    if shrink == 1:
        return tile
    from cloudlabeller.core.raster import resample_label_mask

    n = tile.shape[0] // shrink
    return resample_label_mask(tile, (n, n)).astype(np.int32)


#: A training Tile's mask: class ids fit in 16 bits, half of int32's memory.
MASK_DTYPE = np.int16


def to_uint8(image: np.ndarray) -> np.ndarray:
    """``image`` as uint8: floats in [0, 1] scaled to 0..255."""
    if image.dtype == np.uint8:
        return image
    a = np.asarray(image, np.float32)
    if a.size and float(a.max()) <= 1.0:
        a = a * 255.0
    return np.clip(np.rint(a), 0, 255).astype(np.uint8)


def cut(image: np.ndarray, mask: np.ndarray, tile: int, overlap: float, shrink: int,
        min_labelled: float, weights: np.ndarray | None = None) -> list[tuple]:
    """The training Tiles of one photo: ``((row, col), image, mask)``, each
    shrunk to the network's size, skipping Tiles less than ``min_labelled``
    labelled. The padding is unlabelled. With per-pixel ``weights`` (0..1,
    NaN taken as 1) each Tile also gets its weights, shrunk by area and
    stored as uint8 (255 = 1): ``((row, col), image, mask, weights)``.

    Each Tile owns its memory (uint8 image, int16 mask): a view into the
    padded photo would keep the whole photo alive for as long as one of its
    Tiles is, which at full resolution is ~60 MB per photo."""
    grid = grid_for(image.shape[0], image.shape[1], tile, overlap)
    img = pad(to_uint8(image), grid, 0)
    msk = pad(np.asarray(mask, np.int32), grid, UNLABELLED_ID)
    wts = None
    if weights is not None:
        w = np.nan_to_num(np.asarray(weights, np.float32), nan=1.0)
        wts = pad(np.rint(np.clip(w, 0.0, 1.0) * 255).astype(np.uint8), grid, 0)
    out = []
    for i, j, y, x in grid.origins():
        m = msk[y:y + tile, x:x + tile]
        if (m != UNLABELLED_ID).mean() < max(min_labelled, 1e-9):
            continue
        timg = np.ascontiguousarray(shrink_image(img[y:y + tile, x:x + tile], shrink))
        tmask = shrink_mask(m, shrink).astype(MASK_DTYPE)     # astype copies
        if timg.base is not None:
            timg = timg.copy()
        if wts is None:
            out.append(((i, j), timg, tmask))
        else:
            twts = np.ascontiguousarray(shrink_image(wts[y:y + tile, x:x + tile], shrink))
            out.append(((i, j), timg, tmask, twts.copy() if twts.base is not None else twts))
    return out


def predict_photo(predict_batch, image: np.ndarray, tile: int, overlap: float,
                  shrink: int, n_classes: int, batch: int = 8
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Classes and Uncertainty of one photo, at its size divided by
    ``shrink`` (rounded up). ``predict_batch(x)`` maps (B, n, n, C) float32
    in [0, 1] to (B, n, n, n_classes) probabilities."""
    h, w = image.shape[:2]
    grid = grid_for(h, w, tile, overlap)
    img = pad(image, grid, 0)
    n = tile // shrink
    nb = grid.buffer // shrink
    ns = n - nb                                    # the step, at network size
    weights = feather(n, nb)
    ph, pw = grid.rows * ns + nb, grid.cols * ns + nb
    acc = np.zeros((ph, pw, n_classes), np.float32)
    total = np.zeros((ph, pw), np.float32)
    origins = list(grid.origins())
    for k in range(0, len(origins), batch):
        part = origins[k:k + batch]
        x = np.stack([shrink_image(img[y:y + tile, xx:xx + tile], shrink)
                      for _, _, y, xx in part]).astype(np.float32)
        if x.max() > 1.0:
            x /= 255.0
        probs = np.asarray(predict_batch(x), np.float32)
        for (i, j, _, _), p in zip(part, probs):
            y0, x0 = i * ns, j * ns
            acc[y0:y0 + n, x0:x0 + n] += p * weights[..., None]
            total[y0:y0 + n, x0:x0 + n] += weights
    probs = acc / np.maximum(total, 1e-6)[..., None]
    oh, ow = math.ceil(h / shrink), math.ceil(w / shrink)
    probs = probs[:oh, :ow]
    labels = np.argmax(probs, axis=-1).astype(np.int32)
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-6, 1.0)), axis=-1)
    uncertainty = entropy / math.log(n_classes) if n_classes > 1 else np.zeros_like(entropy)
    return labels, np.clip(uncertainty, 0, 1).astype(np.float16)
