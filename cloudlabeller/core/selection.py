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

"""Choosing points of a cloud: a display subsample and a screen-space lasso.

Pure numpy, no viewer state. The lasso follows the web window's
conventions: a three.js ``projectionMatrix × matrixWorldInverse`` (16
numbers, column-major as ``Matrix4.elements`` gives them) and a polygon in
CSS pixels with y pointing down. Selection goes through the cloud: points
hidden behind others are chosen too. Points behind the camera never are.
"""

from __future__ import annotations

import numpy as np

_CHUNK = 1 << 21          # points projected at once (bounds the float64 scratch)
MAX_POLYGON_VERTICES = 512


def even_subsample(n_points: int, max_points: int, seed: int = 0) -> np.ndarray | None:
    """Indices of a seeded uniform subsample of ``max_points`` points,
    ascending (so the cloud's own order, and its spatial spread, is kept).
    None when no subsample is needed: ``max_points`` is 0 or at least N."""
    if max_points <= 0 or max_points >= n_points:
        return None
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_points, size=max_points, replace=False))


def matrix_from_column_major(elements) -> np.ndarray:
    """A 4×4 matrix from three.js's column-major ``elements``."""
    m = np.asarray(elements, np.float64)
    if m.shape != (16,):
        raise ValueError("a 4x4 matrix needs 16 numbers")
    return m.reshape(4, 4).T


def project_to_css(xyz: np.ndarray, view_projection: np.ndarray,
                   width: float, height: float) -> tuple[np.ndarray, np.ndarray]:
    """World points to CSS pixels (origin top-left, y down).

    Returns ``(screen (N, 2), in_front (N,))``; ``in_front`` is False where
    the clip-space w is not positive (behind the camera), and those screen
    coordinates are meaningless.
    """
    xyz = np.asarray(xyz, np.float64)
    m = view_projection
    clip = xyz @ m[:, :3].T + m[:, 3]                 # (N, 4)
    w = clip[:, 3]
    in_front = w > 0
    w_safe = np.where(in_front, w, 1.0)
    sx = (clip[:, 0] / w_safe + 1.0) * 0.5 * width
    sy = (1.0 - clip[:, 1] / w_safe) * 0.5 * height
    return np.column_stack([sx, sy]), in_front


def points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Even-odd point-in-polygon test, vectorised over the points.

    ``polygon`` is (M, 2), implicitly closed. A self-crossing lasso keeps
    the even-odd meaning: a region circled twice is outside.
    """
    points = np.asarray(points, np.float64)
    poly = np.asarray(polygon, np.float64)
    inside = np.zeros(len(points), bool)
    if len(poly) < 3 or not len(points):
        return inside
    x, y = points[:, 0], points[:, 1]
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        crosses = (yi > y) != (yj > y)
        if yi != yj:
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            inside ^= crosses & (x < x_cross)
        j = i
    return inside


def _decimate(polygon: np.ndarray, max_vertices: int = MAX_POLYGON_VERTICES) -> np.ndarray:
    if len(polygon) <= max_vertices:
        return polygon
    idx = np.linspace(0, len(polygon) - 1, max_vertices).round().astype(int)
    return polygon[idx]


def select_in_polygon(xyz: np.ndarray, view_projection, polygon,
                      width: float, height: float,
                      progress=None) -> np.ndarray:
    """Indices (uint32, ascending) of the points of ``xyz`` whose screen
    projection lies inside ``polygon``, hidden ones included.

    ``view_projection`` is the 16-number column-major matrix or a 4×4 one.
    ``progress(fraction)`` is called between chunks when given.
    """
    m = (np.asarray(view_projection, np.float64) if np.ndim(view_projection) == 2
         else matrix_from_column_major(view_projection))
    poly = _decimate(np.asarray(polygon, np.float64).reshape(-1, 2))
    n = len(xyz)
    if len(poly) < 3 or n == 0:
        return np.empty(0, np.uint32)
    lo, hi = poly.min(axis=0), poly.max(axis=0)
    hits = []
    for start in range(0, n, _CHUNK):
        screen, in_front = project_to_css(xyz[start:start + _CHUNK], m, width, height)
        cand = np.flatnonzero(in_front
                              & (screen[:, 0] >= lo[0]) & (screen[:, 0] <= hi[0])
                              & (screen[:, 1] >= lo[1]) & (screen[:, 1] <= hi[1]))
        inside = points_in_polygon(screen[cand], poly)
        hits.append(cand[inside] + start)
        if progress is not None:
            progress(min(1.0, (start + _CHUNK) / n))
    return np.concatenate(hits).astype(np.uint32)


def pick_nearest(xyz: np.ndarray, view_projection, x: float, y: float,
                 width: float, height: float, radius: float = 8.0) -> int | None:
    """The point drawn nearest the viewer among those within ``radius`` CSS
    pixels of ``(x, y)``, or None: the cloud point a click snaps to."""
    m = view_projection
    if not isinstance(m, np.ndarray) or m.shape != (4, 4):
        m = matrix_from_column_major(m)
    screen, in_front = project_to_css(xyz, m, width, height)
    d2 = (screen[:, 0] - x) ** 2 + (screen[:, 1] - y) ** 2
    near = np.flatnonzero(in_front & (d2 <= radius * radius))
    if near.size == 0:
        return None
    clip = np.asarray(xyz, np.float64)[near] @ m[:, :3].T + m[:, 3]
    w = clip[:, 3]                                     # distance along the view
    # the front surface: of the points within 1% of the nearest depth, the
    # one closest to the click
    front = w <= w.min() * 1.01
    best = np.flatnonzero(front)[np.argmin(d2[near][front])]
    return int(near[best])
