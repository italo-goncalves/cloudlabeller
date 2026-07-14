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

"""Pure lasso-selection math (no Qt/VTK state — testable headlessly).

The viewer grabs the renderer's composite projection matrix once per lasso, we
project every cloud point to display coordinates in one vectorized pass
(~45 ms for 1.1M points), then test them against the drawn path with
matplotlib's C point-in-polygon code.
"""

from __future__ import annotations

import numpy as np


def composite_projection_matrix(renderer) -> np.ndarray:
    """The renderer's world -> NDC 4x4 matrix as a numpy array."""
    m = renderer.GetActiveCamera().GetCompositeProjectionTransformMatrix(
        renderer.GetTiledAspectRatio(), -1.0, 1.0)
    return np.array([[m.GetElement(i, j) for j in range(4)] for i in range(4)])


def project_to_display(xyz: np.ndarray, matrix: np.ndarray,
                       width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Project world points to VTK display coords (origin bottom-left, pixels).

    Returns (screen (N, 2), in_front (N,) bool). Points behind the camera are
    flagged not-in-front (their screen coords are unusable).
    """
    xyz = np.asarray(xyz, np.float64)
    ndc = xyz @ matrix[:3, :3].T + matrix[:3, 3]
    w = xyz @ matrix[3, :3] + matrix[3, 3]
    in_front = w > 1e-9
    w_safe = np.where(in_front, w, 1.0)
    sx = (ndc[:, 0] / w_safe + 1.0) * 0.5 * width
    sy = (ndc[:, 1] / w_safe + 1.0) * 0.5 * height
    return np.column_stack([sx, sy]), in_front


def decimate_path(path: np.ndarray, max_vertices: int = 40) -> np.ndarray:
    """Subsample a freehand path to keep point-in-polygon fast (~100 ms)."""
    path = np.asarray(path, np.float64)
    if len(path) <= max_vertices:
        return path
    idx = np.linspace(0, len(path) - 1, max_vertices).astype(int)
    return path[idx]


def points_in_lasso(xyz: np.ndarray, matrix: np.ndarray, width: int, height: int,
                    path: np.ndarray) -> np.ndarray:
    """Indices of ``xyz`` whose screen projection falls inside the lasso ``path``.

    ``path`` is (M, 2) in the same display coordinate space (bottom-left origin,
    physical pixels). Select-through semantics: occluded points inside the
    region are selected too.
    """
    if len(path) < 3:
        return np.empty(0, dtype=np.int64)
    from matplotlib.path import Path

    screen, in_front = project_to_display(xyz, matrix, width, height)
    inside = Path(decimate_path(path)).contains_points(screen)
    return np.nonzero(inside & in_front)[0]
