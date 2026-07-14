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

"""Extract a :class:`PointCloud` from a COLMAP reconstruction's sparse points."""

from __future__ import annotations

import numpy as np

from cloudlabeller.core.dataset import PointCloud


def reconstruction_to_cloud(reconstruction) -> PointCloud:
    """Pull the sparse 3D points (xyz + RGB) out of a pycolmap reconstruction."""
    points = reconstruction.points3D            # dict {id: Point3D}
    n = len(points)
    xyz = np.empty((n, 3), dtype=np.float32)
    rgb = np.empty((n, 3), dtype=np.uint8)
    for i, p in enumerate(points.values()):
        xyz[i] = p.xyz
        rgb[i] = p.color
    return PointCloud(xyz=xyz, rgb=rgb)


def best_model_dir(sparse_root):
    """Directory of the largest COLMAP model under ``sparse_root`` (matching
    the one ``run_sfm`` extracted the cloud from when mapping split)."""
    from pathlib import Path

    import pycolmap

    root = Path(sparse_root)
    candidates = [d for d in sorted(root.iterdir())
                  if (d / "points3D.bin").exists()] if root.exists() else []
    if not candidates:
        raise FileNotFoundError(f"no COLMAP model found under {root}")
    return max(candidates,
               key=lambda d: pycolmap.Reconstruction(str(d)).num_points3D())


def load_best_model(sparse_root):
    """Load the largest COLMAP model under ``sparse_root``."""
    import pycolmap

    return pycolmap.Reconstruction(str(best_model_dir(sparse_root)))


def sparse_point_stats(reconstruction, xyz: np.ndarray
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Per-point SfM confidence for an existing cloud: (track lengths,
    mean reprojection errors), matched to ``xyz`` rows by exact position.

    Matching by coordinates (KD-tree) instead of iteration order keeps the
    stats aligned with a cloud.npz written by an earlier session — dict order
    across separate loads of a model is not guaranteed.
    """
    from scipy.spatial import cKDTree

    views = np.zeros(len(xyz), np.int32)                  # 0 = no match: filtered
    errors = np.full(len(xyz), np.inf, np.float64)
    tree = cKDTree(np.asarray(xyz, np.float64))
    points = list(reconstruction.points3D.values())
    coords = np.array([p.xyz for p in points], np.float64)
    dist, idx = tree.query(coords, k=1)
    ok = dist < 1e-4                                      # float32 storage jitter
    for j in np.nonzero(ok)[0]:
        i = idx[j]
        views[i] = points[j].track.length()
        errors[i] = points[j].error
    return views, errors
