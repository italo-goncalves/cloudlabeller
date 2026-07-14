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

"""Cloud <-> mesh label mapping via nearest-neighbour transfer.

The KD-tree build + query over a survey-scale dense cloud costs ~1 s and its
result — each mesh vertex's nearest dense point — is pure geometry. It is cached
(self-validating fingerprints) so a label sync is a single indexing op (~1 ms).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cloudlabeller.core.dataset import Mesh, PointCloud
from cloudlabeller.transfer.visibility import array_fingerprint, cloud_fingerprint


def mesh_nn_indices(cloud: PointCloud, mesh: Mesh,
                    cache_path: str | Path | None = None) -> np.ndarray:
    """Index of the nearest cloud point for each mesh vertex, cached on disk."""
    fp = cloud_fingerprint(cloud) + array_fingerprint(
        mesh.vertices, mesh.faces, text=str(len(mesh.vertices)))

    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            try:
                data = np.load(cache_path)
                if str(data["fp"]) == fp:
                    return data["idx"].astype(np.int64)
            except Exception:
                pass                              # stale/corrupt -> recompute

    from scipy.spatial import cKDTree

    _, idx = cKDTree(cloud.xyz).query(mesh.vertices, k=1)
    if cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache_path, idx=idx.astype(np.uint32), fp=np.str_(fp))
        except Exception:
            pass                                  # cache is best-effort
    return np.asarray(idx, np.int64)


def cloud_to_mesh(cloud: PointCloud, cloud_labels: np.ndarray, mesh: Mesh,
                  cache_path: str | Path | None = None) -> np.ndarray:
    """Assign each mesh vertex the label of its nearest cloud point.

    Returns (V,) int32 vertex labels. With ``cache_path`` the nearest-neighbour
    indices are cached, making repeat syncs (labels changed, geometry didn't)
    effectively instant.
    """
    idx = mesh_nn_indices(cloud, mesh, cache_path)
    return cloud_labels[idx].astype(np.int32)


def mesh_to_cloud(mesh: Mesh, vertex_labels: np.ndarray, cloud: PointCloud) -> np.ndarray:
    """Assign each cloud point the label of its nearest mesh vertex."""
    from scipy.spatial import cKDTree

    tree = cKDTree(mesh.vertices)
    _, idx = tree.query(cloud.xyz, k=1)
    return vertex_labels[idx].astype(np.int32)
