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

"""Dataset model: images, cameras, point cloud and mesh.

These are plain data containers. Heavy numerical payloads (point coordinates,
image arrays) are held as numpy arrays or loaded lazily from disk; conversion to
hylite objects happens in :mod:`cloudlabeller.transfer`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Camera:
    """Pinhole camera model (intrinsics + extrinsics), COLMAP-derived.

    Convention: world point ``X`` projects as ``x ~ K @ (R @ X + t)``.
    ``R`` is world->camera rotation (3x3), ``t`` translation (3,).
    Distortion follows the OpenCV model (k1,k2,p1,p2[,k3]).
    """

    K: np.ndarray                      # (3, 3) intrinsics
    R: np.ndarray                      # (3, 3) world->camera rotation
    t: np.ndarray                      # (3,)   world->camera translation
    width: int
    height: int
    distortion: np.ndarray | None = None  # OpenCV dist coeffs or None
    model: str = "PINHOLE"

    @property
    def position(self) -> np.ndarray:
        """Camera centre in world coordinates: C = -R^T t."""
        return -self.R.T @ self.t


@dataclass
class ImageRecord:
    """One source image plus its solved camera (None until SfM has run)."""

    image_id: int
    path: Path
    camera: Camera | None = None

    @property
    def name(self) -> str:
        return self.path.name


@dataclass
class PointCloud:
    """Sparse or dense point cloud. Labels live in :class:`LabelStore`, not here."""

    xyz: np.ndarray                          # (N, 3) float32
    rgb: np.ndarray | None = None            # (N, 3) uint8
    normals: np.ndarray | None = None        # (N, 3) float32

    @property
    def n_points(self) -> int:
        return len(self.xyz)


@dataclass
class Mesh:
    """Triangle mesh, primarily for visualisation."""

    vertices: np.ndarray                     # (V, 3) float32
    faces: np.ndarray                        # (F, 3) int32
    vertex_colors: np.ndarray | None = None  # (V, 3) uint8


@dataclass
class Dataset:
    """Registry of everything reconstructed/loaded for a project."""

    images: list[ImageRecord] = field(default_factory=list)
    cloud: PointCloud | None = None          # sparse cloud (SfM); the label carrier
    dense_cloud: PointCloud | None = None    # dense cloud (MVS), when available
    mesh: Mesh | None = None

    def has(self, representation: str) -> bool:
        """Whether a view representation ('sparse'|'dense'|'mesh') is available."""
        return {
            "sparse": self.cloud is not None,
            "dense": self.dense_cloud is not None,
            "mesh": self.mesh is not None,
        }.get(representation, False)

    def image_by_name(self, name: str) -> ImageRecord | None:
        return next((im for im in self.images if im.name == name), None)

    def solved_images(self) -> list[ImageRecord]:
        """Images that have a solved camera and can participate in transfer."""
        return [im for im in self.images if im.camera is not None]
