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

"""Locating a point seen in several photos (a GCP's photo Marks).

Each Mark's pixel is undistorted through the same COLMAP camera model
:func:`cloudlabeller.transfer.projection.project_points` applies, turned
into a ray, and the rays are met in the least-squares sense; a few
Gauss–Newton steps then minimise the reprojection error in pixels.
"""

from __future__ import annotations

import numpy as np

from cloudlabeller.core.dataset import Camera
from cloudlabeller.transfer.projection import _apply_distortion, project_points


def undistort(camera: Camera, uv: np.ndarray, iterations: int = 30) -> np.ndarray:
    """Normalised camera coordinates (x, y) of pixels ``uv`` (N, 2): the
    inverse of the camera's distortion, by fixed-point iteration."""
    uv = np.asarray(uv, np.float64).reshape(-1, 2)
    K = camera.K
    xd = (uv[:, 0] - K[0, 2]) / K[0, 0]
    yd = (uv[:, 1] - K[1, 2]) / K[1, 1]
    x, y = xd.copy(), yd.copy()
    for _ in range(iterations):
        dx, dy = _apply_distortion(x, y, camera.distortion, camera.model)
        x, y = x + (xd - dx), y + (yd - dy)
    return np.column_stack([x, y])


def _rays(cameras: list[Camera], uvs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centres, dirs = [], []
    for cam, uv in zip(cameras, uvs):
        x, y = undistort(cam, uv)[0]
        d = np.asarray(cam.R, np.float64).T @ np.array([x, y, 1.0])
        centres.append(cam.position)
        dirs.append(d / np.linalg.norm(d))
    return np.array(centres, np.float64), np.array(dirs, np.float64)


def reprojection_errors(cameras: list[Camera], uvs: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Pixel distance between each Mark and where ``X`` projects."""
    out = []
    for cam, uv in zip(cameras, uvs):
        p, _ = project_points(cam, np.asarray(X, np.float64)[None])
        out.append(float(np.hypot(*(p[0, :2] - uv))))
    return np.array(out)


def triangulate(cameras: list[Camera], uvs) -> tuple[np.ndarray, float]:
    """The 3D point seen at pixel ``uvs[i]`` by ``cameras[i]`` (two or more),
    and the RMS reprojection error in pixels."""
    uvs = np.asarray(uvs, np.float64).reshape(-1, 2)
    if len(cameras) < 2 or len(cameras) != len(uvs):
        raise ValueError("triangulating needs two or more photos")
    C, D = _rays(cameras, uvs)
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for c, d in zip(C, D):
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ c
    X = np.linalg.lstsq(A, b, rcond=None)[0]

    def residual(p):
        r = []
        for cam, uv in zip(cameras, uvs):
            q, _ = project_points(cam, p[None])
            r.extend(q[0, :2] - uv)
        return np.array(r)

    scale = max(float(np.linalg.norm(X - C.mean(axis=0))), 1e-6) * 1e-6
    for _ in range(10):                                  # Gauss–Newton, numeric Jacobian
        r = residual(X)
        J = np.column_stack([(residual(X + scale * e) - r) / scale for e in np.eye(3)])
        step = np.linalg.lstsq(J, -r, rcond=None)[0]
        X = X + step
        if np.linalg.norm(step) < scale:
            break
    err = reprojection_errors(cameras, uvs, X)
    return X, float(np.sqrt((err ** 2).mean()))
