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

"""Build PyVista geometry to visualise a camera as a frustum + centre marker.

Given a :class:`Camera` (K, R, t with x_cam = R·X_world + t), we back-project the
four image corners to a fixed depth and draw the pyramid from the camera centre.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv

from cloudlabeller.core.dataset import Camera


def _corner_rays(camera: Camera) -> np.ndarray:
    """Unit-ish camera-frame directions through the four image corners (z=1)."""
    w, h = camera.width, camera.height
    k_inv = np.linalg.inv(camera.K)
    corners_px = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=float)
    rays = np.array([k_inv @ np.array([u, v, 1.0]) for u, v in corners_px])
    return rays  # (4, 3), each with z == 1


def frustum_points(camera: Camera, scale: float) -> np.ndarray:
    """The 5 frustum vertices (centre + 4 image-plane corners) in world space.

    Only these move when the frustum is resized; the line connectivity is fixed,
    so resizing = reassign these points to the existing mesh (no rebuild).
    """
    rays = _corner_rays(camera)                 # (4, 3) in camera frame
    pts_cam = scale * rays                       # corners at depth ~scale
    R, t = camera.R, camera.t
    center = -R.T @ t                            # camera centre in world
    corners_world = (R.T @ (pts_cam - t).T).T    # (4, 3)
    return np.vstack([center[None, :], corners_world])  # 0=centre, 1..4=corners


def camera_frustum(camera: Camera, scale: float = 0.3) -> pv.PolyData:
    """Return a wireframe frustum (centre→corners + image-plane rectangle)."""
    points = frustum_points(camera, scale)
    lines: list[int] = []
    for i in range(1, 5):                        # centre → each corner
        lines += [2, 0, i]
    for a, b in [(1, 2), (2, 3), (3, 4), (4, 1)]:  # image-plane rectangle
        lines += [2, a, b]
    return pv.PolyData(points, lines=np.array(lines))


def camera_center(camera: Camera) -> np.ndarray:
    return -camera.R.T @ camera.t


def merged_frustums(cameras: list[Camera], scale: float) -> pv.PolyData:
    """All cameras' wireframe frustums as ONE PolyData (5 points each).

    One merged actor renders hundreds of cameras as fast as one — per-camera
    actors made opening large projects crawl. Camera ``i`` owns points
    ``5*i .. 5*i+4`` (centre + 4 corners), so resizing = reassigning
    ``frustum_points`` per camera into the merged ``points`` array.
    """
    all_points = np.vstack([frustum_points(cam, scale) for cam in cameras])
    lines: list[int] = []
    for i in range(len(cameras)):
        o = 5 * i
        for c in range(1, 5):                          # centre → each corner
            lines += [2, o, o + c]
        for a, b in [(1, 2), (2, 3), (3, 4), (4, 1)]:  # image-plane rectangle
            lines += [2, o + a, o + b]
    return pv.PolyData(all_points, lines=np.array(lines))
