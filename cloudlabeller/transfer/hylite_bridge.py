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

"""Camera projection for image <-> cloud label transfer.

The projection applies the **full COLMAP camera model** (pinhole + lens
distortion, validated against pycolmap's ``img_from_cam``) so that pixels match
the original photos the user labels — ignoring distortion shifted transfers by
up to ~100 px at the image corners. Occlusion combines per-point normal
back-face culling (hylite-style) with a coarse z-buffer.

Interop note: for distortion-free cameras this projection is pixel-identical to
``hylite.project.proj_persp`` under the conversion below (calibrated to 0 px —
see ``camera_to_hylite`` and its regression test); hylite's model cannot express
lens distortion, which is why the projection itself is computed here.

COLMAP -> hylite camera:
    C   = -R^T t                                  (camera centre)
    R_h = R^T @ diag(1, -1, -1)                   (proper rotation)
    ori = -euler_XYZ(R_h) in degrees              (hylite pitch/roll/yaw)
    fov = 2*atan(H / (2*fy)) in degrees           (vertical fov)
hylite's proj_persp then returns (px, py) == COLMAP's (u, v), top-left origin.
"""

from __future__ import annotations

import numpy as np

from cloudlabeller.core.dataset import Camera, PointCloud


def camera_to_hylite(camera: Camera):
    """Convert a COLMAP-convention camera into a hylite ``Camera`` (see the
    module docstring for the derivation, verified against proj_persp)."""
    from hylite.project import Camera as HCamera
    from scipy.spatial.transform import Rotation

    center = -camera.R.T @ camera.t
    r_h = camera.R.T @ np.diag([1.0, -1.0, -1.0])
    ori = -Rotation.from_matrix(r_h).as_euler("XYZ", degrees=True)
    fov = np.degrees(2.0 * np.arctan(camera.height / (2.0 * camera.K[1, 1])))
    return HCamera(np.asarray(center, float), np.asarray(ori, float),
                   "persp", float(fov), (int(camera.width), int(camera.height)))


def _apply_distortion(x: np.ndarray, y: np.ndarray, dist: np.ndarray | None,
                      model: str) -> tuple[np.ndarray, np.ndarray]:
    """Distort normalized camera coordinates per the COLMAP camera model.

    Coefficient layout comes from ``photogrammetry.cameras.extract_distortion``;
    validated against pycolmap's own ``img_from_cam``.
    """
    if dist is None or len(dist) == 0:
        return x, y
    r2 = x * x + y * y
    if model in ("SIMPLE_RADIAL", "RADIAL"):
        radial = 1.0 + dist[0] * r2
        if len(dist) > 1:
            radial = radial + dist[1] * r2 * r2
        return x * radial, y * radial
    # OPENCV / FULL_OPENCV: radial (k1, k2[, k3..k6]) + tangential (p1, p2)
    k1, k2, p1, p2 = dist[0], dist[1], dist[2], dist[3]
    radial = 1.0 + k1 * r2 + k2 * r2 * r2
    if len(dist) >= 8:                              # FULL_OPENCV rational model
        k3, k4, k5, k6 = dist[4], dist[5], dist[6], dist[7]
        radial = (1.0 + k1 * r2 + k2 * r2**2 + k3 * r2**3) / \
                 (1.0 + k4 * r2 + k5 * r2**2 + k6 * r2**3)
    x_d = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    y_d = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return x_d, y_d


def project_points(camera: Camera, xyz: np.ndarray, normals: np.ndarray | None = None):
    """Return (px, py, depth) (N, 3) and a bool mask of in-front & in-bounds points.

    Applies the **full COLMAP camera model** — pinhole + lens distortion — so
    projections match the original (distorted) photos that the user labels.
    (The previous hylite ``proj_persp`` path is pixel-identical for pinhole
    cameras but cannot express distortion, which shifted transfers by up to
    ~100 px at the image corners.)

    If ``normals`` are given, back-face culling marks points whose surface faces
    away from the camera as not visible.
    """
    xyz = np.asarray(xyz, float)
    cam_pts = xyz @ camera.R.T + camera.t          # world -> camera frame
    z = cam_pts[:, 2]
    in_front = z > 1e-9
    z_safe = np.where(in_front, z, 1.0)

    x, y = cam_pts[:, 0] / z_safe, cam_pts[:, 1] / z_safe

    # The distortion polynomial is only valid within (roughly) the calibrated
    # field of view. Far outside it the radial term extrapolates wildly — with
    # barrel distortion (k < 0) it goes NEGATIVE and FOLDS far-off-axis points
    # back inside the image at shallow depth, creating a bowtie-shaped fan of
    # false occluders through the principal point (label-transfer "cone" bug).
    # Points beyond ~1.5x the corner angle cannot legitimately project
    # in-bounds, so cull them before distorting.
    corner_r2 = (max(camera.K[0, 2], camera.width - camera.K[0, 2]) / camera.K[0, 0]) ** 2 \
        + (max(camera.K[1, 2], camera.height - camera.K[1, 2]) / camera.K[1, 1]) ** 2
    in_fov = x * x + y * y <= corner_r2 * 2.25     # (1.5x radius)**2

    x, y = _apply_distortion(x, y, camera.distortion, camera.model)
    u = camera.K[0, 0] * x + camera.K[0, 2]
    v = camera.K[1, 1] * y + camera.K[1, 2]

    vis = (in_front & in_fov & (u >= 0) & (u < camera.width)
           & (v >= 0) & (v < camera.height))
    if normals is not None:
        view = xyz - camera.position               # camera centre -> point
        vis &= np.einsum("ij,ij->i", np.asarray(normals, float), view) < 0
    return np.column_stack([u, v, z]), vis


# Version of the visibility algorithm below. Participates in the
# VisibilityIndex cache fingerprint so cached entries recompute when the
# algorithm changes.
VISIBILITY_VERSION = 3


def _robust_cell_depth(cell: np.ndarray, depth: np.ndarray, n_cells: int,
                       rank: int) -> np.ndarray:
    """Per-cell occluder depth = the ``rank``-th smallest depth (1-based),
    clamped to each cell's population. ``rank`` > 1 makes the z-buffer immune
    to a few phantom points per cell (see ``visible_point_pixels``)."""
    order = np.lexsort((depth, cell))
    cs, ds = cell[order], depth[order]
    first = np.ones(len(cs), dtype=bool)
    first[1:] = cs[1:] != cs[:-1]
    starts = np.flatnonzero(first)
    counts = np.diff(np.append(starts, len(cs)))
    take = starts + np.minimum(rank - 1, counts - 1)
    zbuf = np.full(n_cells, np.inf)
    zbuf[cs[starts]] = ds[take]
    return zbuf


def visible_point_pixels(cloud: PointCloud, camera: Camera,
                         occlusion_cell: int = 4, depth_tol: float = 0.03,
                         occlusion_rank: int = 1):
    """Indices of cloud points visible in ``camera`` and their (col, row) pixels.

    Occlusion: per-point normals (when the cloud has them, e.g. COLMAP dense
    fusion output) cull back-facing points; a coarse z-buffer then culls points
    more than ``depth_tol`` (relative) behind the occluder depth of their
    ``occlusion_cell``-sized image cell.

    ``occlusion_rank`` optionally hardens the z-buffer against sparse phantom
    points: the occluder depth becomes the rank-th smallest in the cell instead
    of the minimum, so up to rank-1 stray points per cell cannot shadow real
    surface. Default 1 = classic min-depth (``project_points``'s field-of-view
    guard already removes the distortion fold-back ghosts that once required
    this).
    """
    width, height = int(camera.width), int(camera.height)
    pp, vis = project_points(camera, cloud.xyz, normals=cloud.normals)
    idx = np.nonzero(vis)[0]
    if idx.size == 0:
        return idx, np.empty((0, 2), dtype=int)

    cols = np.clip(pp[idx, 0].astype(int), 0, width - 1)
    rows = np.clip(pp[idx, 1].astype(int), 0, height - 1)
    depth = pp[idx, 2]

    gw = (width + occlusion_cell - 1) // occlusion_cell
    gh = (height + occlusion_cell - 1) // occlusion_cell
    cell = (rows // occlusion_cell).astype(np.int64) * gw + cols // occlusion_cell
    zbuf = _robust_cell_depth(cell, depth, gh * gw, occlusion_rank)
    keep = depth <= zbuf[cell] * (1.0 + depth_tol) + 1e-6
    return idx[keep], np.column_stack([cols[keep], rows[keep]])
