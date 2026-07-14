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

"""Georeferencing: align the model to the images' GPS (metric ENU frame).

COLMAP's ``model_aligner`` computes a similarity transform from the solved
camera positions to their EXIF GPS coordinates (converted to a local
East-North-Up frame — metres, true north, small numbers that survive the
float32 storage; ECEF's ~6.4e6 m magnitudes would not). This module holds the
pure logic: EXIF GPS extraction, the similarity fit between the original and
aligned models, and applying that transform to cameras and geometry.

World transform convention: ``X' = s · R @ X + t``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cloudlabeller.core.dataset import Camera

GPS_TAG = 34853              # EXIF GPSInfo IFD


def _dms_to_degrees(dms) -> float:
    d, m, s = (float(v) for v in dms)
    return d + m / 60.0 + s / 3600.0


def parse_gps_info(gps: dict) -> tuple[float, float, float] | None:
    """(lat, lon, alt) from an EXIF GPSInfo dict, or None if incomplete.

    Keys per the EXIF spec: 1/2 = lat ref/value, 3/4 = lon ref/value,
    5/6 = altitude ref/value (ref 1 = below sea level).
    """
    try:
        lat = _dms_to_degrees(gps[2]) * (-1.0 if gps.get(1) == "S" else 1.0)
        lon = _dms_to_degrees(gps[4]) * (-1.0 if gps.get(3) == "W" else 1.0)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    alt = 0.0
    if 6 in gps:
        try:
            ref = gps.get(5, 0)
            if isinstance(ref, (bytes, bytearray)):   # DJI stores it as b'\x00'
                ref = ref[0] if ref else 0
            alt = float(gps[6]) * (-1.0 if int(ref or 0) == 1 else 1.0)
        except (TypeError, ValueError):
            alt = 0.0
    return lat, lon, alt


def exif_gps(image_path: str | Path) -> tuple[float, float, float] | None:
    """Read (lat, lon, alt) from an image's EXIF, or None."""
    from PIL import Image

    try:
        with Image.open(image_path) as im:
            exif = im.getexif()
            gps = exif.get_ifd(GPS_TAG) if exif else None
    except Exception:
        return None
    return parse_gps_info(dict(gps)) if gps else None


def umeyama_similarity(src: np.ndarray, dst: np.ndarray
                       ) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity (s, R, t) with ``dst ≈ s·R@src + t``
    (Umeyama 1991). ``src``/``dst`` are (N, 3), N >= 3 non-degenerate."""
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    u, d, vt = np.linalg.svd(cov)
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[2, 2] = -1.0
    rotation = u @ sign @ vt
    var_s = (xs ** 2).sum() / len(src)
    scale = float(np.trace(np.diag(d) @ sign) / var_s)
    translation = mu_d - scale * rotation @ mu_s
    return scale, rotation, translation


def transform_camera(camera: Camera, s: float, R: np.ndarray, t: np.ndarray) -> Camera:
    """Re-express a world-to-camera pose after the world moves by
    ``X' = s·R@X + t``. Intrinsics are unchanged; depths scale by ``s``."""
    R_new = camera.R @ R.T
    t_new = s * np.asarray(camera.t, np.float64) - R_new @ np.asarray(t, np.float64)
    return Camera(K=camera.K, R=R_new, t=t_new,
                  width=camera.width, height=camera.height,
                  distortion=camera.distortion, model=camera.model)


def transform_points(xyz: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply ``X' = s·R@X + t`` to an (N, 3) array (returns float32)."""
    return (np.asarray(xyz, np.float64) @ (s * R).T + t).astype(np.float32)


def rotate_normals(normals: np.ndarray, R: np.ndarray) -> np.ndarray:
    return (np.asarray(normals, np.float64) @ R.T).astype(np.float32)
