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

A similarity transform is fitted from the solved camera positions to their
EXIF GPS coordinates (converted to a local East-North-Up frame — metres,
true north, small numbers that survive the float32 storage; ECEF's ~6.4e6 m
magnitudes would not). This module holds the pure numpy logic: EXIF GPS
extraction, the robust (RANSAC + Umeyama) similarity fit, and applying that
transform to cameras and geometry. The fit runs natively — no COLMAP
executable involved (the ENU origin convention — first GPS coordinate —
matches COLMAP's ``model_aligner`` for compatibility with older projects).

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


def count_gps_images(paths, need: int = 3, limit: int = 25) -> int:
    """How many of the first ``limit`` images have GPS EXIF, stopping early
    once ``need`` are found. Header-only reads, capped so cloud-synced image
    stores aren't mass-hydrated just to populate a dialog."""
    found = 0
    for path in list(paths)[:limit]:
        if exif_gps(path) is not None:
            found += 1
            if found >= need:
                break
    return found


def ransac_similarity(src: np.ndarray, dst: np.ndarray, max_error: float = 5.0,
                      min_inliers: int = 3, iters: int = 500, seed: int = 0
                      ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
    """Robust similarity fit ``dst ≈ s·R@src + t`` for GPS-grade data.

    Consumer GPS has occasional wild outliers (multipath, cold fixes) that
    wreck a plain least-squares fit, so: RANSAC over minimal 3-point samples,
    inliers within ``max_error`` metres, final Umeyama refit on the inliers.

    Returns ``(s, R, t, inlier_mask, rms)`` — rms over the inliers, metres.
    Raises ``RuntimeError`` when no sample yields ``min_inliers`` inliers.
    """
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    n = len(src)
    if n < 3:
        raise RuntimeError(f"similarity fit needs >= 3 points, got {n}")

    def residuals(s, R, t):
        return np.linalg.norm(src @ (s * R).T + t - dst, axis=1)

    rng = np.random.default_rng(seed)
    best_mask = None
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        try:
            s, R, t = umeyama_similarity(src[idx], dst[idx])
        except np.linalg.LinAlgError:
            continue
        if not (np.isfinite(s) and s > 0):        # degenerate sample
            continue
        mask = residuals(s, R, t) < max_error
        if best_mask is None or mask.sum() > best_mask.sum():
            best_mask = mask
    if best_mask is None or best_mask.sum() < min_inliers:
        raise RuntimeError(
            f"could not align: fewer than {min_inliers} of {n} GPS positions "
            f"agree within {max_error:.1f} m — GPS may be unreliable here")

    # Refit on the consensus set, then let the refined fit re-pick inliers.
    for _ in range(2):
        s, R, t = umeyama_similarity(src[best_mask], dst[best_mask])
        mask = residuals(s, R, t) < max_error
        if mask.sum() < min_inliers or (mask == best_mask).all():
            break
        best_mask = mask
    rms = float(np.sqrt((residuals(s, R, t)[best_mask] ** 2).mean()))
    return s, R, t, best_mask, rms


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
