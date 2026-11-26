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

"""Subprocess entry point for georeferencing: fit an Alignment to EXIF GPS.

Aligns natively (no COLMAP executable): the photos' EXIF GPS is converted to
a local East-North-Up frame (metres, true north) and a robust RANSAC +
Umeyama similarity is fitted from the solved camera centres. Nothing is
written: the similarity is printed, and the Engine keeps it as the
Reconstruction's Alignment (ADR-0005).

The ENU origin is ``--origin`` when the Project already has a frame, else
the first GPS coordinate (COLMAP ``model_aligner``'s convention, kept so old
and new Projects resolve the same origin).

Usage::

    python -m cloudlabeller.photogrammetry.georef_cli SFM_DIR IMAGES_DIR
        [--origin LAT LON ALT] [--max-error 5.0]

Emits ``PROGRESS <frac> <msg>``; prints ``RESULT <json>`` on success with
``s``, ``R``, ``t`` (SfM frame to ENU), the origin and the fit.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from cloudlabeller.logging_setup import setup_logging

log = logging.getLogger("cloudlabeller.georef")


def _parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="georef_cli")
    p.add_argument("sfm_dir", help="the SfM product (cloud.npz, cameras.json)")
    p.add_argument("images_dir")
    p.add_argument("--origin", type=float, nargs=3, default=None,
                   metavar=("LAT", "LON", "ALT"), help="the Project frame's ENU origin")
    p.add_argument("--max-error", type=float, default=5.0,
                   help="RANSAC threshold in metres (GPS accuracy)")
    return p.parse_args(argv[1:])


def main(argv: list[str]) -> int:
    args = _parse(argv)
    setup_logging(logfile=None, console=True)
    from cloudlabeller.workers.resources import limit_subprocess_resources
    limit_subprocess_resources()

    def progress(frac: float, msg: str = "") -> None:
        print(f"PROGRESS {frac:.3f} {msg}", flush=True)

    from cloudlabeller.photogrammetry.crs import lla_to_enu_transform
    from cloudlabeller.photogrammetry.georef import (
        exif_gps,
        ransac_similarity,
    )
    from cloudlabeller.photogrammetry.pipeline import load_result

    # 1. Gather EXIF GPS for the solved images.
    progress(0.02, "Reading EXIF GPS from the image store…")
    result = load_result(args.sfm_dir, args.images_dir)
    gps: dict[str, tuple[float, float, float]] = {}
    for record in result.images:
        lla = exif_gps(record.path)
        if lla is not None:
            gps[Path(record.path).name] = lla
    if len(gps) < 3:
        log.error("georeferencing needs GPS EXIF in at least 3 solved images "
                  "(found %d)", len(gps))
        return 2
    # COLMAP's model_aligner anchors the ENU frame at the FIRST reference
    # coordinate (first line of ref_images.txt), so that — not the mean GPS —
    # is the origin that CRS-aware export must use.
    origin = tuple(args.origin) if args.origin else next(iter(gps.values()))
    progress(0.08, f"{len(gps)} images have GPS")

    # 2. Native similarity fit: solved camera centres -> GPS in local ENU
    #    (origin = first GPS coordinate, model_aligner's convention).
    progress(0.1, "Fitting the model to GPS (RANSAC + Umeyama)…")
    lla_to_enu = lla_to_enu_transform(origin)
    names, centers = [], []
    for record in result.images:
        name = Path(record.path).name
        if record.camera is not None and name in gps:
            names.append(name)
            cam = record.camera
            centers.append(-(cam.R.T @ np.asarray(cam.t, np.float64)))
    if len(centers) < 3:
        log.error("fewer than 3 solved cameras have GPS (%d)", len(centers))
        return 2
    enu = lla_to_enu(np.array([gps[n] for n in names], np.float64))
    try:
        s, R, t, inliers, rms = ransac_similarity(
            np.array(centers), enu, max_error=args.max_error)
    except RuntimeError as exc:
        log.error("%s", exc)
        return 2
    n_inliers = int(inliers.sum())
    log.info("similarity: scale=%.6f m/unit, %d/%d GPS inliers, rms=%.2f m",
             s, n_inliers, len(centers), rms)
    if n_inliers < len(centers):
        residuals = np.linalg.norm(
            np.array(centers) @ (s * R).T + t - enu, axis=1)
        for i in np.argsort(residuals)[::-1][:5]:
            if not inliers[i]:
                log.warning("GPS outlier: %s off by %.1f m",
                            names[i], residuals[i])
    if n_inliers < max(3, len(centers) // 2):
        log.warning("only %d of %d GPS positions agree — the alignment may "
                    "be unreliable; consider re-checking the imagery's GPS",
                    n_inliers, len(centers))

    from cloudlabeller.photogrammetry.crs import ORIGIN_CONVENTION_FIRST_GPS
    print("RESULT " + json.dumps({
        "s": float(s), "R": np.asarray(R).tolist(), "t": np.asarray(t).tolist(),
        "scale_m_per_unit": s,
        "origin_lla": [float(v) for v in origin],
        "origin_convention": ("given" if args.origin else ORIGIN_CONVENTION_FIRST_GPS),
        "n_gps": len(gps),
        "n_inliers": n_inliers,
        "fit_rms_m": rms,
    }), flush=True)
    progress(1.0, "Georeferencing complete")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
