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

"""Subprocess entry point: reproject a georeferenced project to a chosen CRS.

All products (sparse cloud + cameras, dense cloud, mesh, COLMAP model) move
from the current frame — local ENU, or a previous CRS — into the target
projected CRS, minus a km-rounded offset stored in the project settings so
the float32 clouds keep millimetre precision (displayed/exported coordinates
add the offset back).

The exact map projection is approximated by its best-fit similarity over the
site (fitted on the cloud's bounding box + centroid). Projections are
locally conformal, so the residual is centimetres for km-scale sites — and
using one similarity keeps cameras and clouds mutually consistent, which the
2D<->3D label transfer depends on. The residual is reported as
``fit_rms_m``; repeated reprojection accumulates rounding, so prefer doing
it once.

Usage::

    python -m cloudlabeller.photogrammetry.reproject_cli PROJECT_ROOT
        --epsg N [--orthometric]

Emits ``PROGRESS <frac> <msg>``; prints ``RESULT <json>`` on success with
the new frame (epsg, name, orthometric, offset).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from cloudlabeller.logging_setup import setup_logging

log = logging.getLogger("cloudlabeller.reproject")


def _parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="reproject_cli")
    p.add_argument("project_root")
    p.add_argument("--epsg", type=int, required=True,
                   help="target projected CRS (EPSG code)")
    p.add_argument("--orthometric", action="store_true",
                   help="EGM96 sea-level heights (downloads a ~3 MB grid once)")
    return p.parse_args(argv[1:])


def _sample_points(xyz: np.ndarray) -> np.ndarray:
    """Bounding-box corners + centroid — enough to pin down the local
    similarity approximation of the projection over the site."""
    lo = np.asarray(xyz, np.float64).min(axis=0)
    hi = np.asarray(xyz, np.float64).max(axis=0)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                        for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    return np.vstack([corners, xyz.mean(axis=0)])


def main(argv: list[str]) -> int:
    args = _parse(argv)
    setup_logging(logfile=None, console=True)
    from cloudlabeller.workers.resources import limit_subprocess_resources
    limit_subprocess_resources()

    def progress(frac: float, msg: str = "") -> None:
        print(f"PROGRESS {frac:.3f} {msg}", flush=True)

    import pyproj

    from cloudlabeller.photogrammetry.crs import (
        choose_offset,
        download_geoid_grids,
        geoid_ready,
        project_frame_transform,
        resolve_enu_origin,
    )
    from cloudlabeller.photogrammetry.georef import umeyama_similarity
    from cloudlabeller.photogrammetry.project_transform import (
        apply_similarity_to_products,
    )

    root = Path(args.project_root)
    manifest_path = root / "project.json"
    try:
        settings = json.loads(manifest_path.read_text(encoding="utf-8"))["settings"]
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        log.error("could not read project settings: %s", exc)
        return 2
    geo = settings.get("georeferenced")
    if not geo:
        log.error("the project is not georeferenced — align it to GPS first "
                  "(Georeferencing -> Align to EXIF GPS)")
        return 2

    if args.orthometric and not geoid_ready(args.epsg):
        progress(0.02, "Downloading the EGM96 geoid grid (~3 MB, one-time)…")
        download_geoid_grids(args.epsg)

    # 1. Exact mapping stored coords -> absolute target CRS.
    progress(0.05, "Building the coordinate transformation…")
    origin = resolve_enu_origin(geo, root / "reconstruction", root / "images")
    if not geo.get("crs") and not origin.exact:
        log.warning("the exact ENU origin could not be recovered — the "
                    "reprojected coordinates may be offset (see georef log)")
    exact, target_crs = project_frame_transform(geo, origin.lla, args.epsg,
                                                args.orthometric)

    # 2. Offset + best-fit similarity over the site.
    progress(0.1, "Fitting the projection over the site…")
    cloud_xyz = np.load(root / "reconstruction" / "cloud.npz")["xyz"]
    samples = _sample_points(cloud_xyz)
    projected = exact(samples)
    offset = choose_offset(projected)
    s, R, t = umeyama_similarity(samples, projected - offset)
    fit = samples @ (s * R).T + t - (projected - offset)
    rms = float(np.sqrt((fit ** 2).sum(axis=1).mean()))
    log.info("reprojection: scale=%.8f, offset=(%.0f, %.0f, %.0f), "
             "similarity rms=%.3f m over the site",
             s, offset[0], offset[1], offset[2], rms)
    if rms > 0.25:
        log.warning("the projection deviates from a similarity by %.2f m "
                    "across this site — a site this large may be better "
                    "served by a different CRS", rms)

    # 3. Apply to every product (no backup: reprojection is repeatable).
    apply_similarity_to_products(root, s, R, t, progress)

    name = pyproj.CRS.from_epsg(args.epsg).name
    print("RESULT " + json.dumps({
        "epsg": args.epsg,
        "name": name,
        "orthometric": bool(args.orthometric),
        "offset": [float(v) for v in offset],
        "fit_rms_m": rms,
        "scale": s,
    }), flush=True)
    progress(1.0, "Reprojection complete")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
