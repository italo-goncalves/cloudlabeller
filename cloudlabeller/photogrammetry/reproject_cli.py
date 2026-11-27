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

"""Subprocess entry point: fit the move of a Project frame into a projected CRS.

The Project's products stay where they are (ADR-0005): this fits the
similarity that takes coordinates in the current frame (local ENU, or a
previous CRS minus its offset) into the target projected CRS minus a
km-rounded offset (so float32 clouds keep millimetre precision), and the
Engine composes it into every Reconstruction's Alignment.

The exact map projection is approximated by its best-fit similarity over the
site, fitted on the sample points the Engine passes (the aligned clouds'
bounding boxes and centroids). Projections are locally conformal, so the
residual is centimetres for km-scale sites, and one similarity keeps
cameras and clouds consistent, which label transfer depends on. The residual
is reported as ``fit_rms_m``.

Usage::

    python -m cloudlabeller.photogrammetry.reproject_cli INPUT_JSON
        --epsg N [--orthometric]

``INPUT_JSON`` holds ``{"frame": <Project frame>, "origin_lla": [lat, lon,
alt], "samples": [[x, y, z], ...]}``. Emits ``PROGRESS <frac> <msg>``;
prints ``RESULT <json>`` with ``s``, ``R``, ``t`` and the new frame (epsg,
name, orthometric, offset).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from cloudlabeller.logging_setup import setup_logging
from cloudlabeller.workers.result import print_result

log = logging.getLogger("cloudlabeller.reproject")


def _parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="reproject_cli")
    p.add_argument("input_json")
    p.add_argument("--epsg", type=int, required=True,
                   help="target projected CRS (EPSG code)")
    p.add_argument("--orthometric", action="store_true",
                   help="EGM96 sea-level heights (downloads a ~3 MB grid once)")
    return p.parse_args(argv[1:])


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
    )
    from cloudlabeller.photogrammetry.georef import umeyama_similarity

    try:
        given = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
        geo, origin_lla = given["frame"], given["origin_lla"]
        samples = np.asarray(given["samples"], np.float64).reshape(-1, 3)
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        log.error("could not read the input: %s", exc)
        return 2
    if not geo:
        log.error("the Project has no frame: align it to GPS first "
                  "(Georeferencing -> Align to EXIF GPS)")
        return 2
    if len(samples) < 3:
        log.error("no aligned geometry to fit the projection over")
        return 2

    if args.orthometric and not geoid_ready(args.epsg):
        progress(0.02, "Downloading the EGM96 geoid grid (~3 MB, one-time)…")
        download_geoid_grids(args.epsg)

    # 1. Exact mapping stored coords -> absolute target CRS.
    progress(0.05, "Building the coordinate transformation…")
    exact, target_crs = project_frame_transform(geo, origin_lla, args.epsg,
                                                args.orthometric)

    # 2. Offset + best-fit similarity over the site.
    progress(0.1, "Fitting the projection over the site…")
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

    name = pyproj.CRS.from_epsg(args.epsg).name
    print_result({
        "s": float(s), "R": np.asarray(R).tolist(), "t": np.asarray(t).tolist(),
        "epsg": args.epsg,
        "name": name,
        "orthometric": bool(args.orthometric),
        "offset": [float(v) for v in offset],
        "fit_rms_m": rms,
        "scale": s,
    })
    progress(1.0, "Reprojection complete")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
