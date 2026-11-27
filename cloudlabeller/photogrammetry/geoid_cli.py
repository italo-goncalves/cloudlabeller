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

"""Subprocess entry point: fetch the EGM96 geoid grid for orthometric heights.

A one-time download (~3 MB, from cdn.proj.org) into the user's PROJ
directory, so exports and reprojection can write sea-level heights.

Usage::

    python -m cloudlabeller.photogrammetry.geoid_cli --epsg N

Emits ``PROGRESS <frac> <msg>``; prints ``RESULT <json>`` (``{epsg}``) once
the grid is usable.
"""

from __future__ import annotations

import argparse
import logging
import sys

from cloudlabeller.logging_setup import setup_logging
from cloudlabeller.workers.result import print_result

log = logging.getLogger("cloudlabeller.geoid")


def _parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="geoid_cli")
    p.add_argument("--epsg", type=int, required=True,
                   help="the projected CRS the heights go with")
    return p.parse_args(argv[1:])


def main(argv: list[str]) -> int:
    args = _parse(argv)
    setup_logging(logfile=None, console=True)
    from cloudlabeller.photogrammetry.crs import download_geoid_grids, geoid_ready

    print("PROGRESS 0.000 Downloading the EGM96 geoid grid (~3 MB, one-time)…",
          flush=True)
    if not geoid_ready(args.epsg):
        try:
            download_geoid_grids(args.epsg)
        except Exception as exc:
            log.error("%s", exc)
            return 2
    print_result({"epsg": args.epsg})
    print("PROGRESS 1.000 Geoid grid ready", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
