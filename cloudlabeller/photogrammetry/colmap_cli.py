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

"""Subprocess entry point: download the official COLMAP Windows bundle.

Wraps :func:`cloudlabeller.photogrammetry.colmap_fetch.install_colmap`: the
release goes into ``~/.cloudlabeller/colmap``, where the app looks for
``colmap.exe``. Cancel by creating the stop file: the download stops at the
next chunk and leaves no partial install.

Usage::

    python -m cloudlabeller.photogrammetry.colmap_cli --variant cuda|nocuda
        [--stop-file PATH]

Emits ``PROGRESS <frac> <msg>``; prints ``RESULT <json>`` (``{path,
variant}``) on success.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from cloudlabeller.logging_setup import setup_logging

log = logging.getLogger("cloudlabeller.colmap_fetch")


def _parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="colmap_cli")
    p.add_argument("--variant", choices=("cuda", "nocuda"), default="cuda")
    p.add_argument("--stop-file", default=None,
                   help="stop (and clean up) once this file exists")
    return p.parse_args(argv[1:])


def main(argv: list[str]) -> int:
    args = _parse(argv)
    setup_logging(logfile=None, console=True)
    from cloudlabeller.photogrammetry.colmap_fetch import install_colmap

    stop = Path(args.stop_file) if args.stop_file else None

    def progress(frac: float, msg: str = "") -> None:
        print(f"PROGRESS {frac:.3f} {msg}", flush=True)

    try:
        exe = install_colmap(args.variant, progress=progress,
                             should_stop=lambda: stop is not None and stop.exists())
    except Exception as exc:
        log.error("%s", exc)
        return 3 if stop is not None and stop.exists() else 2
    print("RESULT " + json.dumps({"path": exe, "variant": args.variant}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
