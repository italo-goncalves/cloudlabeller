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

"""Centralised logging configuration.

One place to configure where logs go and how they look. Used by both the GUI
(`app.py`) and the reconstruction subprocess (`photogrammetry/run_cli.py`).

Design: the subprocess logs to **stderr** so its Python log lines merge with
COLMAP's native (glog) output; the parent :class:`ProcessJob` captures that
combined stream, tees it to a per-run log file, and forwards it to the in-app Log
panel. The GUI process logs to a rotating user log file.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

DEFAULT_LOG_DIR = Path.home() / ".cloudlabeller" / "logs"
_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    logfile: str | Path | None = None,
    level: int = logging.INFO,
    console: bool = True,
) -> Path | None:
    """Configure the root logger. Idempotent-ish: clears prior handlers first.

    Returns the resolved log file path (or None if ``logfile`` was None).
    """
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter(_FORMAT, _DATEFMT)
    resolved: Path | None = None
    if logfile is not None:
        resolved = Path(logfile)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(resolved, maxBytes=2_000_000, backupCount=3,
                                 encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    if console:
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(fmt)
        root.addHandler(sh)
    return resolved


def install_excepthook(logger: logging.Logger | None = None) -> None:
    """Route uncaught exceptions through logging so crashes leave a trail."""
    log = logger or logging.getLogger("cloudlabeller")

    def hook(exc_type, exc, tb):
        log.critical("Uncaught exception", exc_info=(exc_type, exc, tb))
        sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = hook
