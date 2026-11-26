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

"""Regenerate the desktop app's Environment lock files.

    uv run --no-project tools/lock_environment.py          # rewrite both files
    uv run --no-project tools/lock_environment.py --check  # fail if stale

Two files, both committed:

* ``uv.lock`` — uv's universal lock of ``pyproject.toml``, resolved only for
  the ``[tool.uv] environments`` (Python 3.12 on win_amd64 and manylinux
  x86_64);
* ``apps/desktop/environment/requirements.lock`` — the same pins exported in
  requirements format, with hashes and platform markers, base dependencies
  only (no ``qt``/``dev`` extras, not the project itself). The Tauri core
  ``uv pip sync``-s it into the Environment, then installs the cloudlabeller
  wheel with ``--no-deps``.

Run it after changing the dependencies in ``pyproject.toml``, and to take
upstream bug-fix releases within the pins (``--upgrade``).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOCK = REPO / "uv.lock"
EXPORT = REPO / "apps" / "desktop" / "environment" / "requirements.lock"


def _uv(*args: str) -> None:
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("uv is not on PATH (https://docs.astral.sh/uv/)")
    print("+ uv", " ".join(args), flush=True)
    subprocess.run([uv, *args], cwd=REPO, check=True)


def _export(out: Path) -> None:
    _uv("export", "--quiet", "--frozen", "--format", "requirements-txt", "--no-dev",
        "--no-emit-project", "--no-header", "--output-file", str(out))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--check", action="store_true",
                   help="change nothing; exit 1 if either file is out of date")
    p.add_argument("--upgrade", action="store_true",
                   help="re-resolve every package to its newest allowed version")
    args = p.parse_args(argv)

    if args.check:
        _uv("lock", "--check")
        with tempfile.TemporaryDirectory() as tmp:
            fresh = Path(tmp) / "requirements.lock"
            _export(fresh)
            if not EXPORT.exists() or fresh.read_bytes() != EXPORT.read_bytes():
                print(f"{EXPORT.relative_to(REPO)} is out of date: "
                      "run tools/lock_environment.py", file=sys.stderr)
                return 1
        print("lock files are up to date")
        return 0

    _uv("lock", *(["--upgrade"] if args.upgrade else []))
    EXPORT.parent.mkdir(parents=True, exist_ok=True)
    _export(EXPORT)
    print(f"wrote {LOCK.relative_to(REPO)} and {EXPORT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
