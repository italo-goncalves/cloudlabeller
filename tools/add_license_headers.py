"""Prepend the GPLv3 copyright header to every project .py file.

Idempotent: files already carrying the notice are left alone, so it can be
re-run after adding new modules. A shebang line, when present, stays first.

    python tools/add_license_headers.py [--check]

``--check`` only reports files missing the header (exit 1 if any), for use
as a pre-release check.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HEADER = """\
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

"""

MARKER = "GNU General Public License"
SOURCE_DIRS = ("cloudlabeller", "tests", "tools", "scripts")


def process(path: Path, check: bool) -> bool:
    """Returns True when the file already had (or now has) the header."""
    text = path.read_text(encoding="utf-8")
    if MARKER in text[:1500]:
        return True
    if check:
        return False
    if text.startswith("#!"):                     # keep a shebang first
        shebang, _, rest = text.partition("\n")
        text = f"{shebang}\n{HEADER}{rest}"
    else:
        text = HEADER + text
    path.write_text(text, encoding="utf-8", newline="\n")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="report files missing the header instead of editing")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    missing, done = [], 0
    for d in SOURCE_DIRS:
        for py in sorted((root / d).rglob("*.py")):
            if "__pycache__" in py.parts:
                continue
            if process(py, args.check):
                done += 1
            else:
                missing.append(py.relative_to(root))
    if args.check:
        for p in missing:
            print(f"missing header: {p}")
        print(f"{done} file(s) with header, {len(missing)} missing")
        return 1 if missing else 0
    print(f"{done} file(s) now carry the GPL header")
    return 0


if __name__ == "__main__":
    sys.exit(main())
