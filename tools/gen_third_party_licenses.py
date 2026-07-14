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

"""Generate THIRD_PARTY_LICENSES.txt for a CloudLabeller distribution.

Run it with the TARGET interpreter, so the scan covers exactly the
site-packages being distributed::

    C:\\Programas\\CloudLabeller\\python.exe tools\\gen_third_party_licenses.py ^
        --out C:\\Programas\\CloudLabeller\\THIRD_PARTY_LICENSES.txt

For every installed package it emits name, version, declared license and the
license/notice files shipped in its dist-info. Special sections cover the
components whose obligations are not captured by a wheel's own metadata:
the embedded Python runtime, Qt/PySide6 (LGPLv3), FFmpeg inside the OpenCV
wheel, OpenSSL, and the separately-downloaded COLMAP bundle. The full
LGPL-3.0 / GPL-3.0 texts are appended from tools/licenses/.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from importlib.metadata import distributions
from pathlib import Path

RULE = "=" * 78
THIN = "-" * 78

# dist-info members that count as license material.
_LICENSE_STEMS = ("LICENSE", "LICENCE", "COPYING", "NOTICE", "AUTHORS")

PREAMBLE = """\
CloudLabeller — third-party software licenses
Generated {today} from the Python environment at:
    {prefix}

CloudLabeller bundles the open-source components listed below. Each entry
gives the package, its version, its declared license, and the license /
notice files shipped inside the package's own distribution metadata,
reproduced verbatim. Nothing in this file changes the terms of those
licenses; it collects them in one place as they require.
"""

SPECIAL = """\
{rule}
SPECIAL NOTICES
{rule}

Python runtime
    This application ships the CPython interpreter, licensed under the
    Python Software Foundation License (see LICENSE.txt in the application
    root, distributed with the embedded runtime).

Qt / PySide6 / Shiboken6 (LGPLv3)
    The user interface uses Qt through PySide6, used under the GNU Lesser
    General Public License v3 (full text in the appendix below). Qt and
    PySide6 are dynamically linked and ship as separate, replaceable library
    files (the PySide6/shiboken6 folders in Lib/site-packages): you may
    replace them with your own builds. Their source code is available at
    https://code.qt.io/ and https://pypi.org/project/PySide6/ .

FFmpeg (inside the OpenCV wheel, LGPL)
    The opencv-python wheel bundles an LGPL build of FFmpeg as a separate,
    replaceable DLL. See the OpenCV entry's LICENSE-3RD-PARTY.txt below for
    the exact terms, and https://github.com/opencv/opencv-python for the
    corresponding sources.

OpenSSL
    libcrypto/libssl DLLs ship with the embedded Python runtime and are used
    under the OpenSSL/Apache-2.0 licenses (see the Python runtime's own
    license documentation).

COLMAP (downloaded separately — not part of this distribution)
    CloudLabeller does not ship COLMAP. On request (Photogrammetry →
    Download COLMAP…), the app downloads the official, unmodified release
    bundle from https://github.com/colmap/colmap/releases . COLMAP itself is
    BSD-3-Clause; its official Windows bundle also contains LGPL/GPL-licensed
    third-party libraries (Qt, SuiteSparse components, GMP/MPFR) — their
    terms and sources are published with the COLMAP release and at
    https://github.com/colmap/colmap . If you received a copy of this
    software with a colmap/ folder already included, those same terms apply
    to it.
"""


def _license_of(meta) -> str:
    expr = meta.get("License-Expression")
    if expr:
        return expr
    lic = (meta.get("License") or "").strip()
    if lic and len(lic) < 60 and "\n" not in lic:
        return lic
    classifiers = [c.split("::")[-1].strip()
                   for c in (meta.get_all("Classifier") or [])
                   if c.startswith("License")]
    if classifiers:
        return "; ".join(classifiers)
    if lic:                                   # long free-text License field
        return lic.splitlines()[0][:77]
    return "see license files below"


def _license_files(dist) -> list[tuple[str, str]]:
    """(filename, text) for every license-ish file in the dist-info."""
    out = []
    for f in dist.files or []:
        parts = [p for p in f.parts]
        if not any(p.endswith((".dist-info", ".egg-info")) for p in parts):
            continue
        name = f.name.upper()
        if not name.startswith(_LICENSE_STEMS):
            continue
        try:
            text = dist.locate_file(f).read_text(encoding="utf-8",
                                                 errors="replace")
        except OSError:
            continue
        out.append((f.name, text))
    return sorted(out)


def build(out_path: Path) -> None:
    here = Path(__file__).resolve().parent
    chunks = [PREAMBLE.format(today=date.today().isoformat(),
                              prefix=sys.prefix),
              SPECIAL.format(rule=RULE)]

    chunks.append(f"{RULE}\nPACKAGES\n{RULE}\n")
    dists = sorted(distributions(),
                   key=lambda d: (d.metadata.get("Name") or "?").lower())
    for dist in dists:
        name = dist.metadata.get("Name") or "?"
        home = (dist.metadata.get("Home-page")
                or dist.metadata.get("Project-URL") or "")
        chunks.append(f"{THIN}\n{name} {dist.version}\n"
                      f"License: {_license_of(dist.metadata)}\n"
                      + (f"Home: {home}\n" if home else ""))
        files = _license_files(dist)
        if not files:
            chunks.append("(no license file shipped in the package metadata; "
                          "see the declared license above — full texts of the "
                          "GNU and Apache licenses are in the appendix)\n")
        for fname, text in files:
            chunks.append(f"--- {name}: {fname} ---\n{text.rstrip()}\n")

    chunks.append(f"\n{RULE}\nAPPENDIX — GNU LICENSE TEXTS "
                  f"(referenced above)\n{RULE}\n")
    for txt in ("lgpl-3.0.txt", "gpl-3.0.txt", "apache-2.0.txt"):
        p = here / "licenses" / txt
        chunks.append(f"--- {txt} ---\n"
                      + p.read_text(encoding="utf-8", errors="replace"))

    out_path.write_text("\n".join(chunks), encoding="utf-8")
    print(f"wrote {out_path} "
          f"({out_path.stat().st_size / 1024:.0f} KB, {len(dists)} packages)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True,
                    help="output THIRD_PARTY_LICENSES.txt path")
    args = ap.parse_args()
    build(args.out)
