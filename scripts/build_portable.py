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

"""Build a fully portable CloudLabeller distribution for Windows.

Creates a self-contained folder around the official *embeddable* Python
(which, unlike a venv, is relocatable), pip-installs the requirements into it,
copies the ``cloudlabeller`` package and the CUDA COLMAP binaries, and writes
launchers + README. The result runs from any location on a 64-bit Windows
machine — including the app's subprocess architecture (``python -m …``), which
a frozen (PyInstaller-style) build would break.

Usage (from the dev venv):

    .venv/Scripts/python scripts/build_portable.py [--out C:/Temp/cl_build]

Then zip the resulting ``CloudLabeller`` folder and ship it.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

PYTHON_EMBED_URL = ("https://www.python.org/ftp/python/3.10.11/"
                    "python-3.10.11-embed-amd64.zip")
GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"

PROJECT = Path(__file__).resolve().parents[1]
COLMAP_SRC = Path.home() / ".cloudlabeller" / "colmap"

LAUNCHER_BAT = """@echo off
rem CloudLabeller launcher (console visible - useful for diagnostics)
"%~dp0python.exe" -m cloudlabeller %*
"""

LAUNCHER_VBS = '''\
' CloudLabeller silent launcher (no console window)
Set shell = CreateObject("WScript.Shell")
shell.CurrentDirectory = CreateObject("Scripting.FileSystemObject") _
    .GetParentFolderName(WScript.ScriptFullName)
shell.Run """" & shell.CurrentDirectory & "\\pythonw.exe"" -m cloudlabeller", 0, False
'''

README = """CloudLabeller — portable distribution
=====================================

To run:  double-click CloudLabeller.vbs   (or CloudLabeller.bat to see a console)

Requirements on the target machine
----------------------------------
* 64-bit Windows 10 or 11.
* Microsoft Visual C++ Redistributable 2015-2022 (x64). Most machines have it;
  if the app fails to start, install it from:
  https://aka.ms/vs/17/release/vc_redist.x64.exe
* Dense reconstruction (MVS) additionally needs an NVIDIA GPU with a current
  driver (the bundled COLMAP is a CUDA build). Everything else — SfM,
  labelling, training, prediction — runs on the CPU.

Notes
-----
* The folder is self-contained; move or rename it freely. Settings and logs
  live in the user's home folder under .cloudlabeller/.
* Projects are ordinary folders — copy them between machines as-is.
"""


def download(url: str, dest: Path) -> None:
    print(f"  downloading {url}")
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def run(args: list, **kw) -> None:
    print("  $", " ".join(str(a) for a in args))
    subprocess.run([str(a) for a in args], check=True, **kw)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=r"C:\Temp\cl_build",
                    help="build root (avoid synced folders: the tree is ~3 GB)")
    ap.add_argument("--skip-colmap", action="store_true",
                    help="do not bundle the CUDA COLMAP binaries (~650 MB)")
    args = ap.parse_args()

    out = Path(args.out)
    dist = out / "CloudLabeller"
    if dist.exists():
        print(f"removing previous build at {dist}")
        shutil.rmtree(dist)
    dist.mkdir(parents=True)
    py = dist / "python.exe"

    print("[1/6] embeddable Python")
    zip_path = out / "python-embed.zip"
    if not zip_path.exists():
        download(PYTHON_EMBED_URL, zip_path)
    shutil.unpack_archive(zip_path, dist)
    # Enable site-packages + the app dir (the stock ._pth disables `site`).
    pth = dist / "python310._pth"
    pth.write_text("python310.zip\n.\napp\nLib\\site-packages\nimport site\n",
                   encoding="ascii")

    print("[2/6] pip")
    get_pip = out / "get-pip.py"
    if not get_pip.exists():
        download(GET_PIP_URL, get_pip)
    run([py, get_pip, "--no-warn-script-location"])

    print("[3/6] requirements (this downloads/installs ~2 GB — be patient)")
    run([py, "-m", "pip", "install", "--no-warn-script-location",
         "-r", PROJECT / "requirements.txt"])

    print("[4/6] application package")
    shutil.copytree(PROJECT / "cloudlabeller", dist / "app" / "cloudlabeller",
                    ignore=shutil.ignore_patterns("__pycache__"))

    print("[5/6] COLMAP (CUDA)")
    if args.skip_colmap:
        print("  skipped (--skip-colmap)")
    elif COLMAP_SRC.exists():
        shutil.copytree(COLMAP_SRC, dist / "colmap")
    else:
        print(f"  WARNING: {COLMAP_SRC} not found — dense MVS will be "
              "unavailable on target machines")

    print("[6/6] launchers + README")
    (dist / "CloudLabeller.bat").write_text(LAUNCHER_BAT, encoding="utf-8")
    (dist / "CloudLabeller.vbs").write_text(LAUNCHER_VBS, encoding="utf-8")
    (dist / "README.txt").write_text(README, encoding="utf-8")

    print("smoke test: importing the app (offscreen) in the new distribution…")
    env = {"QT_QPA_PLATFORM": "offscreen", "PYVISTA_OFF_SCREEN": "true",
           "SYSTEMROOT": r"C:\Windows", "PATH": r"C:\Windows\System32"}
    run([py, "-c",
         "import cloudlabeller.ui.main_window, tensorflow, pycolmap, pyvista; "
         "from cloudlabeller.photogrammetry.mvs import find_colmap_binary; "
         "print('smoke OK; colmap:', find_colmap_binary())"], env=env)

    print(f"\nDone: {dist}\nZip that folder to distribute it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
