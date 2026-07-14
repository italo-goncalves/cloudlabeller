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

"""Download the official COLMAP Windows bundle into the app's user directory.

CloudLabeller does not ship COLMAP inside its own distribution: the official
binary bundle links GPL components (SuiteSparse SPQR/CHOLMOD), so instead the
app fetches it straight from the COLMAP GitHub release on demand — one click
via Photogrammetry → Download COLMAP…. The install target is
``~/.cloudlabeller/colmap``, which :func:`cloudlabeller.photogrammetry.mvs.
find_colmap_binary` already searches.

Qt-free: meant to run on a ``workers.job.Job`` (pure network + zip I/O — no
native-library GIL hazard), receiving ``progress`` / ``should_stop`` from the
Job runner.
"""

from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path
from typing import Callable
from urllib.request import Request, urlopen

ProgressFn = Callable[[float, str], None]

COLMAP_VERSION = "4.1.0"

_URL = ("https://github.com/colmap/colmap/releases/download/"
        + COLMAP_VERSION + "/colmap-x64-windows-{variant}.zip")

# Approximate release-asset sizes (MB) — shown in dialogs and used as the
# progress denominator if the server omits Content-Length.
DOWNLOAD_MB = {"cuda": 307, "nocuda": 114}

# The release bundle ships its unit tests: ~140 *_test.exe plus the gtest /
# gmock frameworks — dead weight for running the pipeline.
_TEST_DLLS = {"gtest.dll", "gtest_main.dll", "gmock.dll", "gmock_main.dll"}

_CHUNK = 1 << 20  # 1 MiB


def download_url(variant: str = "cuda") -> str:
    """Release-asset URL for ``variant`` ("cuda" | "nocuda")."""
    if variant not in DOWNLOAD_MB:
        raise ValueError(f"unknown COLMAP variant {variant!r} "
                         f"(expected one of {sorted(DOWNLOAD_MB)})")
    return _URL.format(variant=variant)


def default_install_dir() -> Path:
    """Per-user install location (parent of the ``bin/colmap.exe`` the app
    searches). Kept outside the app folder so reinstalling/updating
    CloudLabeller never re-downloads COLMAP."""
    return Path.home() / ".cloudlabeller" / "colmap"


def cuda_capable() -> bool:
    """Best-effort check for an NVIDIA GPU: nvidia-smi / nvml.dll ship with
    the driver (NOT the CUDA toolkit), so this detects the hardware even on
    machines without CUDA installed — which is all colmap.exe needs, since it
    bundles its own CUDA runtime use via the driver."""
    if shutil.which("nvidia-smi"):
        return True
    root = os.environ.get("SystemRoot", r"C:\Windows")
    return (Path(root) / "System32" / "nvml.dll").exists()


def strip_test_binaries(bundle: Path) -> int:
    """Delete COLMAP's bundled unit tests; returns how many files went."""
    removed = 0
    for f in list(Path(bundle).rglob("*")):
        if f.is_file() and (f.name.endswith("_test.exe")
                            or f.name.lower() in _TEST_DLLS
                            or f.name.upper() == "RUN_TESTS.BAT"):
            f.unlink()
            removed += 1
    return removed


def install_colmap(variant: str = "cuda",
                   dest: str | Path | None = None,
                   progress: ProgressFn = lambda f, m="": None,
                   should_stop: Callable[[], bool] = lambda: False) -> str:
    """Download + unpack the official COLMAP release; return the exe path.

    Any previous install at ``dest`` is replaced (it is an app-managed
    directory). Cancellation (``should_stop``) and errors leave no partial
    install behind: work happens in ``*.part`` / ``*.extract`` siblings and
    only a fully verified bundle is moved into place.
    """
    dest = Path(dest) if dest is not None else default_install_dir()
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = download_url(variant)
    zip_path = dest.parent / f"colmap-{variant}.zip.part"
    tmp = dest.parent / f"colmap-{variant}.extract"

    try:
        # -- download (0 → 0.85) ------------------------------------------
        req = Request(url, headers={"User-Agent": "CloudLabeller"})
        with urlopen(req) as resp, open(zip_path, "wb") as out:
            total = int(resp.headers.get("Content-Length")
                        or DOWNLOAD_MB[variant] * _CHUNK)
            done = 0
            while True:
                if should_stop():
                    raise RuntimeError("COLMAP download cancelled")
                chunk = resp.read(_CHUNK)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                progress(0.85 * min(done / total, 1.0),
                         f"Downloading COLMAP {COLMAP_VERSION} ({variant}) — "
                         f"{done >> 20} / {total >> 20} MB")

        # -- extract (0.85 → 0.97) ----------------------------------------
        if tmp.exists():
            shutil.rmtree(tmp)
        with zipfile.ZipFile(zip_path) as zf:
            members = zf.infolist()
            for i, member in enumerate(members):
                if should_stop():
                    raise RuntimeError("COLMAP download cancelled")
                zf.extract(member, tmp)
                if i % 50 == 0:
                    progress(0.85 + 0.12 * (i + 1) / len(members),
                             f"Unpacking COLMAP… {i + 1}/{len(members)} files")

        exe = next(tmp.rglob("colmap.exe"), None)
        if exe is None:
            raise RuntimeError("downloaded archive contains no colmap.exe — "
                               "release layout changed?")
        # Bundle root = the single top-level wrapper folder if the zip has
        # one (the official releases do), else the archive root itself.
        top = list(tmp.iterdir())
        root = top[0] if len(top) == 1 and top[0].is_dir() else tmp

        n = strip_test_binaries(root)
        progress(0.98, f"Removed {n} bundled test binaries")

        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(root), str(dest))
        final = dest / exe.relative_to(root)
        if not final.exists():
            raise RuntimeError(f"install verification failed: {final} missing")
        progress(1.0, f"COLMAP {COLMAP_VERSION} ({variant}) installed: {final}")
        return str(final)
    finally:
        zip_path.unlink(missing_ok=True)
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
