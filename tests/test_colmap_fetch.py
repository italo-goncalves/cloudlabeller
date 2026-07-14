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

"""Tests for the COLMAP release downloader (photogrammetry/colmap_fetch.py).

The download itself is mocked with an in-memory zip shaped like the official
release asset; an opt-in network test (CLOUDLABELLER_NET_TESTS=1) verifies the
pinned release URL is still live.
"""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

import pytest

from cloudlabeller.photogrammetry import colmap_fetch
from cloudlabeller.photogrammetry.colmap_fetch import (
    COLMAP_VERSION,
    cuda_capable,
    download_url,
    install_colmap,
    strip_test_binaries,
)


def _release_zip(wrapper: str | None = "colmap-x64-windows-cuda") -> bytes:
    """A zip shaped like the official release: bin/ + lib/ (+ optional
    top-level wrapper folder), including test binaries to be stripped."""
    prefix = f"{wrapper}/" if wrapper else ""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(f"{prefix}bin/colmap.exe", b"MZ fake colmap")
        zf.writestr(f"{prefix}bin/ceres.dll", b"MZ fake dll")
        zf.writestr(f"{prefix}bin/bbox_test.exe", b"MZ fake test")
        zf.writestr(f"{prefix}bin/gtest.dll", b"MZ fake gtest")
        zf.writestr(f"{prefix}RUN_TESTS.bat", b"@echo off")
        zf.writestr(f"{prefix}lib/whatever.txt", b"data")
    return buf.getvalue()


class _FakeResponse:
    """Duck-typed urlopen() result: headers + chunked read, context manager."""

    def __init__(self, payload: bytes, content_length: bool = True):
        self._data = io.BytesIO(payload)
        self.headers = ({"Content-Length": str(len(payload))}
                        if content_length else {})

    def read(self, n: int = -1) -> bytes:
        return self._data.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _patch_urlopen(monkeypatch, payload: bytes, seen: list | None = None):
    def fake_urlopen(req):
        if seen is not None:
            seen.append(req.full_url)
        return _FakeResponse(payload)
    monkeypatch.setattr(colmap_fetch, "urlopen", fake_urlopen)


def test_download_url_variants():
    assert COLMAP_VERSION in download_url("cuda")
    assert download_url("cuda").endswith("colmap-x64-windows-cuda.zip")
    assert download_url("nocuda").endswith("colmap-x64-windows-nocuda.zip")
    with pytest.raises(ValueError):
        download_url("gpu")


def test_cuda_capable_returns_bool():
    assert isinstance(cuda_capable(), bool)


def test_install_colmap_from_fake_release(tmp_path, monkeypatch):
    seen: list[str] = []
    _patch_urlopen(monkeypatch, _release_zip(), seen)
    dest = tmp_path / "colmap"
    fractions: list[float] = []
    exe = install_colmap("cuda", dest=dest,
                         progress=lambda f, m="": fractions.append(f))

    assert Path(exe) == dest / "bin" / "colmap.exe"
    assert Path(exe).read_bytes() == b"MZ fake colmap"
    assert (dest / "lib" / "whatever.txt").exists()   # siblings kept
    assert seen == [download_url("cuda")]
    # Test binaries stripped from the install:
    leftovers = [f.name for f in dest.rglob("*")
                 if f.name.endswith("_test.exe") or f.name == "gtest.dll"
                 or f.name.upper() == "RUN_TESTS.BAT"]
    assert leftovers == []
    # No temp files remain, progress is sane and finishes at 1.0:
    assert not list(dest.parent.glob("*.part"))
    assert not list(dest.parent.glob("*.extract"))
    assert fractions == sorted(fractions) and fractions[-1] == 1.0


def test_install_colmap_flat_archive(tmp_path, monkeypatch):
    """A zip without a wrapper folder installs identically."""
    _patch_urlopen(monkeypatch, _release_zip(wrapper=None))
    dest = tmp_path / "colmap"
    exe = install_colmap("cuda", dest=dest)
    assert Path(exe) == dest / "bin" / "colmap.exe"


def test_install_colmap_replaces_previous(tmp_path, monkeypatch):
    _patch_urlopen(monkeypatch, _release_zip())
    dest = tmp_path / "colmap"
    stale = dest / "bin" / "old_marker.dll"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"old install")
    install_colmap("cuda", dest=dest)
    assert not stale.exists()
    assert (dest / "bin" / "colmap.exe").exists()


def test_install_colmap_cancel_leaves_nothing(tmp_path, monkeypatch):
    _patch_urlopen(monkeypatch, _release_zip())
    dest = tmp_path / "colmap"
    with pytest.raises(RuntimeError, match="cancelled"):
        install_colmap("cuda", dest=dest, should_stop=lambda: True)
    assert not dest.exists()
    assert list(dest.parent.iterdir()) == []          # no .part/.extract left


def test_install_colmap_rejects_exe_less_archive(tmp_path, monkeypatch):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("readme.txt", b"no exe here")
    _patch_urlopen(monkeypatch, buf.getvalue())
    dest = tmp_path / "colmap"
    with pytest.raises(RuntimeError, match="colmap.exe"):
        install_colmap("cuda", dest=dest)
    assert not dest.exists()


def test_strip_test_binaries(tmp_path):
    (tmp_path / "bin").mkdir()
    keep = tmp_path / "bin" / "colmap.exe"
    keep.write_bytes(b"x")
    for name in ("bin/bbox_test.exe", "bin/gtest.dll", "bin/gmock_main.dll",
                 "RUN_TESTS.bat"):
        p = tmp_path / name
        p.write_bytes(b"x")
    assert strip_test_binaries(tmp_path) == 4
    assert keep.exists()


def test_find_colmap_binary_managed_dir_fallback(tmp_path, monkeypatch):
    """A restructured downloaded bundle is still discovered by the rglob
    fallback under the app-managed directory."""
    import sys

    from cloudlabeller.photogrammetry import mvs

    exe = tmp_path / "colmap" / "tools" / "colmap.exe"   # not bin/
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"MZ")
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "no-portable-bundle"))
    monkeypatch.setattr(mvs, "_DEFAULT_COLMAP",
                        tmp_path / "colmap" / "bin" / "colmap.exe")
    monkeypatch.setattr(mvs.shutil, "which", lambda _: None)
    assert mvs.find_colmap_binary(None) == str(exe)


@pytest.mark.skipif(not os.environ.get("CLOUDLABELLER_NET_TESTS"),
                    reason="network smoke test — set CLOUDLABELLER_NET_TESTS=1")
def test_release_asset_urls_are_live():
    """One byte from each pinned release asset: catches a deleted/renamed
    release before users hit it in the download dialog."""
    from urllib.request import Request, urlopen

    for variant in ("cuda", "nocuda"):
        req = Request(download_url(variant),
                      headers={"User-Agent": "CloudLabeller",
                               "Range": "bytes=0-0"})
        with urlopen(req, timeout=30) as resp:
            assert resp.status in (200, 206)
