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

"""Cached camera↔point visibility (the transfer hot path).

Which points a camera sees — and at which pixel — is pure geometry: it only
changes when the dense cloud or the cameras change. Computing it costs ~130 ms
per camera at survey scale, and every transfer / propagate / auto-overlay used
to redo it. This index caches the (point_indices, pixel_coords) pair per camera
under ``reconstruction/pmap/`` (~8 ms per camera when warm, ~16x faster).

Entries are **self-validating**: each stores fingerprints of the cloud and the
camera; a new reconstruction, imported cloud, or changed pose simply misses the
cache and recomputes. No manual invalidation hooks required.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np

from cloudlabeller.core.dataset import Camera, PointCloud
from cloudlabeller.transfer.hylite_bridge import VISIBILITY_VERSION, visible_point_pixels

_RAM_ENTRIES = 8          # small LRU of decoded entries kept in memory


def array_fingerprint(*arrays, text: str = "") -> str:
    """Cheap, stable fingerprint of (large) arrays: sizes + sampled bytes."""
    h = hashlib.sha1()
    h.update(text.encode())
    for arr in arrays:
        if arr is None:
            h.update(b"none")
            continue
        arr = np.ascontiguousarray(arr)
        h.update(str(arr.shape).encode() + str(arr.dtype).encode())
        step = max(1, len(arr) // 1024)
        h.update(np.ascontiguousarray(arr[::step]).tobytes())
    return h.hexdigest()[:16]


def cloud_fingerprint(cloud: PointCloud) -> str:
    # Normals participate: they change visibility (back-face culling). The
    # algorithm version participates too, so entries computed by an older
    # visibility algorithm miss and recompute.
    return array_fingerprint(cloud.xyz, cloud.normals,
                             text=f"v{VISIBILITY_VERSION}|{cloud.n_points}")


def camera_fingerprint(camera: Camera) -> str:
    return array_fingerprint(
        np.asarray(camera.K, np.float64), np.asarray(camera.R, np.float64),
        np.asarray(camera.t, np.float64),
        None if camera.distortion is None else np.asarray(camera.distortion, np.float64),
        text=f"{camera.model}|{camera.width}x{camera.height}")


class VisibilityIndex:
    """Per-camera (visible point indices, pixel coords) cache for one project."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.dir = Path(cache_dir)
        self._ram: OrderedDict[tuple, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, name: str, cloud: PointCloud, camera: Camera):
        """Visible (indices, (col,row) pixels) of ``cloud`` in camera ``name``.

        Loads from RAM/disk when the fingerprints match; otherwise computes,
        stores, and returns. Thread-safe.
        """
        cfp = cloud_fingerprint(cloud)
        xfp = camera_fingerprint(camera)
        key = (name, cfp, xfp)
        with self._lock:
            if key in self._ram:
                self._ram.move_to_end(key)
                return self._ram[key]

            path = self.dir / f"{name}.{cfp[:8]}.npz"
            entry = self._load(path, cfp, xfp)
            if entry is None:
                idx, pix = visible_point_pixels(cloud, camera)
                entry = (np.asarray(idx, np.int64), np.asarray(pix, np.int32))
                self._store(name, path, entry, cfp, xfp)
            self._ram[key] = entry
            while len(self._ram) > _RAM_ENTRIES:
                self._ram.popitem(last=False)
            return entry

    # -- disk --------------------------------------------------------------
    def _load(self, path: Path, cfp: str, xfp: str):
        if not path.exists():
            return None
        try:
            data = np.load(path)
            if str(data["cfp"]) != cfp or str(data["xfp"]) != xfp:
                return None                     # stale (camera moved / new cloud)
            return (data["idx"].astype(np.int64), data["pix"].astype(np.int32))
        except Exception:
            return None                          # unreadable -> recompute

    def _store(self, name: str, path: Path, entry, cfp: str, xfp: str) -> None:
        idx, pix = entry
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            for stale in self.dir.glob(f"{name}.*.npz"):   # older-cloud entries
                if stale != path:
                    stale.unlink(missing_ok=True)
            np.savez(path, idx=idx.astype(np.uint32),
                     pix=pix.astype(np.uint16),            # dims < 65536
                     cfp=np.str_(cfp), xfp=np.str_(xfp))
        except Exception:
            pass                                 # cache is best-effort only
