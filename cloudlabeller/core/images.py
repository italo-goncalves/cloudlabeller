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

"""The project image store: listing and ingesting image files.

CloudLabeller keeps a canonical copy of every image inside ``project/images/``
because the images are needed for the whole life of the project (2D labelling,
U-Net training/inference, image<->cloud transfer) — not just for SfM. Ingest is a
real byte copy, so the project is fully self-contained and portable.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# Image formats COLMAP/most tools can read. Everything else in a source folder
# (DJI PPK sidecars .nav/.obs/.bin/.MRK, logs, thumbnails, …) is ignored.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".ppm", ".pgm"}

ProgressFn = Callable[[float, str], None]


def list_image_files(image_dir: str | Path) -> list[str]:
    """Image filenames under ``image_dir`` (recursive), relative & posix-style."""
    root = Path(image_dir)
    if not root.is_dir():
        return []
    return [
        p.relative_to(root).as_posix()
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


@dataclass
class IngestResult:
    copied: list[str] = field(default_factory=list)       # new files
    overwritten: list[str] = field(default_factory=list)  # replaced existing
    skipped: list[str] = field(default_factory=list)      # kept existing / dupes

    @property
    def changed(self) -> bool:
        """Did the store actually change (so a reconstruction would be stale)?"""
        return bool(self.copied or self.overwritten)


def find_conflicts(src_dir: str | Path, store_dir: str | Path) -> list[str]:
    """Basenames in ``src_dir`` that already exist in the store (would collide)."""
    store = Path(store_dir)
    if not store.is_dir():
        return []
    existing = {p.name for p in store.iterdir() if p.is_file()}
    src_names = {Path(rel).name for rel in list_image_files(src_dir)}
    return sorted(src_names & existing)


def ingest_images(
    src_dir: str | Path,
    store_dir: str | Path,
    on_conflict: str = "skip",       # "skip" keeps the existing file, "overwrite" replaces it
    progress: ProgressFn = lambda f, m="": None,
) -> IngestResult:
    """Copy image files from ``src_dir`` into the project store ``store_dir``.

    Only real image files are copied (sidecars excluded). Names are flattened to
    their basename. Behaviour on a name already in the store is controlled by
    ``on_conflict``. Duplicate basenames *within* the source are de-duplicated
    (the first wins; the rest are skipped) so a later file can't silently clobber
    an earlier one.
    """
    src = Path(src_dir)
    store = Path(store_dir)
    store.mkdir(parents=True, exist_ok=True)

    names = list_image_files(src)
    result = IngestResult()
    seen: set[str] = set()
    total = max(len(names), 1)
    for i, rel in enumerate(names, 1):
        base = Path(rel).name
        target = store / base
        if base in seen:
            result.skipped.append(base)            # duplicate name within the source
        elif target.exists():
            if on_conflict == "overwrite":
                shutil.copy2(src / rel, target)
                result.overwritten.append(base)
            else:
                result.skipped.append(base)
        else:
            shutil.copy2(src / rel, target)
            result.copied.append(base)
        seen.add(base)
        progress(i / total, f"Ingested {i}/{len(names)} images")
    return result
