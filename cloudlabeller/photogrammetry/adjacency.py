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

"""Image-overlap (covisibility) graph: which images see the same surface.

Two images overlap iff they observe common 3D points. COLMAP already knows
this: every sparse point's track lists the images observing it, so the graph
falls out of the reconstruction for free at SfM time. It is persisted as
``reconstruction/adjacency.json`` and used to restrict per-image operations
(e.g. propagating one image's labels) to the images that actually share
surface with it, instead of looping over the whole store.

For projects reconstructed before this file existed, the same graph is
derived from the cached camera↔point visibility of the sparse cloud
(:mod:`cloudlabeller.transfer.visibility`) — an occlusion-aware variant of
the same covisibility measure — and saved for reuse.

The graph describes *camera* overlap: it stays valid through label edits and
cloud cleaning, and a new SfM run rewrites it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ADJACENCY_FILE = "adjacency.json"

# An image is a neighbour when it shares at least MIN_SHARED points AND at
# least MIN_FRACTION of the smaller image's observed points. Genuinely
# overlapping survey images share hundreds of sparse points (COLMAP needs
# substantial overlap to register them at all); a handful of shared points is
# incidental (long tracks on distant landmarks).
MIN_SHARED = 10
MIN_FRACTION = 0.02


def _count_shared(rows: np.ndarray, cols: np.ndarray, n_points: int,
                  names: list[str]) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    """Count shared observations from (point, image) pairs.

    ``rows`` = point indices, ``cols`` = image indices into ``names``. Returns
    (per-image totals, symmetric shared-count dict). Uses a sparse
    points×images incidence matrix so the pair counting is one matmul.
    """
    from scipy import sparse as sp

    if len(rows) == 0:
        return {}, {}
    a = sp.coo_matrix((np.ones(len(rows), dtype=np.int32), (rows, cols)),
                      shape=(n_points, len(names))).tocsr()
    a.data[:] = 1                       # dedupe multiple observations per pair
    totals = {names[i]: int(t)
              for i, t in enumerate(np.asarray(a.sum(axis=0)).ravel()) if t}
    c = (a.T @ a).tocoo()
    keep = c.row < c.col                # each pair once; diagonal = totals
    shared: dict[str, dict[str, int]] = {}
    for i, j, v in zip(c.row[keep].tolist(), c.col[keep].tolist(),
                       c.data[keep].tolist()):
        shared.setdefault(names[i], {})[names[j]] = int(v)
        shared.setdefault(names[j], {})[names[i]] = int(v)
    return totals, shared


@dataclass
class ImageAdjacency:
    """Covisibility graph over image names.

    ``totals[name]`` = points observed by that image; ``shared[a][b]`` =
    points observed by both (symmetric).
    """

    totals: dict[str, int] = field(default_factory=dict)
    shared: dict[str, dict[str, int]] = field(default_factory=dict)
    source: str = "sfm"                # "sfm" (tracks) | "visibility" (fallback)

    def overlapping(self, name: str, min_shared: int = MIN_SHARED,
                    min_fraction: float = MIN_FRACTION) -> set[str] | None:
        """Names of images overlapping ``name``, or None when ``name`` is not
        in the graph (caller should fall back to considering every image)."""
        if name not in self.totals:
            return None
        mine = self.totals[name]
        out = set()
        for other, n in self.shared.get(name, {}).items():
            floor = max(min_shared, min_fraction * min(mine, self.totals.get(other, mine)))
            if n >= floor:
                out.add(other)
        return out

    # -- persistence --------------------------------------------------------
    def to_dict(self) -> dict:
        pairs: dict[str, dict[str, int]] = {}
        for a, nbrs in self.shared.items():
            for b, n in nbrs.items():
                if a < b:               # store each pair once
                    pairs.setdefault(a, {})[b] = n
        return {"format": 1, "source": self.source,
                "totals": self.totals, "shared": pairs}

    @classmethod
    def from_dict(cls, data: dict) -> "ImageAdjacency":
        adj = cls(totals={k: int(v) for k, v in data.get("totals", {}).items()},
                  source=data.get("source", "sfm"))
        for a, nbrs in data.get("shared", {}).items():
            for b, n in nbrs.items():
                adj.shared.setdefault(a, {})[b] = int(n)
                adj.shared.setdefault(b, {})[a] = int(n)
        return adj

    def save(self, workspace: str | Path) -> None:
        ws = Path(workspace)
        ws.mkdir(parents=True, exist_ok=True)
        (ws / ADJACENCY_FILE).write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, workspace: str | Path) -> "ImageAdjacency | None":
        path = Path(workspace) / ADJACENCY_FILE
        if not path.exists():
            return None
        try:
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            return None                 # unreadable -> recompute/absent


# select_covering: stop picking when the best remaining image would add fewer
# new points than this fraction of its own footprint / this absolute count.
MIN_NEW_FRACTION = 0.05
MIN_NEW_POINTS = 25


def select_covering(names, adjacency: ImageAdjacency,
                    min_new_fraction: float = MIN_NEW_FRACTION,
                    min_new_points: int = MIN_NEW_POINTS) -> list[str]:
    """Greedy subset of ``names`` that still covers ~the same surface.

    For images that are all carriers of the SAME signal (cloud-projection
    "auto" masks render the same cloud labels), projecting several images of
    one area repeats identical votes — only coverage matters. This picks
    images by estimated new coverage (own point total minus pairwise overlap
    with the already-picked set — an inclusion–exclusion lower bound) until
    the best remaining candidate adds almost nothing new.

    Do NOT use this for sources with independent information (user masks,
    U-Net predictions): majority voting across those is deliberate. Note the
    result also flattens vote *weight* — a region formerly re-affirmed by ten
    auto images now casts one vote there, which is the point.

    Names missing from the graph are kept (no information to prune on).
    Input order is preserved in the result.
    """
    totals = adjacency.totals
    known = [n for n in names if n in totals]
    gain = {n: float(totals[n]) for n in known}
    picked: set[str] = set()
    while len(picked) < len(known):
        best = max((n for n in known if n not in picked), key=gain.__getitem__)
        if gain[best] < max(min_new_points, min_new_fraction * totals[best]):
            break
        picked.add(best)
        for other, shared in adjacency.shared.get(best, {}).items():
            if other in gain and other not in picked:
                gain[other] = max(0.0, gain[other] - shared)
    return [n for n in names if n in picked or n not in totals]


def covisibility_from_reconstruction(reconstruction) -> ImageAdjacency:
    """Build the graph from a pycolmap reconstruction's point tracks.

    Runs at the end of SfM: every 3D point's track lists the observing
    images, which is COLMAP's own covisibility.
    """
    index: dict[int, int] = {}
    names: list[str] = []
    for image_id, image in reconstruction.images.items():
        index[image_id] = len(names)
        names.append(image.name)

    rows: list[int] = []
    cols: list[int] = []
    n_points = 0
    for point in reconstruction.points3D.values():
        for el in point.track.elements:
            j = index.get(el.image_id)
            if j is not None:
                rows.append(n_points)
                cols.append(j)
        n_points += 1

    totals, shared = _count_shared(np.asarray(rows, dtype=np.int64),
                                   np.asarray(cols, dtype=np.int64),
                                   n_points, names)
    return ImageAdjacency(totals=totals, shared=shared, source="sfm")


def covisibility_from_images(cloud, records, visibility=None) -> ImageAdjacency:
    """Build the graph from per-camera visibility of the sparse cloud.

    Fallback for projects whose reconstruction predates ``adjacency.json``:
    ``records`` are solved :class:`ImageRecord`s; ``visibility`` an optional
    :class:`~cloudlabeller.transfer.visibility.VisibilityIndex` (reuses the
    per-camera cache; without it each camera is projected directly).
    """
    from cloudlabeller.transfer.hylite_bridge import visible_point_pixels

    names: list[str] = []
    parts_rows: list[np.ndarray] = []
    parts_cols: list[np.ndarray] = []
    for rec in records:
        if rec.camera is None:
            continue
        j = len(names)
        names.append(rec.name)
        if visibility is not None:
            idx, _ = visibility.get(rec.name, cloud, rec.camera)
        else:
            idx, _ = visible_point_pixels(cloud, rec.camera)
        parts_rows.append(np.asarray(idx, dtype=np.int64))
        parts_cols.append(np.full(len(idx), j, dtype=np.int64))

    rows = np.concatenate(parts_rows) if parts_rows else np.empty(0, np.int64)
    cols = np.concatenate(parts_cols) if parts_cols else np.empty(0, np.int64)
    totals, shared = _count_shared(rows, cols, cloud.n_points, names)
    return ImageAdjacency(totals=totals, shared=shared, source="visibility")
