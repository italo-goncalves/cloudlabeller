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

"""Image labels -> cloud labels by majority vote across all labelled images.

Votes are taken per source, best first (ADR-0006, :mod:`cloudlabeller.core.sources`):
a point any hand-drawn pixel reaches is decided by hand-drawn votes alone,
and worse sources only fill the points better ones didn't reach. Each point
gets the source that decided it and an Uncertainty from those votes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from cloudlabeller.core.dataset import Dataset, PointCloud
from cloudlabeller.core import sources as src
from cloudlabeller.core.label_schema import UNLABELLED_ID
from cloudlabeller.transfer import projection as hb

ProgressFn = Callable[[float, str], None]


class MaskSource:
    """Mapping-like mask source: stored (user) masks plus lazily-rendered auto
    masks (cloud projections / U-Net predictions).

    Auto masks are rendered on access and NOT cached — a full-resolution int32
    mask is ~84 MB, and materialising one per image froze the UI once already.
    ``images_to_cloud`` reads each image's mask exactly once per call, so lazy
    rendering costs one render per auto image per cloud.
    """

    def __init__(self, stored: dict[str, np.ndarray],
                 auto_names: "set[str] | list[str]" = (),
                 render: Callable[[str], np.ndarray | None] | None = None) -> None:
        self._stored = stored
        self._auto = set(auto_names)
        self._render = render

    def get(self, name: str) -> np.ndarray | None:
        mask = self._stored.get(name)
        if mask is None and name in self._auto and self._render is not None:
            mask = self._render(name)
        return mask


@dataclass
class Votes:
    """What voting gives a cloud: per point, a label (−1 none), the source
    that decided it (:mod:`cloudlabeller.core.sources`) and its Uncertainty
    (NaN where unlabelled)."""

    labels: np.ndarray        # (N,) int32
    source: np.ndarray        # (N,) uint8
    uncertainty: np.ndarray   # (N,) float16


@dataclass
class VotePass:
    """One source's votes: the photos that cast them, and how to get each
    photo's mask of that source's pixels (None: nothing to cast)."""

    source: int
    names: Sequence[str]
    mask: Callable[[str], "np.ndarray | None"]


def vote(
    cloud: PointCloud,
    dataset: Dataset,
    passes: Sequence[VotePass],
    n_classes: int,
    progress: ProgressFn = lambda f, m="": None,
    visibility=None,          # optional transfer.visibility.VisibilityIndex
) -> Votes:
    """Vote the photos' labels onto ``cloud``, one source per pass, best
    first. A point that got any vote in a pass is decided by that pass: its
    label is the winning Class, its source the pass's, and its Uncertainty
    comes from that pass's votes and the number of directions they came
    from (:func:`cloudlabeller.core.sources.vote_uncertainty`). Later passes
    only label points no earlier pass reached.

    Each photo's mask is fetched once per pass that lists it, so a lazy
    renderer renders it once per pass.
    """
    n = cloud.n_points
    k = max(n_classes, 1)
    labels = np.full(n, UNLABELLED_ID, np.int32)
    source = np.zeros(n, np.uint8)
    uncertainty = np.full(n, np.nan, np.float16)
    records = {r.name: r for r in dataset.solved_images() if r.camera is not None}
    work = [(p, [records[nm] for nm in p.names if nm in records]) for p in passes]
    total = max(sum(len(rs) for _, rs in work), 1)
    votes = np.zeros((n, k), np.int32)        # reused between passes
    views = np.zeros(n, np.uint64)
    done = 0
    for p, recs in work:
        votes[:] = 0
        views[:] = 0
        used = 0
        for record in recs:
            done += 1
            mask = p.mask(record.name)
            if mask is None or not (mask != UNLABELLED_ID).any():
                progress(done / total, "")
                continue
            used += 1
            if visibility is not None:
                pt_idx, pix = visibility.get(record.name, cloud, record.camera)
            else:
                pt_idx, pix = hb.visible_point_pixels(cloud, record.camera)
            if pt_idx.size:
                h, w = mask.shape
                cols = np.clip(pix[:, 0], 0, w - 1)
                rows = np.clip(pix[:, 1], 0, h - 1)
                sampled = mask[rows, cols]
                keep = (sampled != UNLABELLED_ID) & (labels[pt_idx] == UNLABELLED_ID)
                idx = pt_idx[keep]
                np.add.at(votes, (idx, sampled[keep]), 1)
                views[idx] |= src.view_bits(cloud.xyz[idx], record.camera.position)
            progress(done / total, f"images → cloud ({record.name}, {used} projected)")
        reached = np.flatnonzero(votes.any(axis=1))
        if reached.size:
            labels[reached] = votes[reached].argmax(axis=1)
            source[reached] = p.source
            uncertainty[reached] = src.vote_uncertainty(votes[reached],
                                                        src.popcount(views[reached]))
    return Votes(labels, source, uncertainty)


def images_to_cloud(
    cloud: PointCloud,
    dataset: Dataset,
    image_masks,              # dict[str, np.ndarray] or MaskSource (needs .get)
    n_classes: int,
    progress: ProgressFn = lambda f, m="": None,
    visibility=None,          # optional transfer.visibility.VisibilityIndex
    priority_names: "set[str] | None" = None,
) -> np.ndarray:
    """Accumulate per-image label votes onto cloud points; return (N,) labels.

    For each labelled image we project the cloud into it (occlusion-aware),
    sample the image mask at each visible point, and tally votes per point. The
    winning class per point wins; points with no votes stay unlabelled (-1).

    ``priority_names`` marks images whose labels are HARD data (user-drawn):
    any point receiving at least one vote from them is decided by those votes
    alone — otherwise one hand-labelled image would be outvoted ("drowned out")
    by a crowd of auto/ML-labelled neighbours. The remaining images only label
    points the hard data doesn't reach. (:func:`vote` with one pass per
    source, keeping only the labels.)
    """
    priority_names = priority_names or set()
    names = [r.name for r in dataset.solved_images() if r.camera is not None]
    passes = [VotePass(src.HAND, [nm for nm in names if nm in priority_names],
                       image_masks.get),
              VotePass(src.CLOUD, [nm for nm in names if nm not in priority_names],
                       image_masks.get)]
    return vote(cloud, dataset, passes, n_classes, progress, visibility).labels
