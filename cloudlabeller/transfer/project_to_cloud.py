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

"""Image labels -> cloud labels by majority vote across all labelled images."""

from __future__ import annotations

from typing import Callable

import numpy as np

from cloudlabeller.core.dataset import Dataset, PointCloud
from cloudlabeller.core.label_schema import UNLABELLED_ID
from cloudlabeller.transfer import hylite_bridge as hb

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
    points the hard data doesn't reach.

    Each image's mask is fetched from ``image_masks`` exactly once, so a lazy
    :class:`MaskSource` renders every auto mask a single time per call.
    """
    n = cloud.n_points
    votes = np.zeros((n, max(n_classes, 1)), dtype=np.int32)
    priority_names = priority_names or set()

    candidates = [r for r in dataset.solved_images() if r.camera is not None]
    # Two tiers: hard (priority) images first, the rest after; the votes buffer
    # is reused between tiers to avoid a second (N, n_classes) allocation.
    tiers = ([r for r in candidates if r.name in priority_names],
             [r for r in candidates if r.name not in priority_names])
    total = max(len(candidates), 1)

    def tally() -> np.ndarray:
        out = np.full(n, UNLABELLED_ID, dtype=np.int32)
        voted = votes.sum(axis=1) > 0
        out[voted] = votes[voted].argmax(axis=1)
        return out

    done = 0
    n_used = 0
    labels: np.ndarray | None = None      # hard-tier result, if any
    for tier_i, records in enumerate(tiers):
        for record in records:
            done += 1
            mask = image_masks.get(record.name)
            if mask is None or not (mask != UNLABELLED_ID).any():
                progress(done / total, "")
                continue
            n_used += 1
            if visibility is not None:
                pt_idx, pix = visibility.get(record.name, cloud, record.camera)
            else:
                pt_idx, pix = hb.visible_point_pixels(cloud, record.camera)
            if pt_idx.size:
                h, w = mask.shape
                cols = np.clip(pix[:, 0], 0, w - 1)
                rows = np.clip(pix[:, 1], 0, h - 1)
                sampled = mask[rows, cols]
                keep = sampled != UNLABELLED_ID
                np.add.at(votes, (pt_idx[keep], sampled[keep]), 1)
            progress(done / total,
                     f"images → cloud ({record.name}, {n_used} projected)")
        if tier_i == 0 and priority_names:
            labels = tally()              # hard results, decided by hard votes only
            votes[:] = 0                  # reuse the buffer for the soft tier

    soft = tally()
    if labels is None:
        return soft
    unset = labels == UNLABELLED_ID
    labels[unset] = soft[unset]           # soft labels only fill hard gaps
    return labels
