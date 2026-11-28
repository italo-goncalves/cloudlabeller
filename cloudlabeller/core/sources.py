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

"""Where a label came from, and how sure it is (ADR-0006).

Every label has a **source**, kept as a ``uint8`` code per cloud point and
per pixel. Lower codes are more reliable; ``NONE`` means no label:

``PAINTED``
    painted by the user on the cloud itself (points only). Votes never
    change it.
``HAND``
    hand-drawn: painted on a photo, or a point decided by such photos' votes.
``CLOUD``
    from the cloud: projected into a photo from points whose labels came
    from hand-drawn sources, or a point decided by such projections.
``PREDICTED``
    the U-Net's output, directly or through the cloud. The network never
    learns from it.

The **Uncertainty** of a voted point comes from the votes that decided it:
the normalised entropy of their Class shares smoothed by a Dirichlet prior
(``alpha = 1/K``, about one vote of doubt), with the amount of evidence set
to the number of independent views rather than of photos. Two cameras in
the same direction cell (one of :data:`N_CELLS`, about 25° each) as seen
from the point count as one view.
"""

from __future__ import annotations

import numpy as np

NONE, PAINTED, HAND, CLOUD, PREDICTED = 0, 1, 2, 3, 4

#: The names the Engine and the window use.
NAMES = {NONE: "none", PAINTED: "painted", HAND: "hand", CLOUD: "cloud",
         PREDICTED: "predicted"}

#: Direction cells per point: one bit each in a ``uint64``.
N_CELLS = 64


def rank(source: np.ndarray) -> np.ndarray:
    """A sortable rank, lower better; no label ranks last."""
    s = np.asarray(source, np.uint8)
    return np.where(s == NONE, np.uint8(255), s)


def projected(point_source: np.ndarray) -> np.ndarray:
    """The pixel source of a label projected from points of these sources:
    anything hand-drawn becomes ``CLOUD``, predictions stay ``PREDICTED``."""
    s = np.asarray(point_source, np.uint8)
    out = np.full(s.shape, CLOUD, np.uint8)
    out[s == PREDICTED] = PREDICTED
    out[s == NONE] = NONE
    return out


def trainable(pixel_source: np.ndarray) -> np.ndarray:
    """Pixels the network may learn from: hand-drawn or from the cloud."""
    s = np.asarray(pixel_source, np.uint8)
    return (s != NONE) & (s != PREDICTED)


def legacy(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Source and Uncertainty for labels saved before sources existed:
    hand-drawn where labelled, Uncertainty unknown (NaN)."""
    labelled = np.asarray(labels) != -1
    source = np.where(labelled, np.uint8(HAND), np.uint8(NONE)).astype(np.uint8)
    return source, np.full(len(source), np.nan, np.float16)


# -- direction cells ------------------------------------------------------------
def cell_directions(n: int = N_CELLS) -> np.ndarray:
    """``n`` unit vectors spread evenly over the sphere (a Fibonacci sphere)."""
    k = np.arange(n) + 0.5
    z = 1.0 - 2.0 * k / n
    r = np.sqrt(1.0 - z * z)
    phi = np.pi * (3.0 - np.sqrt(5.0)) * k
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z]).astype(np.float32)


_CELLS = cell_directions()
_BITS = (np.uint64(1) << np.arange(N_CELLS, dtype=np.uint64))


def view_bits(points: np.ndarray, centre: np.ndarray, chunk: int = 1 << 20) -> np.ndarray:
    """The direction cell of a camera at ``centre`` as seen from each point,
    as one set bit of a ``uint64`` per point."""
    out = np.empty(len(points), np.uint64)
    c = np.asarray(centre, np.float32)
    for i in range(0, len(points), chunk):
        d = c[None, :] - np.asarray(points[i:i + chunk], np.float32)
        out[i:i + chunk] = _BITS[np.argmax(d @ _CELLS.T, axis=1)]
    return out


def popcount(bits: np.ndarray) -> np.ndarray:
    """Set bits per ``uint64``: the number of independent views."""
    bits = np.ascontiguousarray(bits, np.uint64)
    if hasattr(np, "bitwise_count"):                 # numpy >= 2.0
        return np.bitwise_count(bits).astype(np.int32)
    table = np.array([bin(i).count("1") for i in range(256)], np.int32)
    return table[bits.view(np.uint8).reshape(-1, 8)].sum(axis=1)


# -- Uncertainty ----------------------------------------------------------------
def vote_uncertainty(counts: np.ndarray, views: np.ndarray, chunk: int = 1 << 20) -> np.ndarray:
    """Uncertainty (0 sure, 1 unsure) of points from their vote counts
    ``(n, K)`` and their number of independent views ``(n,)``.

    The Class shares come from the raw counts; the evidence behind them is
    the number of views (``c̃ = c / N · views``). With ``alpha = 1/K``::

        p = (c̃ + alpha) / (views + K·alpha),   U = H(p) / log K
    """
    counts = np.asarray(counts)
    n, k = counts.shape
    out = np.zeros(n, np.float32)
    if k < 2 or n == 0:
        return out.astype(np.float16)
    alpha = 1.0 / k
    for i in range(0, n, chunk):
        c = counts[i:i + chunk].astype(np.float32)
        total = c.sum(axis=1, keepdims=True)
        v = np.asarray(views[i:i + chunk], np.float32)[:, None]
        shares = np.divide(c, total, out=np.zeros_like(c), where=total > 0)
        p = (shares * v + alpha) / (v + k * alpha)
        out[i:i + chunk] = -(p * np.log(p)).sum(axis=1) / np.log(k)
    return np.clip(out, 0.0, 1.0).astype(np.float16)
