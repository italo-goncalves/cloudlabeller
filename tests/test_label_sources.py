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

"""Label sources and the vote Uncertainty (ADR-0006)."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from cloudlabeller.core import sources as src
from cloudlabeller.core.labels import LabelStore, read_label_files, write_label_files


# -- the Uncertainty ------------------------------------------------------------------
def _u(raw: list[int], views: int, k: int = 10) -> float:
    counts = np.zeros((1, k))
    counts[0, :len(raw)] = raw
    return float(src.vote_uncertainty(counts, np.array([views]))[0])


@pytest.mark.parametrize("raw, views, expected", [
    ([1], 1, 0.73),            # one photo is weak evidence
    ([10], 10, 0.20),          # ten agreeing views
    ([5, 5], 10, 0.46),        # plenty of votes, but split
    ([30], 2, 0.55),           # 30 photos of one flight line: about two looks
    ([30], 5, 0.33),           # three lines at different angles
    ([20, 10], 2, 0.70),
    ([50], 50, 0.05),
])
def test_uncertainty_matches_the_worked_cases(raw, views, expected):
    assert _u(raw, views) == pytest.approx(expected, abs=0.01)


def test_more_views_or_more_agreement_mean_less_uncertainty():
    assert _u([30], 30) < _u([30], 5) < _u([30], 2)
    assert _u([10], 10) < _u([8, 2], 10) < _u([5, 5], 10)


def test_cameras_in_one_direction_count_once():
    pts = np.zeros((4, 3), np.float32)
    near = [src.view_bits(pts, np.array([10.0, 0.1 * i, 0.0])) for i in range(5)]
    bits = np.bitwise_or.reduce(near)
    assert (src.popcount(bits) == 1).all()           # five photos, one view
    opposite = bits | src.view_bits(pts, np.array([-10.0, 0.0, 0.0]))
    assert (src.popcount(opposite) == 2).all()


def test_popcount_without_numpy_2(monkeypatch):
    bits = np.array([0, 1, 3, (1 << 63) | 1], np.uint64)
    want = src.popcount(bits)
    monkeypatch.delattr(np, "bitwise_count", raising=False)
    assert np.array_equal(src.popcount(bits), want)
    assert want.tolist() == [0, 1, 2, 2]


def test_projected_pixels_say_whether_they_can_train():
    points = np.array([src.NONE, src.PAINTED, src.HAND, src.CLOUD, src.PREDICTED], np.uint8)
    pixels = src.projected(points)
    assert pixels.tolist() == [src.NONE, src.CLOUD, src.CLOUD, src.CLOUD, src.PREDICTED]
    assert src.trainable(pixels).tolist() == [False, True, True, True, False]


# -- voting by source ------------------------------------------------------------------
class _FakeVis:
    def __init__(self, table):
        self.table = table

    def get(self, name, cloud, camera):
        idx, pix = self.table[name]
        return np.asarray(idx), np.asarray(pix)


def _scene():
    """Three points; a hand-drawn photo sees 0 and 1, two predicted ones see all."""
    from cloudlabeller.core.dataset import Dataset, ImageRecord, PointCloud

    cloud = PointCloud(xyz=np.zeros((3, 3), np.float32))
    cams = {}
    for n, x in (("u.jpg", 5.0), ("p1.jpg", -5.0), ("p2.jpg", 0.0)):
        cams[n] = mock.MagicMock()
        cams[n].position = np.array([x, 0.0, 5.0])
    dataset = Dataset(images=[ImageRecord(i, Path(n), cams[n]) for i, n in enumerate(cams)])
    vis = _FakeVis({"u.jpg": ([0, 1], [[0, 0], [1, 1]]),
                    "p1.jpg": ([0, 1, 2], [[0, 0], [1, 1], [0, 1]]),
                    "p2.jpg": ([0, 1, 2], [[0, 0], [1, 1], [0, 1]])})
    return cloud, dataset, vis


def test_a_better_source_is_never_outvoted_by_a_worse_one():
    from cloudlabeller.transfer.project_to_cloud import VotePass, vote

    cloud, dataset, vis = _scene()
    hand = np.zeros((2, 2), np.int32)
    predicted = np.ones((2, 2), np.int32)
    out = vote(cloud, dataset, [
        VotePass(src.HAND, ["u.jpg"], {"u.jpg": hand}.get),
        VotePass(src.PREDICTED, ["p1.jpg", "p2.jpg"],
                 {"p1.jpg": predicted, "p2.jpg": predicted}.get),
    ], n_classes=2, visibility=vis)
    assert out.labels.tolist() == [0, 0, 1]          # hand-drawn despite 2 against 1
    assert out.source.tolist() == [src.HAND, src.HAND, src.PREDICTED]
    # One hand-drawn view, against two agreeing predicted views.
    assert out.uncertainty[2] < out.uncertainty[0]
    assert np.isfinite(out.uncertainty.astype(np.float32)).all()


# -- the store ------------------------------------------------------------------------
def test_merging_keeps_the_better_source_and_never_touches_paint():
    store = LabelStore()
    store.init_cloud(4)
    store.paint_cloud(np.array([0]), 1)              # a stroke on the cloud
    store.merge_cloud_labels(np.array([-1, 0, 0, -1], np.int32), source=src.HAND,
                             uncertainty=np.full(4, 0.2, np.float16))
    n = store.merge_cloud_labels(np.array([0, 1, 1, 1], np.int32),
                                 source=np.full(4, src.PREDICTED, np.uint8),
                                 uncertainty=np.full(4, 0.1, np.float16))
    assert n == 1                                    # only the point nothing labelled
    assert store.cloud_labels.tolist() == [1, 0, 0, 1]
    source, unc = store.provenance()
    assert source.tolist() == [src.PAINTED, src.HAND, src.HAND, src.PREDICTED]
    assert float(unc[0]) == 0.0 and float(unc[1]) == pytest.approx(0.2, abs=1e-3)

    store.undo()                                     # provenance comes back too
    assert store.cloud_labels.tolist() == [1, 0, 0, -1]
    assert store.provenance()[0].tolist() == [src.PAINTED, src.HAND, src.HAND, src.NONE]


def test_erasing_on_the_cloud_clears_the_source():
    store = LabelStore()
    store.init_cloud(2)
    store.paint_cloud(np.array([0, 1]), 1)
    store.paint_cloud(np.array([1]), -1)
    assert store.provenance()[0].tolist() == [src.PAINTED, src.NONE]


def test_labels_from_before_sources_are_hand_drawn_of_unknown_uncertainty():
    store = LabelStore()
    store.cloud_labels = np.array([2, -1], np.int32)
    source, unc = store.provenance()
    assert source.tolist() == [src.HAND, src.NONE]
    assert np.isnan(unc.astype(np.float32)).all()


def test_sources_are_saved_beside_the_labels(tmp_path):
    store = LabelStore()
    store.init_dense_cloud(3)
    store.merge_cloud_labels(np.array([0, 1, -1], np.int32), dense=True,
                             source=np.array([src.CLOUD, src.PREDICTED, src.NONE], np.uint8),
                             uncertainty=np.array([0.3, 0.6, np.nan], np.float16))
    write_label_files(store, tmp_path)
    assert (tmp_path / "dense.source.npy").exists()
    back = LabelStore()
    read_label_files(back, tmp_path)
    assert back.dense_cloud_labels.tolist() == [0, 1, -1]
    assert back.provenance(dense=True)[0].tolist() == [src.CLOUD, src.PREDICTED, src.NONE]
    assert float(back.provenance(dense=True)[1][1]) == pytest.approx(0.6, abs=1e-3)


def test_deleting_a_class_clears_its_points_sources():
    store = LabelStore()
    store.init_cloud(3)
    store.merge_cloud_labels(np.array([0, 1, 2], np.int32))
    store.remap_after_delete(1)
    assert store.cloud_labels.tolist() == [0, -1, 1]
    assert store.provenance()[0].tolist() == [src.HAND, src.NONE, src.HAND]


def test_projected_layers_carry_source_and_uncertainty():
    from cloudlabeller.transfer.project_to_image import cloud_layers_to_image, cloud_to_image

    cloud, cam = _camera_scene()
    labels = np.array([0, 1, -1], np.int32)
    source = np.array([src.PAINTED, src.PREDICTED, src.NONE], np.uint8)
    unc = np.array([0.0, 0.5, np.nan], np.float16)
    mask, pix_source, pix_unc = cloud_layers_to_image(cloud, labels, source, unc, cam,
                                                      splat_radius=0)
    assert np.array_equal(mask, cloud_to_image(cloud, labels, cam, splat_radius=0))
    assert set(np.unique(pix_source[mask == 0])) == {src.CLOUD}
    assert set(np.unique(pix_source[mask == 1])) == {src.PREDICTED}
    assert (pix_source[mask == -1] == src.NONE).all()
    assert np.allclose(pix_unc[mask == 1].astype(np.float32), 0.5, atol=1e-3)


def _camera_scene():
    from cloudlabeller.core.dataset import Camera, PointCloud

    K = np.array([[50.0, 0, 32], [0, 50.0, 32], [0, 0, 1]])
    cam = Camera(K=K, R=np.eye(3), t=np.zeros(3), width=64, height=64)
    xyz = np.array([[-0.2, 0.0, 5.0], [0.2, 0.0, 5.0], [0.0, 0.2, 5.0]], np.float32)
    return PointCloud(xyz=xyz), cam
