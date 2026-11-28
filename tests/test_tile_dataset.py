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

"""The tf.data training pipeline (``ml.tile_dataset``). Needs TensorFlow, so
it runs only when opted in: ``CLOUDLABELLER_TF_TESTS=1``."""

import os
import tracemalloc

import numpy as np
import pytest

from cloudlabeller.ml.dataset_builder import Sample, TrainingSet
from cloudlabeller.ml.tile_dataset import n_batches, windows

pytestmark = pytest.mark.skipif(
    not os.environ.get("CLOUDLABELLER_TF_TESTS"),
    reason="TF tests are opt-in: set CLOUDLABELLER_TF_TESTS=1")


def _tiles(n, size=16, seed=0):
    rng = np.random.default_rng(seed)
    return TrainingSet([
        Sample(f"p{k}#0,0", rng.integers(0, 256, (size, size, 3), dtype=np.uint8),
               rng.integers(-1, 3, (size, size)).astype(np.int16), 1.0 if k % 2 else 0.5)
        for k in range(n)])


def _window_reference(image, mask, window, rolls, gamma):
    """The augmentation as ``TrainingSet.augment_data`` did it with numpy
    before 0.4.0: ``window + 1`` rolls of the mirror strip."""
    width = image.shape[1]
    tx = np.concatenate([image, np.flip(image, axis=1)], axis=1)
    ty = np.concatenate([mask, np.flip(mask, axis=1)], axis=1)
    for _ in range(window + 1):
        tx = np.roll(tx, width // rolls, axis=1)
        ty = np.roll(ty, width // rolls, axis=1)
    return tx[:, :width] ** gamma, ty[:, :width]


def test_batches_are_model_ready():
    """Regression (kept from to_arrays): Keras multiplies sample weights
    against the per-pixel loss map (B,H,W); flat (B,) weights crash for batch
    size > 1."""
    from cloudlabeller.ml.tile_dataset import make_dataset

    ds = make_dataset(_tiles(5), batch_size=2, augment_rolls=0, shuffle=False)
    batches = list(ds.as_numpy_iterator())
    assert len(batches) == 3 == n_batches(5, 0, 2) == int(ds.cardinality())
    x, y, w = batches[0]
    assert x.shape == (2, 16, 16, 3) and x.dtype == np.float32
    assert 0.0 <= x.min() and x.max() <= 1.0
    assert y.shape == (2, 16, 16) and y.dtype == np.int32
    assert w.shape == (2, 1, 1) and w[0, 0, 0] == 0.5 and w[1, 0, 0] == 1.0
    ts = _tiles(5)
    np.testing.assert_allclose(x[1], ts.samples[1].image / 255.0, atol=1e-6)
    np.testing.assert_array_equal(y[1], ts.samples[1].mask)


def test_augmentation_matches_the_numpy_original():
    import tensorflow as tf

    from cloudlabeller.ml.tile_dataset import augment

    rng = np.random.default_rng(3)
    image = rng.random((8, 20, 3)).astype(np.float32)
    mask = rng.integers(-1, 4, (8, 20)).astype(np.int32)
    rolls = 3
    for window in range(2 * rolls):
        log_gamma = 0.25 - 0.1 * window
        x, y = augment(tf.constant(image), tf.constant(mask), window, rolls, log_gamma)
        ex, ey = _window_reference(image, mask, window, rolls, np.exp(log_gamma))
        np.testing.assert_allclose(x.numpy(), ex, rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(y.numpy(), ey)


def test_every_tile_gives_each_window_once_per_epoch_in_a_new_order():
    from cloudlabeller.ml.tile_dataset import make_dataset

    ts = _tiles(4)
    rolls = 2
    ds = make_dataset(ts, batch_size=3, augment_rolls=rolls, shuffle=True, seed=0)
    assert int(ds.cardinality()) == n_batches(4, rolls, 3) == 6     # 16 windows / 3

    def epoch():
        masks = np.concatenate([y for _, y, _ in ds.as_numpy_iterator()])
        assert len(masks) == 4 * windows(rolls)
        return masks

    first, second = epoch(), epoch()
    expected = sorted(_window_reference(s.mask, s.mask.astype(np.int32), r, rolls, 1)[1].tobytes()
                      for s in ts.samples for r in range(windows(rolls)))
    for masks in (first, second):
        assert sorted(m.tobytes() for m in masks) == expected
    assert not np.array_equal(first, second)          # reshuffled each epoch


def test_augmented_copies_are_never_held_in_memory():
    """Regression: augmenting by copying the whole set and stacking it as
    float32 needed ~18x the Tiles' memory (~125 GB for 376 drone photos)."""
    from cloudlabeller.ml.tile_dataset import make_dataset

    ts = _tiles(300, size=32)
    rolls = 5
    stacked = len(ts.samples) * windows(rolls) * 32 * 32 * (3 + 1) * 4   # float32 x + int32 y
    tracemalloc.start()
    try:
        ds = make_dataset(ts, batch_size=4, augment_rolls=rolls, shuffle=True, seed=1)
        n = sum(len(x) for x, _, _ in ds.as_numpy_iterator())
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert n == 300 * windows(rolls)
    assert peak < stacked / 10, (peak, stacked)


def test_per_pixel_weights_ride_with_the_mask():
    """ADR-0006: a cloud pixel's weight follows it through the mirror/slide
    augmentation, times its sample's weight; samples without weights weigh 1."""
    from cloudlabeller.ml.tile_dataset import make_dataset

    rng = np.random.default_rng(5)
    a = Sample("p0#0,0", rng.integers(0, 256, (8, 8, 3), dtype=np.uint8),
               np.arange(64, dtype=np.int16).reshape(8, 8), 0.5,
               pixel_weights=np.tile(np.arange(8, dtype=np.uint8) * 30, (8, 1)))
    b = Sample("p1#0,0", rng.integers(0, 256, (8, 8, 3), dtype=np.uint8),
               np.zeros((8, 8), np.int16), 1.0)
    ds = make_dataset(TrainingSet([a, b]), batch_size=2, augment_rolls=2, shuffle=False)
    seen = set()
    for _, y, w in ds.as_numpy_iterator():
        assert w.shape == y.shape
        for yk, wk in zip(y.astype(int), w):
            if yk.max() > 0:                         # sample a, some window
                # each pixel kept its own weight: column c of the mask is c + 8 r
                np.testing.assert_allclose(wk, 0.5 * ((yk % 8) * 30) / 255.0, atol=1e-6)
                seen.add("a")
            else:                                    # sample b: no weights, weighs 1
                np.testing.assert_allclose(wk, 1.0)
                seen.add("b")
    assert seen == {"a", "b"}
