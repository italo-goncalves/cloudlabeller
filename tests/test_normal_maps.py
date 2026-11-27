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


"""Normal maps (``transfer.normal_map``) and the experimental normals input
(``ModelSpec.normals``). The augmentation tests need TensorFlow
(``CLOUDLABELLER_TF_TESTS=1``)."""

import os
from types import SimpleNamespace

import numpy as np
import pytest

from cloudlabeller.ml.model_spec import ModelSpec
from cloudlabeller.transfer.normal_map import NO_NORMAL, encode_normals, normal_map

tf_only = pytest.mark.skipif(not os.environ.get("CLOUDLABELLER_TF_TESTS"),
                             reason="TF tests are opt-in: set CLOUDLABELLER_TF_TESTS=1")


def _decode(enc):
    return enc.astype(np.float32) / 127.5 - 1.0


def test_a_normal_map_holds_rotated_normals_where_points_are_seen():
    # A camera turned 90 degrees about z: world +x is camera +y (down the photo).
    R = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], float).T
    normals = np.array([[2.0, 0, 0], [0, 0, -1]])          # not unit: normalised
    pix = np.array([[1, 1], [8, 6]])                       # (col, row)
    nmap = normal_map(normals, pix, 10, 8, R, splat=1)
    assert nmap.shape == (8, 10, 3) and nmap.dtype == np.uint8
    np.testing.assert_allclose(_decode(nmap[1, 1]), R @ [1, 0, 0], atol=0.01)
    np.testing.assert_allclose(_decode(nmap[0, 2]), R @ [1, 0, 0], atol=0.01)   # splat
    np.testing.assert_allclose(_decode(nmap[6, 8]), [0, 0, -1], atol=0.01)
    assert (nmap[4, 4] == NO_NORMAL).all() and (nmap[7, 0] == NO_NORMAL).all()
    world = normal_map(normals, pix, 10, 8)                # no rotation: as given
    np.testing.assert_allclose(_decode(world[1, 1]), [1, 0, 0], atol=0.01)
    assert encode_normals(np.zeros(3)).tolist() == [NO_NORMAL] * 3
    assert normal_map(np.zeros((0, 3)), np.zeros((0, 2), int), 4, 3).shape == (3, 4, 3)


def test_the_normals_option_adds_three_channels_and_reads_back():
    assert ModelSpec().in_channels == 3 and ModelSpec().normals is False
    spec = ModelSpec.from_dict(ModelSpec(normals=True).to_dict())
    assert spec.normals is True and spec.in_channels == 6
    assert ModelSpec.from_dict({"tile_size": 512}).normals is False     # older settings


def test_a_photo_gets_normals_only_when_the_survey_has_them(tmp_path):
    from PIL import Image

    from cloudlabeller.ml.inputs import can_have_normals, load_input

    path = tmp_path / "a.png"
    Image.new("RGB", (6, 4), (10, 20, 30)).save(path)
    nmap = np.full((4, 6, 3), 7, np.uint8)
    record = SimpleNamespace(camera=object())
    survey = SimpleNamespace(normal_map=lambda name, frame: nmap if frame == "camera" else None,
                             has_normals=lambda: True,
                             dataset=SimpleNamespace(image_by_name=lambda n: record))
    assert load_input(survey, "a", path, normals=False).shape == (4, 6, 3)
    both = load_input(survey, "a", path, normals=True)
    assert both.shape == (4, 6, 6) and (both[..., 3:] == 7).all() and (both[0, 0, :3] == [10, 20, 30]).all()
    assert can_have_normals(survey, "a")
    survey.normal_map = lambda name, frame: None
    assert load_input(survey, "a", path, normals=True) is None
    record.camera = None
    assert not can_have_normals(survey, "a")


@tf_only
def test_with_rgb_alone_the_augmentation_is_unchanged():
    import tensorflow as tf

    from cloudlabeller.ml.tile_dataset import augment

    rng = np.random.default_rng(0)
    image = rng.random((8, 12, 3), np.float32)
    six = np.concatenate([image, rng.random((8, 12, 3), np.float32)], axis=2)
    mask = tf.constant(rng.integers(-1, 3, (8, 12)).astype(np.int32))
    for window in range(6):
        a = augment(tf.constant(image), mask, window, 3, 0.2)
        b = augment(tf.constant(six), mask, window, 3, 0.2)
        np.testing.assert_allclose(a[0].numpy(), b[0].numpy()[..., :3], rtol=1e-6)
        np.testing.assert_array_equal(a[1].numpy(), b[1].numpy())


@tf_only
def test_the_mirrored_half_turns_the_normals_x_around_and_keeps_them_out_of_the_gamma():
    import tensorflow as tf

    from cloudlabeller.ml.tile_dataset import augment

    width, rolls = 12, 3
    image = np.zeros((4, width, 6), np.float32)
    image[..., :3] = 0.25
    image[..., 3] = 0.9                    # every normal points right (+x)
    image[..., 4] = 0.3
    image[..., 5] = 0.6
    x, _ = augment(tf.constant(image), tf.constant(np.zeros((4, width), np.int32)), 0, rolls, 0.4)
    x = x.numpy()
    shift = width // rolls                 # window 0: the first columns are mirrored ones
    np.testing.assert_allclose(x[:, :shift, 3], 0.1, atol=1e-6)     # now pointing left
    np.testing.assert_allclose(x[:, shift:, 3], 0.9, atol=1e-6)
    np.testing.assert_allclose(x[..., 4], 0.3, atol=1e-6)
    np.testing.assert_allclose(x[..., 5], 0.6, atol=1e-6)
    np.testing.assert_allclose(x[..., :3], 0.25 ** np.exp(0.4), rtol=1e-5)


@tf_only
def test_a_six_channel_dataset_and_model_train(tmp_path):
    """The whole path with normals: 6-channel Tiles batch, augment and train."""
    from cloudlabeller.ml import trainer
    from cloudlabeller.ml.dataset_builder import Sample, TrainingSet
    from cloudlabeller.ml.unet import build_model_from_spec

    spec = ModelSpec(tile_size=32, shrink=1, blocks=1, channels=4, filter_size=3, normals=True)
    rng = np.random.default_rng(0)
    ts = TrainingSet([Sample(f"p{k}#0,0", rng.integers(0, 256, (32, 32, 6), dtype=np.uint8),
                             rng.integers(-1, 2, (32, 32)).astype(np.int16)) for k in range(4)])
    model = build_model_from_spec(spec, 2, in_channels=spec.in_channels)
    _, history, outcome = trainer.train(model, ts, None, epochs=1, batch_size=2, augment_rolls=1)
    assert outcome["epochs_run"] == 1 and np.isfinite(history["loss"][0])
