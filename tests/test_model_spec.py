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

"""Tests for the U-Net model spec and Tiles (0.4.0 step 4)."""

import ast
import importlib.util
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from cloudlabeller.ml.dataset_builder import Sample, TrainingSet, build_tile_set, photo_of
from cloudlabeller.ml.model_spec import ModelSpec, is_tiled
from cloudlabeller.ml.tiles import cut, feather, grid_for, pad, predict_photo


class TestSpec:
    def test_the_network_sees_the_tile_shrunk(self):
        spec = ModelSpec(tile_size=1024, shrink=4)
        assert spec.net_size == 256 and spec.buffer == int(1024 * 0.35)
        assert spec.validate() is None

    @pytest.mark.parametrize("fields, words", [
        ({"tile_size": 1000, "shrink": 2, "blocks": 4}, "multiple of 16"),
        ({"tile_size": 1022, "shrink": 4}, "multiple of the shrink"),
        ({"shrink": 3}, "shrink factor"),
        ({"overlap": 0.5}, "overlap"),
    ])
    def test_settings_that_cannot_build_a_network_say_why(self, fields, words):
        assert words in ModelSpec(**fields).validate()

    def test_a_029_spec_reads_back_with_tile_defaults(self):
        spec = ModelSpec.from_dict({"resolution_divisor": 10, "channels": 32})
        assert spec.channels == 32 and spec.tile_size == 1024
        assert not is_tiled({"spec": {"resolution_divisor": 10}})
        assert is_tiled({"spec": spec.to_dict()})


class TestTiles:
    def test_the_grid_covers_the_photo_with_overlapping_tiles(self):
        g = grid_for(3000, 4000, 1024, 0.35)
        assert g.buffer == 358 and g.step == 666
        assert g.padded[0] >= 3000 and g.padded[1] >= 4000
        assert g.padded[0] - g.step < 3000          # no row wasted
        assert len(list(g.origins())) == g.rows * g.cols
        small = grid_for(100, 80, 256, 0.35)        # a photo smaller than a Tile
        assert (small.rows, small.cols) == (1, 1) and small.padded == (256, 256)

    def test_padding_keeps_the_photo_and_marks_the_rest(self):
        g = grid_for(50, 70, 32, 0.25)
        mask = np.zeros((50, 70), np.int32)
        padded = pad(mask, g, -1)
        assert padded.shape == g.padded
        assert (padded[:50, :70] == 0).all() and (padded[50:] == -1).all()

    def test_feathered_weights_ramp_over_the_overlap(self):
        w = feather(10, 3)
        assert w[5, 5] == 1.0 and w[0, 0] < w[1, 1] < w[3, 3]
        assert np.allclose(w, w[::-1, ::-1]) and (w > 0).all()

    def test_cutting_skips_tiles_with_too_few_labels(self):
        image = np.full((64, 96, 3), 200, np.uint8)
        mask = np.full((64, 96), -1, np.int32)
        mask[:20, :20] = 1                            # only the top-left corner
        tiles = cut(image, mask, 32, 0.25, 2, 0.05)
        assert [rc for rc, _, _ in tiles] == [(0, 0)]
        (_, img, m), = tiles
        assert img.shape == (16, 16, 3) and m.shape == (16, 16)
        assert set(np.unique(m)) <= {-1, 1}

    def test_a_constant_prediction_blends_to_itself_at_any_photo_size(self):
        probs = np.array([0.1, 0.7, 0.2], np.float32)
        for h, w in ((100, 140), (61, 37), (256, 256)):
            image = np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)
            seen = []

            def model(x, seen=seen):
                seen.append(x.shape)
                return np.broadcast_to(probs, x.shape[:3] + (3,))
            labels, unsure = predict_photo(model, image, 64, 0.35, 2, 3)
            assert labels.shape == (-(-h // 2), -(-w // 2))
            assert (labels == 1).all()
            want = -(probs * np.log(probs)).sum() / np.log(3)
            assert np.allclose(unsure.astype(np.float32), want, atol=1e-3)
            assert all(s[1:3] == (32, 32) for s in seen)

    def test_blending_fades_neighbours_into_each_other(self):
        """Two Tiles disagreeing: the blend changes class inside the overlap,
        not at a Tile's edge."""
        image = np.zeros((32, 52, 3), np.uint8)

        def model(x):
            out = np.zeros(x.shape[:3] + (2,), np.float32)
            for k in range(len(x)):
                out[k, ..., k % 2] = 1.0              # left Tile says 0, right says 1
            return out
        labels, unsure = predict_photo(model, image, 32, 0.4, 1, 2)
        assert labels[0, 0] == 0 and labels[0, -1] == 1
        switch = int(np.argmax(labels[0] == 1))
        assert 20 <= switch <= 32                     # within the overlap [20, 32)
        assert unsure[0, switch].astype(float) > 0.5


class TestTileSet:
    def test_photos_of_any_size_give_tiles_of_one_size(self):
        spec = ModelSpec(tile_size=32, shrink=2, overlap=0.25, min_labelled=0.0, blocks=2)
        photos = {"s1/a": (40, 60), "s2/b": (90, 33)}
        ts = build_tile_set(list(photos),
                            lambda n: np.zeros(photos[n] + (3,), np.uint8),
                            lambda n: np.zeros(photos[n], np.int32), spec,
                            projected_names={"s2/b"})
        assert {s.image.shape for s in ts.samples} == {(16, 16, 3)}
        assert {photo_of(s.name) for s in ts.samples} == set(photos)
        assert {s.weight for s in ts.samples if photo_of(s.name) == "s2/b"} == {0.5}

    def test_validation_holds_out_whole_photos(self):
        samples = [Sample(f"p{p}#{i},0", np.zeros((2, 2, 3)), np.zeros((2, 2), np.int32))
                   for p in range(5) for i in range(4)]
        train, val = TrainingSet(samples).split(0.4, seed=1)
        a = {photo_of(s.name) for s in train.samples}
        b = {photo_of(s.name) for s in val.samples}
        assert a and b and not a & b and len(train.samples) + len(val.samples) == 20


# The imported package's copy (an installed wheel, or the working tree), read
# as text so TensorFlow isn't imported.
UNET_PY = Path(importlib.util.find_spec("cloudlabeller.ml.unet").origin)


def _function_params(source: str, name: str) -> set[str]:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            args = node.args
            return {a.arg for a in
                    args.posonlyargs + args.args + args.kwonlyargs}
    raise AssertionError(f"function {name!r} not found in {UNET_PY}")


class TestSpecBuildModelLink:
    """Guard the seam between the Model Settings dialog and unet.build_model.

    TensorFlow isn't importable in the test env, so the signature check parses
    unet.py, and the forwarding check runs with a mocked ``tensorflow``.
    """

    def test_unet_kwargs_match_u_net_signature(self):
        # Every kwarg the dialog produces must be a real u_net parameter,
        # so **kwargs forwarding in build_model can never hit a TypeError.
        params = _function_params(UNET_PY.read_text(encoding="utf-8"), "u_net")
        kwargs = ModelSpec().unet_kwargs()
        assert set(kwargs) <= params
        # ...and must not collide with what build_model sets itself.
        assert "end_activation" not in kwargs
        assert not {"input_layer", "output_size"} & set(kwargs)

    def test_build_model_from_spec_forwards_spec_values(self):
        spec = ModelSpec(tile_size=1024, shrink=2, channels=32, blocks=5,
                         dropout=0.25, filter_size=3)
        with mock.patch.dict(sys.modules, {"tensorflow": mock.MagicMock()}):
            sys.modules.pop("cloudlabeller.ml.unet", None)
            import cloudlabeller.ml.unet as unet

            with mock.patch.object(unet, "build_model") as bm:
                unet.build_model_from_spec(spec, n_classes=7)
            sys.modules.pop("cloudlabeller.ml.unet", None)

        bm.assert_called_once_with(
            7, 512, 512, in_channels=3,
            channels=32, blocks=5, dropout_prob=0.25, filter_size=3)
