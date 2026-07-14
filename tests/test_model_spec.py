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

"""Tests for the U-Net model spec: aspect snapping and pow-2 training sizes."""

import ast
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from cloudlabeller.ml.dataset_builder import build_training_set
from cloudlabeller.ml.model_spec import ModelSpec, snap_aspect


class TestSnapAspect:
    @pytest.mark.parametrize("w, h, expected", [
        (5472, 3648, (3, 2)),      # typical drone sensor
        (1920, 1080, (16, 9)),
        (4000, 3000, (4, 3)),
        (1280, 1024, (5, 4)),
        (3648, 5472, (2, 3)),      # portrait
        (2000, 1000, (2, 1)),
    ])
    def test_common_ratios(self, w, h, expected):
        assert snap_aspect(w, h) == expected

    def test_slightly_off_ratio_snaps(self):
        # A cropped 3:2 (a few pixels shaved off) still snaps to 3:2.
        assert snap_aspect(5470, 3648) == (3, 2)

    def test_extreme_ratio_stays_small(self):
        aw, ah = snap_aspect(10000, 300)   # ~33:1 panorama
        assert aw <= 42 and ah <= 21       # bounded terms, not 100:3


class TestTrainingSize:
    def test_size_is_pow2_multiple_of_aspect(self):
        spec = ModelSpec(resolution_divisor=10)
        tw, th = spec.training_size(5472, 3648)
        k = spec.pow2_scale(5472, 3648)
        assert (tw, th) == (3 * 2 ** k, 2 * 2 ** k)
        assert tw % 2 ** spec.blocks == 0 and th % 2 ** spec.blocks == 0

    def test_approximates_requested_divisor(self):
        spec = ModelSpec(resolution_divisor=10)
        tw, th = spec.training_size(5472, 3648)     # target ~547x365
        assert (tw, th) == (768, 512)               # 3:2 x 2^8 (closest pow-2 area)

    def test_divisor_one_keeps_near_full_resolution(self):
        spec = ModelSpec(resolution_divisor=1)
        tw, th = spec.training_size(1920, 1080)
        assert (tw, th) == (2048, 1152)             # 16:9 x 2^7 ~ full res

    def test_validate_flags_shallow_pow2(self):
        # Divisor so large that k drops below the block count.
        spec = ModelSpec(resolution_divisor=100, blocks=4)
        assert spec.pow2_scale(1920, 1080) < 4
        assert spec.validate_size(1920, 1080) is not None

    def test_validate_ok_for_default(self):
        spec = ModelSpec()
        assert spec.validate_size(5472, 3648) is None


class TestBuildTrainingSetResize:
    def test_resizes_to_target_and_preserves_ids(self):
        img = (np.random.rand(30, 45, 3) * 255).astype(np.uint8)
        mask = np.full((30, 45), -1, np.int32)
        mask[:15, :20] = 2
        ts = build_training_set(
            ["a"], lambda n: img, lambda n: mask, target_size=(48, 32))
        s = ts.samples[0]
        assert s.image.shape == (32, 48, 3)
        assert s.mask.shape == (32, 48)
        assert s.mask.dtype == np.int32
        assert set(np.unique(s.mask)) == {-1, 2}

    def test_no_target_keeps_original(self):
        img = np.zeros((10, 20, 3), np.uint8)
        mask = np.zeros((10, 20), np.int32)
        ts = build_training_set(["a"], lambda n: img, lambda n: mask)
        assert ts.samples[0].image.shape == (10, 20, 3)


UNET_PY = Path(__file__).resolve().parents[1] / "cloudlabeller" / "ml" / "unet.py"


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
        spec = ModelSpec(resolution_divisor=10, channels=32, blocks=5,
                         dropout=0.25, filter_size=3)
        with mock.patch.dict(sys.modules, {"tensorflow": mock.MagicMock()}):
            sys.modules.pop("cloudlabeller.ml.unet", None)
            import cloudlabeller.ml.unet as unet

            with mock.patch.object(unet, "build_model") as bm:
                unet.build_model_from_spec(spec, n_classes=7,
                                           source_resolution=(5472, 3648))
            sys.modules.pop("cloudlabeller.ml.unet", None)

        expected_size = spec.training_size(5472, 3648)
        bm.assert_called_once_with(
            7, expected_size[0], expected_size[1], in_channels=3,
            channels=32, blocks=5, dropout_prob=0.25, filter_size=3)
