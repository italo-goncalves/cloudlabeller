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

"""Tests for the ML pipeline plumbing: model store, prediction masks, statuses.

TensorFlow-dependent paths are exercised with mocks; the real training loop has
a separate opt-in smoke test (see test_train_smoke.py).
"""

import json
import sys
from unittest import mock

import numpy as np
import pytest

from cloudlabeller.core.labels import LabelStore
from cloudlabeller.core.project import Project
from cloudlabeller.ml import model_store
from cloudlabeller.ml.model_spec import ModelSpec


class TestModelStore:
    def _fake_model(self):
        m = mock.MagicMock()
        m.save_weights.side_effect = lambda p: open(p, "wb").write(b"w")
        return m

    def test_save_then_list_roundtrip(self, tmp_path):
        spec = ModelSpec(channels=32, blocks=3)
        model_store.save_model(
            self._fake_model(), tmp_path, "rock face", spec=spec, n_classes=4,
            input_size=(768, 512), class_names=["a", "b", "c", "d"])
        models = model_store.list_models(tmp_path)
        assert len(models) == 1
        m = models[0]
        assert m["name"] == "rock face"
        assert m["n_classes"] == 4
        assert m["input_size"] == [768, 512]
        assert ModelSpec.from_dict(m["spec"]) == spec
        assert m["class_names"] == ["a", "b", "c", "d"]

    def test_sanitize_name(self):
        assert model_store.sanitize_name('bad/na:me?') == "bad_na_me_"
        assert model_store.sanitize_name("  ") == "unet"

    def test_overwrite_same_name(self, tmp_path):
        for n_classes in (2, 5):
            model_store.save_model(
                self._fake_model(), tmp_path, "unet", spec=ModelSpec(),
                n_classes=n_classes, input_size=(96, 64))
        models = model_store.list_models(tmp_path)
        assert len(models) == 1
        assert models[0]["n_classes"] == 5

    def test_list_skips_incomplete(self, tmp_path):
        d = tmp_path / "models" / "broken"
        d.mkdir(parents=True)
        (d / model_store.MANIFEST).write_text("{}")   # manifest but no weights
        assert model_store.list_models(tmp_path) == []

    def test_load_rebuilds_from_manifest(self, tmp_path):
        spec = ModelSpec(channels=8, blocks=2, dropout=0.2, filter_size=3)
        model_store.save_model(
            self._fake_model(), tmp_path, "m", spec=spec, n_classes=3,
            input_size=(96, 64))
        fake_unet = mock.MagicMock()
        with mock.patch.dict(sys.modules,
                             {"cloudlabeller.ml.unet": fake_unet}):
            model, manifest = model_store.load_model(tmp_path, "m")
        fake_unet.build_model.assert_called_once_with(
            3, 96, 64, in_channels=3,
            channels=8, blocks=2, dropout_prob=0.2, filter_size=3)
        fake_unet.build_model.return_value.load_weights.assert_called_once()
        assert manifest["name"] == "m"


class TestMlStatus:
    def test_ml_status_never_overrides_user(self):
        ls = LabelStore()
        ls.mark_user_labeled("a")
        ls.mark_ml_labeled("a")
        assert ls.status_of("a") == "user"

    def test_cloud_transfer_supersedes_ml(self):
        ls = LabelStore()
        ls.mark_ml_labeled("a")
        assert ls.status_of("a") == "ml"
        ls.mark_auto_labeled("a")               # newer cloud render wins
        assert ls.status_of("a") == "auto"

    def test_predict_targets_exclude_green(self):
        ls = LabelStore()
        ls.mark_user_labeled("green")
        ls.mark_auto_labeled("yellow")
        names = ["green", "yellow", "red"]
        targets = [n for n in names if ls.status_of(n) != "user"]
        assert targets == ["yellow", "red"]


class TestToArrays:
    def test_weights_have_broadcast_axes(self):
        """Regression: Keras multiplies sample weights against the per-pixel
        loss map (N,H,W); flat (N,) weights crash for batch size > 1."""
        from cloudlabeller.ml.dataset_builder import Sample, TrainingSet
        from cloudlabeller.ml.trainer import to_arrays

        ts = TrainingSet([
            Sample("a", np.zeros((8, 12, 3), np.uint8), np.zeros((8, 12), np.int32), 1.0),
            Sample("b", np.zeros((8, 12, 3), np.uint8), np.zeros((8, 12), np.int32), 0.5),
        ])
        x, y, w = to_arrays(ts)
        assert x.shape == (2, 8, 12, 3) and x.dtype == np.float32
        assert y.shape == (2, 8, 12) and y.dtype == np.int32
        assert w.shape == (2, 1, 1)                  # broadcasts over (N,H,W)
        assert w[0, 0, 0] == 1.0 and w[1, 0, 0] == 0.5


class TestMaskSource:
    def test_stored_masks_win_and_render_is_lazy(self):
        from cloudlabeller.transfer.project_to_cloud import MaskSource

        stored = {"user.jpg": np.ones((2, 2), np.int32)}
        calls = []

        def render(name):
            calls.append(name)
            return np.zeros((2, 2), np.int32)

        src = MaskSource(stored, {"auto.jpg"}, render)
        assert (src.get("user.jpg") == 1).all()
        assert calls == []                        # stored mask: no render
        assert (src.get("auto.jpg") == 0).all()
        assert calls == ["auto.jpg"]
        assert src.get("unknown.jpg") is None
        # No caching (a full-res mask is ~84 MB): rendered again on re-access.
        src.get("auto.jpg")
        assert calls == ["auto.jpg", "auto.jpg"]

    def test_images_to_cloud_fetches_each_mask_once(self):
        """The transfer must read each image's mask exactly once per call, or a
        lazy MaskSource would re-render (slow) instead of staying memory-flat."""
        from pathlib import Path

        from cloudlabeller.core.dataset import Dataset, ImageRecord, PointCloud
        from cloudlabeller.transfer import images_to_cloud

        class CountingSource:
            def __init__(self):
                self.calls = []

            def get(self, name):
                self.calls.append(name)
                return np.full((4, 4), -1, np.int32)   # unlabelled -> skipped

        cam = mock.MagicMock()                          # never projected (mask empty)
        dataset = Dataset(images=[ImageRecord(0, Path("a.jpg"), cam),
                                  ImageRecord(1, Path("b.jpg"), cam)])
        cloud = PointCloud(xyz=np.zeros((5, 3), np.float32))
        src = CountingSource()
        out = images_to_cloud(cloud, dataset, src, n_classes=2)
        assert sorted(src.calls) == ["a.jpg", "b.jpg"]  # exactly once each
        assert (out == -1).all()


class TestPriorityVoting:
    """User-drawn (hard) labels must outrank auto/ML votes wherever they reach —
    one labelled image surrounded by predictions must not be drowned out."""

    class _FakeVis:
        """Injectable visibility: name -> (point indices, pixel coords)."""

        def __init__(self, table):
            self.table = table

        def get(self, name, cloud, camera):
            idx, pix = self.table[name]
            return np.asarray(idx), np.asarray(pix)

    def _scene(self):
        from pathlib import Path

        from cloudlabeller.core.dataset import Dataset, ImageRecord, PointCloud

        cloud = PointCloud(xyz=np.zeros((3, 3), np.float32))
        cam = mock.MagicMock()
        dataset = Dataset(images=[ImageRecord(i, Path(n), cam) for i, n
                                  in enumerate(["u.jpg", "a1.jpg", "a2.jpg"])])
        user = np.zeros((2, 2), np.int32)              # class 0 everywhere
        auto = np.ones((2, 2), np.int32)               # class 1 everywhere
        masks = {"u.jpg": user, "a1.jpg": auto, "a2.jpg": auto}
        vis = self._FakeVis({
            "u.jpg": ([0, 1], [[0, 0], [1, 1]]),       # user sees points 0, 1
            "a1.jpg": ([0, 1, 2], [[0, 0], [1, 1], [0, 1]]),
            "a2.jpg": ([0, 1, 2], [[0, 0], [1, 1], [0, 1]]),
        })
        return cloud, dataset, masks, vis

    def test_hard_labels_win_where_they_reach(self):
        from cloudlabeller.transfer import images_to_cloud

        cloud, dataset, masks, vis = self._scene()
        out = images_to_cloud(cloud, dataset, masks, n_classes=2,
                              visibility=vis, priority_names={"u.jpg"})
        assert out[0] == 0 and out[1] == 0     # user's class despite 2-vs-1 votes
        assert out[2] == 1                      # autos still fill uncovered points

    def test_without_priority_majority_wins(self):
        from cloudlabeller.transfer import images_to_cloud

        cloud, dataset, masks, vis = self._scene()
        out = images_to_cloud(cloud, dataset, masks, n_classes=2, visibility=vis)
        assert out[0] == 1 and out[1] == 1     # drowned out: the old behaviour


class TestPredictionMask:
    def test_upscales_to_source_resolution(self, tmp_path):
        proj = Project.create(tmp_path / "p.clproj")
        proj.settings["source_resolution"] = [40, 20]
        small = np.full((10, 20), -1, np.int32)   # model-res (h=10, w=20)
        small[:5, :10] = 1
        np.save(proj.predictions_dir / "img.npy", small)
        mask = proj.prediction_mask("img")
        assert mask.shape == (20, 40)
        assert mask.dtype == np.int32
        assert set(np.unique(mask)) == {-1, 1}
        assert mask[0, 0] == 1 and mask[-1, -1] == -1

    def test_missing_prediction_is_none(self, tmp_path):
        proj = Project.create(tmp_path / "p.clproj")
        assert proj.prediction_mask("nope") is None

    def test_render_auto_mask_prefers_prediction_for_ml_status(self, tmp_path):
        proj = Project.create(tmp_path / "p.clproj")
        proj.settings["source_resolution"] = [20, 10]
        pred = np.full((10, 20), 2, np.int32)
        np.save(proj.predictions_dir / "img.npy", pred)
        proj.labels.mark_ml_labeled("img")
        mask = proj.render_auto_mask("img")
        assert mask is not None and (mask == 2).all()

    def test_status_ml_survives_manifest_roundtrip(self, tmp_path):
        proj = Project.create(tmp_path / "p.clproj")
        proj.labels.mark_ml_labeled("img")
        proj.save()
        reopened = Project.open(proj.root)
        assert reopened.labels.status_of("img") == "ml"


class TestMinorImprovements:
    def test_delete_model(self, tmp_path):
        m = mock.MagicMock()
        m.save_weights.side_effect = lambda p: open(p, "wb").write(b"w")
        model_store.save_model(m, tmp_path, "doomed", spec=ModelSpec(),
                               n_classes=2, input_size=(96, 64))
        assert model_store.list_models(tmp_path)
        model_store.delete_model(tmp_path, "doomed")
        assert model_store.list_models(tmp_path) == []
        model_store.delete_model(tmp_path, "doomed")   # idempotent

    def test_delete_model_refuses_non_model_dirs(self, tmp_path):
        stray = tmp_path / "models" / "keepme"
        stray.mkdir(parents=True)
        (stray / "data.txt").write_text("x")           # no manifest -> not a model
        model_store.delete_model(tmp_path, "keepme")
        assert (stray / "data.txt").exists()

    def test_dim_other_classes(self):
        from cloudlabeller.ui.viewer3d import dim_other_classes

        colors = np.full((4, 3), 200, np.uint8)
        labels = np.array([0, 1, 0, -1], np.int32)
        out = dim_other_classes(colors, labels, class_id=0)
        assert (out[0] == 200).all() and (out[2] == 200).all()   # isolated: full
        assert (out[1] == 30).all() and (out[3] == 30).all()     # others: dimmed
        assert (colors == 200).all()                             # input untouched
        # mismatched labels -> unchanged
        assert dim_other_classes(colors, np.array([0]), 0) is colors

    def test_format_count(self):
        from cloudlabeller.ui.label_panel import format_count

        assert format_count(500, 1000) == "500 · 50%"
        assert format_count(12_300, 100_000) == "12.3k · 12%"
        assert format_count(12_255_000, 22_751_916) == "12.3M · 54%"
        assert format_count(0, 100) == "0 · 0%"
        assert format_count(5, 0) == ""
