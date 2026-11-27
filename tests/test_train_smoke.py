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

"""End-to-end U-Net smoke test with real TensorFlow: build → train → save →
load → predict. Slow (TF import + fit), so it only runs when opted in:

    CLOUDLABELLER_TF_TESTS=1 .venv/Scripts/python -m pytest tests/test_train_smoke.py -q
"""

import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("CLOUDLABELLER_TF_TESTS"),
    reason="TF smoke test is opt-in: set CLOUDLABELLER_TF_TESTS=1")


def test_train_save_load_predict(tmp_path):
    import cloudlabeller  # noqa: F401  (MSVC-runtime preload before TF)
    from cloudlabeller.ml import inference, model_store, trainer
    from cloudlabeller.ml.dataset_builder import build_tile_set
    from cloudlabeller.ml.model_spec import ModelSpec
    from cloudlabeller.ml.tile_dataset import make_dataset, n_batches
    from cloudlabeller.ml.unet import build_model_from_spec

    rng = np.random.default_rng(0)
    spec = ModelSpec(tile_size=64, shrink=2, overlap=0.25, min_labelled=0.0,
                     channels=4, blocks=2, filter_size=3)
    n = spec.net_size                           # 32
    assert spec.validate() is None

    # Two-class toy scene: left half class 0, right half class 1, some unlabelled.
    def make(name):
        img = rng.integers(0, 255, (64, 96, 3), dtype=np.uint8)
        img[:, 48:] //= 3                       # visibly darker right half
        mask = np.zeros((64, 96), np.int32)
        mask[:, 48:] = 1
        mask[:4, :] = -1                        # unlabelled strip (ignored)
        return img, mask

    # Mixed weights + batch size > 1 + validation: the exact configuration that
    # crashed with flat (N,) sample weights ("Dimensions must be equal").
    data = {k: make(k) for k in ("a", "b", "c", "d")}
    ts = build_tile_set(list(data), lambda k: data[k][0], lambda k: data[k][1], spec,
                        projected_names={"d"}, projected_weight=0.5)
    assert {smp.image.shape for smp in ts.samples} == {(n, n, 3)}
    train_set, val_set = ts.split(val_fraction=0.25)
    model = build_model_from_spec(spec, n_classes=2)

    msgs = []
    per_epoch = n_batches(len(train_set.samples), 1, 2)
    model, metrics, outcome = trainer.train(
        model, train_set, val_set, steps=per_epoch + 1, batch_size=2, augment_rolls=1,
        progress=lambda f, m="": msgs.append((f, m)))
    assert len(metrics["loss"]) == 2                   # one step past an epoch: 2 epochs
    assert outcome["epochs"] == outcome["epochs_run"] == 2
    assert outcome["monitor"] == "val_loss" and "val_loss" in metrics
    assert outcome["best_epoch"] == int(np.argmin(metrics["val_loss"]))
    assert any("epoch 2/2" in m for _, m in msgs)
    # The weights kept are the best epoch's: evaluated again, they score its val_loss.
    again = model.evaluate(make_dataset(val_set, 2, 0, shuffle=False), verbose=0,
                           return_dict=True)
    assert again["loss"] == pytest.approx(metrics["val_loss"][outcome["best_epoch"]], rel=1e-3)

    model_store.save_model(model, tmp_path, "smoke", spec=spec, n_classes=2,
                           input_size=(n, n), history=metrics)
    loaded, manifest = model_store.load_model(tmp_path, "smoke")
    assert manifest["input_size"] == [n, n] and len(manifest["history"]["loss"]) == 2

    labels, unsure = inference.predict(loaded, data["a"][0], spec, 2)
    assert labels.shape == (32, 48)              # the photo / shrink
    assert set(np.unique(labels)) <= {0, 1}
    assert unsure.shape == labels.shape and 0 <= float(unsure.min()) <= float(unsure.max()) <= 1

    # predict_all writes one npz per photo: labels and Uncertainty.
    out = inference.predict_all(loaded, ["a", "b"], lambda k: data[k][0],
                                tmp_path / "pred", spec, 2, "smoke")
    assert sorted(out) == ["a", "b"] and 0 <= out["a"] <= 1
    with np.load(tmp_path / "pred" / "a.npz") as saved:
        assert saved["labels"].shape == (32, 48) and str(saved["model"]) == "smoke"


def test_train_and_predict_cli_subprocess(tmp_path):
    """The real app path: TF runs in child processes (train_cli / predict_cli),
    because in-process TF deadlocked the GIL against Qt."""
    import json
    import subprocess
    import sys

    from PIL import Image

    from cloudlabeller.core.project import Project
    from cloudlabeller.ml.model_spec import ModelSpec

    proj = Project.create(tmp_path / "p.clproj")
    proj.schema.add(name="rock")
    proj.schema.add(name="soil")
    spec = ModelSpec(tile_size=64, shrink=2, overlap=0.25, min_labelled=0.0,
                     channels=4, blocks=2, filter_size=3)
    proj.set_model_spec(spec)
    rng = np.random.default_rng(0)
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        arr = rng.integers(0, 255, (64, 96, 3), dtype=np.uint8)
        Image.fromarray(arr).save(proj.survey("s1").images_dir / name)
    for name in ("a.jpg", "b.jpg"):                     # green images
        mask = np.zeros((64, 96), np.int32)
        mask[:, 48:] = 1
        proj.survey("s1").labels.image_masks[name] = mask
        proj.survey("s1").labels.mark_user_labeled(name)
    proj.survey("s1").load_products()
    proj.save()

    def run(module, *args):
        r = subprocess.run(
            [sys.executable, "-m", module, str(proj.root), *args],
            capture_output=True, text=True, timeout=420)
        assert r.returncode == 0, f"{module} failed:\n{r.stdout}\n{r.stderr}"
        result = [l for l in r.stdout.splitlines() if l.startswith("RESULT ")]
        assert result, f"no RESULT line:\n{r.stdout}"
        assert any(l.startswith("PROGRESS ") for l in r.stdout.splitlines())
        return json.loads(result[-1][7:])

    out = run("cloudlabeller.ml.train_cli", "--name", "cli smoke",
              "--steps", "3", "--batch-size", "1", "--augment-rolls", "1",
              "--val-fraction", "0.34")
    assert out["name"] == "cli smoke"
    assert out["epochs"] == out["epochs_run"] == -(-3 // out["steps_per_epoch"])
    first = out["epochs_run"]
    assert 1 <= out["best_epoch"] <= first
    assert (proj.ml_dir / "models" / "cli smoke" / "model.weights.h5").exists()

    # Resume: load the saved model and train it for more epochs. Architecture
    # comes from the manifest, so this must work even with different current
    # Model Settings.
    proj.set_model_spec(ModelSpec(tile_size=128, shrink=4, channels=8, blocks=1,
                                  filter_size=3))
    proj.save_manifest()
    out = run("cloudlabeller.ml.train_cli", "--name", "cli smoke",
              "--epochs", "1", "--batch-size", "1", "--augment-rolls", "0",
              "--val-fraction", "0", "--resume")
    assert out["resumed"] is True
    assert out["epochs_run"] == 1
    manifest = json.loads((proj.ml_dir / "models" / "cli smoke" /
                           "manifest.json").read_text())
    assert manifest["spec"]["channels"] == 4            # saved arch, not current
    assert manifest["input_size"] == [32, 32] and manifest["surveys"] == ["s1"]
    assert len(manifest["history"]["loss"]) == first + 1   # then 1 more
    assert manifest["monitor"] == "loss"                   # nothing held out
    assert manifest["best_epoch"] == first                 # the resumed run's only epoch

    out = run("cloudlabeller.ml.predict_cli", "--model", "cli smoke", "--surveys", "s1")
    assert [(p["survey"], p["name"]) for p in out["predicted"]] == [("s1", "c.jpg")]
    with np.load(proj.survey("s1").predictions_dir / "c.jpg.npz") as pred:
        assert pred["labels"].shape == (32, 48)         # the photo / shrink
        assert set(np.unique(pred["labels"])) <= {0, 1}


def test_accuracy_skips_unlabelled_pixels():
    """Regression: Keras' "accuracy" counted every unlabelled pixel as a miss,
    so a FIDA run levelled off at 0.64 (its labelled share) while the loss,
    which skips them, fell to 0.007."""
    import cloudlabeller  # noqa: F401  (MSVC-runtime preload before TF)
    import tensorflow as tf

    from cloudlabeller.ml.unet import LabelledAccuracy

    y_true = np.array([[[0, 1, -1, -1]], [[1, 1, 0, -1]]], np.int32)       # (2, 1, 4)
    right = tf.one_hot(np.maximum(y_true, 0), 2)                          # right where labelled
    m = LabelledAccuracy()
    m.update_state(y_true, right)
    assert m.name == "accuracy" and float(m.result()) == 1.0
    plain = tf.keras.metrics.SparseCategoricalAccuracy()
    plain.update_state(y_true, right)
    assert float(plain.result()) < 1.0                     # what it reported before

    # one miss in the second sample; its weight (B, 1, 1) halves it
    pred = right.numpy()
    pred[1, 0, 0] = [1.0, 0.0]
    m.reset_state()
    m.update_state(y_true, pred, sample_weight=np.array([1.0, 0.5], np.float32)[:, None, None])
    assert abs(float(m.result()) - (2 + 0.5 * 2) / (2 + 0.5 * 3)) < 1e-6

    m.reset_state()
    m.update_state(np.full((1, 1, 2), -1, np.int32), right[:1, :, :2])
    assert float(m.result()) == 0.0                        # nothing labelled: no NaN
