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
    from cloudlabeller.ml.dataset_builder import build_training_set
    from cloudlabeller.ml.model_spec import ModelSpec
    from cloudlabeller.ml.unet import build_model_from_spec

    rng = np.random.default_rng(0)
    source_res = (96, 64)                       # pretend photos are 96x64
    spec = ModelSpec(resolution_divisor=2, channels=4, blocks=2, filter_size=3)
    tw, th = spec.training_size(*source_res)
    assert spec.validate_size(*source_res) is None

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
    data = {n: make(n) for n in ("a", "b", "c", "d")}
    ts = build_training_set(list(data), lambda n: data[n][0],
                            lambda n: data[n][1], target_size=(tw, th),
                            projected_names={"d"}, projected_weight=0.5)
    train_set, val_set = ts.split(val_fraction=0.25)
    model = build_model_from_spec(spec, n_classes=2, source_resolution=source_res)

    msgs = []
    model, metrics = trainer.train(
        model, train_set, val_set, epochs=2, batch_size=2, augment_rolls=1,
        progress=lambda f, m="": msgs.append((f, m)))
    assert len(metrics["loss"]) == 2
    assert "val_loss" in metrics
    assert any("epoch 2/2" in m for _, m in msgs)

    model_store.save_model(model, tmp_path, "smoke", spec=spec, n_classes=2,
                           input_size=(tw, th))
    loaded, manifest = model_store.load_model(tmp_path, "smoke")
    assert manifest["input_size"] == [tw, th]

    pred = inference.predict(loaded, data["a"][0])
    assert pred.shape == (th, tw)
    assert pred.dtype == np.int32
    assert set(np.unique(pred)) <= {0, 1}

    # predict_all writes one npy per image at model resolution.
    out = inference.predict_all(loaded, ["a", "b"], lambda n: data[n][0],
                                tmp_path / "pred")
    assert sorted(out) == ["a", "b"]
    saved = np.load(out["a"])
    assert saved.shape == (th, tw)


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
    spec = ModelSpec(resolution_divisor=2, channels=4, blocks=2, filter_size=3)
    proj.set_model_spec(spec)
    tw, th = spec.training_size(96, 64)                 # the model's input size
    rng = np.random.default_rng(0)
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        arr = rng.integers(0, 255, (64, 96, 3), dtype=np.uint8)
        Image.fromarray(arr).save(proj.images_dir / name)
    for name in ("a.jpg", "b.jpg"):                     # green images
        mask = np.zeros((64, 96), np.int32)
        mask[:, 48:] = 1
        proj.labels.image_masks[name] = mask
        proj.labels.mark_user_labeled(name)
    proj.load_products()
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
              "--epochs", "2", "--batch-size", "1", "--augment-rolls", "1",
              "--val-fraction", "0.34")
    assert out["name"] == "cli smoke"
    assert out["epochs_run"] == 2
    assert (proj.ml_dir / "models" / "cli smoke" / "model.weights.h5").exists()

    # Resume: load the saved model and train it for more epochs. Architecture
    # comes from the manifest, so this must work even with different current
    # Model Settings.
    proj.set_model_spec(ModelSpec(resolution_divisor=4, channels=8, blocks=1,
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
    assert manifest["input_size"] == [tw, th]

    out = run("cloudlabeller.ml.predict_cli", "--model", "cli smoke")
    assert out["predicted"] == ["c.jpg"]                # red only; greens kept
    pred = np.load(proj.predictions_dir / "c.jpg.npy")
    assert pred.shape == (th, tw)                       # the MODEL's input size
    assert set(np.unique(pred)) <= {0, 1}
