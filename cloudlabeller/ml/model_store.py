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

"""Named model store — save/load trained U-Nets under ``project/ml/models/``.

Layout::

    ml/models/<name>/
      manifest.json     # name, created, spec, n_classes, input size, classes
      model.weights.h5  # Keras weights

Weights-only saving keeps the format robust across TF versions: loading
rebuilds the architecture from the manifest via ``unet.build_model`` and then
restores the weights. Everything except :func:`load_model` / :func:`save_model`
is TensorFlow-free, so dialogs can list models without importing TF.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from cloudlabeller.ml.model_spec import ModelSpec

MANIFEST = "manifest.json"
WEIGHTS = "model.weights.h5"


def sanitize_name(name: str) -> str:
    """Make a user-typed model name safe as a folder name."""
    name = re.sub(r'[<>:"/\\|?*]+', "_", name.strip())
    return name or "unet"


def models_root(ml_dir: str | Path) -> Path:
    return Path(ml_dir) / "models"


def model_dir(ml_dir: str | Path, name: str) -> Path:
    return models_root(ml_dir) / sanitize_name(name)


def list_models(ml_dir: str | Path) -> list[dict]:
    """Manifests of every saved model (newest first)."""
    manifests = []
    root = models_root(ml_dir)
    if not root.exists():
        return manifests
    for d in sorted(root.iterdir()):
        f = d / MANIFEST
        if d.is_dir() and f.exists() and (d / WEIGHTS).exists():
            try:
                manifests.append(json.loads(f.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
    manifests.sort(key=lambda m: m.get("created", ""), reverse=True)
    return manifests


def save_model(model, ml_dir: str | Path, name: str, *, spec: ModelSpec,
               n_classes: int, input_size: tuple[int, int],
               class_names: list[str] | None = None,
               in_channels: int = 3, metrics: dict | None = None) -> Path:
    """Save weights + manifest; overwrites a model of the same name."""
    d = model_dir(ml_dir, name)
    d.mkdir(parents=True, exist_ok=True)
    model.save_weights(d / WEIGHTS)
    manifest = {
        "name": sanitize_name(name),
        "created": datetime.now().isoformat(timespec="seconds"),
        "spec": spec.to_dict(),
        "n_classes": int(n_classes),
        "input_size": [int(input_size[0]), int(input_size[1])],   # (width, height)
        "in_channels": int(in_channels),
        "class_names": list(class_names or []),
        "metrics": metrics or {},
    }
    (d / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return d


def delete_model(ml_dir: str | Path, name: str) -> None:
    """Remove a saved model (weights + manifest folder)."""
    import shutil

    d = model_dir(ml_dir, name)
    if (d / MANIFEST).exists():                    # only ever delete model dirs
        shutil.rmtree(d)


def load_model(ml_dir: str | Path, name: str):
    """Rebuild the architecture from the manifest and restore the weights.
    Returns (model, manifest)."""
    d = model_dir(ml_dir, name)
    manifest = json.loads((d / MANIFEST).read_text(encoding="utf-8"))
    from cloudlabeller.ml.unet import build_model   # lazy: imports TensorFlow

    spec = ModelSpec.from_dict(manifest.get("spec"))
    width, height = manifest["input_size"]
    model = build_model(manifest["n_classes"], width, height,
                        in_channels=manifest.get("in_channels", 3),
                        **spec.unet_kwargs())
    model.load_weights(d / WEIGHTS)
    return model, manifest
