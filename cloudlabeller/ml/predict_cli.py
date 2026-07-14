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

"""Subprocess entry point for U-Net prediction.

TensorFlow runs in its own process (see ``train_cli`` for why: GIL deadlock
against Qt when run on an in-process thread).

Usage::

    python -m cloudlabeller.ml.predict_cli PROJECT_ROOT --model NAME

Predicts every image whose status is not 'user' (red + yellow), writing masks
at model resolution into ``project/ml/predictions/``. Statuses are NOT written
here — the parent owns the manifest and marks images 'ml' on success. Emits
``PROGRESS <frac> <msg>`` lines; prints ``RESULT <json>`` with the predicted
names on success.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from cloudlabeller.logging_setup import setup_logging

log = logging.getLogger("cloudlabeller.ml")


def _load_rgb(path):
    import numpy as np
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def _parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="predict_cli")
    p.add_argument("project_root")
    p.add_argument("--model", required=True, help="saved model name (ml/models/<name>)")
    return p.parse_args(argv[1:])


def main(argv: list[str]) -> int:
    args = _parse(argv)
    setup_logging(logfile=None, console=True)     # stderr; parent captures it
    from cloudlabeller.workers.resources import limit_subprocess_resources
    limit_subprocess_resources()

    def progress(frac: float, msg: str = "") -> None:
        print(f"PROGRESS {frac:.4f} {msg}", flush=True)

    from cloudlabeller.core.project import Project
    from cloudlabeller.ml import inference, model_store

    progress(0.0, "Opening project…")
    proj = Project.open(args.project_root)

    manifests = {m["name"]: m for m in model_store.list_models(proj.ml_dir)}
    manifest = manifests.get(model_store.sanitize_name(args.model))
    if manifest is None:
        log.error("model %r not found under %s", args.model, proj.ml_dir / "models")
        return 2
    if manifest["n_classes"] != len(proj.schema.classes):
        log.error("model has %d classes but the project has %d — retrain first",
                  manifest["n_classes"], len(proj.schema.classes))
        return 2

    labels = proj.labels
    targets = [r.name for r in proj.dataset.images
               if labels.status_of(r.name) != "user"]     # red + yellow only
    if not targets:
        print("RESULT " + json.dumps({"predicted": []}), flush=True)
        return 0

    records = {r.name: r.path for r in proj.dataset.images}
    progress(0.01, f"Loading model '{manifest['name']}'…")
    model, _ = model_store.load_model(proj.ml_dir, manifest["name"])

    written = inference.predict_all(
        model, targets, lambda n: _load_rgb(records[n]), proj.predictions_dir,
        progress=lambda f, m="": progress(0.05 + 0.95 * f, m))
    print("RESULT " + json.dumps({"predicted": sorted(written)}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
