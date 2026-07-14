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

"""Subprocess entry point for U-Net training.

TensorFlow runs in its own process: running it on a QThread inside the GUI
process deadlocked the GIL against Qt's event loop (main thread stuck in
``PyGILState_Ensure``, TF worker stuck in ``PyEval_AcquireThread``), and a
native TF crash/OOM now fails the job instead of the app — same rationale as
COLMAP's ``run_cli``.

Usage::

    python -m cloudlabeller.ml.train_cli PROJECT_ROOT --name unet
        [--epochs 100] [--batch-size 2] [--val-fraction 0.2]
        [--augment-rolls 5] [--include-auto] [--stop-file PATH]

Reads labels/statuses from the project on disk (the parent saves first).
Emits ``PROGRESS <frac> <msg>`` lines; graceful cancel: the parent creates
``--stop-file`` and the trainer halts after the current epoch, still saving
the model. Prints ``RESULT <json>`` on success.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from cloudlabeller.logging_setup import setup_logging

log = logging.getLogger("cloudlabeller.ml")


def _load_rgb(path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def _parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="train_cli")
    p.add_argument("project_root")
    p.add_argument("--name", default="unet")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--augment-rolls", type=int, default=5)
    p.add_argument("--include-auto", action="store_true",
                   help="add cloud-projected (auto) images at half weight")
    p.add_argument("--resume", action="store_true",
                   help="load the saved model of this name and train it for "
                        "more epochs (architecture/size come from its manifest)")
    p.add_argument("--stop-file", default=None,
                   help="halt after the current epoch when this file appears")
    return p.parse_args(argv[1:])


def main(argv: list[str]) -> int:
    args = _parse(argv)
    setup_logging(logfile=None, console=True)     # stderr; parent captures it
    # Cap threadpools (before TF initialises) + below-normal priority: keep
    # the OS/UI responsive while training.
    from cloudlabeller.workers.resources import limit_subprocess_resources
    limit_subprocess_resources()

    def progress(frac: float, msg: str = "") -> None:
        print(f"PROGRESS {frac:.4f} {msg}", flush=True)

    stop_path = Path(args.stop_file) if args.stop_file else None
    if stop_path is not None:
        stop_path.unlink(missing_ok=True)          # stale flag from a past run
    should_stop = (lambda: stop_path.exists()) if stop_path else (lambda: False)

    from cloudlabeller.core.project import Project
    from cloudlabeller.ml import model_store, trainer
    from cloudlabeller.ml.dataset_builder import build_training_set
    from cloudlabeller.ml.unet import build_model_from_spec

    progress(0.0, "Opening project…")
    proj = Project.open(args.project_root)
    labels = proj.labels
    source_res = proj.source_resolution()
    if not proj.schema.classes or source_res is None:
        log.error("project has no label classes or no images")
        return 2
    n_classes = len(proj.schema.classes)

    manifest = None
    if args.resume:
        # Continue training: architecture, spec and training size come from
        # the saved model's manifest, NOT the current Model Settings.
        import json as _json

        manifest_path = model_store.model_dir(proj.ml_dir, args.name) / model_store.MANIFEST
        if not manifest_path.exists():
            log.error("no saved model named %r to resume", args.name)
            return 2
        manifest = _json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["n_classes"] != n_classes:
            log.error("saved model has %d classes but the project now has %d — "
                      "cannot resume; train a fresh model",
                      manifest["n_classes"], n_classes)
            return 2
        from cloudlabeller.ml.model_spec import ModelSpec

        spec = ModelSpec.from_dict(manifest.get("spec"))
    else:
        spec = proj.model_spec()
        error = spec.validate_size(*source_res)
        if error:
            log.error("model settings invalid: %s", error)
            return 2

    user_names = [n for n, m in labels.image_masks.items()
                  if labels.status_of(n) == "user" and (m != -1).any()]
    auto_names: list[str] = []
    if args.include_auto:
        # Cloud-projected supplements only; never train on U-Net output ('ml').
        auto_names = [r.name for r in proj.dataset.solved_images()
                      if r.camera is not None and labels.status_of(r.name) == "auto"]
    names = user_names + auto_names
    if not names:
        log.error("no labelled images to train on")
        return 2

    records = {r.name: r.path for r in proj.dataset.images}
    # When resuming, the dataset must be built at the SAVED model's input size.
    target_size = (tuple(manifest["input_size"]) if manifest is not None
                   else spec.training_size(*source_res))

    def mask_loader(name):
        mask = labels.image_masks.get(name)
        return mask if mask is not None else proj.render_auto_mask(name)

    progress(0.01, f"Loading {len(names)} images at "
                   f"{target_size[0]}×{target_size[1]} px…")
    ts = build_training_set(
        names, lambda n: _load_rgb(records[n]), mask_loader,
        projected_weight=0.5, projected_names=set(auto_names),
        target_size=target_size)
    train_set, val_set = (ts.split(args.val_fraction)
                          if args.val_fraction > 0 and len(ts.samples) > 1
                          else (ts, None))

    if args.resume:
        progress(0.03, f"Loading model '{args.name}' to continue training…")
        model, _ = model_store.load_model(proj.ml_dir, args.name)
    else:
        progress(0.03, f"Building U-Net ({spec.channels} ch × {spec.blocks} "
                       f"blocks, {n_classes} classes)…")
        model = build_model_from_spec(spec, n_classes, source_res)

    model, metrics = trainer.train(
        model, train_set, val_set,
        epochs=args.epochs, batch_size=args.batch_size,
        augment_rolls=args.augment_rolls,
        progress=lambda f, m="": progress(0.03 + 0.94 * f, m),
        should_stop=should_stop)

    progress(0.98, f"Saving model '{args.name}'…")
    last = {k: v[-1] for k, v in metrics.items()}
    model_store.save_model(
        model, proj.ml_dir, args.name, spec=spec, n_classes=n_classes,
        input_size=target_size,
        class_names=[c.name for c in proj.schema.classes], metrics=last)
    print("RESULT " + json.dumps({
        "name": model_store.sanitize_name(args.name),
        "epochs_run": len(metrics.get("loss", [])),
        "n_train": len(train_set.samples),
        "n_val": len(val_set.samples) if val_set else 0,
        "resumed": bool(args.resume),
        "metrics": last,
    }), flush=True)
    progress(1.0, "Training complete")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
