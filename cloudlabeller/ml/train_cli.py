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
        [--steps 50000 | --epochs N] [--patience 10] [--batch-size 2]
        [--val-fraction 0.2] [--augment-rolls 5] [--include-auto]
        [--weight-by-uncertainty] [--stop-file PATH]

Reads labels/statuses from the project on disk (the parent saves first).
It learns only from hand-drawn labels and from labels the cloud got from
them (ADR-0006): a cloud-made photo's pixels projected from predicted
points are left out, and validation holds out hand-drawn photos only.
Trains for ``--steps`` batches (as many epochs as that takes, rounded up),
stops early after ``--patience`` epochs without a better validation loss,
and saves the best epoch's weights. Emits ``PROGRESS <frac> <msg>`` lines;
graceful cancel: the parent creates ``--stop-file`` and the trainer halts
after the current epoch, still saving the model (its best epoch). Prints
``RESULT <json>`` on success.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from cloudlabeller.logging_setup import setup_logging
from cloudlabeller.workers.result import print_result

log = logging.getLogger("cloudlabeller.ml")


#: The share of the Job's progress bar the cutting takes.
CUTTING = 0.09


def _history(manifest: dict | None, metrics: dict) -> dict:
    """Per-epoch metrics: a resumed model's earlier epochs, then these."""
    before = (manifest or {}).get("history") or {}
    return {k: list(before.get(k, [])) + [float(v) for v in vals]
            for k, vals in metrics.items()}


def training_layers(survey, name: str, by_uncertainty: bool):
    """A cloud-made photo's mask for training, and its per-pixel weights.

    Pixels projected from predicted points are left out (the network never
    learns from its own output, ADR-0006). With ``by_uncertainty`` each
    pixel weighs 1 − its Uncertainty (unknown: 1); otherwise there are no
    per-pixel weights. Returns the mask, ``(mask, weights)``, or None."""
    from cloudlabeller.core import sources as src

    layers = survey.render_auto_layers(name)
    if layers is None:
        return None
    mask, source, unc = layers
    mask = mask.copy()
    mask[~src.trainable(source)] = -1
    if not by_uncertainty:
        return mask
    import numpy as np

    weights = 1.0 - unc.astype(np.float32)
    return mask, np.where(np.isfinite(weights), weights, 1.0)


def _parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="train_cli")
    p.add_argument("project_root")
    p.add_argument("--name", default="unet")
    p.add_argument("--steps", type=int, default=50000,
                   help="batches to train for; the epochs are as many as that takes, "
                        "rounded up")
    p.add_argument("--epochs", type=int, default=None,
                   help="train for this many epochs instead of --steps")
    p.add_argument("--patience", type=int, default=10,
                   help="stop after this many epochs without a better validation "
                        "loss (0: never stop early)")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--augment-rolls", type=int, default=5)
    p.add_argument("--include-auto", action="store_true",
                   help="add cloud-projected (auto) images at half weight")
    p.add_argument("--weight-by-uncertainty", action="store_true",
                   help="weigh each cloud-projected pixel by 1 - its Uncertainty as well")
    p.add_argument("--resume", action="store_true",
                   help="load the saved model of this name and train it for "
                        "more epochs (architecture/size come from its manifest)")
    p.add_argument("--stop-file", default=None,
                   help="halt after the current epoch when this file appears")
    p.add_argument("--surveys", default=None,
                   help="comma-separated Survey ids to train on (default: every one)")
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
    from cloudlabeller.ml.dataset_builder import build_tile_set
    from cloudlabeller.ml.unet import build_model_from_spec

    progress(0.0, "Opening project…")
    proj = Project.open(args.project_root)
    if not proj.schema.classes:
        log.error("project has no label classes")
        return 2
    n_classes = len(proj.schema.classes)
    wanted = [s for s in (args.surveys or "").split(",") if s] or list(proj.surveys)
    unknown = [s for s in wanted if s not in proj.surveys]
    if unknown:
        log.error("no Survey %s in the project", ", ".join(unknown))
        return 2

    manifest = None
    if args.resume:
        # Continue training: architecture, spec and Tiles come from the saved
        # model's manifest, NOT the current Model Settings.
        import json as _json

        from cloudlabeller.ml.model_spec import ModelSpec, is_tiled

        manifest_path = model_store.model_dir(proj.ml_dir, args.name) / model_store.MANIFEST
        if not manifest_path.exists():
            log.error("no saved model named %r to resume", args.name)
            return 2
        manifest = _json.loads(manifest_path.read_text(encoding="utf-8"))
        if not is_tiled(manifest):
            log.error("model %r was trained on whole photos (before Tiles): "
                      "train a fresh one", args.name)
            return 2
        if manifest["n_classes"] != n_classes:
            log.error("saved model has %d classes but the project now has %d — "
                      "cannot resume; train a fresh model",
                      manifest["n_classes"], n_classes)
            return 2
        spec = ModelSpec.from_dict(manifest.get("spec"))
    else:
        spec = proj.model_spec()
        error = spec.validate()
        if error:
            log.error("model settings invalid: %s", error)
            return 2

    # Photos are pooled over the chosen Surveys, keyed "<survey>/<name>".
    user_names: list[str] = []
    auto_names: list[str] = []
    records: dict[str, tuple] = {}
    for sid in wanted:
        survey = proj.survey(sid)
        labels = survey.labels
        for r in survey.dataset.images:
            records[f"{sid}/{r.name}"] = (survey, r.name, r.path)
        user_names += [f"{sid}/{n}" for n, m in labels.image_masks.items()
                       if labels.status_of(n) == "user" and (m != -1).any()]
        if args.include_auto:
            # Cloud-projected supplements only; never train on U-Net output ('ml').
            auto_names += [f"{sid}/{r.name}" for r in survey.dataset.solved_images()
                           if labels.status_of(r.name) == "auto"]
    if spec.normals:
        # Experimental: only photos the dense cloud's normals reach.
        from cloudlabeller.ml.inputs import can_have_normals

        def reached(key):
            survey, name, _ = records[key]
            return can_have_normals(survey, name)
        left_out = [k for k in user_names + auto_names if not reached(k)]
        if left_out:
            log.warning("%d labelled photos have no normals (no camera or no dense "
                        "cloud) and are left out: %s", len(left_out), ", ".join(left_out[:10]))
        user_names = [k for k in user_names if reached(k)]
        auto_names = [k for k in auto_names if reached(k)]
    names = user_names + auto_names
    if not names:
        log.error("no labelled images to train on"
                  + (" that the dense cloud's normals reach" if spec.normals else ""))
        return 2

    auto_keys = set(auto_names)

    def mask_loader(key):
        survey, name, _ = records[key]
        if key not in auto_keys:
            return survey.labels.image_masks.get(name)
        return training_layers(survey, name, args.weight_by_uncertainty)

    n = spec.net_size
    progress(0.01, f"Cutting {len(names)} photos into {spec.tile_size} px Tiles "
                   f"({n}×{n} px for the network)…")

    def cutting(done: int, total: int) -> None:
        # Rendering a photo's mask from the cloud takes seconds when its
        # visibility isn't cached yet: show how far the cutting has got.
        progress(0.01 + CUTTING * done / total,
                 f"Cutting photos into Tiles: {done} of {total}")

    from cloudlabeller.ml.inputs import load_input

    def image_loader(key):
        survey, name, path = records[key]
        return load_input(survey, name, path, spec.normals)

    ts = build_tile_set(
        names, image_loader, mask_loader, spec,
        projected_weight=0.5, projected_names=set(auto_names), progress=cutting)
    if not ts.samples:
        log.error("no Tile is labelled enough to train on (at least %.0f%% of a Tile)",
                  100 * spec.min_labelled)
        return 2
    # Validation scores hand-drawn labels only (ADR-0006).
    train_set, val_set = (ts.split(args.val_fraction, among=set(user_names))
                          if args.val_fraction > 0 and len(names) > 1 and len(user_names) > 1
                          else (ts, None))
    if val_set is not None and not val_set.samples:
        val_set = None

    if args.resume:
        progress(0.01 + CUTTING, f"Loading model '{args.name}' to continue training…")
        model, _ = model_store.load_model(proj.ml_dir, args.name)
    else:
        progress(0.01 + CUTTING, f"Building U-Net ({spec.channels} ch × {spec.blocks} "
                       f"blocks, {n_classes} classes)…")
        model = build_model_from_spec(spec, n_classes, in_channels=spec.in_channels)

    model, metrics, outcome = trainer.train(
        model, train_set, val_set,
        steps=None if args.epochs else args.steps, epochs=args.epochs,
        batch_size=args.batch_size, augment_rolls=args.augment_rolls,
        patience=args.patience,
        progress=lambda f, m="": progress(0.01 + CUTTING + (0.97 - 0.01 - CUTTING) * f, m),
        should_stop=should_stop)

    best = outcome["best_epoch"]
    progress(0.98, f"Saving model '{args.name}' as of epoch {best + 1} "
                   f"(the lowest {outcome['monitor']})…")
    kept = {k: v[best] for k, v in metrics.items() if len(v) > best}
    history = _history(manifest, metrics)
    before = len(history.get("loss", [])) - len(metrics.get("loss", []))
    model_store.save_model(
        model, proj.ml_dir, args.name, spec=spec, n_classes=n_classes,
        input_size=(n, n), in_channels=spec.in_channels,
        class_names=[c.name for c in proj.schema.classes], metrics=kept,
        history=history, surveys=wanted, best_epoch=before + best,
        monitor=outcome["monitor"])
    print_result({
        "name": model_store.sanitize_name(args.name),
        "steps_per_epoch": outcome["steps_per_epoch"],
        "epochs": outcome["epochs"],
        "epochs_run": outcome["epochs_run"],
        "best_epoch": best + 1,
        "stopped_early": outcome["stopped_early"],
        "n_train": len(train_set.samples),
        "n_val": len(val_set.samples) if val_set else 0,
        "resumed": bool(args.resume),
        "metrics": kept,
    })
    progress(1.0, "Training complete")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
