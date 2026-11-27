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

"""Training loop around the user's U-Net (``ml.unet``).

Runs in ``train_cli``'s process: the caller passes ``progress`` /
``should_stop`` hooks so training reports per-epoch metrics and stays
cancellable. TensorFlow is imported lazily so the module can be imported
(e.g. by tests) without TF.

Training is asked for in **steps** (batches): the epochs are as many as it
takes to make that many steps, rounded up, an epoch being every Tile's
augmented windows once. The weights kept are those of the **best epoch**,
the one with the lowest validation loss (training loss when nothing is held
out), and training **stops early** after ``patience`` epochs without a
better one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from cloudlabeller.ml.dataset_builder import TrainingSet

ProgressFn = Callable[[float, str], None]
StopFn = Callable[[], bool]


def epochs_for_steps(steps: int, steps_per_epoch: int) -> int:
    """How many epochs make ``steps`` batches, rounded up (at least one)."""
    return max(1, math.ceil(max(1, int(steps)) / max(1, int(steps_per_epoch))))


@dataclass
class BestEpoch:
    """Follows the monitored loss epoch by epoch: which epoch is best, and
    whether ``patience`` epochs (0: never) have passed without a better one."""

    patience: int = 0
    best: float = math.inf
    epoch: int = -1             # the best epoch, 0-based; -1 before any
    waited: int = 0

    def update(self, epoch: int, value) -> tuple[bool, bool]:
        """(this epoch is the best so far, it is time to stop)."""
        if value is None or not math.isfinite(float(value)):
            self.waited += 1
            return False, bool(self.patience) and self.waited >= self.patience
        if float(value) < self.best:
            self.best, self.epoch, self.waited = float(value), int(epoch), 0
            return True, False
        self.waited += 1
        return False, bool(self.patience) and self.waited >= self.patience


def train(
    model,
    train_set: TrainingSet,
    val_set: TrainingSet | None = None,
    *,
    steps: int | None = None,
    epochs: int | None = None,
    batch_size: int = 2,
    augment_rolls: int = 5,
    patience: int = 0,
    progress: ProgressFn = lambda f, m: None,
    should_stop: StopFn = lambda: False,
    **hyperparams,
) -> tuple[object, dict, dict]:
    """Train ``model`` for ``steps`` batches (or ``epochs`` epochs).

    Returns (model with the best epoch's weights, history of per-epoch
    metrics, outcome): the outcome is ``{steps_per_epoch, epochs,
    epochs_run, best_epoch (0-based), monitor, stopped_early}``.

    The Tiles go through ``tile_dataset.make_dataset``: with ``augment_rolls``
    > 0 each Tile gives ``2 * augment_rolls`` mirror/slide/gamma windows per
    epoch, made batch by batch; validation stays unaugmented.
    """
    import tensorflow as tf

    from cloudlabeller.ml.tile_dataset import make_dataset, n_batches, windows

    data = make_dataset(train_set, batch_size, augment_rolls, shuffle=True)
    validation = None
    if val_set is not None and val_set.samples:
        validation = make_dataset(val_set, batch_size, 0, shuffle=False)
    per_epoch = n_batches(len(train_set.samples), augment_rolls, batch_size)
    if epochs is None:
        epochs = epochs_for_steps(steps if steps is not None else per_epoch, per_epoch)
    monitor = "val_loss" if validation is not None else "loss"

    h, w = train_set.samples[0].image.shape[:2]
    n = len(train_set.samples)
    progress(0.0, f"Training on {n} Tiles of {w}×{h} px"
                  + (f", {windows(augment_rolls)} augmented windows each per epoch"
                     if augment_rolls > 0 else "")
                  + (f" (validating on {len(val_set.samples)})" if validation else "")
                  + f": {per_epoch} steps per epoch, {epochs} epochs"
                  + (f" for {steps} steps" if steps is not None else "")
                  + (f", stopping after {patience} without a better {monitor}"
                     if patience else ""))

    tracker = BestEpoch(patience=max(0, int(patience)))
    kept: dict = {"weights": None, "stopped_early": False}

    class _KeepBest(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            best, stop = tracker.update(epoch, (logs or {}).get(monitor))
            if best:
                kept["weights"] = self.model.get_weights()
            if stop:
                kept["stopped_early"] = True
                self.model.stop_training = True

        def on_train_end(self, logs=None):
            if kept["weights"] is not None:
                self.model.set_weights(kept["weights"])

    class _Hooks(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            msg = f"epoch {epoch + 1}/{epochs}: " + ", ".join(
                f"{k}={v:.4f}" for k, v in logs.items())
            if tracker.epoch == epoch:
                msg += "  (best so far)"
            if kept["stopped_early"]:
                msg += (f"  [no better {monitor} for {tracker.patience} epochs: "
                        f"stopping, keeping epoch {tracker.epoch + 1}]")
            elif should_stop():
                self.model.stop_training = True
                msg += f"  [stopped by user — keeping epoch {tracker.epoch + 1}]"
            progress((epoch + 1) / epochs, msg)

        def on_train_batch_end(self, batch, logs=None):
            if should_stop():                      # responsive mid-epoch cancel
                self.model.stop_training = True

    history = model.fit(
        data,
        validation_data=validation,
        epochs=epochs,
        callbacks=[_KeepBest(), _Hooks()],
        verbose=0,
    )
    metrics = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    outcome = {
        "steps_per_epoch": per_epoch,
        "epochs": epochs,
        "epochs_run": len(metrics.get("loss", [])),
        "best_epoch": max(tracker.epoch, 0) if kept["weights"] is not None
        else len(metrics.get("loss", [])) - 1,
        "monitor": monitor,
        "stopped_early": kept["stopped_early"],
    }
    return model, metrics, outcome
