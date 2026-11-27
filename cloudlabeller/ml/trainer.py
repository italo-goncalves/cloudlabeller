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

Runs on a worker thread: the UI passes ``progress`` / ``should_stop`` hooks so
training reports per-epoch metrics and stays cancellable. TensorFlow is
imported lazily so the module can be imported (e.g. by tests) without TF.
"""

from __future__ import annotations

from typing import Callable

from cloudlabeller.ml.dataset_builder import TrainingSet

ProgressFn = Callable[[float, str], None]
StopFn = Callable[[], bool]


def train(
    model,
    train_set: TrainingSet,
    val_set: TrainingSet | None = None,
    *,
    epochs: int = 100,
    batch_size: int = 2,
    augment_rolls: int = 5,
    progress: ProgressFn = lambda f, m: None,
    should_stop: StopFn = lambda: False,
    **hyperparams,
) -> tuple[object, dict]:
    """Train ``model``; returns (model, history dict of per-epoch metrics).

    The Tiles go through ``tile_dataset.make_dataset``: with ``augment_rolls``
    > 0 each Tile gives ``2 * augment_rolls`` mirror/slide/gamma windows per
    epoch, made batch by batch; validation stays unaugmented.
    """
    import tensorflow as tf

    from cloudlabeller.ml.tile_dataset import make_dataset, windows

    data = make_dataset(train_set, batch_size, augment_rolls, shuffle=True)
    validation = None
    if val_set is not None and val_set.samples:
        validation = make_dataset(val_set, batch_size, 0, shuffle=False)

    h, w = train_set.samples[0].image.shape[:2]
    n = len(train_set.samples)
    progress(0.0, f"Training on {n} Tiles of {w}×{h} px"
                  + (f", {windows(augment_rolls)} augmented windows each per epoch"
                     if augment_rolls > 0 else "")
                  + (f" (validating on {len(val_set.samples)})" if validation else ""))

    class _Hooks(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            msg = f"epoch {epoch + 1}/{epochs}: " + ", ".join(
                f"{k}={v:.4f}" for k, v in logs.items())
            if should_stop():
                self.model.stop_training = True
                msg += "  [stopped by user — saving the model as-is]"
            progress((epoch + 1) / epochs, msg)

        def on_train_batch_end(self, batch, logs=None):
            if should_stop():                      # responsive mid-epoch cancel
                self.model.stop_training = True

    history = model.fit(
        data,
        validation_data=validation,
        epochs=epochs,
        callbacks=[_Hooks()],
        verbose=0,
    )
    metrics = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    return model, metrics
