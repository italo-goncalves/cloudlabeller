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

import numpy as np

from cloudlabeller.ml.dataset_builder import TrainingSet

ProgressFn = Callable[[float, str], None]
StopFn = Callable[[], bool]


def to_arrays(ts: TrainingSet) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack a TrainingSet into model-ready arrays:
    x (N,H,W,C) float32 in [0,1]; y (N,H,W) int32 (-1 = unlabelled, ignored by
    the loss); w (N,1,1) float32 per-sample weights.

    The weights carry two singleton axes because Keras multiplies them against
    the per-PIXEL loss map (N,H,W): a flat (N,) vector fails to broadcast for
    any batch size > 1 ("Dimensions must be equal, ... [5], [5,256,384]")."""
    x = np.stack([s.image for s in ts.samples]).astype(np.float32)
    if x.max() > 1.0:
        x /= 255.0
    y = np.stack([s.mask for s in ts.samples]).astype(np.int32)
    w = np.asarray([s.weight for s in ts.samples], dtype=np.float32)
    return x, y, w[:, None, None]


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

    ``augment_rolls`` > 0 expands the training set with the mirror/slide/gamma
    augmentation (``TrainingSet.augment_data``); validation stays unaugmented.
    """
    import tensorflow as tf

    if augment_rolls > 0:
        progress(0.0, f"Augmenting {len(train_set.samples)} samples "
                      f"(x{2 * augment_rolls})…")
        train_set = train_set.augment_data(n_rolls=augment_rolls)

    x, y, w = to_arrays(train_set)
    validation = None
    if val_set is not None and val_set.samples:
        vx, vy, vw = to_arrays(val_set)
        validation = (vx, vy, vw)

    progress(0.0, f"Training on {len(x)} samples of {x.shape[2]}×{x.shape[1]} px"
                  + (f" (validating on {len(validation[0])})" if validation else ""))

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
        x, y,
        sample_weight=w,
        validation_data=validation,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[_Hooks()],
        verbose=0,
    )
    metrics = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    return model, metrics
