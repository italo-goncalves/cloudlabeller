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

"""The training input pipeline: Tiles to batches with ``tf.data``.

The Tiles stay in memory once, as uint8 images and int16 masks. Each epoch a
generator hands ``tf.data`` one (Tile, augmentation) pair at a time in a fresh
random order; the augmentation, the conversion to float32 and the batching
run inside the pipeline, in parallel and a few batches ahead of the GPU. So
memory holds the Tiles plus a few batches, whatever the number of
augmentation rolls (stacking every augmented copy as float32 needed ~18x the
Tiles' size: ~125 GB for 376 drone photos).

The augmentation is the mirror/slide/gamma one of ``unet_tools``: the Tile is
joined to its mirror image into a seamless strip twice as wide, a window as
wide as the Tile slides across it (``2 * rolls`` positions, ``width / rolls``
pixels apart), and the image (not the mask) gets a random gamma,
``exp(U(-0.4, 0.4))``, per window.

TensorFlow is imported inside the functions, so the module imports without it.
"""

from __future__ import annotations

import numpy as np

from cloudlabeller.ml.dataset_builder import TrainingSet

#: The gamma of an augmented window is exp(U(-GAMMA_LOG, GAMMA_LOG)).
GAMMA_LOG = 0.4


def windows(augment_rolls: int) -> int:
    """How many samples each Tile gives per epoch."""
    return 2 * augment_rolls if augment_rolls > 0 else 1


def n_batches(n_tiles: int, augment_rolls: int, batch_size: int) -> int:
    return -(-n_tiles * windows(augment_rolls) // max(1, batch_size))


def augment(image, mask, window, rolls: int, log_gamma, weights=None):
    """Window ``window`` (0 .. 2*rolls-1) of a Tile's mirror strip.

    Per-pixel ``weights`` (H, W), when given, move with the mask and come
    back third: ``(x, y, w)``.

    ``image`` (H, W, C) float32 in [0, 1], ``mask`` (H, W); TF tensors or
    arrays. The window is shifted by ``(window + 1) * (W // rolls)`` pixels
    and the RGB raised to ``exp(log_gamma)``.

    Channels past RGB are the camera-frame normals (``ModelSpec.normals``),
    encoded so 0.5 is 0: the gamma leaves them alone, and mirroring the Tile
    left to right turns their x component around (``x -> 1 - x``)."""
    import tensorflow as tf

    width = tf.shape(image)[1]
    shift = (tf.cast(window, tf.int32) + 1) * (width // rolls)
    mirrored = tf.reverse(image, axis=[1])
    extra = image.shape[-1] is not None and image.shape[-1] > 3
    if extra:
        mirrored = tf.concat([mirrored[..., :3], 1.0 - mirrored[..., 3:4], mirrored[..., 4:]],
                             axis=-1)
    strip_x = tf.concat([image, mirrored], axis=1)
    strip_y = tf.concat([mask, tf.reverse(mask, axis=[1])], axis=1)
    x = tf.roll(strip_x, shift, axis=1)[:, :width]
    y = tf.roll(strip_y, shift, axis=1)[:, :width]
    gamma = tf.exp(tf.cast(log_gamma, x.dtype))
    if extra:
        x = tf.concat([tf.pow(x[..., :3], gamma), x[..., 3:]], axis=-1)
    else:
        x = tf.pow(x, gamma)
    if weights is None:
        return x, y
    strip_w = tf.concat([weights, tf.reverse(weights, axis=[1])], axis=1)
    return x, y, tf.roll(strip_w, shift, axis=1)[:, :width]


def make_dataset(ts: TrainingSet, batch_size: int, augment_rolls: int = 0, *,
                 shuffle: bool = True, seed: int | None = None):
    """``ts`` as a ``tf.data.Dataset`` of ``(x, y, w)`` batches: x (B, H, W, C)
    float32 in [0, 1], y (B, H, W) int32 (-1 unlabelled, ignored by the loss),
    w (B, 1, 1) float32 per-sample weights, or (B, H, W) per-pixel ones when
    any sample has them (``Sample.pixel_weights`` times its weight).

    The weights carry two singleton axes because Keras multiplies them against
    the per-PIXEL loss map (B, H, W): a flat (B,) vector fails to broadcast for
    any batch size > 1 ("Dimensions must be equal, ... [5], [5,256,384]").

    With ``augment_rolls`` > 0 every Tile gives ``2 * augment_rolls`` windows
    per epoch; otherwise it is passed as is. ``shuffle`` draws a new order
    each epoch."""
    import tensorflow as tf

    samples = ts.samples
    if not samples:
        raise ValueError("no samples to train on")
    h, w, c = samples[0].image.shape
    per = windows(augment_rolls)
    per_pixel = any(s.pixel_weights is not None for s in samples)
    ones = np.full((h, w), 255, np.uint8)
    total = len(samples) * per
    rng = np.random.default_rng(seed)

    def generate():
        order = rng.permutation(total) if shuffle else range(total)
        for k in order:
            i, r = divmod(int(k), per)
            s = samples[i]
            if per_pixel:
                pw = s.pixel_weights if s.pixel_weights is not None else ones
                yield s.image, s.mask, np.float32(s.weight), np.int32(r), pw
            else:
                yield s.image, s.mask, np.float32(s.weight), np.int32(r)

    image_dtype = tf.as_dtype(samples[0].image.dtype)
    mask_dtype = tf.as_dtype(samples[0].mask.dtype)
    ds = tf.data.Dataset.from_generator(generate, output_signature=(
        tf.TensorSpec((h, w, c), image_dtype),
        tf.TensorSpec((h, w), mask_dtype),
        tf.TensorSpec((), tf.float32),
        tf.TensorSpec((), tf.int32),
    ) + ((tf.TensorSpec((h, w), tf.uint8),) if per_pixel else ()))

    def to_float(image):
        x = tf.cast(image, tf.float32)
        return x / 255.0 if image_dtype == tf.uint8 else x

    def prepare(image, mask, weight, window, pixel_weights=None):
        x, y = to_float(image), tf.cast(mask, tf.int32)
        pw = None if pixel_weights is None else tf.cast(pixel_weights, tf.float32) / 255.0
        if augment_rolls > 0:
            log_gamma = tf.random.uniform((), -GAMMA_LOG, GAMMA_LOG)
            if pw is None:
                x, y = augment(x, y, window, augment_rolls, log_gamma)
            else:
                x, y, pw = augment(x, y, window, augment_rolls, log_gamma, pw)
        if pw is None:
            return x, y, tf.reshape(weight, (1, 1))
        return x, y, pw * weight

    ds = (ds.map(prepare, num_parallel_calls=tf.data.AUTOTUNE, deterministic=not shuffle)
          .batch(batch_size)
          .prefetch(tf.data.AUTOTUNE))
    # from_generator can't know its length; Keras uses it for the epoch's steps.
    return ds.apply(tf.data.experimental.assert_cardinality(
        n_batches(len(samples), augment_rolls, batch_size)))
