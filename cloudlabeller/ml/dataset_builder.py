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

"""Assemble (image, mask) training pairs from labels.

Training data comes from two sources, unified here:
  1. Images the user labelled directly in 2D.
  2. Masks synthesised by projecting 3D cloud labels into images
     (``transfer.cloud_to_image``) — so labelling in 3D trains the 2D model too.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import skimage.exposure as ske
import cv2


@dataclass
class Sample:
    """One training example: an image, its label mask, and a loss weight."""

    name: str
    image: np.ndarray          # (H, W, C)
    mask: np.ndarray           # (H, W) int32 class ids
    weight: float = 1.0        # e.g. down-weight projection-derived masks


class TrainingSet:
    """A lazily-iterable collection of :class:`Sample`s with a train/val split."""

    def __init__(self, samples: list[Sample]) -> None:
        self.samples = samples

    def split(self, val_fraction: float = 0.2, seed: int = 0) -> tuple["TrainingSet", "TrainingSet"]:
        """Random (train, validation) split — seeded, so re-runs are stable."""
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(self.samples))
        cut = int(len(idx) * (1 - val_fraction))
        train = [self.samples[i] for i in idx[:cut]]
        val = [self.samples[i] for i in idx[cut:]]
        return TrainingSet(train), TrainingSet(val)

    def augment_data(self, n_rolls=5):
        """Augment by mirroring + sliding + gamma correction.

        Each image is concatenated with its mirror to form a seamless
        double-width panorama, then a full-width window is slid across it
        (``2 * n_rolls`` positions); the image (only) gets a random gamma
        tweak per window. Masks follow the same geometry untouched.
        """
        photos = [sample.image for sample in self.samples]
        labels = [sample.mask for sample in self.samples]
        names = [sample.name for sample in self.samples]
        weights = [sample.weight for sample in self.samples]

        resolution = photos[0].shape
        width = resolution[1]

        pixels_rolled = int(resolution[1] / n_rolls)
        aug_x, aug_y, aug_names, aug_w = [], [], [], []
        for tx, ty, name, w in zip(photos, labels, names, weights):
            flipped_x = np.flip(tx, axis=1)
            flipped_y = np.flip(ty, axis=1)

            tx = np.concatenate([tx, flipped_x], axis=1)
            ty = np.concatenate([ty, flipped_y], axis=1)

            for r in range(n_rolls * 2):
                tx = np.roll(tx, pixels_rolled, axis=1)
                ty = np.roll(ty, pixels_rolled, axis=1)

                tx_gamma = ske.adjust_gamma(tx[:, :width], np.exp(np.random.uniform(-0.4, 0.4)))

                aug_x.append(tx_gamma)
                aug_y.append(ty[:, :width])         # mask is (H, W): no channel axis
                aug_names.append(f'{name}_aug_{r}')
                aug_w.append(w)

        augmented_dataset = [Sample(name, image, mask, weight)
                             for name, image, mask, weight
                             in zip(aug_names, aug_x, aug_y, aug_w)]
        return TrainingSet(augmented_dataset)


#: A target pixel is kept labelled when labelled classes together cover at
#: least this fraction of its area (below it, it becomes -1/unlabelled).
MASK_LABEL_FRACTION = 0.2


def resize_mask(mask: np.ndarray, size: tuple[int, int],
                min_label_fraction: float = MASK_LABEL_FRACTION) -> np.ndarray:
    """Resize an int label mask to ``size`` = (width, height), preserving
    labelled area far better than nearest-neighbour.

    Each labelled class (id >= 0) is resized as a one-hot channel with area
    averaging, so every target pixel gets the fraction of its area covered by
    each class. A pixel takes the majority labelled class whenever labelled
    classes *together* cover at least ``min_label_fraction`` of it, and -1
    (unlabelled) otherwise — i.e. the unlabelled background only wins when
    labels are genuinely sparse there.

    This keeps thin labels that nearest-neighbour would drop on downscaling,
    and closes the scattered -1 holes typical of projection-derived masks
    (which matters when those auto-labelled images are used as training data).
    """
    import cv2

    labelled_ids = [int(c) for c in np.unique(mask) if c >= 0]
    if not labelled_ids:                          # nothing labelled to preserve
        width, height = size
        return np.full((height, width), -1, dtype=mask.dtype)
    coverage = np.stack(
        [cv2.resize((mask == c).astype(np.float32), size,
                    interpolation=cv2.INTER_AREA) for c in labelled_ids],
        axis=-1)                                  # (H, W, K) per-class area fraction
    labelled_fraction = coverage.sum(axis=-1)
    winner = np.asarray(labelled_ids)[coverage.argmax(axis=-1)]
    resized = np.where(labelled_fraction >= min_label_fraction, winner, -1)
    return resized.astype(mask.dtype)


def resize_pair(image: np.ndarray, mask: np.ndarray, size: tuple[int, int]
                ) -> tuple[np.ndarray, np.ndarray]:
    """Resize an (image, mask) pair to ``size`` = (width, height).

    The image uses area averaging (best for downscaling); the mask uses the
    area-based one-hot method in :func:`resize_mask`, which preserves labelled
    pixels (including sparse projection labels) better than nearest-neighbour.
    The target usually stretches the aspect ratio slightly (it is the snapped
    U-Net-compatible size from ``ModelSpec.training_size``).
    """
    import cv2

    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return image, resize_mask(mask, size)


def build_training_set(
    labelled_image_names: list[str],
    image_loader,                       # name -> (H, W, C)
    mask_loader,                        # name -> (H, W) int32 (user + projected)
    projected_weight: float = 0.5,
    projected_names: set[str] | None = None,
    target_size: tuple[int, int] | None = None,   # (width, height) from ModelSpec.training_size
) -> TrainingSet:
    """Collect samples from labelled + projection-derived masks.

    With ``target_size`` set, every sample is resized to it on load, so the
    dataset lives in memory at the (U-Net-compatible) size the model trains on.
    """
    projected_names = projected_names or set()
    samples: list[Sample] = []
    for name in labelled_image_names:
        image = image_loader(name)
        mask = mask_loader(name)
        if target_size is not None and tuple(target_size) != (image.shape[1], image.shape[0]):
            image, mask = resize_pair(image, mask, tuple(target_size))
        samples.append(
            Sample(
                name=name,
                image=image,
                mask=mask,
                weight=projected_weight if name in projected_names else 1.0,
            )
        )
    return TrainingSet(samples)
