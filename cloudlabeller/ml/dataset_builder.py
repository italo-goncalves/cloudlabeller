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

from typing import Callable

import numpy as np


@dataclass
class Sample:
    """One training example: an image, its label mask, and a loss weight."""

    name: str
    image: np.ndarray          # (H, W, C), uint8 for Tiles
    mask: np.ndarray           # (H, W) class ids (int16 for Tiles), -1 unlabelled
    weight: float = 1.0        # e.g. down-weight projection-derived masks


class TrainingSet:
    """A lazily-iterable collection of :class:`Sample`s with a train/val split."""

    def __init__(self, samples: list[Sample]) -> None:
        self.samples = samples

    def split(self, val_fraction: float = 0.2, seed: int = 0) -> tuple["TrainingSet", "TrainingSet"]:
        """Random (train, validation) split by photo — seeded, so re-runs are
        stable. The Tiles of one photo (``name#row,col``) stay together, so
        validation never sees a photo training saw."""
        photos = sorted({photo_of(s.name) for s in self.samples})
        rng = np.random.default_rng(seed)
        order = [photos[i] for i in rng.permutation(len(photos))]
        cut = int(len(order) * (1 - val_fraction))
        if len(order) > 1:
            cut = min(max(cut, 1), len(order) - 1)
        held = set(order[cut:])
        train = [s for s in self.samples if photo_of(s.name) not in held]
        val = [s for s in self.samples if photo_of(s.name) in held]
        return TrainingSet(train), TrainingSet(val)


# The mirror/slide/gamma augmentation runs inside the input pipeline, one
# batch at a time (``ml.tile_dataset``), never as copies of the whole set.


def photo_of(sample_name: str) -> str:
    """The photo a sample came from (a Tile is named ``photo#row,col``)."""
    return sample_name.split("#", 1)[0]


MASK_LABEL_FRACTION = 0.2


def resize_mask(mask: np.ndarray, size: tuple[int, int],
                min_label_fraction: float = MASK_LABEL_FRACTION) -> np.ndarray:
    """Resize a training-data label mask to ``size`` = (width, height),
    preserving labelled area far better than nearest-neighbour.

    A pixel takes the majority labelled class whenever labelled classes
    *together* cover at least ``min_label_fraction`` of it, and -1 (unlabelled)
    otherwise — so the unlabelled background only wins where labels are
    genuinely sparse. This keeps thin labels that nearest-neighbour drops on
    downscaling and closes the scattered -1 holes of projection-derived masks
    (which matters when those auto-labelled images are reused as training
    data). See :func:`cloudlabeller.core.raster.resample_label_mask` for the
    shared one-hot resampling.
    """
    from cloudlabeller.core.raster import resample_label_mask

    return resample_label_mask(mask, size, min_label_fraction=min_label_fraction)


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


def build_tile_set(
    labelled_image_names: list[str],
    image_loader,                       # name -> (H, W, C)
    mask_loader,                        # name -> (H, W) int32 (user + projected)
    spec,                               # ModelSpec: tile_size, overlap, shrink, min_labelled
    projected_weight: float = 0.5,
    projected_names: set[str] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> TrainingSet:
    """The Tiles of every labelled photo, at the network's size: one sample
    per Tile with enough labels, named ``photo#row,col``. ``progress(done,
    total)`` is called after each photo: rendering a photo's mask from the
    cloud can take seconds the first time.

    Tiles are uint8 images and int16 masks; nothing is augmented here."""
    from cloudlabeller.ml.tiles import cut

    projected_names = projected_names or set()
    samples: list[Sample] = []
    total = len(labelled_image_names)
    for k, name in enumerate(labelled_image_names):
        image, mask = image_loader(name), mask_loader(name)
        if progress is not None:
            progress(k + 1, total)
        if mask is None or image is None:          # no image: no normals for it
            continue
        weight = projected_weight if name in projected_names else 1.0
        for (i, j), timg, tmask in cut(image, mask, int(spec.tile_size), float(spec.overlap),
                                       int(spec.shrink), float(spec.min_labelled)):
            samples.append(Sample(name=f"{name}#{i},{j}", image=timg, mask=tmask,
                                  weight=weight))
    return TrainingSet(samples)
