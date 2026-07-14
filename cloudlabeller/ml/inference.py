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

"""Inference — apply the trained U-Net to images.

``predict`` returns the mask at the MODEL's resolution (e.g. 768x512), not the
photo's: predictions are stored at that size (a full-res 21 Mpx int32 mask is
~84 MB) and upscaled on demand by ``Project.prediction_mask``. ``predict_all``
orchestrates a whole batch and writes the .npy files the project reads back.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

ProgressFn = Callable[[float, str], None]


def predict(model, image: np.ndarray) -> np.ndarray:
    """Run the model on one (H, W, C) uint8 image; return an (h, w) int32 mask
    at the model's input resolution (argmax over class probabilities)."""
    import cv2

    _, in_h, in_w, _ = model.input_shape
    x = image
    if (image.shape[0], image.shape[1]) != (in_h, in_w):
        x = cv2.resize(image, (in_w, in_h), interpolation=cv2.INTER_AREA)
    x = x.astype(np.float32)
    if x.max() > 1.0:
        x /= 255.0
    prob = model.predict(x[None], verbose=0)[0]      # (h, w, n_classes)
    return np.argmax(prob, axis=-1).astype(np.int32)


def predict_all(
    model,
    image_names: list[str],
    image_loader,                       # name -> (H, W, C) uint8
    out_dir: str | Path,
    progress: ProgressFn = lambda f, m: None,
) -> dict[str, Path]:
    """Predict masks for every image; save as .npy (at model resolution);
    return name -> path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    total = len(image_names)
    for i, name in enumerate(image_names, 1):
        mask = predict(model, image_loader(name))
        path = out_dir / f"{name}.npy"
        np.save(path, mask.astype(np.int32))
        written[name] = path
        progress(i / total, f"Predicted {name} ({i}/{total})")
    return written
