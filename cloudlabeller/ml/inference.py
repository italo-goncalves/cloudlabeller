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

"""Inference — apply the trained U-Net to photos, Tile by Tile.

``predict_photo`` cuts a photo into the model's Tiles, predicts them and
blends them (:mod:`cloudlabeller.ml.tiles`), returning the classes and the
Uncertainty at the photo's size divided by the model's shrink factor.
Predictions are stored at that size (a full-res 21 Mpx int32 mask is
~84 MB) and upscaled on demand by ``Survey.prediction_mask``.
``predict_all`` writes one ``<name>.npz`` per photo: ``labels``,
``uncertainty`` and the model's name.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

ProgressFn = Callable[[float, str], None]


def predict(model, image: np.ndarray, spec, n_classes: int) -> tuple[np.ndarray, np.ndarray]:
    """One photo's (classes, Uncertainty) at its size / ``spec.shrink``."""
    from cloudlabeller.ml.tiles import predict_photo

    return predict_photo(lambda x: model.predict(x, verbose=0), image,
                         int(spec.tile_size), float(spec.overlap), int(spec.shrink), n_classes)


def predict_all(
    model,
    image_names: list[str],
    image_loader,                       # name -> (H, W, C) uint8
    out_dir: str | Path,
    spec,
    n_classes: int,
    model_name: str = "",
    progress: ProgressFn = lambda f, m: None,
) -> dict[str, float]:
    """Predict every photo; save ``<name>.npz``; return name -> its mean
    Uncertainty."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, float] = {}
    total = len(image_names)
    for i, name in enumerate(image_names, 1):
        image = image_loader(name)
        if image is None:                          # a normals model, no normals here
            progress(i / total, f"Skipped {name}: no normals ({i}/{total})")
            continue
        labels, uncertainty = predict(model, image, spec, n_classes)
        np.savez_compressed(out_dir / f"{name}.npz", labels=labels.astype(np.int16),
                            uncertainty=uncertainty, model=np.array(model_name))
        (out_dir / f"{name}.npy").unlink(missing_ok=True)      # a 0.3.0 prediction
        written[name] = float(np.mean(uncertainty, dtype=np.float64))
        progress(i / total, f"Predicted {name} ({i}/{total})")
    return written
