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

"""Cloud labels -> per-image mask by splatting visible points into pixels."""

from __future__ import annotations

import numpy as np

from cloudlabeller.core.dataset import Camera, PointCloud
from cloudlabeller.core.label_schema import UNLABELLED_ID
from cloudlabeller.transfer import projection as hb


def cloud_to_image(
    cloud: PointCloud,
    cloud_labels: np.ndarray,
    camera: Camera,
    splat_radius: int = 2,
    name: str | None = None,
    visibility=None,          # optional transfer.visibility.VisibilityIndex
) -> np.ndarray:
    """Render the cloud's labels into an (H, W) int32 mask for one image.

    Each visible, labelled point paints its class into the pixel it projects to
    (with a square splat to fill the gaps between sparse points). Pass ``name``
    and a :class:`VisibilityIndex` to reuse the cached camera↔point visibility.
    """
    h, w = int(camera.height), int(camera.width)
    mask = np.full((h, w), UNLABELLED_ID, dtype=np.int32)

    if visibility is not None and name is not None:
        pt_idx, pix = visibility.get(name, cloud, camera)
    else:
        pt_idx, pix = hb.visible_point_pixels(cloud, camera)
    if pt_idx.size == 0:
        return mask
    classes = cloud_labels[pt_idx]
    keep = classes != UNLABELLED_ID
    cols, rows, classes = pix[keep, 0], pix[keep, 1], classes[keep]

    r = max(splat_radius, 0)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            rr = np.clip(rows + dy, 0, h - 1)
            cc = np.clip(cols + dx, 0, w - 1)
            mask[rr, cc] = classes
    return mask


def cloud_layers_to_image(
    cloud: PointCloud,
    cloud_labels: np.ndarray,
    point_source: np.ndarray,
    point_uncertainty: np.ndarray,
    camera: Camera,
    splat_radius: int = 2,
    name: str | None = None,
    visibility=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """:func:`cloud_to_image`, with each pixel's source and Uncertainty.

    Returns ``(mask, source, uncertainty)``, each ``(H, W)``: the labels as
    int32; the pixel source as uint8 (``CLOUD`` where the point's label was
    hand-drawn, ``PREDICTED`` where it came from predictions, ``NONE``
    unlabelled; :func:`cloudlabeller.core.sources.projected`); the points'
    Uncertainty as float16 (NaN unlabelled or unknown). Every array takes the
    same point at each pixel.
    """
    from cloudlabeller.core import sources as src

    h, w = int(camera.height), int(camera.width)
    mask = np.full((h, w), UNLABELLED_ID, dtype=np.int32)
    source = np.zeros((h, w), np.uint8)
    unc = np.full((h, w), np.nan, np.float16)
    if visibility is not None and name is not None:
        pt_idx, pix = visibility.get(name, cloud, camera)
    else:
        pt_idx, pix = hb.visible_point_pixels(cloud, camera)
    if pt_idx.size == 0:
        return mask, source, unc
    classes = cloud_labels[pt_idx]
    keep = classes != UNLABELLED_ID
    idx = pt_idx[keep]
    cols, rows, classes = pix[keep, 0], pix[keep, 1], classes[keep]
    kinds = src.projected(point_source[idx])
    doubt = np.asarray(point_uncertainty[idx], np.float16)
    r = max(splat_radius, 0)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            rr = np.clip(rows + dy, 0, h - 1)
            cc = np.clip(cols + dx, 0, w - 1)
            mask[rr, cc] = classes
            source[rr, cc] = kinds
            unc[rr, cc] = doubt
    return mask, source, unc
