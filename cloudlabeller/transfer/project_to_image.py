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
from cloudlabeller.transfer import hylite_bridge as hb


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
