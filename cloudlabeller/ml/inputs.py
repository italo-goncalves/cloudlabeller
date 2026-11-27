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


"""What the U-Net is given for one photo: its RGB, and with the
experimental ``ModelSpec.normals`` its normal map in the camera's frame
(``Survey.normal_map``) as three more channels."""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("cloudlabeller.ml")


def load_rgb(path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def load_input(survey, name: str, path, normals: bool) -> np.ndarray | None:
    """(H, W, 3) RGB, or (H, W, 6) RGB + normals; None when the normals are
    asked for and the photo has none (no camera, or no dense cloud)."""
    rgb = load_rgb(path)
    if not normals:
        return rgb
    nmap = survey.normal_map(name, "camera")
    if nmap is None:
        return None
    if nmap.shape[:2] != rgb.shape[:2]:
        log.warning("%s: the photo is %s but its camera %s; no normals", name,
                    rgb.shape[:2], nmap.shape[:2])
        return None
    return np.concatenate([rgb, nmap], axis=2)


def can_have_normals(survey, name: str) -> bool:
    """Whether ``name`` can get a normal map: it has a camera and the Survey
    a dense cloud with normals."""
    record = survey.dataset.image_by_name(name)
    return record is not None and record.camera is not None and survey.has_normals()
