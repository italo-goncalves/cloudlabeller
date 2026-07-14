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

"""Convert COLMAP cameras into the app's :class:`Camera` model.

Verified against pycolmap 4.0.4:
  * ``Camera.calibration_matrix()`` → K (3x3)
  * ``Image.cam_from_world`` → Rigid3d (world→camera); ``.rotation.matrix()`` = R,
    ``.translation`` = t
  * ``Image.projection_center`` → camera centre in world coords (== -R^T t)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cloudlabeller.core.dataset import Camera, ImageRecord


def _resolve(attr):
    """Return ``attr()`` if it's a method, else ``attr``.

    pycolmap exposes the pose as ``Image.cam_from_world`` — a **method** in 4.x
    (call it) but a property in older builds. Resolving defensively keeps this
    working across versions. ``.rotation`` / ``.translation`` are plain
    attributes (not callable), so they pass through unchanged.
    """
    return attr() if callable(attr) else attr


def colmap_to_cameras(reconstruction, image_dir: str | Path) -> list[ImageRecord]:
    """Build :class:`ImageRecord`s (with solved cameras) from a reconstruction.

    Only images with a solved pose are returned. ``path`` resolves the image
    name against ``image_dir``.
    """
    image_dir = Path(image_dir)
    records: list[ImageRecord] = []
    for image_id, image in reconstruction.images.items():
        if not image.has_pose:
            continue
        cam = reconstruction.cameras[image.camera_id]
        cfw = _resolve(image.cam_from_world)  # Rigid3d, world→camera
        R = np.asarray(_resolve(cfw.rotation).matrix(), dtype=float)
        t = np.asarray(_resolve(cfw.translation), dtype=float)
        K = np.asarray(cam.calibration_matrix(), dtype=float)
        records.append(
            ImageRecord(
                image_id=image_id,
                path=image_dir / image.name,
                camera=Camera(
                    K=K, R=R, t=t,
                    width=cam.width, height=cam.height,
                    distortion=extract_distortion(cam.model_name, cam.params),
                    model=cam.model_name,
                ),
            )
        )
    return records


def extract_distortion(model_name: str, params) -> np.ndarray | None:
    """Distortion coefficients from COLMAP camera params, by model.

    Returned in the model's own coefficient order (interpreted by
    ``transfer.hylite_bridge._apply_distortion``):
      SIMPLE_RADIAL -> [k]              RADIAL -> [k1, k2]
      OPENCV        -> [k1, k2, p1, p2] FULL_OPENCV -> [k1, k2, p1, p2, k3..k6]
    Pinhole models return None. Ignoring these was the source of large
    (~60-100 px at the borders) transfer offsets.
    """
    params = np.asarray(params, dtype=float)
    coeffs = {
        "SIMPLE_RADIAL": params[3:4],       # f, cx, cy, k
        "RADIAL": params[3:5],              # f, cx, cy, k1, k2
        "OPENCV": params[4:8],              # fx, fy, cx, cy, k1, k2, p1, p2
        "FULL_OPENCV": params[4:12],        # ... k1, k2, p1, p2, k3, k4, k5, k6
    }.get(model_name)
    if coeffs is None or not np.any(coeffs):
        return None                          # pinhole or all-zero coefficients
    return coeffs.copy()


def camera_to_hylite(camera: Camera):
    """Convert an app :class:`Camera` to a ``hylite.project.Camera`` (PLACEHOLDER).

    Centralised here so the projection convention is defined in one place. Pin to
    the installed hylite API when wiring label transfer.
    """
    raise NotImplementedError("Camera → hylite.Camera conversion (see DESIGN.md §6).")
