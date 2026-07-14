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

"""Headless reconstruction entry point: images → (sparse cloud, cameras).

This is decoupled from Qt so it can be tested/scripted directly and run on a
worker thread by the UI.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from cloudlabeller.core.dataset import Camera, ImageRecord, PointCloud
from cloudlabeller.photogrammetry.adjacency import ImageAdjacency, covisibility_from_reconstruction
from cloudlabeller.photogrammetry.cameras import colmap_to_cameras
from cloudlabeller.photogrammetry.extract import reconstruction_to_cloud
from cloudlabeller.photogrammetry.sfm import run_sfm

ProgressFn = Callable[[float, str], None]

CLOUD_FILE = "cloud.npz"
CAMERAS_FILE = "cameras.json"


@dataclass
class ReconstructResult:
    cloud: PointCloud
    images: list[ImageRecord]
    adjacency: ImageAdjacency | None = None    # image-overlap graph (covisibility)

    def summary(self) -> str:
        return f"{self.cloud.n_points} points, {len(self.images)} cameras"


def reconstruct(
    image_dir: str | Path,
    workspace: str | Path,
    matcher: str = "spatial",
    progress: ProgressFn = lambda f, m="": None,
    use_gpu: bool = False,
    single_camera: bool = True,
    camera_model: str = "SIMPLE_RADIAL",
    max_image_size: int = 3200,
    colmap_binary: str | None = None,
) -> ReconstructResult:
    """Run SfM and return the sparse cloud plus solved cameras.

    Note: COLMAP's GPU SIFT creates its own OpenGL context. Run this in a
    *separate process* (see :mod:`cloudlabeller.photogrammetry.run_cli`) when a
    GUI with a live OpenGL/VTK context is present, or two GL contexts in one
    process can crash. ``use_gpu`` defaults to False (CPU SIFT) for safety.
    See :func:`run_sfm` for the remaining options.
    """
    sfm = run_sfm(image_dir, workspace, matcher=matcher, progress=progress,
                  use_gpu=use_gpu, single_camera=single_camera,
                  camera_model=camera_model, max_image_size=max_image_size,
                  colmap_binary=colmap_binary)
    cloud = reconstruction_to_cloud(sfm.reconstruction)
    images = colmap_to_cameras(sfm.reconstruction, image_dir)
    adjacency = None
    try:
        progress(0.99, "Building the image-overlap graph…")
        adjacency = covisibility_from_reconstruction(sfm.reconstruction)
    except Exception:                  # the graph is an optimisation, never fatal
        pass
    return ReconstructResult(cloud=cloud, images=images, adjacency=adjacency)


# -- serialisation: lets the subprocess hand results back to the GUI ------
# cameras.json stores image *filenames* (not absolute paths); they resolve
# against the project image store, so a project stays portable if moved.
def _default_images_dir(workspace: str | Path) -> Path:
    """The project image store sits next to ``reconstruction/`` → ``../images``."""
    return Path(workspace).parent / "images"


def save_result(result: ReconstructResult, workspace: str | Path) -> None:
    ws = Path(workspace)
    ws.mkdir(parents=True, exist_ok=True)
    c = result.cloud
    np.savez(
        ws / CLOUD_FILE,
        xyz=c.xyz,
        rgb=c.rgb if c.rgb is not None else np.empty((0, 3), np.uint8),
    )
    cams = [
        {
            "image_id": r.image_id,
            "name": Path(r.path).name,        # filename only; resolved on load
            "width": r.camera.width,
            "height": r.camera.height,
            "model": r.camera.model,
            "K": r.camera.K.tolist(),
            "R": r.camera.R.tolist(),
            "t": r.camera.t.tolist(),
            # Distortion coefficients — dropping them shifted transfers ~100 px
            # at the image corners.
            "dist": None if r.camera.distortion is None else r.camera.distortion.tolist(),
        }
        for r in result.images
        if r.camera is not None
    ]
    (ws / CAMERAS_FILE).write_text(json.dumps(cams), encoding="utf-8")
    if result.adjacency is not None:
        result.adjacency.save(ws)


def load_cameras(workspace: str | Path) -> dict[str, Camera]:
    """Map image filename -> solved :class:`Camera` from ``cameras.json``.

    Handles the current "name" format and the legacy "path" format.
    """
    path = Path(workspace) / CAMERAS_FILE
    if not path.exists():
        return {}
    out: dict[str, Camera] = {}
    for c in json.loads(path.read_text(encoding="utf-8")):
        name = c.get("name") or Path(c["path"]).name
        dist = c.get("dist")
        out[name] = Camera(
            K=np.asarray(c["K"]), R=np.asarray(c["R"]), t=np.asarray(c["t"]),
            width=c["width"], height=c["height"], model=c["model"],
            distortion=None if dist is None else np.asarray(dist, dtype=float),
        )
    return out


def load_result(workspace: str | Path, images_dir: str | Path | None = None) -> ReconstructResult:
    """Load the saved cloud + solved cameras; image paths resolve to the store."""
    ws = Path(workspace)
    img_dir = Path(images_dir) if images_dir is not None else _default_images_dir(ws)
    data = np.load(ws / CLOUD_FILE)
    rgb = data["rgb"]
    cloud = PointCloud(xyz=data["xyz"], rgb=rgb if len(rgb) else None)
    images = [
        ImageRecord(image_id=i, path=img_dir / name, camera=cam)
        for i, (name, cam) in enumerate(load_cameras(ws).items())
    ]
    return ReconstructResult(cloud=cloud, images=images)
