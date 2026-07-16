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

"""Apply a similarity transform to EVERY product of a project, in place.

Shared by georeferencing (align to GPS) and reprojection (change of CRS):
sparse cloud + cameras, dense cloud (points + normals), mesh, and the COLMAP
model on disk (via pycolmap, so later MVS runs / confidence cleaning match
the new coordinates). Label arrays are per-point/per-vertex and follow their
geometry unchanged; the visibility cache is wiped (its entries can never
match the moved geometry again).

World transform convention: ``X' = s · R @ X + t``.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np

ProgressFn = Callable[[float, str], None]


def apply_similarity_to_products(root: str | Path, s: float, R: np.ndarray,
                                 t: np.ndarray, progress: ProgressFn,
                                 model_backup_dir: Path | None = None) -> None:
    """Transform sparse+cameras, dense, mesh and the COLMAP model by
    ``X' = s·R@X + t``; wipe the visibility cache.

    ``model_backup_dir``: move the original COLMAP model there before the
    swap (georeferencing keeps ``sparse_prealigned/`` as the restore point);
    None replaces the model without a backup (reprojection).
    """
    import pycolmap

    from cloudlabeller.io.geometry import (
        load_cloud_npz,
        load_mesh_npz,
        save_cloud_npz,
        save_mesh_npz,
    )
    from cloudlabeller.photogrammetry.extract import best_model_dir
    from cloudlabeller.photogrammetry.georef import (
        rotate_normals,
        transform_camera,
        transform_points,
    )
    from cloudlabeller.photogrammetry.pipeline import (
        ReconstructResult,
        load_result,
        save_result,
    )

    root = Path(root)
    rec_dir = root / "reconstruction"
    model_dir = best_model_dir(rec_dir / "sparse")
    with tempfile.TemporaryDirectory(prefix="cl_transform_") as tmp:
        # COLMAP model first, into a temp dir — the swap happens last, so a
        # crash mid-way never leaves a half-transformed model on disk.
        progress(0.3, "Transforming the COLMAP model…")
        transformed_dir = Path(tmp) / "transformed"
        transformed_dir.mkdir()
        rec = pycolmap.Reconstruction(str(model_dir))
        rec.transform(pycolmap.Sim3d(
            np.hstack([s * R, np.asarray(t, np.float64)[:, None]])))
        rec.write(str(transformed_dir))

        progress(0.5, "Transforming sparse cloud + cameras…")
        result = load_result(rec_dir, root / "images")
        result.cloud.xyz = transform_points(result.cloud.xyz, s, R, t)
        for record in result.images:
            if record.camera is not None:
                record.camera = transform_camera(record.camera, s, R, t)
        save_result(ReconstructResult(cloud=result.cloud, images=result.images),
                    rec_dir)

        dense_path = rec_dir / "dense.npz"
        if dense_path.exists():
            progress(0.6, "Transforming dense cloud…")
            dense = load_cloud_npz(dense_path)
            dense.xyz = transform_points(dense.xyz, s, R, t)
            if dense.normals is not None:
                dense.normals = rotate_normals(dense.normals, R)
            save_cloud_npz(dense, dense_path)

        mesh_path = root / "products" / "mesh.npz"
        if mesh_path.exists():
            progress(0.8, "Transforming mesh…")
            mesh = load_mesh_npz(mesh_path)
            mesh.vertices = transform_points(mesh.vertices, s, R, t)
            save_mesh_npz(mesh, mesh_path)

        progress(0.9, "Replacing the COLMAP model…")
        if model_backup_dir is not None:
            model_backup_dir.mkdir(parents=True)
            shutil.move(str(model_dir), str(model_backup_dir / model_dir.name))
        else:
            shutil.rmtree(model_dir)
        shutil.move(str(transformed_dir), str(model_dir))

    # Old visibility cache entries can never match again — free the space.
    pmap = rec_dir / "pmap"
    if pmap.exists():
        for f in pmap.glob("*.npz"):
            f.unlink(missing_ok=True)
