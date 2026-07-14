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

"""Export project products to common interchange formats.

Clouds carry their labels (-1 = unlabelled) in every format:
  * CSV — ``x,y,z,red,green,blue,label`` header + one row per point.
  * LAS — RGB point format, label as an extra ``label`` (int32) dimension.
  * PLY — binary, with a ``label`` (int32) vertex property.
Meshes export via trimesh: PLY (binary, vertex colours), OBJ, STL (no colours).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from cloudlabeller.core.dataset import Mesh, PointCloud

ProgressFn = Callable[[float, str], None]
_NOOP: ProgressFn = lambda f, m="": None

CLOUD_FORMATS = (".las", ".csv", ".ply")
MESH_FORMATS = (".ply", ".obj", ".stl")


def _cloud_columns(cloud: PointCloud, labels: np.ndarray | None):
    rgb = (cloud.rgb if cloud.rgb is not None
           else np.zeros((cloud.n_points, 3), np.uint8))
    lab = (np.asarray(labels, np.int32) if labels is not None
           and len(labels) == cloud.n_points
           else np.full(cloud.n_points, -1, np.int32))
    return rgb, lab


def export_cloud_csv(cloud: PointCloud, labels: np.ndarray | None,
                     path: str | Path, progress: ProgressFn = _NOOP) -> None:
    rgb, lab = _cloud_columns(cloud, labels)
    n = cloud.n_points
    chunk = 500_000
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("x,y,z,red,green,blue,label\n")
        for i in range(0, n, chunk):
            j = min(i + chunk, n)
            block = np.column_stack([cloud.xyz[i:j], rgb[i:j], lab[i:j]])
            np.savetxt(f, block, fmt="%.6f,%.6f,%.6f,%d,%d,%d,%d")
            progress(j / n, f"CSV: {j:,} / {n:,} points")


def export_cloud_las(cloud: PointCloud, labels: np.ndarray | None,
                     path: str | Path, progress: ProgressFn = _NOOP) -> None:
    import laspy

    rgb, lab = _cloud_columns(cloud, labels)
    xyz = np.asarray(cloud.xyz, np.float64)
    progress(0.1, "LAS: building header…")
    header = laspy.LasHeader(point_format=2, version="1.2")   # xyz + RGB
    header.offsets = xyz.min(axis=0)
    # Scale so the full extent fits int32 with headroom; floor at 1 µm.
    extent = float((xyz.max(axis=0) - xyz.min(axis=0)).max())
    header.scales = np.full(3, max(1e-6, extent / 2e9))
    header.add_extra_dim(laspy.ExtraBytesParams(name="label", type=np.int32,
                                                description="CloudLabeller class id"))
    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    las.red, las.green, las.blue = (rgb.astype(np.uint16) * 257).T   # 8 -> 16 bit
    las["label"] = lab
    progress(0.5, f"LAS: writing {cloud.n_points:,} points…")
    las.write(str(path))
    progress(1.0, "LAS written")


def export_cloud_ply(cloud: PointCloud, labels: np.ndarray | None,
                     path: str | Path, progress: ProgressFn = _NOOP) -> None:
    from plyfile import PlyData, PlyElement

    rgb, lab = _cloud_columns(cloud, labels)
    fields = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    if cloud.normals is not None:
        fields += [("nx", "f4"), ("ny", "f4"), ("nz", "f4")]
    fields += [("red", "u1"), ("green", "u1"), ("blue", "u1"), ("label", "i4")]
    progress(0.2, f"PLY: packing {cloud.n_points:,} points…")
    data = np.zeros(cloud.n_points, dtype=fields)
    for i, k in enumerate(("x", "y", "z")):
        data[k] = cloud.xyz[:, i]
    if cloud.normals is not None:
        for i, k in enumerate(("nx", "ny", "nz")):
            data[k] = cloud.normals[:, i]
    for i, k in enumerate(("red", "green", "blue")):
        data[k] = rgb[:, i]
    data["label"] = lab
    progress(0.6, "PLY: writing…")
    PlyData([PlyElement.describe(data, "vertex")]).write(str(path))
    progress(1.0, "PLY written")


def export_cloud(cloud: PointCloud, labels: np.ndarray | None,
                 path: str | Path, progress: ProgressFn = _NOOP) -> None:
    """Dispatch on the file extension (one of CLOUD_FORMATS)."""
    ext = Path(path).suffix.lower()
    writer = {".csv": export_cloud_csv, ".las": export_cloud_las,
              ".ply": export_cloud_ply}.get(ext)
    if writer is None:
        raise ValueError(f"Unsupported cloud format {ext!r} — use one of "
                         f"{', '.join(CLOUD_FORMATS)}")
    writer(cloud, labels, path, progress)


def export_mesh(mesh: Mesh, path: str | Path, progress: ProgressFn = _NOOP) -> None:
    """Export via trimesh; format from the extension (PLY keeps vertex colours,
    OBJ mostly does, STL is geometry-only)."""
    ext = Path(path).suffix.lower()
    if ext not in MESH_FORMATS:
        raise ValueError(f"Unsupported mesh format {ext!r} — use one of "
                         f"{', '.join(MESH_FORMATS)}")
    import trimesh

    progress(0.2, f"Building mesh ({len(mesh.faces):,} faces)…")
    tm = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces,
                         vertex_colors=mesh.vertex_colors, process=False)
    progress(0.6, f"Writing {ext.upper().lstrip('.')}…")
    tm.export(str(path))
    progress(1.0, "Mesh written")
