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

"""Point cloud and mesh I/O.

Two roles:
  * **import** external geometry (PLY/LAS/OBJ/STL …) via PyVista/laspy into the
    plain :class:`PointCloud` / :class:`Mesh` containers;
  * **persist** the project's own dense cloud / mesh as compact ``.npz`` files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cloudlabeller.core.dataset import Mesh, PointCloud


# -- RGB / normal extraction from a PyVista dataset -----------------------
def _extract_rgb(ds) -> np.ndarray | None:
    """Point colours from a PyVista dataset as (N, 3) uint8, or None.

    Checks the common conventions in turn: a packed RGB array under one of
    the usual names, separate red/green/blue scalars, then a 3+-component
    active-scalars array."""
    pd = ds.point_data
    for key in ("RGB", "rgb", "Colors", "colors", "diffuse_color"):
        arr = pd.get(key)
        if arr is not None and arr.ndim == 2 and arr.shape[1] >= 3:
            return np.asarray(arr[:, :3], np.uint8)
    if all(k in pd for k in ("red", "green", "blue")):
        return np.column_stack([pd["red"], pd["green"], pd["blue"]]).astype(np.uint8)
    scal = ds.active_scalars
    if scal is not None and scal.ndim == 2 and scal.shape[1] >= 3:
        return np.asarray(scal[:, :3], np.uint8)
    return None


def _extract_normals(ds) -> np.ndarray | None:
    arr = ds.point_data.get("Normals")
    return np.asarray(arr, np.float32) if arr is not None else None


# -- import (external files) ----------------------------------------------
def load_cloud(path: str | Path) -> PointCloud:
    """Read a point cloud (.ply/.vtk/… via PyVista, .las/.laz via laspy)."""
    path = Path(path)
    if path.suffix.lower() in (".las", ".laz"):
        return _load_las(path)
    import pyvista as pv

    ds = pv.read(str(path))
    return PointCloud(
        xyz=np.asarray(ds.points, np.float32),
        rgb=_extract_rgb(ds),
        normals=_extract_normals(ds),
    )


def _load_las(path: Path) -> PointCloud:
    """Read a .las/.laz cloud via laspy (16-bit colours scaled to 8-bit)."""
    import laspy

    las = laspy.read(str(path))
    xyz = np.column_stack([las.x, las.y, las.z]).astype(np.float32)
    rgb = None
    if {"red", "green", "blue"} <= set(las.point_format.dimension_names):
        # LAS colours are 16-bit; scale to 8-bit.
        rgb = (np.column_stack([las.red, las.green, las.blue]) >> 8).astype(np.uint8)
    return PointCloud(xyz=xyz, rgb=rgb)


def load_mesh(path: str | Path) -> Mesh:
    """Read a triangle mesh (.ply/.obj/.stl/… via PyVista)."""
    import pyvista as pv

    ds = pv.read(str(path)).triangulate()
    if ds.n_cells == 0:
        raise ValueError(f"{path} contains no faces (not a mesh)")
    return Mesh(
        vertices=np.asarray(ds.points, np.float32),
        faces=np.asarray(ds.regular_faces, np.int32),
        vertex_colors=_extract_rgb(ds),
    )


def save_cloud_ply(cloud: PointCloud, path: str | Path) -> None:
    """Write a binary PLY (x,y,z[,nx,ny,nz][,rgb]) — COLMAP poisson_mesher input."""
    from plyfile import PlyData, PlyElement

    fields = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    if cloud.normals is not None:
        fields += [("nx", "f4"), ("ny", "f4"), ("nz", "f4")]
    if cloud.rgb is not None:
        fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    data = np.zeros(cloud.n_points, dtype=fields)
    for i, k in enumerate(("x", "y", "z")):
        data[k] = cloud.xyz[:, i]
    if cloud.normals is not None:
        for i, k in enumerate(("nx", "ny", "nz")):
            data[k] = cloud.normals[:, i]
    if cloud.rgb is not None:
        for i, k in enumerate(("red", "green", "blue")):
            data[k] = cloud.rgb[:, i]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(data, "vertex")]).write(str(path))


# -- internal persistence (.npz) ------------------------------------------
def save_cloud_npz(cloud: PointCloud, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        xyz=cloud.xyz,
        rgb=cloud.rgb if cloud.rgb is not None else np.empty((0, 3), np.uint8),
        normals=cloud.normals if cloud.normals is not None else np.empty((0, 3), np.float32),
    )


def load_cloud_npz(path: str | Path) -> PointCloud:
    d = np.load(path)
    rgb, nrm = d["rgb"], d["normals"]
    return PointCloud(xyz=d["xyz"], rgb=rgb if len(rgb) else None,
                      normals=nrm if len(nrm) else None)


def save_mesh_npz(mesh: Mesh, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        vertices=mesh.vertices,
        faces=mesh.faces,
        vertex_colors=mesh.vertex_colors if mesh.vertex_colors is not None
        else np.empty((0, 3), np.uint8),
    )


def load_mesh_npz(path: str | Path) -> Mesh:
    d = np.load(path)
    vc = d["vertex_colors"]
    return Mesh(vertices=d["vertices"], faces=d["faces"],
               vertex_colors=vc if len(vc) else None)
