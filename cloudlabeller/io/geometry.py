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
  * **import** external geometry (PLY/OBJ/STL/OFF … via trimesh, LAS/LAZ via
    laspy) into the plain :class:`PointCloud` / :class:`Mesh` containers;
  * **persist** the project's own dense cloud / mesh as compact ``.npz`` files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cloudlabeller.core.dataset import Mesh, PointCloud

# Per-vertex colour property names, in the order they are tried.
_RGB_NAMES = (("red", "green", "blue"), ("r", "g", "b"),
              ("diffuse_red", "diffuse_green", "diffuse_blue"))


# -- RGB / normal extraction from a trimesh geometry ----------------------
def _ply_vertex(geom) -> np.ndarray | None:
    """The raw PLY vertex record array trimesh keeps, or None (not a PLY)."""
    raw = (getattr(geom, "metadata", None) or {}).get("_ply_raw") or {}
    vertex = raw.get("vertex") or {}
    data = vertex.get("data")
    return data if data is not None and data.dtype.names else None


def _as_rgb(arr) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.dtype.kind == "f":          # float colours in [0, 1]
        arr = np.clip(arr * 255.0 + 0.5, 0, 255)
    return arr.astype(np.uint8)


def _extract_rgb(geom, n: int) -> np.ndarray | None:
    """Vertex colours as (n, 3) uint8, or None when the file has none."""
    raw = _ply_vertex(geom)
    if raw is not None:
        names = set(raw.dtype.names)
        for keys in _RGB_NAMES:
            if set(keys) <= names:
                return _as_rgb(np.column_stack([raw[k] for k in keys]))
        return None
    if hasattr(geom, "faces"):           # a Trimesh
        visual = geom.visual
        if getattr(visual, "kind", None) == "vertex":
            return _as_rgb(visual.vertex_colors[:, :3])
        return None
    colors = getattr(geom, "colors", None)  # a PointCloud
    if colors is not None and len(colors) == n and np.ndim(colors) == 2:
        return _as_rgb(colors[:, :3])
    return None


def _extract_normals(geom) -> np.ndarray | None:
    """Per-vertex normals stored in the file (never computed), or None."""
    raw = _ply_vertex(geom)
    if raw is not None and {"nx", "ny", "nz"} <= set(raw.dtype.names):
        return np.column_stack([raw["nx"], raw["ny"], raw["nz"]]).astype(np.float32)
    return None


def _read_geometry(path: Path, force: str | None = None):
    import trimesh

    geom = trimesh.load(str(path), process=False, force=force)
    if isinstance(geom, trimesh.Scene):
        geom = geom.to_geometry() if hasattr(geom, "to_geometry") else geom.dump(concatenate=True)
    return geom


# -- import (external files) ----------------------------------------------
def load_cloud(path: str | Path) -> PointCloud:
    """Read a point cloud (.ply/.obj/… via trimesh, .las/.laz via laspy).

    For a mesh file the cloud is its vertices."""
    path = Path(path)
    if path.suffix.lower() in (".las", ".laz"):
        return _load_las(path)
    geom = _read_geometry(path)
    xyz = np.asarray(geom.vertices, np.float32)
    return PointCloud(
        xyz=xyz,
        rgb=_extract_rgb(geom, len(xyz)),
        normals=_extract_normals(geom),
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
    """Read a triangle mesh (.ply/.obj/.stl/.off/… via trimesh)."""
    geom = _read_geometry(Path(path), force="mesh")
    faces = np.asarray(getattr(geom, "faces", np.empty((0, 3))), np.int32)
    if len(faces) == 0:
        raise ValueError(f"{path} contains no faces (not a mesh)")
    vertices = np.asarray(geom.vertices, np.float32)
    return Mesh(
        vertices=vertices,
        faces=faces.reshape(-1, 3),
        vertex_colors=_extract_rgb(geom, len(vertices)),
    )


def save_cloud_ply(cloud: PointCloud, path: str | Path) -> None:
    """Write a binary PLY (x,y,z[,nx,ny,nz][,rgb]) — COLMAP poisson_mesher input."""
    from cloudlabeller.io.ply import write_ply

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
    write_ply(path, data)


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
