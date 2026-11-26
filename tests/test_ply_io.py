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

"""PLY I/O: io/ply.py writes, trimesh reads (io/geometry.py)."""

import numpy as np
import pytest

from cloudlabeller.io.geometry import load_cloud, load_mesh
from cloudlabeller.io.ply import read_ply, write_ply


def _vertices(n: int, rng) -> np.ndarray:
    v = np.zeros(n, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                           ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
                           ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    for k in ("x", "y", "z"):
        v[k] = rng.normal(size=n)
    nrm = rng.normal(size=(n, 3))
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    for i, k in enumerate(("nx", "ny", "nz")):
        v[k] = nrm[:, i]
    for k in ("red", "green", "blue"):
        v[k] = rng.integers(0, 256, n)
    return v


def _cols(v, keys):
    return np.column_stack([v[k] for k in keys])


def test_binary_ply_round_trip_with_normals_colours_and_faces(tmp_path):
    rng = np.random.default_rng(7)
    v = _vertices(40, rng)
    faces = rng.integers(0, 40, (25, 3)).astype(np.int32)
    path = tmp_path / "m.ply"
    write_ply(path, v, faces)
    assert path.read_bytes().startswith(b"ply\nformat binary_little_endian 1.0\n")

    cloud = load_cloud(path)
    assert cloud.n_points == 40
    np.testing.assert_array_equal(cloud.xyz, _cols(v, "xyz"))
    np.testing.assert_array_equal(cloud.normals, _cols(v, ("nx", "ny", "nz")))
    assert cloud.rgb.dtype == np.uint8
    np.testing.assert_array_equal(cloud.rgb, _cols(v, ("red", "green", "blue")))

    mesh = load_mesh(path)
    np.testing.assert_array_equal(mesh.vertices, _cols(v, "xyz"))
    np.testing.assert_array_equal(mesh.faces, faces)
    np.testing.assert_array_equal(mesh.vertex_colors, _cols(v, ("red", "green", "blue")))

    back = read_ply(path)                       # our own reader agrees
    assert back["vertex"].dtype == v.dtype
    np.testing.assert_array_equal(back["vertex"], v)
    np.testing.assert_array_equal(back["face"]["vertex_indices"], faces)


def test_cloud_without_colours_or_normals(tmp_path):
    v = np.zeros(5, dtype=[("x", "f8"), ("y", "f8"), ("z", "f8")])
    v["x"] = np.arange(5) + 1e6                 # float64 survives
    path = tmp_path / "c.ply"
    write_ply(path, v)
    cloud = load_cloud(path)
    assert cloud.rgb is None and cloud.normals is None
    np.testing.assert_allclose(cloud.xyz[:, 0], v["x"], rtol=1e-7)
    assert read_ply(path)["vertex"].dtype["x"] == np.float64


def test_load_mesh_rejects_a_point_cloud(tmp_path):
    path = tmp_path / "c.ply"
    write_ply(path, _vertices(5, np.random.default_rng(0)))
    with pytest.raises(ValueError, match="no faces"):
        load_mesh(path)


def test_load_mesh_reads_obj(tmp_path):
    import trimesh

    path = tmp_path / "box.obj"
    trimesh.creation.box().export(str(path))
    mesh = load_mesh(path)
    assert mesh.faces.shape == (12, 3) and len(mesh.vertices) == 8
