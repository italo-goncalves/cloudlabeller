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

"""Binary PLY writing (and reading back what we write), with no dependency.

CloudLabeller writes PLY for COLMAP's Poisson mesher and for exports: one
``vertex`` element with scalar properties and, for meshes, a triangle
``face`` element. :func:`write_ply` takes the vertex data as a numpy
structured array whose field names and dtypes become the PLY properties.
:func:`read_ply` reads such files back (binary little-endian, triangle faces);
it is meant for round-trips and tests. External PLY files of any flavour are
imported through trimesh in :mod:`cloudlabeller.io.geometry`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# numpy dtype char/size -> PLY type name, and back.
_TO_PLY = {"i1": "char", "u1": "uchar", "i2": "short", "u2": "ushort",
           "i4": "int", "u4": "uint", "f4": "float", "f8": "double"}
_FROM_PLY = {v: k for k, v in _TO_PLY.items()}
_FROM_PLY.update({"int8": "i1", "uint8": "u1", "int16": "i2", "uint16": "u2",
                  "int32": "i4", "uint32": "u4", "float32": "f4", "float64": "f8"})


def _ply_type(dt: np.dtype) -> str:
    key = f"{dt.kind}{dt.itemsize}"
    if key not in _TO_PLY:
        raise TypeError(f"no PLY type for numpy dtype {dt}")
    return _TO_PLY[key]


def write_ply(path: str | Path, vertex: np.ndarray,
              faces: np.ndarray | None = None) -> None:
    """Write a binary little-endian PLY.

    ``vertex`` is a structured array (one field per scalar property, e.g.
    ``x, y, z`` as ``f4``/``f8``, ``red, green, blue`` as ``u1``); ``faces``
    is an optional ``(M, 3)`` integer array of triangle vertex indices."""
    vertex = np.asarray(vertex)
    if vertex.dtype.names is None:
        raise TypeError("vertex must be a numpy structured array")
    lines = ["ply", "format binary_little_endian 1.0",
             f"element vertex {len(vertex)}"]
    le_fields = []
    for name in vertex.dtype.names:
        dt = vertex.dtype[name]
        lines.append(f"property {_ply_type(dt)} {name}")
        le_fields.append((name, dt.newbyteorder("<")))
    face_block = b""
    if faces is not None:
        faces = np.asarray(faces)
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"faces must be (M, 3), got {faces.shape}")
        lines += [f"element face {len(faces)}",
                  "property list uchar int vertex_indices"]
        rec = np.empty(len(faces), dtype=[("n", "u1"), ("idx", "<i4", (3,))])
        rec["n"] = 3
        rec["idx"] = faces
        face_block = rec.tobytes()
    lines.append("end_header")
    header = ("\n".join(lines) + "\n").encode("ascii")
    body = vertex.astype(np.dtype(le_fields), copy=False).tobytes()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header)
        f.write(body)
        f.write(face_block)


def read_ply(path: str | Path) -> dict[str, np.ndarray]:
    """Read a binary little-endian PLY into ``{element name: structured array}``.

    Scalar properties become fields; a ``vertex_indices`` list property must
    hold triangles and becomes an ``(3,)``-shaped ``i4`` field."""
    data = Path(path).read_bytes()
    end = data.find(b"end_header")
    if not data.startswith(b"ply") or end < 0:
        raise ValueError(f"{path} is not a PLY file")
    body_start = data.index(b"\n", end) + 1
    elements: list[tuple[str, int, list]] = []
    for raw in data[:end].decode("ascii").splitlines():
        tok = raw.split()
        if not tok:
            continue
        if tok[0] == "format" and tok[1] != "binary_little_endian":
            raise ValueError(f"{path}: only binary_little_endian is supported")
        if tok[0] == "element":
            elements.append((tok[1], int(tok[2]), []))
        elif tok[0] == "property":
            elements[-1][2].append(tok[1:])
    out: dict[str, np.ndarray] = {}
    offset = body_start
    for name, count, props in elements:
        if any(p[0] == "list" for p in props):
            if len(props) != 1:
                raise ValueError(f"{path}: element {name!r} mixes list and scalars")
            _, cnt_t, idx_t, pname = props[0]
            rec_dt = np.dtype([("n", "<" + _FROM_PLY[cnt_t]),
                               ("idx", "<" + _FROM_PLY[idx_t], (3,))])
            rec = np.frombuffer(data, rec_dt, count, offset)
            if count and not np.all(rec["n"] == 3):
                raise ValueError(f"{path}: only triangle faces are supported")
            arr = np.empty(count, dtype=[(pname, "i4", (3,))])
            arr[pname] = rec["idx"]
            offset += rec_dt.itemsize * count
        else:
            dt = np.dtype([(p[1], "<" + _FROM_PLY[p[0]]) for p in props])
            arr = np.frombuffer(data, dt, count, offset).copy()
            offset += dt.itemsize * count
        out[name] = arr
    return out
