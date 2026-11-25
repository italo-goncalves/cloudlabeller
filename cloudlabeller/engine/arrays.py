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

"""Array files: how large arrays cross between the window and the Engine.

An Array file is raw little-endian, C-ordered bytes with no header, in the
Project's ``.arrays/`` folder. The message carries its descriptor::

    {"path": ".arrays/mask/IMG_0001.JPG.bin", "dtype": "int32", "shape": [3000, 4000]}

``path`` is relative to the Project root and uses ``/``, so it means the
same file on both sides of the Windows/WSL boundary.

- Outbound (Engine → window): the Engine writes the file at a stable name per
  purpose; it stays valid until the next request for the same array.
- Inbound (window → Engine): the window writes the file under ``.arrays/``;
  the Engine deletes it once read.

``.arrays/`` holds nothing that isn't derived or in flight: the Engine empties
it when a Project opens and when it closes.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from cloudlabeller.engine.protocol import InvalidParams

ARRAYS_DIR = ".arrays"

DTYPES = ("uint8", "int8", "uint16", "int16", "uint32", "int32", "float32", "float64")


def arrays_dir(root: Path) -> Path:
    return Path(root) / ARRAYS_DIR


def clear(root: Path) -> None:
    """Empty the Project's ``.arrays/`` folder (on open and on close)."""
    shutil.rmtree(arrays_dir(root), ignore_errors=True)


def write(root: Path, name: str, array: np.ndarray) -> dict:
    """Write ``array`` as ``.arrays/<name>.bin``; return its descriptor."""
    array = np.asarray(array)
    if array.dtype.name not in DTYPES:
        raise TypeError(f"dtype {array.dtype.name} is not an Array file dtype")
    rel = f"{ARRAYS_DIR}/{name}.bin"
    path = Path(root) / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.ascontiguousarray(array.astype(array.dtype.newbyteorder("<"), copy=False))
    tmp = path.with_suffix(".bin.part")
    data.tofile(tmp)
    tmp.replace(path)                       # a reader never sees a half-written file
    return {"path": rel, "dtype": array.dtype.name, "shape": list(array.shape)}


def read(root: Path, ref: object, *, dtype: str | None = None,
         ndim: int | None = None, consume: bool = True) -> np.ndarray:
    """Read an inbound Array file described by ``ref``.

    The path must lie inside ``.arrays/``; the size must match dtype × shape.
    ``consume`` deletes the file once read.
    """
    if not isinstance(ref, dict) or not {"path", "dtype", "shape"} <= set(ref):
        raise InvalidParams("an Array file needs path, dtype and shape")
    if ref["dtype"] not in DTYPES:
        raise InvalidParams(f"dtype {ref['dtype']!r} is not an Array file dtype",
                            dtype=ref["dtype"])
    if dtype is not None and ref["dtype"] != dtype:
        raise InvalidParams(f"expected dtype {dtype}, got {ref['dtype']}",
                            expected=dtype, got=ref["dtype"])
    shape = ref["shape"]
    if (not isinstance(shape, list)
            or not all(isinstance(s, int) and s >= 0 for s in shape)):
        raise InvalidParams("shape must be a list of non-negative integers")
    if ndim is not None and len(shape) != ndim:
        raise InvalidParams(f"expected {ndim} dimensions, got {len(shape)}",
                            expected=ndim, got=len(shape))

    base = arrays_dir(root).resolve()
    path = (Path(root) / str(ref["path"])).resolve()
    if base not in path.parents:
        raise InvalidParams("an Array file must be inside the Project's .arrays/ folder",
                            path=str(ref["path"]))
    if not path.is_file():
        raise InvalidParams("the Array file does not exist", path=str(ref["path"]))

    dt = np.dtype(ref["dtype"]).newbyteorder("<")
    expected = int(np.prod(shape, dtype=np.int64)) * dt.itemsize
    if path.stat().st_size != expected:
        raise InvalidParams("the Array file's size does not match its dtype and shape",
                            path=str(ref["path"]), expected=expected,
                            got=path.stat().st_size)
    array = np.fromfile(path, dtype=dt).reshape(shape).astype(ref["dtype"])
    if consume:
        path.unlink(missing_ok=True)
    return array
