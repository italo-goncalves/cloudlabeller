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

"""A Reconstruction: one run of photogrammetry for a Survey (ADR-0005).

It names the products it uses (one per step: ``sfm``, ``mvs``, ``mesh``,
each a folder under the Survey's ``products/``), the sparse points it has
removed, and its Alignment: the similarity ``X' = s·R@X + t`` that carries
its products, kept in their own SfM frame, into the Project frame. Products
are never transformed; every reader of geometry that needs the Project
frame goes through :meth:`Reconstruction.frame_points` and friends.

On disk, ``surveys/<sid>/reconstructions/<rid>/`` holds ``recon.json``, the
removed-points mask ``keep.npy`` and the label arrays in ``labels/``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cloudlabeller.core.dataset import Camera

RECON_FILE = "recon.json"
KEEP_FILE = "keep.npy"
SLOTS = ("sfm", "mvs", "mesh")


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# -- similarity transforms ------------------------------------------------------
def identity_alignment(source: str) -> dict:
    return {"s": 1.0, "R": np.eye(3).tolist(), "t": [0.0, 0.0, 0.0], "source": source}


def alignment_parts(alignment: dict) -> tuple[float, np.ndarray, np.ndarray]:
    return (float(alignment["s"]), np.asarray(alignment["R"], np.float64).reshape(3, 3),
            np.asarray(alignment["t"], np.float64).reshape(3))


def compose(outer: dict, inner: dict) -> dict:
    """The similarity ``outer ∘ inner`` (apply ``inner`` first), keeping
    ``inner``'s other fields (where it came from, its fit)."""
    s1, R1, t1 = alignment_parts(outer)
    s2, R2, t2 = alignment_parts(inner)
    out = dict(inner)
    out.update(s=s1 * s2, R=(R1 @ R2).tolist(), t=(s1 * R1 @ t2 + t1).tolist())
    return out


def is_identity(alignment: dict) -> bool:
    s, R, t = alignment_parts(alignment)
    return abs(s - 1.0) < 1e-12 and np.allclose(R, np.eye(3)) and not np.any(t)


def apply_points(alignment: dict | None, xyz: np.ndarray) -> np.ndarray:
    """``xyz`` (N, 3) in the Project frame, as float32. ``None`` = unaligned."""
    if alignment is None or is_identity(alignment):
        return np.asarray(xyz, np.float32)
    s, R, t = alignment_parts(alignment)
    return (np.asarray(xyz, np.float64) @ (s * R).T + t).astype(np.float32)


def apply_normals(alignment: dict | None, normals: np.ndarray) -> np.ndarray:
    if alignment is None or is_identity(alignment):
        return np.asarray(normals, np.float32)
    _, R, _ = alignment_parts(alignment)
    return (np.asarray(normals, np.float64) @ R.T).astype(np.float32)


def apply_camera(alignment: dict | None, camera: Camera) -> Camera:
    """The camera's pose once the world moves by the alignment."""
    if alignment is None or is_identity(alignment):
        return camera
    s, R, t = alignment_parts(alignment)
    R_new = np.asarray(camera.R, np.float64) @ R.T
    t_new = s * np.asarray(camera.t, np.float64) - R_new @ t
    return Camera(K=camera.K, R=R_new, t=t_new, width=camera.width,
                  height=camera.height, distortion=camera.distortion, model=camera.model)


# -- the record -----------------------------------------------------------------
@dataclass
class Reconstruction:
    id: str
    dir: Path
    name: str = ""
    parent: str | None = None
    products: dict = field(default_factory=lambda: {s: None for s in SLOTS})
    alignment: dict | None = None
    created: str = field(default_factory=now_iso)
    options: dict = field(default_factory=dict)
    marks: dict = field(default_factory=dict)     # gcp id -> [x, y, z], own frame

    @property
    def labels_dir(self) -> Path:
        return self.dir / "labels"

    @property
    def keep_path(self) -> Path:
        return self.dir / KEEP_FILE

    @property
    def aligned(self) -> bool:
        return self.alignment is not None

    def load_keep(self) -> np.ndarray | None:
        """The sparse points this Reconstruction keeps (bool over the
        product's cloud), or None when it keeps them all."""
        if self.keep_path.exists():
            return np.load(self.keep_path).astype(bool)
        return None

    def save_keep(self, keep: np.ndarray | None) -> None:
        if keep is None or keep.all():
            self.keep_path.unlink(missing_ok=True)
        else:
            self.dir.mkdir(parents=True, exist_ok=True)
            np.save(self.keep_path, np.asarray(keep, bool))

    # -- the frame ------------------------------------------------------------
    def frame_points(self, xyz: np.ndarray) -> np.ndarray:
        return apply_points(self.alignment, xyz)

    def frame_normals(self, normals: np.ndarray) -> np.ndarray:
        return apply_normals(self.alignment, normals)

    def frame_camera(self, camera: Camera) -> Camera:
        return apply_camera(self.alignment, camera)

    # -- persistence ------------------------------------------------------------
    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "parent": self.parent,
                "products": dict(self.products), "alignment": self.alignment,
                "created": self.created, "options": self.options, "marks": self.marks}

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / RECON_FILE).write_text(json.dumps(self.to_dict(), indent=2),
                                           encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Reconstruction":
        data = json.loads((path / RECON_FILE).read_text(encoding="utf-8"))
        products = {s: None for s in SLOTS}
        products.update(data.get("products") or {})
        return cls(id=data.get("id", path.name), dir=path, name=data.get("name", ""),
                   parent=data.get("parent"), products=products,
                   alignment=data.get("alignment"), created=data.get("created", ""),
                   options=data.get("options") or {}, marks=data.get("marks") or {})
