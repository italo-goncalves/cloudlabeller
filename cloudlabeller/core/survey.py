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

"""A Survey: one acquisition of a Project's site (see GLOSSARY.md).

On disk, ``surveys/<sid>/`` holds::

    survey.json                metadata, settings, image status, active Reconstruction
    images/                    the photos
    masks/<name>.npy           the user's Masks
    predictions/<name>.npy     U-Net predictions (at model resolution)
    products/<slot>-<n>/       what photogrammetry produced (never edited)
    reconstructions/<rid>/     each Reconstruction (recon.json, keep.npy, labels/)
    cache/                     visibility (pmap/) and derived adjacency

In memory a Survey bundles its photos and the active Reconstruction's
geometry (``dataset``), its labels (``labels``: the Masks and the active
Reconstruction's cloud and mesh labels) and its ``settings``. Geometry is
held in the Reconstruction's own SfM frame; the active Reconstruction's
``frame_*`` methods carry it into the Project frame.
"""

from __future__ import annotations

import json
import re
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cloudlabeller.core.dataset import Dataset
from cloudlabeller.core.labels import LabelStore, load_label_array, save_label_array
from cloudlabeller.core.reconstruction import SLOTS, Reconstruction, now_iso

SURVEY_FILE = "survey.json"
CLOUD_FILE = "cloud.npz"
DENSE_FILE = "dense.npz"
MESH_FILE = "mesh.npz"
META_KEYS = ("name", "timestamp", "operator", "camera", "notes")
_ID = re.compile(r"^[a-z]+(-?)(\d+)$")


def next_id(prefix: str, taken, sep: str = "") -> str:
    """``prefix`` + the next free number (``s3``, ``sfm-2``)."""
    numbers = [int(m.group(2)) for t in taken
               if (m := _ID.match(t)) and t.startswith(prefix + sep)]
    return f"{prefix}{sep}{max(numbers, default=0) + 1}"


@dataclass
class Survey:
    id: str
    dir: Path
    name: str = ""
    timestamp: str | None = None
    operator: str = ""
    camera: str = ""
    notes: str = ""
    settings: dict = field(default_factory=dict)
    reconstructions: dict[str, Reconstruction] = field(default_factory=dict)
    active_id: str | None = None
    marks: dict = field(default_factory=dict)     # gcp id -> {photo: [u, v]}
    dataset: Dataset = field(default_factory=Dataset)
    labels: LabelStore = field(default_factory=LabelStore)
    loaded: bool = False
    _visibility: object = field(default=None, init=False, repr=False)
    _adjacency: object = field(default=None, init=False, repr=False)

    # -- paths ------------------------------------------------------------------
    @property
    def images_dir(self) -> Path:
        return self.dir / "images"

    @property
    def masks_dir(self) -> Path:
        return self.dir / "masks"

    @property
    def predictions_dir(self) -> Path:
        return self.dir / "predictions"

    @property
    def products_dir(self) -> Path:
        return self.dir / "products"

    @property
    def reconstructions_dir(self) -> Path:
        return self.dir / "reconstructions"

    @property
    def cache_dir(self) -> Path:
        return self.dir / "cache"

    @property
    def active(self) -> Reconstruction | None:
        return self.reconstructions.get(self.active_id) if self.active_id else None

    def product_dir(self, slot: str) -> Path | None:
        """The active Reconstruction's product folder for ``slot``, or None."""
        rec = self.active
        pid = rec.products.get(slot) if rec is not None else None
        return self.products_dir / pid if pid else None

    @property
    def sfm_dir(self) -> Path | None:
        return self.product_dir("sfm")

    @property
    def mvs_dir(self) -> Path | None:
        return self.product_dir("mvs")

    @property
    def dense_cloud_path(self) -> Path | None:
        d = self.mvs_dir
        return d / DENSE_FILE if d else None

    @property
    def mesh_path(self) -> Path | None:
        d = self.product_dir("mesh")
        return d / MESH_FILE if d else None

    @property
    def mesh_nn_cache_path(self) -> Path | None:
        rec = self.active
        return rec.labels_dir / "mesh_nn.npz" if rec is not None else None

    def new_product_dir(self, slot: str) -> Path:
        """A fresh product folder for ``slot`` (``sfm-3``); not made yet."""
        taken = [p.name for p in self.products_dir.iterdir()] if self.products_dir.exists() else []
        return self.products_dir / next_id(slot, taken, sep="-")

    def plan_step(self, slot: str) -> tuple[Path, bool]:
        """Where a step writes, and whether finishing it makes a new
        Reconstruction. A step whose slot the active Reconstruction hasn't
        filled fills it; one it has filled makes a child that shares the
        earlier steps' products (nothing is overwritten, ADR-0005)."""
        rec = self.active
        fork = rec is not None and rec.products.get(slot) is not None
        return self.new_product_dir(slot), fork

    def _new_reconstruction(self, parent: str | None = None) -> Reconstruction:
        rid = next_id("r", self.reconstructions)
        rec = Reconstruction(id=rid, dir=self.reconstructions_dir / rid,
                             name=f"Reconstruction {rid[1:]}", parent=parent)
        self.reconstructions[rid] = rec
        return rec

    def ensure_active(self) -> Reconstruction:
        """The active Reconstruction, made (empty) when there is none."""
        if self.active is None:
            self.active_id = self._new_reconstruction().id
        return self.active

    def fork(self, slot: str) -> Reconstruction:
        """A child of the active Reconstruction sharing the products of the
        steps before ``slot`` (and their labels, kept points and Alignment),
        made active. Re-running SfM shares nothing: a new solve is a new
        frame."""
        from cloudlabeller.core.reconstruction import SLOTS

        parent = self.active
        shared = SLOTS[:SLOTS.index(slot)]
        child = self._new_reconstruction(parent.id if parent is not None and shared else None)
        if parent is not None:
            for s in shared:
                child.products[s] = parent.products.get(s)
            if "sfm" in shared:
                child.alignment = parent.alignment
                child.marks = {g: list(x) for g, x in parent.marks.items()}
                child.save_keep(parent.load_keep())
            files = {"sfm": "sparse.npy", "mvs": "dense.npy"}
            self.save()                              # the parent's labels, as they are
            for s in shared:
                src = parent.labels_dir / files.get(s, "")
                if s in files and src.exists():
                    child.labels_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, child.labels_dir / files[s])
        child.save()
        self.activate(child.id, save=False)
        return child

    def activate(self, rid: str, save: bool = True) -> None:
        """Make ``rid`` the active Reconstruction: the current one's labels
        are saved, the new one's geometry and labels loaded. The undo
        history is cleared (it refers to the other one's arrays)."""
        if rid not in self.reconstructions:
            raise KeyError(rid)
        if save:
            self.save()
        self.active_id = rid
        lb = self.labels
        lb.cloud_labels = lb.dense_cloud_labels = lb.mesh_labels = None
        lb._undo_stack.clear()
        lb._redo_stack.clear()
        rec = self.active
        for attr, fname in (("cloud_labels", "sparse.npy"),
                            ("dense_cloud_labels", "dense.npy"),
                            ("mesh_labels", "mesh.npy")):
            if (rec.labels_dir / fname).exists():
                setattr(lb, attr, load_label_array(rec.labels_dir / fname))
        self.load_products()
        self.save_manifest()

    # -- sizes and deleting ----------------------------------------------------------
    def used_products(self, but: str | None = None) -> set[str]:
        """The product folders some Reconstruction (other than ``but``) uses."""
        return {pid for rid, rec in self.reconstructions.items() if rid != but
                for pid in rec.products.values() if pid}

    def freed_by_deleting(self, rid: str) -> int:
        """Bytes deleting Reconstruction ``rid`` would free: its own folder
        and the products no other Reconstruction uses."""
        rec = self.reconstructions[rid]
        others = self.used_products(but=rid)
        total = dir_size(rec.dir)
        for pid in set(rec.products.values()) - others - {None}:
            total += dir_size(self.products_dir / pid)
        return total

    def delete_reconstruction(self, rid: str) -> int:
        """Delete Reconstruction ``rid`` and the products only it used;
        returns the bytes freed. Deleting the active one activates its
        parent, else the newest left, else none."""
        rec = self.reconstructions[rid]
        freed = self.freed_by_deleting(rid)
        others = self.used_products(but=rid)
        was_active = rid == self.active_id
        del self.reconstructions[rid]
        for pid in set(rec.products.values()) - others - {None}:
            shutil.rmtree(self.products_dir / pid, ignore_errors=True)
        shutil.rmtree(rec.dir, ignore_errors=True)
        for other in self.reconstructions.values():
            if other.parent == rid:
                other.parent = rec.parent
        if was_active:
            nxt = rec.parent if rec.parent in self.reconstructions else (
                max(self.reconstructions, key=lambda r: int(r[1:])) if self.reconstructions
                else None)
            self.active_id = None
            if nxt is not None:
                self.activate(nxt, save=False)
            else:
                lb = self.labels
                lb.cloud_labels = lb.dense_cloud_labels = lb.mesh_labels = None
                self.load_products()
        self.save_manifest()
        return freed

    def reconstruction_rows(self) -> list[dict]:
        """Each Reconstruction as the window lists it, oldest first."""
        rows = []
        for rid in sorted(self.reconstructions, key=lambda r: int(r[1:])):
            rec = self.reconstructions[rid]
            rows.append({"id": rid, "name": rec.name, "parent": rec.parent,
                         "products": dict(rec.products), "aligned": rec.aligned,
                         "created": rec.created, "active": rid == self.active_id,
                         "freed_bytes": self.freed_by_deleting(rid)})
        return rows

    # -- a look without loading -------------------------------------------------------
    def peek_images(self) -> list[tuple[str, bool]]:
        """(name, solved) of each photo, as :meth:`load_products` would list
        them, reading only the photo folder and ``cameras.json``."""
        if self.loaded:
            return [(r.name, r.camera is not None) for r in self.dataset.images]
        from cloudlabeller.core.images import list_image_files

        solved: set[str] = set()
        if self.sfm_dir and (self.sfm_dir / "cameras.json").exists():
            for c in json.loads((self.sfm_dir / "cameras.json").read_text(encoding="utf-8")):
                solved.add(c.get("name") or Path(c.get("path", "")).name)
        store = ({Path(n).name for n in list_image_files(self.images_dir)}
                 if self.images_dir.exists() else set())
        return [(n, n in solved) for n in sorted(store | solved)]

    def peek_resolution(self) -> tuple[int, int] | None:
        """The photos' size without loading: cached, else the first solved
        camera's."""
        if self.loaded:
            return self.source_resolution(cache=False)
        cached = self.settings.get("source_resolution")
        if cached:
            return (int(cached[0]), int(cached[1]))
        if self.sfm_dir and (self.sfm_dir / "cameras.json").exists():
            cams = json.loads((self.sfm_dir / "cameras.json").read_text(encoding="utf-8"))
            if cams:
                return (int(cams[0]["width"]), int(cams[0]["height"]))
        return None

    def peek_counts(self) -> dict:
        """Point and vertex counts of the active Reconstruction, read from
        the files' headers when the Survey isn't loaded."""
        if self.loaded:
            ds = self.dataset
            return {"sparse": ds.cloud.n_points if ds.cloud is not None else None,
                    "dense": ds.dense_cloud.n_points if ds.dense_cloud is not None else None,
                    "mesh": ({"vertices": len(ds.mesh.vertices), "faces": len(ds.mesh.faces)}
                             if ds.mesh is not None else None)}
        sparse = npz_rows(self.sfm_dir / CLOUD_FILE, "xyz") if self.sfm_dir else None
        if sparse is not None and self.active is not None:
            keep = self.active.load_keep()
            if keep is not None and len(keep) == sparse:
                sparse = int(keep.sum())
        dense = npz_rows(self.dense_cloud_path, "xyz") if self.dense_cloud_path else None
        mesh = None
        if self.mesh_path:
            v, f = npz_rows(self.mesh_path, "vertices"), npz_rows(self.mesh_path, "faces")
            mesh = {"vertices": v, "faces": f} if v is not None else None
        return {"sparse": sparse, "dense": dense, "mesh": mesh}

    # -- the frame ------------------------------------------------------------------
    @property
    def aligned(self) -> bool:
        rec = self.active
        return rec is not None and rec.aligned

    # -- caches ---------------------------------------------------------------------
    def visibility_index(self):
        """Lazy camera↔point visibility cache (``cache/pmap``), keyed by the
        cloud's and camera's fingerprints, so it serves any Reconstruction."""
        if self._visibility is None:
            from cloudlabeller.transfer.visibility import VisibilityIndex

            self._visibility = VisibilityIndex(self.cache_dir / "pmap")
        return self._visibility

    def image_adjacency(self):
        """The image-overlap (covisibility) graph, or None if underivable.

        Loaded from the SfM product's ``adjacency.json``. For products made
        before that file existed it is derived once from the sparse cloud's
        visibility and saved in ``cache/``. May take a moment the first
        time: call it from a worker thread.
        """
        if self._adjacency is None:
            from cloudlabeller.photogrammetry.adjacency import (
                ImageAdjacency,
                covisibility_from_images,
            )

            adj = ImageAdjacency.load(self.sfm_dir) if self.sfm_dir else None
            if adj is None:
                adj = ImageAdjacency.load(self.cache_dir)
            solved = self.dataset.solved_images()
            if adj is None and self.dataset.cloud is not None and solved:
                adj = covisibility_from_images(self.dataset.cloud, solved,
                                               self.visibility_index())
                adj.save(self.cache_dir)
            self._adjacency = adj
        return self._adjacency

    def overlapping_images(self, name: str) -> "set[str] | None":
        """Names of photos that see surface in common with ``name``; None
        means unknown (treat every photo as a candidate)."""
        adj = self.image_adjacency()
        return None if adj is None else adj.overlapping(name)

    # -- auto (derived) masks -------------------------------------------------------
    # Auto masks are derived data: rendered from the cloud labels on demand
    # instead of stored for every photo (a 21 Mpx int32 mask is ~84 MB).
    def auto_mask_source(self):
        """The labelled cloud auto masks are rendered from: dense first, else
        sparse. Returns (cloud, labels) or None if nothing is labelled."""
        ds, lb = self.dataset, self.labels
        if (ds.dense_cloud is not None and lb.dense_cloud_labels is not None
                and len(lb.dense_cloud_labels) == ds.dense_cloud.n_points
                and (lb.dense_cloud_labels != -1).any()):
            return ds.dense_cloud, lb.dense_cloud_labels
        if (ds.cloud is not None and lb.cloud_labels is not None
                and (lb.cloud_labels != -1).any()):
            return ds.cloud, lb.cloud_labels
        return None

    def render_auto_mask(self, name: str):
        """``name``'s auto mask: its U-Net prediction when its status is
        'ml', else projected from the cloud (or None)."""
        if self.labels.status_of(name) == "ml":
            pred = self.prediction_mask(name)
            if pred is not None:
                return pred
        source = self.auto_mask_source()
        record = self.dataset.image_by_name(name)
        if source is None or record is None or record.camera is None:
            return None
        from cloudlabeller.transfer import cloud_to_image  # lazy: heavy deps

        return cloud_to_image(source[0], source[1], record.camera,
                              name=name, visibility=self.visibility_index())

    def overlay_below(self, name: str):
        """What shows under ``name``'s Mask: the cloud's projection where it
        reaches, else the U-Net's prediction (ADR-0006: the better source
        wins per pixel). Returns ``(mask, source)``, with ``source`` 'auto'
        when the cloud labels any pixel, else 'ml', or ``(None, None)``."""
        pred = self.prediction_mask(name)
        proj = None
        source = self.auto_mask_source()
        record = self.dataset.image_by_name(name)
        if source is not None and record is not None and record.camera is not None:
            from cloudlabeller.transfer import cloud_to_image  # lazy: heavy deps

            proj = cloud_to_image(source[0], source[1], record.camera,
                                  name=name, visibility=self.visibility_index())
        if proj is not None and (proj != -1).any():
            if pred is not None and pred.shape == proj.shape:
                proj = np.where(proj != -1, proj, pred)
            return proj, "auto"
        if pred is not None:
            return pred, "ml"
        return None, None

    def normal_map(self, name: str, frame: str = "camera"):
        """``name``'s normal map from the dense cloud (``transfer.normal_map``):
        (H, W, 3) uint8 in the ``camera`` frame (what the U-Net takes) or the
        ``world`` one (the Project frame, for showing), or None when the photo
        has no camera or the Survey no dense cloud with normals."""
        ds = self.dataset
        record = ds.image_by_name(name)
        cloud = ds.dense_cloud
        if record is None or record.camera is None or cloud is None or cloud.normals is None:
            return None
        from cloudlabeller.transfer.normal_map import normal_map

        idx, pix = self.visibility_index().get(name, cloud, record.camera)
        normals = cloud.normals[idx]
        if frame == "world":
            normals, rotation = self.active.frame_normals(normals), None
        elif frame == "camera":
            rotation = record.camera.R
        else:
            raise ValueError(f"frame must be 'camera' or 'world', not {frame!r}")
        return normal_map(normals, pix, record.camera.width, record.camera.height, rotation)

    def has_normals(self) -> bool:
        """Whether the active Reconstruction's dense cloud has normals (read
        from the file's header when the cloud isn't loaded yet)."""
        cloud = self.dataset.dense_cloud if self.loaded else None
        if cloud is not None and not getattr(cloud, "_unread", False):
            return cloud.normals is not None          # read already: no cost
        path = self.dense_cloud_path
        if not path or not path.exists():
            return False
        known = getattr(self, "_normals_known", None)
        if known is not None and known[0] == str(path):
            return known[1]
        import zipfile

        from numpy.lib import format as npy

        try:
            with zipfile.ZipFile(path) as z, z.open("normals.npy") as f:
                shape, _, _ = npy._read_array_header(f, npy.read_magic(f))
            has = bool(shape and shape[0] > 0)
        except (KeyError, OSError, ValueError, zipfile.BadZipFile):
            has = False
        self._normals_known = (str(path), has)       # products are never edited
        return has

    def prediction_mask(self, name: str):
        """``name``'s U-Net prediction upscaled to the photo's resolution:
        (H, W) int32 or None. The upscale is a per-class one-hot resample
        (:func:`~cloudlabeller.core.raster.resample_label_mask`): smooth
        class boundaries, never a spurious in-between class."""
        small = self._prediction(name, "labels")
        if small is None:
            return None
        wh = self.source_resolution()
        record = self.dataset.image_by_name(name)
        if record is not None and record.camera is not None:
            wh = (record.camera.width, record.camera.height)
        if wh is None or (small.shape[1], small.shape[0]) == wh:
            return small.astype(np.int32)
        from cloudlabeller.core.raster import resample_label_mask

        return resample_label_mask(small, (int(wh[0]), int(wh[1]))).astype(np.int32)

    def _prediction(self, name: str, key: str):
        """An array of ``name``'s stored prediction (``labels`` or
        ``uncertainty``), or None; 0.3.0 predictions (``.npy``) are labels only."""
        npz = self.predictions_dir / f"{name}.npz"
        if npz.exists():
            with np.load(npz) as data:
                return data[key] if key in data else None
        npy = self.predictions_dir / f"{name}.npy"
        if key == "labels" and npy.exists():
            return np.load(npy)
        return None

    def prediction_uncertainty(self, name: str):
        """``name``'s Uncertainty (0 sure .. 1 unsure) at the prediction's
        size, as float32, or None."""
        u = self._prediction(name, "uncertainty")
        return None if u is None else u.astype(np.float32)

    def source_resolution(self, cache: bool = True) -> tuple[int, int] | None:
        """(width, height) of the Survey's photos: from a solved camera, else
        an image header; cached in ``settings`` once known."""
        cached = self.settings.get("source_resolution")
        if cached:
            return (int(cached[0]), int(cached[1]))
        wh: tuple[int, int] | None = None
        for rec in self.dataset.images:
            if rec.camera is not None:
                wh = (int(rec.camera.width), int(rec.camera.height))
                break
        if wh is None:
            from PIL import Image

            for rec in self.dataset.images:
                if rec.path and Path(rec.path).exists():
                    with Image.open(rec.path) as im:
                        wh = (int(im.size[0]), int(im.size[1]))
                    break
        if wh and cache:
            self.settings["source_resolution"] = list(wh)
        return wh

    # -- reconstruction freshness ---------------------------------------------------
    def mark_reconstruction_current(self) -> None:
        self.settings["reconstruction_outdated"] = False

    def mark_reconstruction_outdated(self) -> None:
        self.settings["reconstruction_outdated"] = True

    def reconstruction_status(self) -> str:
        """'none' (no cloud), 'outdated' (photos changed since SfM) or 'current'."""
        if self.dataset.cloud is None:
            return "none"
        return "outdated" if self.settings.get("reconstruction_outdated") else "current"

    def filter_sparse_cloud(self, keep: np.ndarray) -> int:
        """Keep only the sparse points where ``keep`` is True. The product is
        not touched: the Reconstruction records which of the product's
        points it keeps. The labels follow (which clears the undo history).
        Returns how many points went; the caller saves the Survey."""
        from cloudlabeller.core.dataset import PointCloud

        cloud, rec = self.dataset.cloud, self.active
        keep = np.asarray(keep, bool)
        if cloud is None or rec is None or len(keep) != cloud.n_points:
            raise ValueError("keep must have one entry per sparse point")
        before = rec.load_keep()
        if before is None:
            full = keep
        else:
            full = before.copy()
            full[np.flatnonzero(before)] = keep
        rec.save_keep(full)
        self.dataset.cloud = PointCloud(
            xyz=cloud.xyz[keep], rgb=cloud.rgb[keep] if cloud.rgb is not None else None)
        self.labels.filter_cloud(keep)
        self._adjacency = None
        return cloud.n_points - self.dataset.cloud.n_points

    # -- loading ---------------------------------------------------------------------
    def load_products(self) -> None:
        """Rebuild the dataset from the photos and the active Reconstruction.

        The photos are the union of the image store and the solved cameras,
        each with its camera when one was solved.
        """
        from cloudlabeller.core.dataset import ImageRecord, PointCloud
        from cloudlabeller.core.images import list_image_files
        from cloudlabeller.io.geometry import load_cloud_npz, load_mesh_npz
        from cloudlabeller.photogrammetry.pipeline import load_cameras

        self._adjacency = None
        sfm = self.sfm_dir
        cameras = load_cameras(sfm) if sfm else {}
        store = ({Path(n).name for n in list_image_files(self.images_dir)}
                 if self.images_dir.exists() else set())
        names = sorted(store | set(cameras))
        ds = self.dataset
        ds.images = [ImageRecord(image_id=i, path=self.images_dir / name,
                                 camera=cameras.get(name))
                     for i, name in enumerate(names)]

        ds.cloud = None
        if sfm and (sfm / CLOUD_FILE).exists():
            data = np.load(sfm / CLOUD_FILE)
            xyz, rgb = data["xyz"], data["rgb"]
            rgb = rgb if len(rgb) else None
            keep = self.active.load_keep()
            if keep is not None and len(keep) == len(xyz):
                xyz = xyz[keep]
                rgb = rgb[keep] if rgb is not None else None
            ds.cloud = PointCloud(xyz=xyz, rgb=rgb)
            if (self.labels.cloud_labels is None
                    or len(self.labels.cloud_labels) != ds.cloud.n_points):
                self.labels.init_cloud(ds.cloud.n_points)
        dense, mesh = self.dense_cloud_path, self.mesh_path
        ds.dense_cloud = load_cloud_npz(dense) if dense and dense.exists() else None
        ds.mesh = load_mesh_npz(mesh) if mesh and mesh.exists() else None

    def _load_labels(self) -> None:
        """Masks, then the active Reconstruction's label arrays. Stale
        materialised auto masks (an older app stored them) are deleted."""
        lb = self.labels
        rec = self.active
        if rec is not None:
            d = rec.labels_dir
            for attr, fname in (("cloud_labels", "sparse.npy"),
                                ("dense_cloud_labels", "dense.npy"),
                                ("mesh_labels", "mesh.npy")):
                if (d / fname).exists():
                    setattr(lb, attr, load_label_array(d / fname))
        if self.masks_dir.exists():
            for f in self.masks_dir.glob("*.npy"):
                if lb.status_of(f.stem) == "auto":
                    f.unlink(missing_ok=True)
                    continue
                lb.image_masks[f.stem] = load_label_array(f)

    def ensure_loaded(self) -> "Survey":
        if not self.loaded:
            self._load_labels()
            self.load_products()
            self.loaded = True
        return self

    # -- saving ----------------------------------------------------------------------
    def to_dict(self) -> dict:
        out = {"id": self.id}
        out.update({k: getattr(self, k) for k in META_KEYS})
        out.update(settings=self.settings, active=self.active_id,
                   image_status=self.labels.image_status, marks=self.marks)
        return out

    def save_manifest(self) -> None:
        """Write ``survey.json`` and the Reconstructions' records."""
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / SURVEY_FILE).write_text(json.dumps(self.to_dict(), indent=2),
                                            encoding="utf-8")
        for rec in self.reconstructions.values():
            rec.save()

    def save_mask(self, name: str) -> None:
        mask = self.labels.image_masks.get(name)
        if mask is not None:
            self.masks_dir.mkdir(parents=True, exist_ok=True)
            save_label_array(self.masks_dir / f"{name}.npy", mask)

    def save(self) -> None:
        """Manifest plus every label array (the heavy part)."""
        lb = self.labels
        if (self.loaded and self.active is None
                and any(a is not None for a in
                        (lb.cloud_labels, lb.dense_cloud_labels, lb.mesh_labels))):
            self.ensure_active()
        self.save_manifest()
        if not self.loaded:
            return
        rec = self.active
        if rec is not None:
            rec.labels_dir.mkdir(parents=True, exist_ok=True)
            for attr, fname in (("cloud_labels", "sparse.npy"),
                                ("dense_cloud_labels", "dense.npy"),
                                ("mesh_labels", "mesh.npy")):
                arr = getattr(lb, attr)
                if arr is not None:
                    save_label_array(rec.labels_dir / fname, arr)
        for name in lb.image_masks:
            self.save_mask(name)

    # -- lifecycle -------------------------------------------------------------------
    @classmethod
    def create(cls, dir: Path, sid: str, name: str = "", **meta) -> "Survey":
        survey = cls(id=sid, dir=Path(dir), name=name or f"Survey {sid[1:]}", **meta)
        survey.timestamp = survey.timestamp or None
        for d in (survey.images_dir, survey.masks_dir, survey.predictions_dir,
                  survey.products_dir, survey.reconstructions_dir):
            d.mkdir(parents=True, exist_ok=True)
        survey.loaded = True
        survey.save_manifest()
        return survey

    @classmethod
    def load(cls, dir: Path) -> "Survey":
        """The Survey's manifest and Reconstructions; its photos, geometry
        and labels load on :meth:`ensure_loaded`."""
        dir = Path(dir)
        data = json.loads((dir / SURVEY_FILE).read_text(encoding="utf-8"))
        survey = cls(id=data.get("id", dir.name), dir=dir,
                     settings=data.get("settings") or {})
        for k in META_KEYS:
            if data.get(k) is not None:
                setattr(survey, k, data[k])
        survey.labels.image_status = dict(data.get("image_status") or {})
        survey.marks = {g: dict(m) for g, m in (data.get("marks") or {}).items()}
        rdir = survey.reconstructions_dir
        if rdir.exists():
            for p in sorted(rdir.iterdir()):
                if (p / "recon.json").exists():
                    rec = Reconstruction.load(p)
                    survey.reconstructions[rec.id] = rec
        active = data.get("active")
        survey.active_id = active if active in survey.reconstructions else None
        return survey

    # -- summary ---------------------------------------------------------------------
    def summary_rows(self) -> list[tuple[str, str]]:
        ds, lb = self.dataset, self.labels
        solved = sum(1 for r in ds.images if r.camera is not None)
        rows = [("Photos", f"{len(ds.images)} in store, {solved} with solved cameras")]
        statuses = [lb.status_of(r.name) for r in ds.images]
        rows.append(("Label status",
                     f"{statuses.count('user')} user-labelled, "
                     f"{statuses.count('auto') + statuses.count('ml')} auto-labelled "
                     f"({statuses.count('ml')} by U-Net), "
                     f"{statuses.count('none')} unlabelled"))
        rows.append(("Reconstruction", self.reconstruction_status()))
        if ds.cloud is not None:
            rows.append(("Sparse cloud", f"{ds.cloud.n_points:,} points"))
        if ds.dense_cloud is not None:
            rows.append(("Dense cloud", f"{ds.dense_cloud.n_points:,} points"))
        if ds.mesh is not None:
            rows.append(("Mesh", f"{len(ds.mesh.vertices):,} vertices, "
                                 f"{len(ds.mesh.faces):,} faces"))
        rec = self.active
        if rec is not None and rec.alignment:
            a = rec.alignment
            src = {"gps": "EXIF GPS", "legacy": "aligned before 0.4.0"}.get(
                a.get("source"), a.get("source", "?"))
            text = f"{src}, scale {float(a['s']):.4f}"
            if a.get("n_gps"):
                text += f", {a.get('n_inliers', '?')}/{a['n_gps']} GPS photos"
            if a.get("rms") is not None:
                text += f", rms {float(a['rms']):.2f} m"
            rows.append(("Alignment", text))
        elif ds.cloud is not None:
            rows.append(("Alignment", "none (its own local frame)"))
        return rows


def dir_size(path: Path) -> int:
    """Bytes in the files under ``path`` (0 when it doesn't exist)."""
    path = Path(path)
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def npz_rows(path: Path | None, key: str) -> int | None:
    """The first dimension of array ``key`` in an ``.npz``, read from its
    header without loading it; None when the file or array isn't there."""
    import numpy.lib.format as fmt

    if path is None or not Path(path).exists():
        return None
    try:
        with zipfile.ZipFile(path) as z, z.open(f"{key}.npy") as f:
            version = fmt.read_magic(f)
            read = fmt.read_array_header_1_0 if version == (1, 0) else fmt.read_array_header_2_0
            shape, _, _ = read(f)
    except (KeyError, OSError, ValueError, zipfile.BadZipFile):
        return None
    return int(shape[0]) if shape else 0


__all__ = ["Survey", "SLOTS", "dir_size", "next_id", "now_iso", "npz_rows"]
