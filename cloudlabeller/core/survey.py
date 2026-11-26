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
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cloudlabeller.core.dataset import Dataset
from cloudlabeller.core.labels import LabelStore
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

    def output_dir(self, slot: str) -> Path:
        """Where a step writes: the active Reconstruction's product for
        ``slot`` (running a step again replaces it), else a new product."""
        current = self.product_dir(slot)
        if current is not None:
            return current
        taken = [p.name for p in self.products_dir.iterdir()] if self.products_dir.exists() else []
        return self.products_dir / next_id(slot, taken, sep="-")

    def ensure_active(self) -> Reconstruction:
        """The active Reconstruction, made (empty) when there is none."""
        if self.active is None:
            rid = next_id("r", self.reconstructions)
            rec = Reconstruction(id=rid, dir=self.reconstructions_dir / rid,
                                 name=f"Reconstruction {rid[1:]}")
            self.reconstructions[rid] = rec
            self.active_id = rid
        return self.active

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

    def prediction_mask(self, name: str):
        """``name``'s U-Net prediction upscaled to the photo's resolution:
        (H, W) int32 or None. The upscale is a per-class one-hot resample
        (:func:`~cloudlabeller.core.raster.resample_label_mask`): smooth
        class boundaries, never a spurious in-between class."""
        path = self.predictions_dir / f"{name}.npy"
        if not path.exists():
            return None
        small = np.load(path)
        wh = self.source_resolution()
        record = self.dataset.image_by_name(name)
        if record is not None and record.camera is not None:
            wh = (record.camera.width, record.camera.height)
        if wh is None or (small.shape[1], small.shape[0]) == wh:
            return small.astype(np.int32)
        from cloudlabeller.core.raster import resample_label_mask

        return resample_label_mask(small, (int(wh[0]), int(wh[1]))).astype(np.int32)

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
                    setattr(lb, attr, np.load(d / fname))
        if self.masks_dir.exists():
            for f in self.masks_dir.glob("*.npy"):
                if lb.status_of(f.stem) == "auto":
                    f.unlink(missing_ok=True)
                    continue
                lb.image_masks[f.stem] = np.load(f)

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
                   image_status=self.labels.image_status)
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
            np.save(self.masks_dir / f"{name}.npy", mask)

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
                    np.save(rec.labels_dir / fname, arr)
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


__all__ = ["Survey", "SLOTS", "next_id", "now_iso"]
