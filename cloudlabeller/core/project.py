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

"""Project = a folder on disk. Handles the manifest and (de)serialisation.

Layout (see DESIGN.md §4)::

    my_project.clproj/
      project.json
      images/
      reconstruction/
      products/{cloud.ply, mesh.ply}
      labels/{cloud_labels.npy, images/<name>.npy}
      ml/{checkpoints/, predictions/}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cloudlabeller.core.dataset import Dataset
from cloudlabeller.core.label_schema import LabelSchema
from cloudlabeller.core.labels import LabelStore

MANIFEST = "project.json"
FORMAT_VERSION = 1


@dataclass
class Project:
    """A CloudLabeller project: a plain folder plus its in-memory state.

    On disk the ``root`` folder holds ``project.json`` (schema, settings,
    per-image label status), ``images/`` (the image store), ``reconstruction/``
    (sparse cloud, cameras, dense cloud, visibility cache), ``products/``
    (mesh), ``labels/`` (per-point/-vertex/-image label arrays) and ``ml/``
    (checkpoints, predictions). Projects are portable — copy the folder as-is.

    In memory it bundles the label ``schema``, the reconstructed ``dataset``
    (images, clouds, mesh), the ``labels`` store, and free-form ``settings``
    (e.g. ``settings["georeferenced"]`` with the ENU origin / CRS frame).
    """

    root: Path
    schema: LabelSchema = field(default_factory=LabelSchema)
    dataset: Dataset = field(default_factory=Dataset)
    labels: LabelStore = field(default_factory=LabelStore)
    settings: dict = field(default_factory=dict)
    _visibility: object = field(default=None, init=False, repr=False)
    _adjacency: object = field(default=None, init=False, repr=False)

    def visibility_index(self):
        """Lazy per-project camera↔point visibility cache (reconstruction/pmap)."""
        if self._visibility is None:
            from cloudlabeller.transfer.visibility import VisibilityIndex

            self._visibility = VisibilityIndex(self.reconstruction_dir / "pmap")
        return self._visibility

    def image_adjacency(self):
        """The image-overlap (covisibility) graph, or None if underivable.

        Loaded from ``reconstruction/adjacency.json`` (written by SfM). For
        projects reconstructed before that file existed it is derived once
        from the sparse cloud's cached per-camera visibility and saved. May
        take a moment on that first derivation — call from a worker thread.
        """
        if self._adjacency is None:
            from cloudlabeller.photogrammetry.adjacency import (
                ImageAdjacency,
                covisibility_from_images,
            )

            adj = ImageAdjacency.load(self.reconstruction_dir)
            solved = self.dataset.solved_images()
            if adj is None and self.dataset.cloud is not None and solved:
                adj = covisibility_from_images(self.dataset.cloud, solved,
                                               self.visibility_index())
                adj.save(self.reconstruction_dir)
            self._adjacency = adj
        return self._adjacency

    def overlapping_images(self, name: str) -> "set[str] | None":
        """Names of images that see surface in common with ``name``.

        None means "unknown" (no graph, or ``name`` not in it) — treat every
        image as a candidate in that case.
        """
        adj = self.image_adjacency()
        return None if adj is None else adj.overlapping(name)

    @property
    def mesh_nn_cache_path(self) -> Path:
        return self.labels_dir / "mesh_nn.npz"

    # -- standard sub-paths -----------------------------------------------
    @property
    def images_dir(self) -> Path:
        return self.root / "images"

    @property
    def reconstruction_dir(self) -> Path:
        return self.root / "reconstruction"

    @property
    def products_dir(self) -> Path:
        return self.root / "products"

    @property
    def labels_dir(self) -> Path:
        return self.root / "labels"

    @property
    def ml_dir(self) -> Path:
        return self.root / "ml"

    @property
    def dense_cloud_path(self) -> Path:
        return self.reconstruction_dir / "dense.npz"

    @property
    def mesh_path(self) -> Path:
        return self.products_dir / "mesh.npz"

    # -- label management --------------------------------------------------
    def add_label(self, name: str | None = None, color: str | None = None):
        return self.schema.add(name=name, color=color)

    def rename_label(self, class_id: int, name: str) -> None:
        self.schema.rename(class_id, name)

    def set_label_color(self, class_id: int, color: str) -> None:
        self.schema.set_color(class_id, color)

    def delete_label(self, class_id: int) -> None:
        """Delete a class and remap the stored labels to stay contiguous."""
        self.labels.remap_after_delete(class_id)
        self.schema.remove(class_id)

    # -- auto (derived) image masks -----------------------------------------
    # Auto-labelled masks are DERIVED data: they are re-rendered from the cloud
    # labels on demand instead of being materialised for every image (a 21 Mpx
    # int32 mask is ~84 MB — storing 50+ of them froze the UI and exhausted RAM).
    def auto_mask_source(self):
        """The labelled cloud used to render auto masks: dense first, else sparse.
        Returns (cloud, labels) or None if nothing is labelled."""
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
        """Render ``name``'s auto label mask: a stored U-Net prediction when the
        image's status is 'ml', else projected from the cloud (or None)."""
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

    # -- U-Net predictions (stored at model resolution) ----------------------
    @property
    def predictions_dir(self) -> Path:
        return self.ml_dir / "predictions"

    def prediction_mask(self, name: str):
        """Load ``name``'s U-Net prediction and upscale it to the photo's
        resolution. Returns (H, W) int32 or None. Predictions are stored at
        the model's training size — full-resolution int32 masks are ~84 MB
        each, which froze the UI when materialised.

        The upscale is a per-class one-hot resample (see
        :func:`~cloudlabeller.core.raster.resample_label_mask`) — smooth
        class boundaries instead of the blocky staircase nearest-neighbour
        baked in, and never a spurious in-between class. This mask also feeds
        cloud transfer and retraining, so the boundary quality carries through.
        """
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

    # -- model / ML settings ----------------------------------------------
    def model_spec(self):
        """The persisted U-Net :class:`ModelSpec` (defaults if never set)."""
        from cloudlabeller.ml.model_spec import ModelSpec

        return ModelSpec.from_dict(self.settings.get("model_spec"))

    def set_model_spec(self, spec) -> None:
        self.settings["model_spec"] = spec.to_dict()

    def source_resolution(self, cache: bool = True) -> tuple[int, int] | None:
        """Original (width, height) of the project's images.

        Stored so model predictions — produced at the reduced training size —
        can be upscaled back to full resolution. Resolved from a solved camera
        first, else by reading an image header; cached in settings once known.
        """
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

    # -- summary -------------------------------------------------------------
    def summary_info(self) -> list[tuple[str, str]]:
        """(label, value) rows describing the project — shown in the Project
        Info dialog and copyable as plain text."""
        ds, lb = self.dataset, self.labels
        rows: list[tuple[str, str]] = [("Project", str(self.root))]

        solved = sum(1 for r in ds.images if r.camera is not None)
        rows.append(("Images", f"{len(ds.images)} in store, {solved} with solved cameras"))
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

        geo = self.settings.get("georeferenced")
        if geo:
            crs_info = geo.get("crs")
            if crs_info:
                off = crs_info.get("offset") or [0, 0, 0]
                rows.append(("Coordinates",
                             f"EPSG:{crs_info.get('epsg')} — "
                             f"{crs_info.get('name', '?')}"))
                rows.append(("Heights", "EGM96 (sea level)"
                             if crs_info.get("orthometric") else "ellipsoidal"))
                if any(off):
                    rows.append(("Stored offset",
                                 f"{off[0]:.0f}, {off[1]:.0f}, {off[2]:.0f} "
                                 "(exports add it back)"))
            else:
                rows.append(("Coordinates",
                             f"{geo.get('frame', 'ENU')} — metres, true north"))
            lat, lon, alt = (geo.get("origin_lla") or [0, 0, 0])
            rows.append(("Origin (lat, lon, alt)",
                         f"{lat:.7f}°, {lon:.7f}°, {alt:.1f} m"))
            scale = geo.get("scale_m_per_unit")
            if scale:
                rows.append(("Alignment scale",
                             f"1 model unit was {scale:.4f} m "
                             f"({geo.get('n_gps', '?')} GPS images)"))
        else:
            rows.append(("Coordinates", "arbitrary local frame "
                                        "(not georeferenced)"))

        if self.schema.classes:
            rows.append(("Classes", ", ".join(
                f"{c.id}: {c.name}" for c in self.schema.classes)))
        spec = self.model_spec()
        rows.append(("U-Net spec", f"1/{spec.resolution_divisor} resolution, "
                                   f"{spec.channels} ch × {spec.blocks} blocks, "
                                   f"filter {spec.filter_size}"))
        from cloudlabeller.ml.model_store import list_models

        models = list_models(self.ml_dir)
        if models:
            rows.append(("Saved models", ", ".join(m["name"] for m in models)))
        return rows

    # -- reconstruction freshness -----------------------------------------
    def mark_reconstruction_current(self) -> None:
        self.settings["reconstruction_outdated"] = False

    def mark_reconstruction_outdated(self) -> None:
        self.settings["reconstruction_outdated"] = True

    def reconstruction_status(self) -> str:
        """One of: 'none' (no cloud), 'outdated' (store changed since SfM),
        'current'."""
        if self.dataset.cloud is None:
            return "none"
        return "outdated" if self.settings.get("reconstruction_outdated") else "current"

    # -- lifecycle ---------------------------------------------------------
    @classmethod
    def create(cls, root: str | Path, schema: LabelSchema | None = None) -> "Project":
        """Create the folder structure for a new project and save its manifest."""
        root = Path(root)
        proj = cls(root=root, schema=schema or LabelSchema())
        for d in (proj.images_dir, proj.reconstruction_dir, proj.products_dir,
                  proj.labels_dir, proj.labels_dir / "images",
                  proj.ml_dir / "checkpoints", proj.ml_dir / "predictions"):
            d.mkdir(parents=True, exist_ok=True)
        proj.save()
        return proj

    @classmethod
    def open(cls, root: str | Path) -> "Project":
        """Load an existing project: manifest, then products, then labels."""
        root = Path(root)
        data = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
        proj = cls(
            root=root,
            schema=LabelSchema.from_dict(data.get("schema", {})),
            settings=data.get("settings", {}),
        )
        proj.labels.image_status = dict(data.get("image_status", {}))
        proj.load_products()
        proj._load_labels()
        return proj

    def save_manifest(self) -> None:
        """Write just project.json (schema, settings, image status). Cheap — used
        to persist the frustum-dot statuses promptly, without the heavy npy dump."""
        manifest = {
            "format_version": FORMAT_VERSION,
            "schema": self.schema.to_dict(),
            "settings": self.settings,
            "images": [im.name for im in self.dataset.images],
            "image_status": self.labels.image_status,   # drives the frustum dot colours
        }
        (self.root / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def save(self) -> None:
        """Full save: manifest plus every label array (the heavy part)."""
        self.save_manifest()
        self._save_labels()

    # -- label persistence -------------------------------------------------
    def _save_labels(self) -> None:
        """Dump cloud/mesh label arrays and user-drawn image masks as .npy."""
        if self.labels.cloud_labels is not None:
            np.save(self.labels_dir / "cloud_labels.npy", self.labels.cloud_labels)
        if self.labels.dense_cloud_labels is not None:
            np.save(self.labels_dir / "dense_cloud_labels.npy", self.labels.dense_cloud_labels)
        if self.labels.mesh_labels is not None:
            np.save(self.labels_dir / "mesh_labels.npy", self.labels.mesh_labels)
        for name, mask in self.labels.image_masks.items():
            np.save(self.labels_dir / "images" / f"{name}.npy", mask)

    def _load_labels(self) -> None:
        """Load label arrays back; stale auto-generated image masks (from an
        older app version that materialised them) are deleted, not loaded."""
        cloud_path = self.labels_dir / "cloud_labels.npy"
        if cloud_path.exists():
            self.labels.cloud_labels = np.load(cloud_path)
        dense_path = self.labels_dir / "dense_cloud_labels.npy"
        if dense_path.exists():
            self.labels.dense_cloud_labels = np.load(dense_path)
        mesh_path = self.labels_dir / "mesh_labels.npy"
        if mesh_path.exists():
            self.labels.mesh_labels = np.load(mesh_path)
        img_dir = self.labels_dir / "images"
        if img_dir.exists():
            for f in img_dir.glob("*.npy"):
                if self.labels.status_of(f.stem) == "auto":
                    # Stale derived mask (an older version materialised auto
                    # masks). It is regenerated from the cloud on demand —
                    # loading 50+ full-res masks exhausted RAM.
                    f.unlink(missing_ok=True)
                    continue
                self.labels.image_masks[f.stem] = np.load(f)

    def load_products(self) -> None:
        """Rebuild the dataset (image store + sparse cloud + cameras) on open.

        The dataset's images come from a union of (a) the image files present in
        ``project/images/`` and (b) the solved cameras in ``cameras.json`` — so
        both reconstructed and not-yet-reconstructed images appear, each with a
        camera when one was solved. The cloud is read from ``cloud.npz``.
        """
        self._adjacency = None      # a new reconstruction brings a new overlap graph

        # Lazy imports keep the core package free of a hard photogrammetry dep.
        import numpy as np

        from cloudlabeller.core.dataset import ImageRecord
        from cloudlabeller.core.images import list_image_files
        from cloudlabeller.photogrammetry.pipeline import CLOUD_FILE, load_cameras

        rec = self.reconstruction_dir

        # Images: union of the store contents and any solved cameras.
        cameras = load_cameras(rec)                       # filename -> Camera
        store = {Path(n).name for n in list_image_files(self.images_dir)}
        names = sorted(store | set(cameras))
        self.dataset.images = [
            ImageRecord(image_id=i, path=self.images_dir / name, camera=cameras.get(name))
            for i, name in enumerate(names)
        ]

        # Sparse cloud.
        cloud_path = rec / CLOUD_FILE
        if cloud_path.exists():
            data = np.load(cloud_path)
            from cloudlabeller.core.dataset import PointCloud
            rgb = data["rgb"]
            self.dataset.cloud = PointCloud(xyz=data["xyz"], rgb=rgb if len(rgb) else None)
            if (self.labels.cloud_labels is None
                    or len(self.labels.cloud_labels) != self.dataset.cloud.n_points):
                self.labels.init_cloud(self.dataset.cloud.n_points)

        # Dense cloud / mesh (from MVS or import), if present.
        from cloudlabeller.io.geometry import load_cloud_npz, load_mesh_npz
        if self.dense_cloud_path.exists():
            self.dataset.dense_cloud = load_cloud_npz(self.dense_cloud_path)
        if self.mesh_path.exists():
            self.dataset.mesh = load_mesh_npz(self.mesh_path)
