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

"""The Engine's methods: thin handlers over ``core/``, ``transfer/`` and the Jobs.

Each handler takes the Engine and the request's params and returns a
JSON-able result, or a :class:`Deferred` for work that takes a while. The
handlers carry over what the Qt window decided around each action (which
images a transfer reads, what a status change persists), not any UI.

Persistence follows the Qt app: a status change writes ``project.json`` at
once; a painted image mask is written at once; other label arrays are marked
dirty and saved within a minute, on ``project.save`` and on close.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cloudlabeller.core.label_schema import UNLABELLED_ID
from cloudlabeller.engine import arrays
from cloudlabeller.engine.protocol import InvalidParams, Problem
from cloudlabeller.engine.server import Deferred, Engine, method, optional, require

CLOUDS = ("sparse", "dense")


# -- engine -------------------------------------------------------------------
@method("engine.hello", needs_project=False)
def engine_hello(engine: Engine, params: dict) -> dict:
    return engine.hello()


@method("engine.shutdown", needs_project=False)
def engine_shutdown(engine: Engine, params: dict) -> dict:
    """Save and exit after answering."""
    engine.stop()
    return {}


@method("engine.cancel", needs_project=False)
def engine_cancel(engine: Engine, params: dict) -> dict:
    """Cancel a pending request (a transfer). Its own response is the
    ``request.cancelled`` Problem."""
    req_id = params.get("id")
    if req_id is None:
        raise InvalidParams("missing param 'id'", param="id")
    return {"cancelled": engine.cancel_request(req_id)}


# -- project ------------------------------------------------------------------
def _schema(project) -> list[dict]:
    return [{"id": c.id, "name": c.name, "color": c.color, "hotkey": c.hotkey}
            for c in project.schema.classes]


def project_state(engine: Engine) -> dict:
    project = engine.require_project()
    ds, lb = project.dataset, project.labels
    return {
        "root": str(project.root),
        "schema": _schema(project),
        "images": [{"name": r.name, "status": lb.status_of(r.name),
                    "solved": r.camera is not None,
                    "user_mask": r.name in lb.image_masks}
                   for r in ds.images],
        "reconstruction": project.reconstruction_status(),
        "sparse": ds.cloud.n_points if ds.cloud is not None else None,
        "dense": ds.dense_cloud.n_points if ds.dense_cloud is not None else None,
        "mesh": ({"vertices": len(ds.mesh.vertices), "faces": len(ds.mesh.faces)}
                 if ds.mesh is not None else None),
        "georeferenced": project.settings.get("georeferenced"),
        "source_resolution": project.source_resolution(cache=False),
        "can_undo": lb.can_undo(),
        "can_redo": lb.can_redo(),
        "dirty": engine.dirty,
    }


@method("project.create", needs_project=False)
def project_create(engine: Engine, params: dict) -> dict:
    from cloudlabeller.core.label_schema import LabelSchema
    from cloudlabeller.core.project import MANIFEST, Project

    root = Path(require(params, "root", str))
    if (root / MANIFEST).exists():
        raise Problem("project.exists", "A Project already exists there.", root=str(root))
    _no_outstanding_jobs(engine)
    schema = LabelSchema()
    for c in optional(params, "classes", list, []):
        if not isinstance(c, dict):
            raise InvalidParams("each class is an object with name and color")
        schema.add(name=c.get("name"), color=c.get("color"))
    engine.close_project()
    engine.project = Project.create(root, schema)
    arrays.clear(root)
    return project_state(engine)


@method("project.open", needs_project=False)
def project_open(engine: Engine, params: dict) -> dict:
    from cloudlabeller.core.project import MANIFEST, Project

    root = Path(require(params, "root", str))
    if not (root / MANIFEST).is_file():
        raise Problem("project.not_found", "No Project there.", root=str(root))
    _no_outstanding_jobs(engine)
    engine.close_project()
    engine.project = Project.open(root)
    arrays.clear(root)
    return project_state(engine)


def _no_outstanding_jobs(engine: Engine) -> None:
    if engine.jobs.outstanding:
        raise Problem("project.jobs_running", "Jobs are still running.",
                      jobs=engine.jobs.list())


@method("project.close")
def project_close(engine: Engine, params: dict) -> dict:
    _no_outstanding_jobs(engine)
    engine.close_project()
    return {}


@method("project.save")
def project_save(engine: Engine, params: dict) -> dict:
    engine.save_project()
    return {}


@method("project.state")
def project_state_method(engine: Engine, params: dict) -> dict:
    return project_state(engine)


@method("project.summary")
def project_summary(engine: Engine, params: dict) -> dict:
    return {"rows": [[k, v] for k, v in engine.project.summary_info()]}


# -- schema ----------------------------------------------------------------------
def _class_id(engine: Engine, params: dict) -> int:
    cid = require(params, "id", int)
    if not 0 <= cid < len(engine.project.schema.classes):
        raise Problem("schema.no_class", "No class has that id.", id=cid)
    return cid


def _schema_changed(engine: Engine) -> dict:
    engine.project.save_manifest()
    engine.changed("schema")
    return {"schema": _schema(engine.project)}


@method("schema.add")
def schema_add(engine: Engine, params: dict) -> dict:
    engine.project.add_label(name=optional(params, "name", str, None),
                             color=optional(params, "color", str, None))
    return _schema_changed(engine)


@method("schema.rename")
def schema_rename(engine: Engine, params: dict) -> dict:
    engine.project.rename_label(_class_id(engine, params), require(params, "name", str))
    return _schema_changed(engine)


@method("schema.set_color")
def schema_set_color(engine: Engine, params: dict) -> dict:
    engine.project.set_label_color(_class_id(engine, params), require(params, "color", str))
    return _schema_changed(engine)


@method("schema.delete")
def schema_delete(engine: Engine, params: dict) -> dict:
    """Delete a class: its points and pixels become unlabelled, higher ids
    shift down by one, and the undo history is cleared."""
    engine.project.delete_label(_class_id(engine, params))
    engine.dirty = True
    result = _schema_changed(engine)
    engine.changed("labels", images=engine.project.labels.image_masks)
    return result


# -- undo / redo -------------------------------------------------------------------
def _after_history(engine: Engine, changed) -> dict:
    from cloudlabeller.core.labels import Modality

    if changed is None:
        return {"changed": None}
    modality, key = changed
    engine.dirty = True
    if modality is Modality.IMAGE and key:
        engine.changed("labels", images=[key])
    else:
        engine.changed("labels")
        sync_mesh_labels(engine)
    return {"changed": {"modality": modality.value, "image": key}}


@method("labels.undo")
def labels_undo(engine: Engine, params: dict) -> dict:
    return _after_history(engine, engine.project.labels.undo())


@method("labels.redo")
def labels_redo(engine: Engine, params: dict) -> dict:
    return _after_history(engine, engine.project.labels.redo())


# -- images ------------------------------------------------------------------------
def _image(engine: Engine, params: dict):
    name = require(params, "name", str)
    record = engine.project.dataset.image_by_name(name)
    if record is None:
        raise Problem("image.not_found", "No image has that name.", name=name)
    return record


def _image_size(engine: Engine, record) -> tuple[int, int]:
    """(height, width) of an image: its camera, else its file header."""
    if record.camera is not None:
        return int(record.camera.height), int(record.camera.width)
    from PIL import Image

    with Image.open(record.path) as im:
        return int(im.size[1]), int(im.size[0])


@method("image.mask")
def image_mask(engine: Engine, params: dict) -> dict:
    """The mask to show over a photo: the user's own, else the U-Net's
    prediction or the cloud's projection (``source`` says which)."""
    record = _image(engine, params)
    project, name = engine.project, record.name
    mask = project.labels.image_masks.get(name)
    source = "user" if mask is not None else None
    if mask is None:
        mask = project.render_auto_mask(name)
        if mask is not None:
            source = "ml" if project.labels.status_of(name) == "ml" else "auto"
    if mask is None:
        return {"source": None, "mask": None}
    return {"source": source,
            "mask": arrays.write(project.root, f"mask/{name}", mask.astype(np.int32))}


@method("image.save_mask")
def image_save_mask(engine: Engine, params: dict) -> dict:
    """Store the mask the user painted on a photo (one undoable edit).

    The window sends the whole mask, already seeded from whatever overlay it
    showed; only the pixels that differ are recorded for undo.
    """
    record = _image(engine, params)
    project, name = engine.project, record.name
    h, w = _image_size(engine, record)
    new = arrays.read(project.root, require(params, "mask", dict), dtype="int32", ndim=2)
    if new.shape != (h, w):
        raise Problem("image.mask_shape", "The mask is not the photo's size.",
                      expected=[h, w], got=list(new.shape))
    n_classes = len(project.schema.classes)
    if new.size and (new.min() < UNLABELLED_ID or new.max() >= n_classes):
        raise Problem("image.mask_values", "The mask holds ids outside the schema.",
                      n_classes=n_classes)
    labels = project.labels
    if name not in labels.image_masks:
        labels.init_image(name, h, w)
    old = labels.image_masks[name]
    changed = np.flatnonzero(old.reshape(-1) != new.reshape(-1))
    if changed.size:
        labels.paint_image(name, changed, new.reshape(-1)[changed])
    labels.mark_user_labeled(name)
    project.save_image_mask(name)
    project.save_manifest()
    engine.changed("labels", "status", images=[name])
    return {"changed_pixels": int(changed.size), "status": labels.status_of(name)}


# -- clouds and mesh -----------------------------------------------------------------
def _which(params: dict) -> str:
    which = optional(params, "cloud", str, "sparse")
    if which not in CLOUDS:
        raise InvalidParams("cloud must be 'sparse' or 'dense'", param="cloud")
    return which


def _cloud(engine: Engine, which: str):
    ds = engine.project.dataset
    cloud = ds.cloud if which == "sparse" else ds.dense_cloud
    if cloud is None:
        raise Problem("cloud.none", "That cloud doesn't exist yet.", cloud=which)
    return cloud


def _cloud_labels(engine: Engine, which: str, n: int) -> np.ndarray:
    lb = engine.project.labels
    labels = lb.cloud_labels if which == "sparse" else lb.dense_cloud_labels
    if labels is None or len(labels) != n:
        if which == "sparse":
            lb.init_cloud(n)
            return lb.cloud_labels
        lb.init_dense_cloud(n)
        return lb.dense_cloud_labels
    return labels


@method("cloud.points")
def cloud_points(engine: Engine, params: dict) -> dict:
    which = _which(params)
    cloud = _cloud(engine, which)
    root = engine.project.root
    return {"cloud": which, "n_points": cloud.n_points,
            "xyz": arrays.write(root, f"cloud/{which}/xyz", cloud.xyz.astype(np.float32)),
            "rgb": (arrays.write(root, f"cloud/{which}/rgb", cloud.rgb.astype(np.uint8))
                    if cloud.rgb is not None else None)}


@method("cloud.labels")
def cloud_labels(engine: Engine, params: dict) -> dict:
    which = _which(params)
    cloud = _cloud(engine, which)
    labels = _cloud_labels(engine, which, cloud.n_points)
    return {"cloud": which,
            "labels": arrays.write(engine.project.root, f"cloud/{which}/labels",
                                   labels.astype(np.int32))}


@method("cloud.paint")
def cloud_paint(engine: Engine, params: dict) -> dict:
    """Give the chosen points one class (the lasso): one undoable edit."""
    which = _which(params)
    cloud = _cloud(engine, which)
    class_id = require(params, "class_id", int)
    if not UNLABELLED_ID <= class_id < len(engine.project.schema.classes):
        raise Problem("schema.no_class", "No class has that id.", id=class_id)
    ref = require(params, "indices", dict)
    if ref.get("dtype") not in ("int32", "uint32"):
        raise InvalidParams("indices must be int32 or uint32", param="indices")
    indices = arrays.read(engine.project.root, ref, ndim=1).astype(np.int64)
    if indices.size and (indices.min() < 0 or indices.max() >= cloud.n_points):
        raise Problem("cloud.bad_indices", "Some indices are outside the cloud.",
                      n_points=cloud.n_points)
    _cloud_labels(engine, which, cloud.n_points)
    lb = engine.project.labels
    if indices.size:
        if which == "sparse":
            lb.paint_cloud(indices, class_id)
        else:
            lb.paint_dense_cloud(indices, class_id)
        engine.dirty = True
        engine.changed("labels")
        if which == "dense":
            sync_mesh_labels(engine)
    return {"painted": int(indices.size)}


@method("mesh.geometry")
def mesh_geometry(engine: Engine, params: dict) -> dict:
    mesh = engine.project.dataset.mesh
    if mesh is None:
        raise Problem("mesh.none", "There is no mesh yet.")
    root = engine.project.root
    return {"vertices": arrays.write(root, "mesh/vertices", mesh.vertices.astype(np.float32)),
            "faces": arrays.write(root, "mesh/faces", mesh.faces.astype(np.int32)),
            "colors": (arrays.write(root, "mesh/colors", mesh.vertex_colors.astype(np.uint8))
                       if mesh.vertex_colors is not None else None)}


@method("mesh.labels")
def mesh_labels(engine: Engine, params: dict) -> dict:
    mesh = engine.project.dataset.mesh
    if mesh is None:
        raise Problem("mesh.none", "There is no mesh yet.")
    labels = engine.project.labels.mesh_labels
    if labels is None or len(labels) != len(mesh.vertices):
        labels = np.full(len(mesh.vertices), UNLABELLED_ID, np.int32)
    return {"labels": arrays.write(engine.project.root, "mesh/labels", labels.astype(np.int32))}


def sync_mesh_labels(engine: Engine) -> None:
    """Recompute the mesh's labels from the dense cloud (nearest point), in
    the background. Called whenever the dense labels or the mesh change."""
    project = engine.project
    mesh, dense = project.dataset.mesh, project.dataset.dense_cloud
    if mesh is None or dense is None:
        return
    labels = project.labels.dense_cloud_labels
    if labels is None or len(labels) != dense.n_points or not (labels != UNLABELLED_ID).any():
        project.labels.mesh_labels = np.full(len(mesh.vertices), UNLABELLED_ID, np.int32)
        engine.dirty = True
        engine.changed("labels")
        return
    from cloudlabeller.transfer.mesh_cloud import cloud_to_mesh

    snapshot = labels.copy()
    cache = project.mesh_nn_cache_path

    def finish(mesh_labels) -> None:
        if project.dataset.mesh is mesh:
            project.labels.mesh_labels = mesh_labels
            engine.dirty = True
            engine.changed("labels")

    engine.run_background(lambda: cloud_to_mesh(dense, snapshot, mesh, cache_path=cache),
                          finish)


# -- transfers -----------------------------------------------------------------------
def _labelled_source(project):
    """The cloud whose labels are projected onto photos: dense when labelled."""
    ds, lb = project.dataset, project.labels
    if (ds.dense_cloud is not None and lb.dense_cloud_labels is not None
            and len(lb.dense_cloud_labels) == ds.dense_cloud.n_points
            and (lb.dense_cloud_labels != UNLABELLED_ID).any()):
        return "dense", ds.dense_cloud, lb.dense_cloud_labels.copy()
    if lb.cloud_labels is not None and (lb.cloud_labels != UNLABELLED_ID).any():
        return "sparse", ds.cloud, lb.cloud_labels.copy()
    return None


def _mark_auto(engine: Engine, counts: dict, min_px: int) -> list[str]:
    marked = [name for name, n_px in counts.items() if n_px >= min_px]
    for name in marked:
        engine.project.labels.mark_auto_labeled(name)
    engine.project.save_manifest()
    return marked


@method("transfer.images_to_cloud")
def transfer_images_to_cloud(engine: Engine, params: dict) -> Deferred:
    """Vote the photos' labels onto the sparse cloud and the dense one.

    User-drawn masks outrank auto ones. With ``include_auto`` (default true)
    the U-Net's predictions vote too, and the cloud-projected photos that
    cover the surface. The result is merged: points no photo reached keep
    their labels.
    """
    from cloudlabeller.transfer import images_to_cloud
    from cloudlabeller.transfer.project_to_cloud import MaskSource

    project = engine.project
    ds, labels = project.dataset, project.labels
    if ds.cloud is None:
        raise Problem("transfer.no_cloud", "Run SfM first.")
    include_auto = optional(params, "include_auto", bool, True)
    masks = dict(labels.image_masks)
    n_classes = len(project.schema.classes)
    n_user = sum(1 for m in masks.values() if (m != UNLABELLED_ID).any())
    auto_names = [r.name for r in ds.solved_images()
                  if r.name not in masks and labels.status_of(r.name) in ("auto", "ml")]
    if not include_auto:
        auto_names = []
    if not auto_names and n_user == 0:
        raise Problem("transfer.no_labelled_images", "No photo is labelled yet.")
    pure_auto = [n for n in auto_names if labels.status_of(n) == "auto"]
    ml_names = [n for n in auto_names if labels.status_of(n) == "ml"]
    sparse, dense = ds.cloud, ds.dense_cloud
    vis = project.visibility_index()
    hard = set(masks)

    def work(progress):
        selected = auto_names
        if pure_auto:
            adj = project.image_adjacency()
            if adj is not None:
                from cloudlabeller.photogrammetry.adjacency import select_covering

                selected = ml_names + select_covering(pure_auto, adj)
        source = MaskSource(masks, selected, project.render_auto_mask)
        progress(0.0, f"Transferring labels from {n_user + len(selected)} images")
        out = {"images": n_user + len(selected)}
        out["sparse"] = images_to_cloud(
            sparse, ds, source, n_classes,
            lambda f, m="": progress(0.02 + 0.46 * f, ""), visibility=vis,
            priority_names=hard)
        if dense is not None:
            out["dense"] = images_to_cloud(
                dense, ds, source, n_classes,
                lambda f, m="": progress(0.5 + 0.48 * f, ""), visibility=vis,
                priority_names=hard)
        progress(1.0, "Images → cloud complete")
        return out

    def finish(out):
        result = {"images": out["images"],
                  "sparse": labels.merge_cloud_labels(out["sparse"])}
        if "dense" in out:
            result["dense"] = labels.merge_cloud_labels(out["dense"], dense=True)
        engine.dirty = True
        engine.changed("labels")
        sync_mesh_labels(engine)
        return result

    return Deferred(work, finish)


@method("transfer.cloud_to_images")
def transfer_cloud_to_images(engine: Engine, params: dict) -> Deferred:
    """Mark every solved photo the user hasn't drawn on as auto-labelled
    wherever the labelled cloud reaches it. No masks are stored: overlays
    are rendered from the cloud on request (``image.mask``)."""
    from cloudlabeller.transfer import cloud_to_image

    project = engine.project
    if project.dataset.cloud is None:
        raise Problem("transfer.no_cloud", "Run SfM first.")
    source = _labelled_source(project)
    if source is None:
        raise Problem("transfer.cloud_unlabelled", "Label the cloud first.")
    kind, cloud, cloud_labels_ = source
    targets = [(r.name, r.camera) for r in project.dataset.solved_images()
               if project.labels.status_of(r.name) != "user"]
    vis = project.visibility_index()

    def work(progress):
        counts = {}
        for i, (name, cam) in enumerate(targets, 1):
            mask = cloud_to_image(cloud, cloud_labels_, cam, name=name, visibility=vis)
            counts[name] = int((mask != UNLABELLED_ID).sum())
            progress(i / max(len(targets), 1), f"{name}: {counts[name]:,} px")
        return counts

    def finish(counts):
        marked = _mark_auto(engine, counts, 1)
        engine.changed("status", images=counts)
        return {"cloud": kind, "auto_labelled": marked}

    return Deferred(work, finish)


PROPAGATE_MIN_PX = 500


@method("transfer.propagate")
def transfer_propagate(engine: Engine, params: dict) -> Deferred:
    """Spread one photo's labels, through the cloud, to the photos that see
    the same surface. Merged into the cloud labels; nothing is erased."""
    from cloudlabeller.core.dataset import Dataset
    from cloudlabeller.transfer import cloud_to_image, images_to_cloud

    record = _image(engine, params)
    project, name = engine.project, record.name
    labels, ds = project.labels, project.dataset
    mask = labels.image_masks.get(name)
    if mask is None or not (mask != UNLABELLED_ID).any():
        raise Problem("transfer.image_unlabelled", "That photo has no labels.", name=name)
    if record.camera is None:
        raise Problem("transfer.image_unsolved", "That photo has no solved camera.",
                      name=name)
    cloud = ds.dense_cloud if ds.dense_cloud is not None else ds.cloud
    if cloud is None:
        raise Problem("transfer.no_cloud", "Run SfM first.")
    kind = "dense" if cloud is ds.dense_cloud else "sparse"
    n_classes = len(project.schema.classes)
    targets = [(r.name, r.camera) for r in ds.solved_images()
               if r.name != name and labels.status_of(r.name) != "user"]
    vis = project.visibility_index()
    mask = mask.copy()

    def work(progress):
        progress(0.0, f"Finding images overlapping {name}")
        overlap = project.overlapping_images(name)
        cand = targets if overlap is None else [t for t in targets if t[0] in overlap]
        temp = images_to_cloud(cloud, Dataset(images=[record]), {name: mask},
                               n_classes, visibility=vis)
        progress(0.15, f"{int((temp != UNLABELLED_ID).sum()):,} cloud points labelled")
        counts = {}
        for i, (nm, cam) in enumerate(cand, 1):
            m = cloud_to_image(cloud, temp, cam, name=nm, visibility=vis)
            counts[nm] = int((m != UNLABELLED_ID).sum())
            progress(0.15 + 0.85 * i / max(len(cand), 1), f"{nm}: {counts[nm]:,} px")
        return {"labels": temp, "counts": counts}

    def finish(out):
        n = labels.merge_cloud_labels(out["labels"], dense=(kind == "dense"))
        marked = _mark_auto(engine, out["counts"], PROPAGATE_MIN_PX)
        if n:
            engine.dirty = True
            engine.changed("labels")
            sync_mesh_labels(engine)
        engine.changed("status", images=out["counts"])
        return {"cloud": kind, "points": n, "auto_labelled": marked}

    return Deferred(work, finish)


# -- model --------------------------------------------------------------------------
@method("model.spec")
def model_spec(engine: Engine, params: dict) -> dict:
    project = engine.project
    spec = project.model_spec()
    res = project.source_resolution()
    return {"spec": spec.to_dict(),
            "training_size": list(spec.training_size(*res)) if res else None,
            "problem": spec.validate_size(*res) if res else None}


@method("model.set_spec")
def model_set_spec(engine: Engine, params: dict) -> dict:
    from cloudlabeller.ml.model_spec import ModelSpec

    try:
        spec = ModelSpec.from_dict(require(params, "spec", dict))
    except (TypeError, ValueError) as exc:
        raise InvalidParams(f"not a model spec: {exc}") from exc
    engine.project.set_model_spec(spec)
    engine.project.save_manifest()
    engine.changed("settings")
    return model_spec(engine, {})


@method("models.list")
def models_list(engine: Engine, params: dict) -> dict:
    from cloudlabeller.ml.model_store import list_models

    return {"models": list_models(engine.project.ml_dir)}


# -- jobs ---------------------------------------------------------------------------
@method("jobs.kinds", needs_project=False)
def jobs_kinds(engine: Engine, params: dict) -> dict:
    from cloudlabeller.engine.jobs import KINDS

    return {"kinds": [{"kind": k.name, "group": k.group, "description": k.description}
                      for k in KINDS.values()]}


@method("jobs.plan")
def jobs_plan(engine: Engine, params: dict) -> dict:
    """Check a Job can run and say how to run it (the core starts it)."""
    return engine.jobs.plan(require(params, "kind", str),
                            optional(params, "options", dict, {}))


@method("jobs.finish")
def jobs_finish(engine: Engine, params: dict) -> dict:
    """The core reports a planned Job's end; apply it to the Project."""
    exit_code = params.get("exit")
    if exit_code is not None and (isinstance(exit_code, bool)
                                  or not isinstance(exit_code, int)):
        raise InvalidParams("param 'exit' must be an integer or null", param="exit")
    results = optional(params, "results", dict, {})
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in results.items()):
        raise InvalidParams("results maps each RESULT key to the rest of its line",
                            param="results")
    return engine.jobs.finish(require(params, "plan", int), exit_code, results,
                              optional(params, "cancelled", bool, False))


@method("jobs.list", needs_project=False)
def jobs_list(engine: Engine, params: dict) -> dict:
    """The plans not yet finished."""
    return {"jobs": engine.jobs.list()}
