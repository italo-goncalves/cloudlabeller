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
handlers carry over what the 0.2.x Qt window decided around each action (which
images a transfer reads, what a status change persists), not any UI.

Every method about photos, clouds or the mesh names its Survey (``survey``)
and works on that Survey's active Reconstruction. Geometry sent to the
window is in the Project frame (the Reconstruction's Alignment applied);
transfers work in the Reconstruction's own frame, where cameras and clouds
agree.

Persistence is as it was in the 0.2.x Qt app: a status change writes the
manifests at once; a painted image mask is written at once; other label
arrays are marked dirty and saved within a minute, on ``project.save`` and
on close.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cloudlabeller.core import sources as src
from cloudlabeller.core.label_schema import UNLABELLED_ID
from cloudlabeller.core.labels import narrow
from cloudlabeller.engine import arrays
from cloudlabeller.engine.protocol import InvalidParams, Problem
from cloudlabeller.engine.server import Deferred, Engine, method, optional, require

CLOUDS = ("sparse", "dense")
#: What an inbound mask may be; outbound masks and labels come narrowed
#: (``core.labels.narrow``: int8 for up to 127 classes).
MASK_DTYPES = ("int8", "int16", "int32")


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


# -- surveys ------------------------------------------------------------------
def _survey(engine: Engine, params: dict):
    """The Survey a request names (``survey``), loaded."""
    sid = require(params, "survey", str)
    project = engine.require_project()
    if sid not in project.surveys:
        raise Problem("survey.not_found", "No Survey has that id.", survey=sid)
    return project.survey(sid)


def survey_state(survey) -> dict:
    """A Survey as ``project.state`` reports it. A Survey not loaded yet is
    described from its files, without loading its clouds."""
    lb = survey.labels
    rec = survey.active
    if survey.loaded:
        ds = survey.dataset
        images = [{"name": r.name, "status": lb.status_of(r.name),
                   "solved": r.camera is not None, "user_mask": r.name in lb.image_masks}
                  for r in ds.images]
        unsure = survey.settings.get("uncertainty") or {}
        for row in images:
            row["uncertainty"] = unsure.get(row["name"]) if row["status"] == "ml" else None
    else:
        masks = ({f.stem for f in survey.masks_dir.glob("*.npy")}
                 if survey.masks_dir.exists() else set())
        unsure = survey.settings.get("uncertainty") or {}
        images = [{"name": name, "status": lb.status_of(name), "solved": solved,
                   "user_mask": name in masks and lb.status_of(name) == "user",
                   "uncertainty": unsure.get(name) if lb.status_of(name) == "ml" else None}
                  for name, solved in survey.peek_images()]
    counts = survey.peek_counts()
    outdated = survey.settings.get("reconstruction_outdated")
    return {
        "id": survey.id, "name": survey.name, "timestamp": survey.timestamp,
        "operator": survey.operator, "camera": survey.camera, "notes": survey.notes,
        "images": images,
        "reconstruction": ("none" if counts["sparse"] is None
                           else "outdated" if outdated else "current"),
        "active": rec.id if rec is not None else None,
        "reconstructions": len(survey.reconstructions),
        "aligned": survey.aligned,
        **counts,
        "normals": survey.has_normals(),
        "source_resolution": survey.peek_resolution(),
        "can_undo": lb.can_undo(),
        "can_redo": lb.can_redo(),
    }


# -- project ------------------------------------------------------------------
def _schema(project) -> list[dict]:
    return [{"id": c.id, "name": c.name, "color": c.color, "hotkey": c.hotkey}
            for c in project.schema.classes]


def project_state(engine: Engine) -> dict:
    project = engine.require_project()
    return {
        "root": str(project.root),
        "schema": _schema(project),
        "surveys": [survey_state(survey) for survey in project.surveys.values()],
        "georeferenced": project.frame,
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
    from cloudlabeller.core.project import MANIFEST, Project, ProjectFormatError

    root = Path(require(params, "root", str))
    if not (root / MANIFEST).is_file():
        raise Problem("project.not_found", "No Project there.", root=str(root))
    _no_outstanding_jobs(engine)
    engine.close_project()
    try:
        engine.project = Project.open(root)
    except ProjectFormatError as exc:
        raise Problem("project.format", str(exc), root=str(root)) from exc
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
    project = engine.project
    project.delete_label(_class_id(engine, params))
    engine.dirty = True
    result = _schema_changed(engine)
    for sid, survey in project.surveys.items():
        engine.changed("labels", survey=sid, images=survey.labels.image_masks)
    return result


# -- undo / redo -------------------------------------------------------------------
def _after_history(engine: Engine, survey, changed) -> dict:
    from cloudlabeller.core.labels import Modality

    if changed is None:
        return {"changed": None}
    modality, key = changed
    engine.dirty = True
    if modality is Modality.IMAGE and key:
        engine.changed("labels", survey=survey.id, images=[key])
    else:
        engine.changed("labels", survey=survey.id)
        sync_mesh_labels(engine, survey)
    return {"changed": {"modality": modality.value, "image": key}}


@method("labels.undo")
def labels_undo(engine: Engine, params: dict) -> dict:
    survey = _survey(engine, params)
    return _after_history(engine, survey, survey.labels.undo())


@method("labels.redo")
def labels_redo(engine: Engine, params: dict) -> dict:
    survey = _survey(engine, params)
    return _after_history(engine, survey, survey.labels.redo())


# -- images ------------------------------------------------------------------------
def _image(survey, params: dict):
    name = require(params, "name", str)
    record = survey.dataset.image_by_name(name)
    if record is None:
        raise Problem("image.not_found", "No image has that name.", name=name,
                      survey=survey.id)
    return record


def _image_size(record) -> tuple[int, int]:
    """(height, width) of an image: its camera, else its file header."""
    if record.camera is not None:
        return int(record.camera.height), int(record.camera.width)
    from PIL import Image

    with Image.open(record.path) as im:
        return int(im.size[1]), int(im.size[0])


def _below(engine: Engine, survey, name: str):
    """What shows under a photo's Mask (``Survey.overlay_below``), kept for
    the last photo asked: ``image.save_mask`` compares against what the
    window was shown. ``Engine.changed`` drops it."""
    key = (survey.id, name)
    kept = engine.cache.get("overlay_below")
    if kept is not None and kept[0] == key:
        return kept[1]
    layers = survey.overlay_below(name)
    engine.cache["overlay_below"] = (key, layers)
    return layers


def _composed(user, below):
    """The user's Mask over what shows under it (either may be None)."""
    if user is None:
        return below
    if below is None or below.shape != user.shape:
        return user
    return np.where(user != UNLABELLED_ID, user, below)


@method("image.mask")
def image_mask(engine: Engine, params: dict) -> dict:
    """The mask to show over a photo: the user's own labels, over the
    cloud's projection, over the U-Net's prediction (ADR-0006). ``source``
    is the best one present."""
    survey = _survey(engine, params)
    record = _image(survey, params)
    name = record.name
    user = survey.labels.image_masks.get(name)
    below, below_source, _ = _below(engine, survey, name)
    source = None
    if user is not None and (user != UNLABELLED_ID).any():
        source = "user"
    elif below_source is not None and (below_source == src.CLOUD).any():
        source = "auto"
    elif below_source is not None and (below_source == src.PREDICTED).any():
        source = "ml"
    mask = _composed(user, below)
    if mask is None:
        return {"source": None, "mask": None}
    if source is None:
        source = "user"
    return {"source": source,
            "mask": arrays.write(engine.project.root, f"mask/{survey.id}/{name}",
                                 narrow(mask))}


@method("image.save_mask")
def image_save_mask(engine: Engine, params: dict) -> dict:
    """Store the mask the user painted on a photo (one undoable edit).

    The window sends the whole mask, seeded from the overlay ``image.mask``
    showed. Only the user's strokes are kept: the pixels that differ from
    that overlay (ADR-0006). The cloud's projection and the U-Net's
    prediction under them are not copied in as hand-drawn labels.
    """
    survey = _survey(engine, params)
    record = _image(survey, params)
    project, name = engine.project, record.name
    h, w = _image_size(record)
    ref = require(params, "mask", dict)
    if ref.get("dtype") not in MASK_DTYPES:
        raise InvalidParams(f"a mask is {', '.join(MASK_DTYPES)}, not {ref.get('dtype')!r}",
                            dtype=ref.get("dtype"))
    new = arrays.read(project.root, ref, ndim=2).astype(np.int32, copy=False)
    if new.shape != (h, w):
        raise Problem("image.mask_shape", "The mask is not the photo's size.",
                      expected=[h, w], got=list(new.shape))
    n_classes = len(project.schema.classes)
    if new.size and (new.min() < UNLABELLED_ID or new.max() >= n_classes):
        raise Problem("image.mask_values", "The mask holds ids outside the schema.",
                      n_classes=n_classes)
    labels = survey.labels
    user = labels.image_masks.get(name)
    below = _below(engine, survey, name)[0]
    shown = _composed(user, below)
    if shown is None or shown.shape != new.shape:
        mine = new
    else:
        strokes = new != shown                        # what the user changed
        mine = (user.copy() if user is not None
                else np.full((h, w), UNLABELLED_ID, dtype=np.int32))
        mine[strokes] = new[strokes]
    if user is None:
        if not (mine != UNLABELLED_ID).any():
            return {"changed_pixels": 0, "status": labels.status_of(name)}
        labels.init_image(name, h, w)
    old = labels.image_masks[name]
    changed = np.flatnonzero(old.reshape(-1) != mine.reshape(-1))
    if changed.size:
        labels.paint_image(name, changed, mine.reshape(-1)[changed])
    labels.mark_user_labeled(name)
    survey.save_mask(name)
    survey.save_manifest()
    kept = engine.cache.get("overlay_below")
    engine.changed("labels", "status", survey=survey.id, images=[name])
    if kept is not None and kept[0] == (survey.id, name):
        engine.cache["overlay_below"] = kept         # the user's Mask isn't part of it
    return {"changed_pixels": int(changed.size), "status": labels.status_of(name)}


@method("image.uncertainty")
def image_uncertainty(engine: Engine, params: dict) -> dict:
    """How unsure the U-Net's prediction of a photo is, per pixel (0 sure, 1
    unsure), at the prediction's size; null when it has none."""
    survey = _survey(engine, params)
    record = _image(survey, params)
    u = survey.prediction_uncertainty(record.name)
    if u is None:
        return {"uncertainty": None, "mean": None}
    return {"uncertainty": arrays.write(engine.project.root,
                                        f"uncertainty/{survey.id}/{record.name}", u),
            "mean": float(u.mean(dtype=np.float64))}


# -- clouds and mesh -----------------------------------------------------------------
def _which(params: dict) -> str:
    which = optional(params, "cloud", str, "sparse")
    if which not in CLOUDS:
        raise InvalidParams("cloud must be 'sparse' or 'dense'", param="cloud")
    return which


def _cloud(survey, which: str):
    ds = survey.dataset
    cloud = ds.cloud if which == "sparse" else ds.dense_cloud
    if cloud is None:
        raise Problem("cloud.none", "That cloud doesn't exist yet.", cloud=which,
                      survey=survey.id)
    return cloud


def framed_xyz(engine: Engine, survey, which: str) -> np.ndarray:
    """A cloud's points in the Project frame (cached for this cloud and
    Alignment)."""
    cloud = _cloud(survey, which)
    rec = survey.active
    key = ("framed", survey.id, which)
    cached = engine.cache.get(key)
    if cached is not None and cached[0] is cloud and cached[1] == rec.alignment:
        return cached[2]
    xyz = rec.frame_points(cloud.xyz)
    engine.cache[key] = (cloud, rec.alignment, xyz)
    return xyz


def _cloud_labels(survey, which: str, n: int) -> np.ndarray:
    lb = survey.labels
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
    survey = _survey(engine, params)
    which = _which(params)
    cloud = _cloud(survey, which)
    root, base = engine.project.root, f"cloud/{survey.id}/{which}"
    return {"cloud": which, "n_points": cloud.n_points,
            "xyz": arrays.write(root, f"{base}/xyz", framed_xyz(engine, survey, which)),
            "rgb": (arrays.write(root, f"{base}/rgb", cloud.rgb.astype(np.uint8))
                    if cloud.rgb is not None else None)}


@method("cloud.labels")
def cloud_labels(engine: Engine, params: dict) -> dict:
    survey = _survey(engine, params)
    which = _which(params)
    cloud = _cloud(survey, which)
    labels = _cloud_labels(survey, which, cloud.n_points)
    return {"cloud": which,
            "labels": arrays.write(engine.project.root, f"cloud/{survey.id}/{which}/labels",
                                   narrow(labels))}


@method("cloud.sources")
def cloud_sources(engine: Engine, params: dict) -> dict:
    """Where each point's label came from and how sure it is (ADR-0006)."""
    survey = _survey(engine, params)
    which = _which(params)
    cloud = _cloud(survey, which)
    _cloud_labels(survey, which, cloud.n_points)
    source, unc = survey.labels.provenance(dense=which == "dense")
    base = f"cloud/{survey.id}/{which}"
    return {"cloud": which,
            "source": arrays.write(engine.project.root, f"{base}/source", source),
            "uncertainty": arrays.write(engine.project.root, f"{base}/uncertainty",
                                        unc.astype(np.float32))}


@method("cloud.paint")
def cloud_paint(engine: Engine, params: dict) -> dict:
    """Give the chosen points one class (the lasso): one undoable edit."""
    survey = _survey(engine, params)
    which = _which(params)
    cloud = _cloud(survey, which)
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
    _cloud_labels(survey, which, cloud.n_points)
    lb = survey.labels
    if indices.size:
        if which == "sparse":
            lb.paint_cloud(indices, class_id)
        else:
            lb.paint_dense_cloud(indices, class_id)
        engine.dirty = True
        engine.changed("labels", survey=survey.id)
        if which == "dense":
            sync_mesh_labels(engine, survey)
    return {"painted": int(indices.size)}


def _mesh(survey):
    mesh = survey.dataset.mesh
    if mesh is None:
        raise Problem("mesh.none", "There is no mesh yet.", survey=survey.id)
    return mesh


@method("mesh.geometry")
def mesh_geometry(engine: Engine, params: dict) -> dict:
    survey = _survey(engine, params)
    mesh = _mesh(survey)
    root, base = engine.project.root, f"mesh/{survey.id}"
    vertices = survey.active.frame_points(mesh.vertices)
    return {"vertices": arrays.write(root, f"{base}/vertices", vertices),
            "faces": arrays.write(root, f"{base}/faces", mesh.faces.astype(np.int32)),
            "colors": (arrays.write(root, f"{base}/colors",
                                    mesh.vertex_colors.astype(np.uint8))
                       if mesh.vertex_colors is not None else None)}


@method("mesh.labels")
def mesh_labels(engine: Engine, params: dict) -> dict:
    survey = _survey(engine, params)
    mesh = _mesh(survey)
    labels = survey.labels.mesh_labels
    if labels is None or len(labels) != len(mesh.vertices):
        labels = np.full(len(mesh.vertices), UNLABELLED_ID, np.int32)
    return {"labels": arrays.write(engine.project.root, f"mesh/{survey.id}/labels",
                                   narrow(labels))}


def sync_mesh_labels(engine: Engine, survey) -> None:
    """Recompute the mesh's labels from the dense cloud (nearest point), in
    the background. Called whenever the dense labels or the mesh change."""
    mesh, dense = survey.dataset.mesh, survey.dataset.dense_cloud
    if mesh is None or dense is None:
        return
    labels = survey.labels.dense_cloud_labels
    if labels is None or len(labels) != dense.n_points or not (labels != UNLABELLED_ID).any():
        survey.labels.mesh_labels = np.full(len(mesh.vertices), UNLABELLED_ID, np.int32)
        engine.dirty = True
        engine.changed("labels", survey=survey.id)
        return
    from cloudlabeller.transfer.mesh_cloud import cloud_to_mesh

    snapshot = labels.copy()
    cache = survey.mesh_nn_cache_path
    cache.parent.mkdir(parents=True, exist_ok=True)

    def finish(mesh_labels) -> None:
        if survey.dataset.mesh is mesh:
            survey.labels.mesh_labels = mesh_labels
            engine.dirty = True
            engine.changed("labels", survey=survey.id)

    engine.run_background(lambda: cloud_to_mesh(dense, snapshot, mesh, cache_path=cache),
                          finish)


# -- transfers -----------------------------------------------------------------------
def _labelled_source(survey):
    """The cloud whose labels are projected onto photos: dense when labelled."""
    ds, lb = survey.dataset, survey.labels
    if (ds.dense_cloud is not None and lb.dense_cloud_labels is not None
            and len(lb.dense_cloud_labels) == ds.dense_cloud.n_points
            and (lb.dense_cloud_labels != UNLABELLED_ID).any()):
        return "dense", ds.dense_cloud, lb.dense_cloud_labels.copy()
    if lb.cloud_labels is not None and (lb.cloud_labels != UNLABELLED_ID).any():
        return "sparse", ds.cloud, lb.cloud_labels.copy()
    return None


def _mark_auto(survey, counts: dict, min_px: int) -> list[str]:
    marked = [name for name, n_px in counts.items() if n_px >= min_px]
    for name in marked:
        survey.labels.mark_auto_labeled(name)
    survey.save_manifest()
    return marked


@method("transfer.images_to_cloud")
def transfer_images_to_cloud(engine: Engine, params: dict) -> Deferred:
    """Vote the photos' labels onto the sparse cloud and the dense one.

    User-drawn masks outrank auto ones. With ``include_auto`` (default true)
    the cloud-projected photos that cover the surface vote too. The U-Net's
    predictions vote only with ``include_predictions`` (default false), so
    the network's output doesn't come back as cloud-made labels it then
    trains on (ADR-0006). The result is merged: points no photo reached keep
    their labels.
    """
    from cloudlabeller.transfer.project_to_cloud import VotePass, vote

    survey = _survey(engine, params)
    ds, labels = survey.dataset, survey.labels
    if ds.cloud is None:
        raise Problem("transfer.no_cloud", "Run SfM first.")
    include_auto = optional(params, "include_auto", bool, True)
    include_predictions = optional(params, "include_predictions", bool, False)
    masks = dict(labels.image_masks)
    n_classes = len(engine.project.schema.classes)
    user_names = [n for n, m in masks.items() if (m != UNLABELLED_ID).any()]
    solved = [r.name for r in ds.solved_images() if r.name not in masks]
    auto_names = [n for n in solved if labels.status_of(n) == "auto"] if include_auto else []
    ml_names = [n for n in solved if labels.status_of(n) == "ml"] if include_predictions else []
    if not auto_names and not ml_names and not user_names:
        raise Problem("transfer.no_labelled_images", "No photo is labelled yet.")
    sparse, dense = ds.cloud, ds.dense_cloud
    vis = survey.visibility_index()
    found = survey._auto_cloud()
    projected_from = None if found is None else ("dense" if found[2] else "sparse")

    def only(layers, code):
        """A layered mask's labels of one source (None: none of them)."""
        if layers is None:
            return None
        mask = layers[0].copy()
        mask[layers[1] != code] = UNLABELLED_ID
        return mask

    def work(progress):
        covering = auto_names
        if auto_names:
            adj = survey.image_adjacency()
            if adj is not None:
                from cloudlabeller.photogrammetry.adjacency import select_covering

                covering = select_covering(auto_names, adj)
        n_images = len(user_names) + len(covering) + len(ml_names)
        progress(0.0, f"Transferring labels from {n_images} images")
        out = {"images": n_images}
        targets = [("sparse", sparse)] + ([("dense", dense)] if dense is not None else [])
        for t, (kind, cloud) in enumerate(targets):
            # A projection doesn't vote on the cloud it came from (ADR-0006).
            projected = covering if kind != projected_from else []
            passes = [
                VotePass(src.HAND, user_names, masks.get),
                VotePass(src.CLOUD, projected,
                         lambda n: only(survey.cloud_layers(n), src.CLOUD)),
                VotePass(src.PREDICTED, ml_names + (projected if include_predictions else []),
                         lambda n: (only(survey.render_auto_layers(n), src.PREDICTED))),
            ]
            share = 0.96 / len(targets)
            out[kind] = vote(cloud, ds, passes, n_classes,
                             lambda f, m="", t=t, share=share: progress(0.02 + share * (t + f), ""),
                             visibility=vis)
        progress(1.0, "Images → cloud complete")
        return out

    def finish(out):
        result = {"images": out["images"]}
        for kind in ("sparse", "dense"):
            if kind in out:
                v = out[kind]
                result[kind] = labels.merge_cloud_labels(v.labels, dense=kind == "dense",
                                                         source=v.source,
                                                         uncertainty=v.uncertainty)
        engine.dirty = True
        engine.changed("labels", survey=survey.id)
        sync_mesh_labels(engine, survey)
        return result

    return Deferred(work, finish)


@method("transfer.cloud_to_images")
def transfer_cloud_to_images(engine: Engine, params: dict) -> Deferred:
    """Mark every solved photo the user hasn't drawn on as auto-labelled
    wherever the labelled cloud reaches it. No masks are stored: overlays
    are rendered from the cloud on request (``image.mask``)."""
    from cloudlabeller.transfer import cloud_to_image

    survey = _survey(engine, params)
    if survey.dataset.cloud is None:
        raise Problem("transfer.no_cloud", "Run SfM first.")
    source = _labelled_source(survey)
    if source is None:
        raise Problem("transfer.cloud_unlabelled", "Label the cloud first.")
    kind, cloud, cloud_labels_ = source
    targets = [(r.name, r.camera) for r in survey.dataset.solved_images()
               if survey.labels.status_of(r.name) != "user"]
    vis = survey.visibility_index()

    def work(progress):
        counts = {}
        for i, (name, cam) in enumerate(targets, 1):
            mask = cloud_to_image(cloud, cloud_labels_, cam, name=name, visibility=vis)
            counts[name] = int((mask != UNLABELLED_ID).sum())
            progress(i / max(len(targets), 1), f"{name}: {counts[name]:,} px")
        return counts

    def finish(counts):
        marked = _mark_auto(survey, counts, 1)
        engine.changed("status", survey=survey.id, images=counts)
        return {"cloud": kind, "auto_labelled": marked}

    return Deferred(work, finish)


PROPAGATE_MIN_PX = 500


@method("transfer.propagate")
def transfer_propagate(engine: Engine, params: dict) -> Deferred:
    """Spread one photo's labels, through the cloud, to the photos that see
    the same surface. Merged into the cloud labels; nothing is erased."""
    from cloudlabeller.core.dataset import Dataset
    from cloudlabeller.transfer import cloud_to_image
    from cloudlabeller.transfer.project_to_cloud import VotePass, vote

    survey = _survey(engine, params)
    record = _image(survey, params)
    name = record.name
    labels, ds = survey.labels, survey.dataset
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
    n_classes = len(engine.project.schema.classes)
    targets = [(r.name, r.camera) for r in ds.solved_images()
               if r.name != name and labels.status_of(r.name) != "user"]
    vis = survey.visibility_index()
    mask = mask.copy()

    def work(progress):
        progress(0.0, f"Finding images overlapping {name}")
        overlap = survey.overlapping_images(name)
        cand = targets if overlap is None else [t for t in targets if t[0] in overlap]
        votes = vote(cloud, Dataset(images=[record]),
                     [VotePass(src.HAND, [name], {name: mask}.get)], n_classes, visibility=vis)
        temp = votes.labels
        progress(0.15, f"{int((temp != UNLABELLED_ID).sum()):,} cloud points labelled")
        counts = {}
        for i, (nm, cam) in enumerate(cand, 1):
            m = cloud_to_image(cloud, temp, cam, name=nm, visibility=vis)
            counts[nm] = int((m != UNLABELLED_ID).sum())
            progress(0.15 + 0.85 * i / max(len(cand), 1), f"{nm}: {counts[nm]:,} px")
        return {"votes": votes, "counts": counts}

    def finish(out):
        v = out["votes"]
        n = labels.merge_cloud_labels(v.labels, dense=(kind == "dense"), source=v.source,
                                      uncertainty=v.uncertainty)
        marked = _mark_auto(survey, out["counts"], PROPAGATE_MIN_PX)
        if n:
            engine.dirty = True
            engine.changed("labels", survey=survey.id)
            sync_mesh_labels(engine, survey)
        engine.changed("status", survey=survey.id, images=out["counts"])
        return {"cloud": kind, "points": n, "auto_labelled": marked}

    return Deferred(work, finish)


# -- model --------------------------------------------------------------------------
@method("model.spec")
def model_spec(engine: Engine, params: dict) -> dict:
    spec = engine.project.model_spec()
    n = spec.net_size
    return {"spec": spec.to_dict(), "training_size": [n, n], "problem": spec.validate(),
            "default_tile_size": engine.project.default_tile_size()}


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
    from cloudlabeller.ml.model_spec import is_tiled
    from cloudlabeller.ml.model_store import list_models

    return {"models": [{**m, "tiled": is_tiled(m)} for m in list_models(engine.project.ml_dir)]}


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


@method("jobs.facts")
def jobs_facts(engine: Engine, params: dict) -> Deferred:
    """What a Job's dialog shows before planning it (GPS, GPU, depth, counts).
    ``survey`` is needed for sfm, mvs and mesh."""
    from cloudlabeller.engine.jobs import facts

    kind_name = require(params, "kind", str)
    survey = _survey(engine, params) if kind_name in ("sfm", "mvs", "mesh") else None
    work, finish = facts(engine, engine.project, kind_name, survey)
    return Deferred(work, finish)


# -- ingest -------------------------------------------------------------------------
@method("ingest.conflicts")
def ingest_conflicts(engine: Engine, params: dict) -> dict:
    """The photos in a folder whose names are already in the Survey's photos."""
    from cloudlabeller.core.images import find_conflicts

    survey = _survey(engine, params)
    folder = require(params, "folder", str)
    if not Path(folder).is_dir():
        raise Problem("job.ingest.no_folder", "The folder does not exist.", folder=folder)
    return {"conflicts": find_conflicts(folder, survey.images_dir)}


# -- models -------------------------------------------------------------------------
@method("models.delete")
def models_delete(engine: Engine, params: dict) -> dict:
    from cloudlabeller.ml.model_store import delete_model, list_models

    name = require(params, "name", str)
    ml_dir = engine.project.ml_dir
    if not any(m.get("name") == name for m in list_models(ml_dir)):
        raise Problem("models.not_found", "No saved model has that name.", name=name)
    if any(o.kind.group == "ml" for o in engine.jobs.outstanding.values()):
        raise Problem("models.in_use", "A training or prediction Job is running.")
    delete_model(ml_dir, name)
    engine.changed("settings")
    return {"models": list_models(ml_dir)}


# The scene (clouds, cameras, mesh) and the CRS/export methods live in their
# own modules; importing them registers them.
from cloudlabeller.engine import geo as _geo  # noqa: F401
from cloudlabeller.engine import scene as _scene  # noqa: F401
from cloudlabeller.engine import gcps as _gcps  # noqa: F401
from cloudlabeller.engine import surveys as _surveys  # noqa: F401
