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

"""The Engine's methods for Surveys and their Reconstructions.

Making, describing and deleting Surveys; listing, choosing, renaming and
deleting a Survey's Reconstructions; and copying labels into the active
Reconstruction from another one. Registered by importing
:mod:`cloudlabeller.engine.methods`.
"""

from __future__ import annotations

import shutil

import numpy as np

from cloudlabeller.core.label_schema import UNLABELLED_ID
from cloudlabeller.core.labels import load_label_array
from cloudlabeller.engine.methods import _survey, survey_state
from cloudlabeller.engine.protocol import InvalidParams, Problem
from cloudlabeller.engine.server import Deferred, Engine, method, optional, require

META = ("name", "timestamp", "operator", "camera", "notes")


def _meta(params: dict) -> dict:
    out = {}
    for k in META:
        v = optional(params, k, str, None)
        if v is not None:
            out[k] = v.strip()
    if out.get("name") == "":
        raise InvalidParams("a Survey needs a name", param="name")
    return out


def _idle(engine: Engine, what: str) -> None:
    if engine.jobs.outstanding:
        raise Problem("project.jobs_running", f"Jobs are still running: {what} later.",
                      jobs=engine.jobs.list())


# -- surveys ------------------------------------------------------------------------
@method("survey.create")
def survey_create(engine: Engine, params: dict) -> dict:
    """A new, empty Survey, after the others."""
    project = engine.project
    meta = _meta(params)
    survey = project.add_survey(meta.pop("name", ""), **meta)
    engine.changed("surveys")
    return survey_state(project.survey(survey.id))


@method("survey.update")
def survey_update(engine: Engine, params: dict) -> dict:
    """Change a Survey's name, timestamp, operator, camera or notes."""
    survey = _survey(engine, params)
    for k, v in _meta(params).items():
        setattr(survey, k, (v or None) if k == "timestamp" else v)
    survey.save_manifest()
    engine.changed("surveys", survey=survey.id)
    return survey_state(survey)


@method("survey.delete")
def survey_delete(engine: Engine, params: dict) -> dict:
    """Delete a Survey: its photos, Masks, predictions, Reconstructions and
    products. The last Survey of a Project can't go."""
    from cloudlabeller.core.survey import dir_size

    survey = _survey(engine, params)
    project = engine.project
    if len(project.surveys) == 1:
        raise Problem("survey.last", "A Project keeps at least one Survey.")
    _idle(engine, "delete the Survey")
    freed = dir_size(survey.dir)
    del project.surveys[survey.id]
    shutil.rmtree(survey.dir, ignore_errors=True)
    project.save_manifest()
    for key in [k for k in engine.cache if isinstance(k, tuple) and survey.id in k]:
        engine.cache.pop(key, None)
    engine.changed("surveys")
    return {"freed_bytes": freed, "surveys": list(project.surveys)}


# -- reconstructions ----------------------------------------------------------------
def _reconstruction(survey, params: dict, key: str = "reconstruction"):
    rid = require(params, key, str)
    if rid not in survey.reconstructions:
        raise Problem("reconstruction.not_found", "No Reconstruction has that id.",
                      reconstruction=rid, survey=survey.id)
    return survey.reconstructions[rid]


def _list(survey) -> dict:
    return {"reconstructions": survey.reconstruction_rows(), "active": survey.active_id}


@method("reconstruction.list")
def reconstruction_list(engine: Engine, params: dict) -> dict:
    return _list(_survey(engine, params))


@method("reconstruction.activate")
def reconstruction_activate(engine: Engine, params: dict) -> dict:
    """Show, label and export another of the Survey's Reconstructions."""
    survey = _survey(engine, params)
    rec = _reconstruction(survey, params)
    _idle(engine, "switch Reconstructions")
    if rec.id != survey.active_id:
        survey.activate(rec.id)
        engine.dirty = False
        engine.cache.clear()
        engine.changed("sparse", "dense", "mesh", "labels", "images", "surveys",
                       survey=survey.id)
    return _list(survey)


@method("reconstruction.rename")
def reconstruction_rename(engine: Engine, params: dict) -> dict:
    survey = _survey(engine, params)
    rec = _reconstruction(survey, params)
    name = require(params, "name", str).strip()
    if not name:
        raise InvalidParams("a Reconstruction needs a name", param="name")
    rec.name = name
    rec.save()
    engine.changed("surveys", survey=survey.id)
    return _list(survey)


@method("reconstruction.delete")
def reconstruction_delete(engine: Engine, params: dict) -> dict:
    """Delete a Reconstruction and the products no other one uses."""
    survey = _survey(engine, params)
    rec = _reconstruction(survey, params)
    _idle(engine, "delete the Reconstruction")
    was_active = rec.id == survey.active_id
    if was_active:
        engine.save_project()
    freed = survey.delete_reconstruction(rec.id)
    if was_active:
        engine.cache.clear()
        engine.changed("sparse", "dense", "mesh", "labels", "images", survey=survey.id)
    engine.changed("surveys", survey=survey.id)
    return {"freed_bytes": freed, **_list(survey)}


# -- copying labels between Reconstructions ----------------------------------------------
def _geometry(survey, rec):
    """A Reconstruction's sparse cloud (its kept points) and dense cloud, and
    their labels, read from disk."""
    from cloudlabeller.io.geometry import load_cloud_npz

    def products(slot):
        pid = rec.products.get(slot)
        return survey.products_dir / pid if pid else None

    sparse = dense = None
    sfm, mvs = products("sfm"), products("mvs")
    if sfm and (sfm / "cloud.npz").exists():
        xyz = np.load(sfm / "cloud.npz")["xyz"]
        keep = rec.load_keep()
        sparse = xyz[keep] if keep is not None and len(keep) == len(xyz) else xyz
    if mvs and (mvs / "dense.npz").exists():
        dense = load_cloud_npz(mvs / "dense.npz").xyz

    def labels(name, n):
        f = rec.labels_dir / name
        if n is None or not f.exists():
            return None
        arr = load_label_array(f)
        return arr if len(arr) == n else None
    return {"sparse": (sparse, labels("sparse.npy", None if sparse is None else len(sparse))),
            "dense": (dense, labels("dense.npy", None if dense is None else len(dense)))}


def nearest_labels(source_xyz: np.ndarray, source_labels: np.ndarray,
                   target_xyz: np.ndarray) -> np.ndarray:
    """Each target point takes the label of the nearest labelled source
    point, when that point is within three times the source's typical
    spacing; others stay unlabelled."""
    from scipy.spatial import cKDTree

    out = np.full(len(target_xyz), UNLABELLED_ID, np.int32)
    have = source_labels != UNLABELLED_ID
    if not have.any() or not len(target_xyz):
        return out
    src = np.asarray(source_xyz, np.float64)
    tree_all = cKDTree(src)
    sample = src[np.random.default_rng(0).choice(len(src), min(len(src), 2000), replace=False)]
    spacing = float(np.median(tree_all.query(sample, k=2)[0][:, 1])) if len(src) > 1 else 0.0
    tree = cKDTree(src[have])
    dist, idx = tree.query(np.asarray(target_xyz, np.float64), k=1,
                           distance_upper_bound=max(3.0 * spacing, 1e-9))
    hit = np.isfinite(dist)
    out[hit] = source_labels[have][idx[hit]]
    return out


@method("transfer.reconstruction")
def transfer_reconstruction(engine: Engine, params: dict) -> Deferred:
    """Copy labels from another Reconstruction of the Survey into the active
    one, point to nearest point (sparse to sparse, dense to dense), merged
    as one undoable edit. The two must share their SfM product (one frame)
    or both be aligned into the Project frame."""
    from cloudlabeller.engine.methods import sync_mesh_labels

    survey = _survey(engine, params)
    target = survey.active
    source = _reconstruction(survey, params, "from")
    if target is None or source.id == target.id:
        raise Problem("transfer.same_reconstruction", "Choose another Reconstruction.")
    same_frame = source.products.get("sfm") and source.products.get("sfm") == \
        target.products.get("sfm")
    if not same_frame and not (source.aligned and target.aligned):
        raise Problem("transfer.not_comparable",
                      "These Reconstructions don't share a frame: align both first.")
    ds = survey.dataset
    targets = {"sparse": ds.cloud, "dense": ds.dense_cloud}

    def work(progress):
        progress(0.0, "Reading the other Reconstruction")
        geo = _geometry(survey, source)
        out = {}
        for i, (kind, cloud) in enumerate(targets.items()):
            src_xyz, src_labels = geo[kind]
            if cloud is None:
                continue
            if src_labels is None or not (src_labels != UNLABELLED_ID).any():
                # no labels of that kind there: fall back to its other cloud
                other = geo["dense" if kind == "sparse" else "sparse"]
                src_xyz, src_labels = other
            if src_xyz is None or src_labels is None:
                continue
            a = src_xyz if same_frame else source.frame_points(src_xyz)
            b = cloud.xyz if same_frame else target.frame_points(cloud.xyz)
            out[kind] = nearest_labels(a, src_labels, b)
            progress(0.5 + 0.5 * i, f"{kind} cloud done")
        return out

    def finish(out):
        if survey.dataset is not ds or survey.active is not target:
            raise Problem("cloud.changed", "The Reconstruction changed meanwhile. Try again.")
        if not out:
            raise Problem("transfer.cloud_unlabelled", "That Reconstruction has no labels.")
        lb = survey.labels
        result = {"sparse": lb.merge_cloud_labels(out["sparse"]) if "sparse" in out else 0}
        if "dense" in out:
            result["dense"] = lb.merge_cloud_labels(out["dense"], dense=True)
        engine.dirty = True
        engine.changed("labels", survey=survey.id)
        sync_mesh_labels(engine, survey)
        return result

    return Deferred(work, finish)
