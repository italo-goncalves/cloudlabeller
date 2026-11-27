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

"""The Engine's methods for the 3D scene: what the viewer shows and edits.

A display subsample of a cloud, the lasso, deleting and confidence-cleaning
sparse points, the solved cameras, a photo's file, and the mesh's cost
estimate. Registered by importing :mod:`cloudlabeller.engine.methods`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cloudlabeller.engine import arrays
from cloudlabeller.engine.methods import (
    _cloud,
    _image,
    _image_size,
    _survey,
    _which,
    framed_xyz,
)
from cloudlabeller.engine.protocol import InvalidParams, Problem
from cloudlabeller.engine.server import Deferred, Engine, method, optional, require


#: What ``cloud.view`` colours the points by: the photos, or the normals as a
#: false-colour RGB (``transfer.normal_map.encode_normals``, Project frame).
COLOURS = ("photo", "normals")

#: A photo's normal map is sent at most this many pixels on its longer side.
NORMALS_SIDE = 2048


def _number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidParams(f"param {name!r} must be a number", param=name)
    return float(value)


# -- clouds ----------------------------------------------------------------------
@method("cloud.view")
def cloud_view(engine: Engine, params: dict) -> dict:
    """A seeded, evenly spread subsample of a cloud for display."""
    from cloudlabeller.core.selection import even_subsample

    survey = _survey(engine, params)
    which = _which(params)
    cloud = _cloud(survey, which)
    max_points = optional(params, "max_points", int, 0)
    if max_points < 0:
        raise InvalidParams("max_points must be 0 or more", param="max_points")
    colour = optional(params, "colour", str, "photo")
    if colour not in COLOURS:
        raise InvalidParams(f"colour must be one of {', '.join(COLOURS)}", param="colour")
    root, base = engine.project.root, f"view/{survey.id}/{which}"
    index = even_subsample(cloud.n_points, max_points)
    xyz = cloud.xyz if index is None else cloud.xyz[index]
    xyz = survey.active.frame_points(xyz)
    if colour == "normals":
        if cloud.normals is None:
            raise Problem("cloud.no_normals", "This cloud has no normals: only a dense "
                          "cloud does.", cloud=which)
        from cloudlabeller.transfer.normal_map import encode_normals, unit

        normals = cloud.normals if index is None else cloud.normals[index]
        rgb = encode_normals(unit(survey.active.frame_normals(normals)))
    else:
        rgb = cloud.rgb if (index is None or cloud.rgb is None) else cloud.rgb[index]
    return {"cloud": which, "n_points": cloud.n_points, "shown": len(xyz),
            "xyz": arrays.write(root, f"{base}/xyz", xyz.astype(np.float32)),
            "rgb": (arrays.write(root, f"{base}/rgb", rgb.astype(np.uint8))
                    if rgb is not None else None),
            "index": (arrays.write(root, f"{base}/index", index.astype(np.int32))
                      if index is not None else None)}


@method("image.normals")
def image_normals(engine: Engine, params: dict) -> dict:
    """A photo coloured by the dense cloud's normals, as the cloud is."""
    survey = _survey(engine, params)
    record = _image(survey, params)
    nmap = survey.normal_map(record.name, "world")
    if nmap is None:
        return {"normals": None, "step": None}
    step = max(1, -(-max(nmap.shape[:2]) // NORMALS_SIDE))
    small = np.ascontiguousarray(nmap[::step, ::step])
    return {"normals": arrays.write(engine.project.root,
                                    f"normals/{survey.id}/{record.name}", small),
            "step": step}


@method("cloud.select")
def cloud_select(engine: Engine, params: dict) -> Deferred:
    """The full cloud's points inside a screen-space lasso, hidden ones too."""
    from cloudlabeller.core.selection import select_in_polygon

    survey = _survey(engine, params)
    which = _which(params)
    vp = require(params, "view_projection", list)
    if len(vp) != 16:
        raise InvalidParams("view_projection needs 16 numbers", param="view_projection")
    vp = [_number(v, "view_projection") for v in vp]
    poly = require(params, "polygon", list)
    if not all(isinstance(p, list) and len(p) == 2 for p in poly):
        raise InvalidParams("polygon is a list of [x, y] pairs", param="polygon")
    poly = np.array([[_number(x, "polygon"), _number(y, "polygon")] for x, y in poly],
                    np.float64).reshape(-1, 2)
    width = _number(require(params, "width", (int, float)), "width")
    height = _number(require(params, "height", (int, float)), "height")
    if width <= 0 or height <= 0:
        raise InvalidParams("width and height must be positive", param="width")
    xyz, root = framed_xyz(engine, survey, which), engine.project.root

    def work(progress):
        return select_in_polygon(xyz, vp, poly, width, height,
                                 progress=lambda f: progress(f, ""))

    def finish(indices):
        return {"count": int(indices.size),
                "indices": arrays.write(root, f"select/{survey.id}/{which}", indices)}

    return Deferred(work, finish)


def _sparse_only(params: dict) -> None:
    if optional(params, "cloud", str, "sparse") != "sparse":
        raise InvalidParams("only the sparse cloud can lose points", param="cloud")


def _delete_points(engine: Engine, survey, keep: np.ndarray) -> dict:
    """Drop the sparse points where ``keep`` is False, as the 0.2.x Qt window's
    ``_apply_sparse_clean`` did: the Reconstruction stops showing them (its
    product is not edited), the labels follow, and the undo history is
    cleared."""
    removed = survey.filter_sparse_cloud(keep) if not keep.all() else 0
    if removed:
        engine.cache.pop(("sparse_stats", survey.id), None)
        engine.save_project()
        engine.changed("sparse", "labels", survey=survey.id)
    return {"removed": int(removed), "n_points": survey.dataset.cloud.n_points}


@method("cloud.delete")
def cloud_delete(engine: Engine, params: dict) -> dict:
    _sparse_only(params)
    survey = _survey(engine, params)
    cloud = _cloud(survey, "sparse")
    ref = require(params, "indices", dict)
    if ref.get("dtype") not in ("int32", "uint32"):
        raise InvalidParams("indices must be int32 or uint32", param="indices")
    indices = arrays.read(engine.project.root, ref, ndim=1).astype(np.int64)
    if indices.size and (indices.min() < 0 or indices.max() >= cloud.n_points):
        raise Problem("cloud.bad_indices", "Some indices are outside the cloud.",
                      n_points=cloud.n_points)
    keep = np.ones(cloud.n_points, bool)
    keep[indices] = False
    return _delete_points(engine, survey, keep)


def _with_stats(engine: Engine, survey, then) -> Deferred:
    """A Deferred that runs ``then(views, errors)`` on the loop thread with
    the sparse cloud's SfM statistics, read from the saved COLMAP model
    (and cached for this cloud)."""
    from cloudlabeller.photogrammetry.extract import MIN_MODEL_MATCH

    cloud = _cloud(survey, "sparse")
    key = ("sparse_stats", survey.id)
    cached = engine.cache.get(key)
    sparse_root = survey.sfm_dir / "sparse"

    def work(progress):
        if cached is not None and cached[0] is cloud:
            return cached[1], cached[2]
        from cloudlabeller.photogrammetry.extract import load_best_model, sparse_point_stats

        progress(0.0, "Reading the COLMAP model")
        try:
            rec = load_best_model(sparse_root)
        except FileNotFoundError as exc:
            raise Problem("cloud.no_model",
                          "There is no saved COLMAP model: run SfM again.") from exc
        progress(0.5, "Matching the points")
        return sparse_point_stats(rec, cloud.xyz)

    def finish(stats):
        views, errors = stats
        if survey.dataset.cloud is not cloud:
            raise Problem("cloud.changed", "The cloud changed meanwhile. Try again.")
        matched = float((views > 0).mean()) if len(views) else 0.0
        if matched < MIN_MODEL_MATCH:
            raise Problem("cloud.model_mismatch",
                          "The saved COLMAP model doesn't match the cloud: run SfM again.",
                          matched=matched)
        engine.cache[key] = (cloud, views, errors)
        return then(views, errors)

    return Deferred(work, finish)


@method("cloud.stats")
def cloud_stats(engine: Engine, params: dict) -> Deferred:
    _sparse_only(params)
    survey = _survey(engine, params)
    root, base = engine.project.root, f"stats/{survey.id}"

    def then(views, errors):
        return {"n_points": len(views), "matched": float((views > 0).mean()),
                "views": arrays.write(root, f"{base}/views", views.astype(np.int32)),
                "errors": arrays.write(root, f"{base}/errors", errors.astype(np.float32))}
    return _with_stats(engine, survey, then)


def _thresholds(params: dict) -> tuple[int, float]:
    min_images = require(params, "min_images", int)
    max_error = _number(require(params, "max_error", (int, float)), "max_error")
    return min_images, max_error


@method("cloud.clean_preview")
def cloud_clean_preview(engine: Engine, params: dict) -> Deferred:
    from cloudlabeller.photogrammetry.extract import confidence_keep

    survey = _survey(engine, params)
    min_images, max_error = _thresholds(params)
    root = engine.project.root

    def then(views, errors):
        keep = confidence_keep(views, errors, min_images, max_error)
        return {"keep": int(keep.sum()), "total": len(keep),
                "removed": arrays.write(root, f"clean/{survey.id}/removed",
                                        np.flatnonzero(~keep).astype(np.uint32))}
    return _with_stats(engine, survey, then)


@method("cloud.clean")
def cloud_clean(engine: Engine, params: dict) -> Deferred:
    from cloudlabeller.photogrammetry.extract import confidence_keep

    survey = _survey(engine, params)
    min_images, max_error = _thresholds(params)
    return _with_stats(engine, survey, lambda views, errors: _delete_points(
        engine, survey, confidence_keep(views, errors, min_images, max_error)))


# -- cameras and photos ----------------------------------------------------------
@method("cameras.list")
def cameras_list(engine: Engine, params: dict) -> dict:
    """The solved cameras, and the scene's up for the home view."""
    from cloudlabeller.photogrammetry.cameras import scene_up
    from cloudlabeller.photogrammetry.crs import frame_offset

    survey = _survey(engine, params)
    geo = engine.project.frame
    offset = frame_offset(geo) if survey.aligned else np.zeros(3)
    rec = survey.active
    solved = survey.dataset.solved_images()
    framed = [rec.frame_camera(r.camera) for r in solved]
    cams = []
    for r, c in zip(solved, framed):
        R = np.asarray(c.R, np.float64)
        t = np.asarray(c.t, np.float64)
        position = -(R.T @ t)
        cams.append({"name": r.name, "width": int(c.width), "height": int(c.height),
                     "K": np.asarray(c.K, np.float64).tolist(), "R": R.tolist(),
                     "t": t.tolist(), "position": position.tolist(),
                     "position_crs": (position + offset).tolist()})
    up = scene_up([c.R for c in framed], bool(geo) and survey.aligned)
    return {"cameras": cams, "up": up.tolist() if up is not None else None}


@method("image.info")
def image_info(engine: Engine, params: dict) -> dict:
    survey = _survey(engine, params)
    record = _image(survey, params)
    h, w = _image_size(record)
    path = Path(record.path)
    try:
        rel = path.relative_to(engine.project.root).as_posix()
    except ValueError:
        rel = path.as_posix()
    return {"path": rel, "width": w, "height": h}


# -- mesh ------------------------------------------------------------------------
@method("mesh.estimate")
def mesh_estimate(engine: Engine, params: dict) -> Deferred:
    """A rough cost for meshing the dense cloud: time, RAM, vertices."""
    from cloudlabeller.photogrammetry.meshing import estimate_mesh_cost, total_ram_gb

    method_ = optional(params, "method", str, "poisson")
    if method_ not in ("poisson", "delaunay"):
        raise InvalidParams("method must be 'poisson' or 'delaunay'", param="method")
    depth = optional(params, "depth", int, None)
    survey = _survey(engine, params)
    dense = survey.dataset.dense_cloud
    if dense is None:
        raise Problem("job.mesh.no_dense", "Meshing needs the dense cloud.")
    cached = engine.cache.get("poisson_depth")

    def work(progress):
        if cached is not None and cached[0] is dense:
            suggested = cached[1]
        else:
            from cloudlabeller.photogrammetry.meshing import suggest_poisson_depth

            suggested = suggest_poisson_depth(dense)
        est = estimate_mesh_cost(dense.n_points,
                                 depth if depth is not None else suggested,
                                 suggested, method=method_)
        return suggested, est

    def finish(out):
        suggested, est = out
        if survey.dataset.dense_cloud is dense:
            engine.cache["poisson_depth"] = (dense, suggested)
        return {"minutes": float(est["minutes"]), "ram_gb": float(est["ram_gb"]),
                "vertices": float(est["vertices"]), "suggested_depth": suggested,
                "machine_ram_gb": total_ram_gb()}

    return Deferred(work, finish)
