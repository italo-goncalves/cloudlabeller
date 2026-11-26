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

"""The Engine's methods for GCPs, their Marks, and aligning to them.

GCPs are the Project's; photo Marks the Survey's; cloud Marks the active
Reconstruction's, kept in its own frame. ``cloud.pick`` is how a click on
the 3D view finds the cloud point a Mark snaps to. ``align.fit`` fits every
Survey's Alignment to the GCPs (:mod:`cloudlabeller.photogrammetry.align`).
Registered by importing :mod:`cloudlabeller.engine.methods`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cloudlabeller.core.gcp import GCP, USES, CsvError, next_gcp_id, parse_csv
from cloudlabeller.engine.methods import _cloud, _image, _survey, _which, framed_xyz
from cloudlabeller.engine.protocol import InvalidParams, Problem
from cloudlabeller.engine.server import Deferred, Engine, method, optional, require


def _number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidParams(f"param {name!r} must be a number", param=name)
    return float(value)


def _gcp(engine: Engine, params: dict, key: str = "id") -> GCP:
    gid = require(params, key, str)
    gcp = engine.project.gcps.get(gid)
    if gcp is None:
        raise Problem("gcp.not_found", "No GCP has that id.", id=gid)
    return gcp


def _marks_of(project, gid: str) -> dict:
    out = {}
    for sid, survey in project.surveys.items():
        rec = survey.active
        photos = sorted(survey.marks.get(gid, {}))
        cloud = rec is not None and gid in rec.marks
        if photos or cloud:
            out[sid] = {"photos": photos, "cloud": cloud}
    return out


def _gcp_row(project, g: GCP) -> dict:
    return {**g.to_dict(), "marks": _marks_of(project, g.id)}


def gcp_list_result(engine: Engine) -> dict:
    project = engine.project
    return {"gcps": [_gcp_row(project, g) for g in project.gcps.values()],
            "fit": project.settings.get("gcp_fit")}


def _fields(params: dict, gcp: GCP) -> None:
    """Apply the GCP fields a request gives to ``gcp``."""
    if "name" in params:
        name = require(params, "name", str).strip()
        if not name:
            raise InvalidParams("a GCP needs a name", param="name")
        gcp.name = name
    if "xyz" in params:
        xyz = params["xyz"]
        if xyz is None:
            gcp.xyz = None
        else:
            if not isinstance(xyz, list) or len(xyz) != 3:
                raise InvalidParams("xyz is three numbers or null", param="xyz")
            gcp.xyz = [_number(v, "xyz") for v in xyz]
    if "epsg" in params:
        gcp.epsg = optional(params, "epsg", int, None)
    if "orthometric" in params:
        gcp.orthometric = bool(optional(params, "orthometric", bool, False))
    if "accuracy" in params:
        acc = params["accuracy"]
        gcp.accuracy = None if acc is None else _number(acc, "accuracy")
    if "use" in params:
        use = require(params, "use", str)
        if use not in USES:
            raise InvalidParams("use is 'control' or 'check'", param="use")
        gcp.use = use
    if gcp.xyz is not None and gcp.epsg is None:
        raise InvalidParams("surveyed coordinates need their CRS (epsg)", param="epsg")


def _saved(engine: Engine) -> dict:
    engine.project.save_manifest()
    engine.changed("gcps")
    return gcp_list_result(engine)


# -- GCPs ---------------------------------------------------------------------------
@method("gcp.list")
def gcp_list(engine: Engine, params: dict) -> dict:
    return gcp_list_result(engine)


@method("gcp.create")
def gcp_create(engine: Engine, params: dict) -> dict:
    project = engine.project
    gcp = GCP(id=next_gcp_id(project.gcps), name="")
    _fields({"name": f"GCP {len(project.gcps) + 1}", **params}, gcp)
    project.gcps[gcp.id] = gcp
    return {"id": gcp.id, **_saved(engine)}


@method("gcp.update")
def gcp_update(engine: Engine, params: dict) -> dict:
    gcp = _gcp(engine, params)
    _fields(params, gcp)
    return _saved(engine)


@method("gcp.delete")
def gcp_delete(engine: Engine, params: dict) -> dict:
    """Delete a GCP and every Mark of it."""
    gcp = _gcp(engine, params)
    project = engine.project
    del project.gcps[gcp.id]
    for survey in project.surveys.values():
        survey.marks.pop(gcp.id, None)
        for rec in survey.reconstructions.values():
            rec.marks.pop(gcp.id, None)
    return _saved(engine)


@method("gcp.import_csv")
def gcp_import_csv(engine: Engine, params: dict) -> dict:
    """GCPs from a coordinates file (``name, x, y, z[, accuracy]``); a name
    already there takes the new coordinates."""
    path = Path(require(params, "path", str))
    epsg = require(params, "epsg", int)
    ortho = optional(params, "orthometric", bool, False)
    use = optional(params, "use", str, "control")
    if use not in USES:
        raise InvalidParams("use is 'control' or 'check'", param="use")
    try:
        rows = parse_csv(path.read_text(encoding="utf-8-sig"))
    except OSError as exc:
        raise Problem("gcp.no_file", "The file can't be read.", path=str(path)) from exc
    except CsvError as exc:
        raise Problem("gcp.bad_csv", str(exc), line=exc.line) from exc
    project = engine.project
    by_name = {g.name: g for g in project.gcps.values()}
    added = updated = 0
    for name, xyz, acc in rows:
        gcp = by_name.get(name)
        if gcp is None:
            gcp = GCP(id=next_gcp_id(project.gcps), name=name, use=use)
            project.gcps[gcp.id] = gcp
            by_name[name] = gcp
            added += 1
        else:
            updated += 1
        gcp.xyz, gcp.epsg, gcp.orthometric = xyz, epsg, ortho
        if acc is not None:
            gcp.accuracy = acc
    return {"added": added, "updated": updated, **_saved(engine)}


# -- Marks --------------------------------------------------------------------------
@method("mark.set")
def mark_set(engine: Engine, params: dict) -> dict:
    """Place a GCP's Mark: on photo ``name`` at pixel ``uv``, or on the cloud
    at point ``index`` (as ``cloud.pick`` found it). Placing it again moves it."""
    survey = _survey(engine, params)
    gcp = _gcp(engine, params, "gcp")
    if "name" in params:
        record = _image(survey, params)
        uv = require(params, "uv", list)
        if len(uv) != 2:
            raise InvalidParams("uv is two numbers", param="uv")
        u, v = (_number(x, "uv") for x in uv)
        survey.marks.setdefault(gcp.id, {})[record.name] = [u, v]
    else:
        which = _which(params)
        cloud = _cloud(survey, which)
        index = require(params, "index", int)
        if not 0 <= index < cloud.n_points:
            raise Problem("cloud.bad_indices", "That point is outside the cloud.",
                          n_points=cloud.n_points)
        survey.active.marks[gcp.id] = [float(v) for v in cloud.xyz[index]]
    survey.save_manifest()
    engine.changed("gcps", survey=survey.id)
    return {"gcp": _gcp_row(engine.project, gcp)}


@method("mark.delete")
def mark_delete(engine: Engine, params: dict) -> dict:
    """Remove a GCP's Mark on photo ``name``, or (no ``name``) its cloud Mark."""
    survey = _survey(engine, params)
    gcp = _gcp(engine, params, "gcp")
    if "name" in params:
        name = require(params, "name", str)
        photos = survey.marks.get(gcp.id, {})
        photos.pop(name, None)
        if not photos:
            survey.marks.pop(gcp.id, None)
    elif survey.active is not None:
        survey.active.marks.pop(gcp.id, None)
    survey.save_manifest()
    engine.changed("gcps", survey=survey.id)
    return {"gcp": _gcp_row(engine.project, gcp)}


def _targets(project) -> dict:
    """Surveyed GCPs in the Project frame (those that can be converted)."""
    from cloudlabeller.photogrammetry.crs import to_project_frame

    geo = project.frame
    out = {}
    if not geo:
        return out
    for g in project.gcps.values():
        if g.surveyed:
            try:
                out[g.id] = to_project_frame(geo, g.epsg, g.orthometric, g.xyz)[0]
            except RuntimeError:
                continue
    return out


@method("marks.photo")
def marks_photo(engine: Engine, params: dict) -> dict:
    """A photo's Marks, and where the other GCPs should be in it (``hints``):
    projected from where the Reconstruction puts them, or, when it is
    aligned, from their surveyed coordinates."""
    from cloudlabeller.photogrammetry.align import gcp_positions, inverse_points
    from cloudlabeller.transfer.projection import project_points

    survey = _survey(engine, params)
    record = _image(survey, params)
    project = engine.project
    marks = [{"gcp": gid, "uv": photos[record.name]}
             for gid, photos in survey.marks.items()
             if record.name in photos and gid in project.gcps]
    hints = []
    cam, rec = record.camera, survey.active
    if cam is not None and rec is not None:
        pos = gcp_positions(survey, list(project.gcps))
        targets = _targets(project) if rec.aligned else {}
        marked = {m["gcp"] for m in marks}
        for gid in project.gcps:
            if gid in marked:
                continue
            if gid in pos:
                X = pos[gid]["xyz"]
            elif gid in targets:
                X = inverse_points(rec.alignment, targets[gid])[0]
            else:
                continue
            p, ok = project_points(cam, np.asarray(X, np.float64)[None])
            if ok[0]:
                hints.append({"gcp": gid, "uv": [float(p[0, 0]), float(p[0, 1])]})
    return {"marks": marks, "hints": hints}


@method("marks.cloud")
def marks_cloud(engine: Engine, params: dict) -> dict:
    """Where the active Reconstruction puts each GCP, in the Project frame:
    ``{marks: [{gcp, xyz, from}]}``, ``from`` being ``cloud`` or ``photos``."""
    from cloudlabeller.photogrammetry.align import gcp_positions

    survey = _survey(engine, params)
    rec = survey.active
    if rec is None:
        return {"marks": []}
    pos = gcp_positions(survey, list(engine.project.gcps))
    return {"marks": [{"gcp": gid, "xyz": rec.frame_points(p["xyz"][None])[0].tolist(),
                       "from": p["from"]} for gid, p in pos.items()]}


@method("cloud.pick")
def cloud_pick(engine: Engine, params: dict) -> dict:
    """The cloud point a click snaps to: the front-most point within
    ``radius`` CSS pixels (default 8) of ``(x, y)``, or ``index`` null."""
    from cloudlabeller.core.selection import pick_nearest

    survey = _survey(engine, params)
    which = _which(params)
    _cloud(survey, which)
    vp = require(params, "view_projection", list)
    if len(vp) != 16:
        raise InvalidParams("view_projection needs 16 numbers", param="view_projection")
    vp = [_number(v, "view_projection") for v in vp]
    x = _number(require(params, "x", (int, float)), "x")
    y = _number(require(params, "y", (int, float)), "y")
    width = _number(require(params, "width", (int, float)), "width")
    height = _number(require(params, "height", (int, float)), "height")
    radius = _number(optional(params, "radius", (int, float), 8.0), "radius")
    xyz = framed_xyz(engine, survey, which)
    index = pick_nearest(xyz, vp, x, y, width, height, radius)
    return {"cloud": which, "index": index,
            "xyz": xyz[index].tolist() if index is not None else None}


# -- the fit ------------------------------------------------------------------------
@method("align.fit")
def align_fit(engine: Engine, params: dict) -> Deferred:
    """Fit every Survey's Alignment to the GCPs; see ``docs/engine.md``."""
    from cloudlabeller.photogrammetry.align import fit, gcp_positions
    from cloudlabeller.photogrammetry.crs import frame_from_coordinates, to_project_frame

    project = engine.project
    if not project.gcps:
        raise Problem("align.no_gcps", "Add GCPs and mark them first.")
    if engine.jobs.outstanding:
        raise Problem("project.jobs_running", "Jobs are still running.",
                      jobs=engine.jobs.list())
    reference = optional(params, "reference", str, None)
    surveys = {sid: project.survey(sid) for sid in project.surveys}
    gcps = dict(project.gcps)
    geo = project.frame
    surveyed = [g for g in gcps.values() if g.surveyed]
    new_frame = None
    if surveyed and not geo:
        first = surveyed[0]
        same = [g.xyz for g in surveyed if (g.epsg, g.orthometric) == (first.epsg, first.orthometric)]
        try:
            new_frame = frame_from_coordinates(first.epsg, first.orthometric, same)
        except RuntimeError as exc:
            raise Problem("export.needs_geoid", str(exc), epsg=first.epsg) from exc
        geo = new_frame
    current = {sid: s.active.alignment if s.active else None for sid, s in surveys.items()}

    def work(progress):
        targets = {}
        for g in surveyed:
            try:
                targets[g.id] = to_project_frame(geo, g.epsg, g.orthometric, g.xyz)[0]
            except RuntimeError as exc:
                raise Problem("export.needs_geoid", str(exc), epsg=g.epsg) from exc
        progress(0.2, "Locating the GCPs")
        positions = {sid: gcp_positions(s, gcps) for sid, s in surveys.items()
                     if s.active is not None}
        progress(0.7, "Fitting")
        return fit(gcps, positions, targets, current, reference), positions

    def finish(out):
        result, positions = out
        if new_frame is not None:
            project.settings["georeferenced"] = new_frame
        for sid, a in result["alignments"].items():
            surveys[sid].active.alignment = a
        located = {sid: {gid: {"from": p["from"], "photos": p.get("photos", 0),
                               **({"rms_px": p["rms_px"]} if "rms_px" in p else {})}
                         for gid, p in pos.items()} for sid, pos in positions.items()}
        summary = {"surveys": result["surveys"], "residuals": result["residuals"],
                   "located": located, "frame_created": new_frame is not None}
        project.settings["gcp_fit"] = summary
        engine.cache.clear()
        engine.save_project()
        for sid in result["alignments"]:
            engine.changed("sparse", "dense", "mesh", "images", "surveys", survey=sid)
        engine.changed("gcps", "settings")
        return summary

    return Deferred(work, finish)
