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

"""The Engine's methods for coordinate frames and export.

Searching the projected-CRS catalogue, suggesting the site's UTM zone,
describing the Project's frame, the export settings kept in the manifest,
and exporting the cloud or the mesh (in the local frame, the stored CRS, or
a projected CRS from the ENU frame). An export names its Survey and writes
its active Reconstruction in the Project frame; an unaligned one exports its
own local frame. Registered by importing :mod:`cloudlabeller.engine.methods`.
"""

from __future__ import annotations

from pathlib import Path

from cloudlabeller.engine.protocol import InvalidParams, Problem
from cloudlabeller.engine.server import Deferred, Engine, method, optional, require

CLOUD_FORMATS = ("las", "csv", "ply")
MESH_FORMATS = ("ply", "obj", "stl")
EXPORT_MODES = ("local", "projected")


# -- CRS ---------------------------------------------------------------------------
@method("crs.search", needs_project=False)
def crs_search(engine: Engine, params: dict) -> dict:
    from cloudlabeller.photogrammetry.crs import search_projected_crs

    query = optional(params, "query", str, "")
    limit = optional(params, "limit", int, 50)
    if limit < 1:
        raise InvalidParams("limit must be at least 1", param="limit")
    return {"results": [{"epsg": code, "name": name}
                        for code, name in search_projected_crs(query, limit)]}


def _geo(engine: Engine) -> dict:
    geo = engine.project.frame
    if not geo:
        raise Problem("crs.not_georeferenced", "The Project isn't georeferenced.")
    return geo


def _origin(engine: Engine, geo: dict):
    """The exact ENU origin; a legacy Project's is healed and saved."""
    from cloudlabeller.photogrammetry.crs import heal_enu_origin

    project = engine.project
    # A legacy origin is recovered from the first SfM product's camera order.
    sfm = next((s for s in project.surveys.values() if s.sfm_dir is not None), None)
    workspace = sfm.sfm_dir if sfm else project.root
    images = sfm.images_dir if sfm else project.root
    origin, healed = heal_enu_origin(geo, workspace, images)
    if healed:
        project.save_manifest()
        engine.changed("settings")
    return origin


@method("crs.suggest")
def crs_suggest(engine: Engine, params: dict) -> dict:
    """The UTM zone covering the site, in the national datum where one exists."""
    from cloudlabeller.photogrammetry.crs import projected_crs_catalogue, suggest_projected_epsg

    geo = _geo(engine)
    origin = _origin(engine, geo)
    epsg = suggest_projected_epsg(origin.lla)
    name = dict(projected_crs_catalogue()).get(epsg)
    return {"epsg": epsg, "name": name, "origin_lla": list(origin.lla),
            "exact": origin.exact}


@method("crs.frame")
def crs_frame(engine: Engine, params: dict) -> dict:
    from cloudlabeller.photogrammetry.crs import frame_labels

    short, full = frame_labels(engine.project.frame)
    return {"short": short, "full": full}


# -- export settings ---------------------------------------------------------------
def _settings(engine: Engine) -> dict:
    geo = engine.project.frame or {}
    crs_info = geo.get("crs")
    frame = "local" if not geo else ("crs" if crs_info else "enu")
    return {"frame": frame,
            "crs": ({"epsg": crs_info.get("epsg"), "name": crs_info.get("name"),
                     "orthometric": bool(crs_info.get("orthometric", False)),
                     "offset": list(crs_info.get("offset") or [0.0, 0.0, 0.0])}
                    if crs_info else None),
            "export_mode": geo.get("export_mode"),
            "export_epsg": geo.get("export_epsg"),
            "export_orthometric": bool(geo.get("export_orthometric", False))}


@method("export.settings")
def export_settings(engine: Engine, params: dict) -> dict:
    return _settings(engine)


def _mode_choice(params: dict, default_mode, default_epsg, default_ortho):
    mode = optional(params, "export_mode", str, default_mode) or "local"
    if mode not in EXPORT_MODES:
        raise InvalidParams("export_mode must be 'local' or 'projected'",
                            param="export_mode")
    epsg = optional(params, "export_epsg", int, default_epsg)
    ortho = optional(params, "export_orthometric", bool, bool(default_ortho))
    if mode == "projected" and epsg is None:
        raise InvalidParams("a projected export needs export_epsg", param="export_epsg")
    return mode, epsg, ortho


@method("export.set_settings")
def export_set_settings(engine: Engine, params: dict) -> dict:
    geo = engine.project.frame
    if not geo:
        raise Problem("export.not_georeferenced",
                      "An unreferenced Project exports its local frame only.")
    mode, epsg, ortho = _mode_choice(params, geo.get("export_mode"),
                                     geo.get("export_epsg"),
                                     geo.get("export_orthometric", False))
    geo.update(export_mode=mode, export_epsg=epsg, export_orthometric=ortho)
    engine.project.save_manifest()
    engine.changed("settings")
    return _settings(engine)


# -- export ------------------------------------------------------------------------
def _target(params: dict, formats: tuple) -> Path:
    fmt = require(params, "format", str)
    if fmt not in formats:
        raise InvalidParams(f"format must be one of {', '.join(formats)}",
                            param="format", choices=list(formats))
    path = Path(require(params, "path", str))
    if path.suffix.lower() != "." + fmt:
        path = path.with_name(path.name + "." + fmt)
    if not path.parent.is_dir():
        raise Problem("export.no_folder", "The folder does not exist.",
                      folder=str(path.parent))
    return path


def _export_spec(engine: Engine, params: dict, survey) -> dict:
    """How this export writes coordinates (see ``crs.export_spec``). A
    choice made for an ENU Project is kept in the manifest, as the Qt app
    did (0.2.x); ``crs`` null reuses the kept one. An unaligned Survey
    exports its own local frame."""
    from cloudlabeller.photogrammetry.crs import export_needs_geoid, export_spec

    geo = engine.project.frame
    if not survey.aligned:
        return export_spec(None)
    if not geo or geo.get("crs"):
        return export_spec(geo)
    choice = optional(params, "crs", dict, None)
    if choice is None:
        mode, epsg, ortho = (geo.get("export_mode") or "local", geo.get("export_epsg"),
                             geo.get("export_orthometric", False))
        if mode == "projected" and epsg is None:
            mode = "local"
    else:
        mode, epsg, ortho = _mode_choice(
            {"export_mode": choice.get("mode"), "export_epsg": choice.get("epsg"),
             "export_orthometric": choice.get("orthometric")}, "local", None, False)
        geo.update(export_mode=mode)
        if mode == "projected":
            geo.update(export_epsg=epsg, export_orthometric=ortho)
        engine.project.save_manifest()
        engine.changed("settings")
    if mode != "projected":
        return export_spec(geo)
    spec = export_spec(geo, "projected", epsg, ortho, _origin(engine, geo).lla)
    if export_needs_geoid(spec):
        raise Problem("export.needs_geoid",
                      "Sea-level heights need the EGM96 geoid grid: run the geoid Job.",
                      epsg=spec["epsg"])
    return spec


def _crs_text(spec: dict) -> str | None:
    if spec["kind"] == "local":
        return None
    from cloudlabeller.photogrammetry.crs import crs_display

    return crs_display(spec["epsg"]) + (" + EGM96 heights" if spec["orthometric"] else "")


@method("export.cloud")
def export_cloud_method(engine: Engine, params: dict) -> Deferred:
    """Write a cloud and its labels to LAS, CSV or PLY."""
    from cloudlabeller.core.dataset import PointCloud
    from cloudlabeller.engine.methods import _survey

    survey = _survey(engine, params)
    ds, lb = survey.dataset, survey.labels
    which = optional(params, "cloud", str, None)
    if which not in (None, "sparse", "dense"):
        raise InvalidParams("cloud must be 'sparse' or 'dense'", param="cloud")
    if which is None:
        which = "dense" if ds.dense_cloud is not None else "sparse"
    cloud = ds.dense_cloud if which == "dense" else ds.cloud
    if cloud is None:
        raise Problem("cloud.none", "That cloud doesn't exist yet.", cloud=which)
    labels = lb.dense_cloud_labels if which == "dense" else lb.cloud_labels
    labels = labels.copy() if labels is not None and len(labels) == cloud.n_points else None
    path = _target(params, CLOUD_FORMATS)
    spec = _export_spec(engine, params, survey)
    rec = survey.active
    raw = cloud

    def framed():
        return PointCloud(xyz=rec.frame_points(raw.xyz), rgb=raw.rgb,
                          normals=(rec.frame_normals(raw.normals)
                                   if raw.normals is not None else None))

    def work(progress):
        from cloudlabeller.io.export import export_cloud
        from cloudlabeller.photogrammetry.crs import build_export_transform

        transform, crs = build_export_transform(spec, progress)
        progress(0.05, f"Exporting the {which} cloud ({cloud.n_points:,} points)")
        export_cloud(framed(), labels, path, progress, transform=transform, crs=crs)

    def finish(_):
        return {"path": str(path), "cloud": which, "n_points": cloud.n_points,
                "crs": _crs_text(spec)}

    return Deferred(work, finish)


@method("export.mesh")
def export_mesh_method(engine: Engine, params: dict) -> Deferred:
    """Write the mesh to PLY, OBJ or STL."""
    from cloudlabeller.core.dataset import Mesh
    from cloudlabeller.engine.methods import _survey

    survey = _survey(engine, params)
    raw = survey.dataset.mesh
    if raw is None:
        raise Problem("mesh.none", "There is no mesh yet.")
    path = _target(params, MESH_FORMATS)
    spec = _export_spec(engine, params, survey)
    rec = survey.active
    mesh = raw

    def work(progress):
        from cloudlabeller.io.export import export_mesh
        from cloudlabeller.photogrammetry.crs import build_export_transform

        transform, crs = build_export_transform(spec, progress)
        progress(0.05, f"Exporting the mesh ({len(mesh.faces):,} faces)")
        framed = Mesh(vertices=rec.frame_points(raw.vertices), faces=raw.faces,
                      vertex_colors=raw.vertex_colors)
        export_mesh(framed, path, progress, transform=transform, crs=crs)

    def finish(_):
        return {"path": str(path), "vertices": len(mesh.vertices),
                "faces": len(mesh.faces), "crs": _crs_text(spec)}

    return Deferred(work, finish)
