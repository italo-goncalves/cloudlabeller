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

"""Jobs: the Engine plans them, the Tauri core runs them (ADR-0004).

A Job kind knows three things: how to build its command line from the
window's options (refusing with a Problem when it can't run), whether it is
a GPU Job, and what to do to the open Project when it ends well (reload
products, record a new frame, mark predicted images). That last part is
what the 0.2.x Qt window's ``_on_*_done`` handlers did.

The core runs ``<Environment python> -m <module> <args…>``, writes the
output to the plan's log file, and reads the ``*_cli`` line protocol:
``PROGRESS <frac> <msg>`` lines, ``RESULT…`` lines with the outcome. It
hands the ``RESULT…`` lines back to :meth:`JobPlans.finish` as
``{key: rest of line}``.

A plan's ``stage`` names the Project directories (relative to its root) the
Job reads (``in``) and writes (``out``). In the WSL Environment the core
copies the ``in`` dirs to WSL's own disk, runs the Job there with the
Project-root prefix of ``args`` rewritten, and copies the ``out`` dirs back
(WSL reads and writes Windows files slowly). Kinds without heavy file I/O
have no stage and run on the Project in place.

Two Jobs may be outstanding at once only if they are in different groups:
``reconstruction`` Jobs rewrite the Project's products, ``ml`` Jobs read
labels and write models and predictions, ``download`` Jobs fetch the geoid
grid or COLMAP into the user's own folders.
"""

from __future__ import annotations

import itertools
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from cloudlabeller.engine.protocol import InvalidParams, Problem

if TYPE_CHECKING:
    from cloudlabeller.core.project import Project
    from cloudlabeller.engine.server import Engine

log = logging.getLogger("cloudlabeller.engine.jobs")


@dataclass
class Plan:
    """Everything needed to start one Job and to finish it."""

    args: list[str]
    log_path: Path
    gpu: bool = False
    stop_file: Path | None = None        # graceful cancel: touch instead of kill
    context: dict = field(default_factory=dict)
    stage: dict | None = None            # {"in": [dirs], "out": [dirs]}, root-relative


@dataclass
class JobKind:
    name: str
    module: str
    group: str
    prepare: Callable[["Engine", "Project", dict], Plan]
    apply: Callable[["Engine", "Project", dict, Plan], Any]
    description: str = ""


KINDS: dict[str, JobKind] = {}


def kind(name: str, module: str, group: str, description: str):
    """Register a Job kind from its ``prepare`` function; its ``apply`` is
    registered separately with :func:`applies`."""
    def register(prepare):
        KINDS[name] = JobKind(name, module, group, prepare, _nothing, description)
        return prepare
    return register


def applies(name: str):
    def register(fn):
        KINDS[name].apply = fn
        return fn
    return register


def _nothing(engine, project, results, plan):
    return None


# -- option helpers -----------------------------------------------------------
def _opt(options: dict, key: str, kind_: type | tuple, default: Any) -> Any:
    value = options.get(key, default)
    if value is None:
        return default
    kinds = kind_ if isinstance(kind_, tuple) else (kind_,)
    if isinstance(value, bool) and bool not in kinds:      # bool is an int in Python
        raise InvalidParams(f"option {key!r} has the wrong type", option=key)
    if not isinstance(value, kinds):
        raise InvalidParams(f"option {key!r} has the wrong type", option=key)
    return value


def _choice(options: dict, key: str, choices: tuple, default: str) -> str:
    value = _opt(options, key, str, default)
    if value not in choices:
        raise InvalidParams(f"option {key!r} must be one of {', '.join(choices)}",
                            option=key, choices=list(choices))
    return value


def _colmap_binary() -> str | None:
    from cloudlabeller.config import AppConfig
    from cloudlabeller.photogrammetry.mvs import find_colmap_binary

    return find_colmap_binary(AppConfig.load().colmap_binary)


def _pycolmap_has_cuda() -> bool:
    """Whether this Environment's pycolmap is a CUDA build (WSL)."""
    try:
        import pycolmap
    except ImportError:
        return False
    return bool(pycolmap.has_cuda)


def _stage(project: "Project", reads: list[Path], writes: list[Path]) -> dict:
    """A plan's ``stage``: the directories as POSIX paths relative to the root."""
    root = Path(project.root)

    def rel(paths):
        return [p.relative_to(root).as_posix() for p in paths]
    return {"in": rel(reads), "out": rel(writes)}


def parse_result(results: dict, key: str = "RESULT") -> dict:
    """The JSON payload of a ``RESULT <json>`` line, or ``{}`` when there is
    none. A long payload comes in numbered pieces, ``RESULT.0``,
    ``RESULT.1``, … (``workers.result.print_result``), joined here.

    A payload that is there but isn't JSON is a Problem, never ``{}``: in
    0.4.0 a 65 KB prediction result arrived cut in two (the core tells longer
    lines in pieces), parsed as nothing, and no predicted photo was marked."""
    pieces = sorted((int(k[len(key) + 1:]), v) for k, v in results.items()
                    if k.startswith(key + ".") and k[len(key) + 1:].isdigit())
    text = "".join(v for _, v in pieces) if pieces else results.get(key, "")
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise Problem("job.bad_result", "The Job's result could not be read: "
                      f"{exc.msg} at character {exc.pos} of {len(text)}.",
                      key=key, length=len(text)) from exc
    return value if isinstance(value, dict) else {}


# -- surveys --------------------------------------------------------------------
def _survey(engine, project, options):
    """The Survey a reconstruction Job works on (option ``survey``)."""
    sid = _opt(options, "survey", str, None)
    if not sid:
        raise InvalidParams("option 'survey' is required", option="survey")
    if sid not in project.surveys:
        raise Problem("survey.not_found", "No Survey has that id.", survey=sid)
    return project.survey(sid)


def _logs(survey_or_project) -> Path:
    d = Path(getattr(survey_or_project, "dir", None) or survey_or_project.root) / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _planned(project, plan: Plan):
    """The Survey a finished plan was for."""
    return project.survey(plan.context["survey"])


# -- reconstruction -----------------------------------------------------------
@kind("ingest", "cloudlabeller.photogrammetry.run_cli", "reconstruction",
      "Copy a folder of photos into a Survey's photos")
def _prepare_ingest(engine, project, options):
    survey = _survey(engine, project, options)
    folder = _opt(options, "folder", str, None)
    if not folder:
        raise InvalidParams("option 'folder' is required", option="folder")
    if not Path(folder).is_dir():
        raise Problem("job.ingest.no_folder", "The folder does not exist.", folder=folder)
    policy = _choice(options, "on_conflict", ("skip", "overwrite"), "skip")
    survey.images_dir.mkdir(parents=True, exist_ok=True)
    args = [str(survey.images_dir), str(_logs(survey)),
            "--ingest-from", folder, "--ingest-only", "--on-conflict", policy]
    return Plan(args, _logs(survey) / "ingest.log", context={"survey": survey.id})


@applies("ingest")
def _apply_ingest(engine, project, results, plan):
    survey = _planned(project, plan)
    fields_ = dict(part.split("=", 1) for part in results.get("RESULT_INGEST", "").split()
                   if "=" in part)
    changed = fields_.get("changed") == "1"
    if changed and survey.dataset.cloud is not None:
        survey.mark_reconstruction_outdated()
    survey.load_products()
    if not survey.timestamp:
        from cloudlabeller.core.images import median_photo_time

        survey.timestamp = median_photo_time(survey.images_dir)
    engine.save_project()
    engine.changed("images", "settings", "surveys", survey=survey.id)
    return {"copied": int(fields_.get("copied", 0)),
            "overwritten": int(fields_.get("overwritten", 0)),
            "changed": changed,
            "n_images": len(survey.dataset.images),
            "reconstruction": survey.reconstruction_status()}


def _product_plan(survey, slot: str) -> tuple[Path, dict]:
    """Where a step writes (always a new product folder), and the plan
    context that says how finishing it records it."""
    out, fork = survey.plan_step(slot)
    out.mkdir(parents=True, exist_ok=True)
    return out, {"survey": survey.id, "slot": slot, "product": out.name, "fork": fork}


def _record_product(survey, plan: Plan, clear: tuple[str, ...]) -> None:
    """Record the product a step wrote: in the active Reconstruction when
    its slot was empty, else in a new child of it (ADR-0005). Labels of the
    new product start empty."""
    ctx = plan.context
    rec = survey.fork(ctx["slot"]) if ctx.get("fork") else survey.ensure_active()
    rec.products[ctx["slot"]] = ctx["product"]
    for attr in clear:
        setattr(survey.labels, attr, None)


@kind("sfm", "cloudlabeller.photogrammetry.run_cli", "reconstruction",
      "Structure from motion: solve the cameras and the sparse cloud")
def _prepare_sfm(engine, project, options):
    from cloudlabeller.photogrammetry.options import CAMERA_MODELS, MATCHERS, SfmOptions

    survey = _survey(engine, project, options)
    if not survey.dataset.images:
        raise Problem("job.sfm.no_images", "The Survey has no photos.")
    opts = SfmOptions(
        matcher=_choice(options, "matcher", MATCHERS, "spatial"),
        use_gpu=_opt(options, "gpu", bool, False),
        single_camera=_opt(options, "single_camera", bool, True),
        camera_model=_choice(options, "camera_model", CAMERA_MODELS, "SIMPLE_RADIAL"),
        max_image_size=_opt(options, "max_image_size", int, 3200),
    )
    out, context = _product_plan(survey, "sfm")
    args = [str(survey.images_dir), str(out), *opts.to_cli_args()]
    binary = _colmap_binary()
    if binary:
        args += ["--colmap-binary", binary]
    return Plan(args, _logs(survey) / "sfm.log", gpu=opts.use_gpu, context=context,
                stage=_stage(project, [survey.images_dir], [out]))


@applies("sfm")
def _apply_sfm(engine, project, results, plan):
    survey = _planned(project, plan)
    # A fresh solve is a new local frame: the earlier Reconstruction keeps
    # its Alignment, the new one has none until it is aligned.
    cleared = survey.aligned
    _record_product(survey, plan, ("cloud_labels",))
    rec = survey.active
    rec.save_keep(None)
    rec.alignment = None
    survey.load_products()
    if survey.dataset.cloud is None:
        raise Problem("job.sfm.no_cloud", "No point cloud was produced.")
    survey.mark_reconstruction_current()
    engine.save_project()
    engine.changed("sparse", "images", "labels", "settings", survey=survey.id)
    return {"n_points": survey.dataset.cloud.n_points,
            "n_cameras": len(survey.dataset.solved_images()),
            "georeferencing_cleared": cleared}


@kind("georef", "cloudlabeller.photogrammetry.georef_cli", "reconstruction",
      "Align a Survey to its photos' EXIF GPS (ENU, metres)")
def _prepare_georef(engine, project, options):
    survey = _survey(engine, project, options)
    if survey.dataset.cloud is None:
        raise Problem("job.no_reconstruction", "Run SfM first.")
    max_error = _opt(options, "max_error", (int, float), 5.0)
    args = [str(survey.sfm_dir), str(survey.images_dir), "--max-error", str(float(max_error))]
    if project.frame:
        from cloudlabeller.engine.geo import _origin

        args += ["--origin", *(repr(float(v)) for v in _origin(engine, project.frame).lla)]
    return Plan(args, _logs(survey) / "georef.log", context={"survey": survey.id})


@applies("georef")
def _apply_georef(engine, project, results, plan):
    from cloudlabeller.core.reconstruction import apply_points, compose

    survey = _planned(project, plan)
    out = parse_result(results)
    if not {"s", "R", "t"} <= set(out):
        raise Problem("job.georef.no_fit", "The fit printed no transform.")
    alignment = {"s": out["s"], "R": out["R"], "t": out["t"], "source": "gps",
                 "n_gps": out.get("n_gps"), "n_inliers": out.get("n_inliers"),
                 "rms": out.get("fit_rms_m"), "scale_m_per_unit": out.get("scale_m_per_unit")}
    geo = project.frame
    if not geo:
        geo = project.settings["georeferenced"] = {
            "frame": "ENU", "origin_lla": out.get("origin_lla"),
            "origin_convention": out.get("origin_convention"),
        }
    elif geo.get("crs"):
        # The fit is into ENU about the frame's origin; carry it on into the CRS.
        from cloudlabeller.photogrammetry.crs import enu_to_frame_similarity, site_samples

        samples = site_samples(apply_points(alignment, survey.dataset.cloud.xyz))
        try:
            s, R, t = enu_to_frame_similarity(geo, out.get("origin_lla"), samples)
        except RuntimeError as exc:
            raise Problem("export.needs_geoid", str(exc),
                          epsg=geo["crs"].get("epsg")) from exc
        alignment = compose({"s": s, "R": np.asarray(R).tolist(),
                             "t": np.asarray(t).tolist()}, alignment)
    survey.active.alignment = alignment
    engine.save_project()
    engine.changed("sparse", "dense", "mesh", "images", "settings", survey=survey.id)
    return out


def _aligned_samples(project) -> np.ndarray:
    """Points spread over every aligned Survey, in the Project frame."""
    from cloudlabeller.photogrammetry.crs import site_samples

    parts = []
    for sid in project.surveys:
        survey = project.survey(sid)
        if survey.aligned and survey.dataset.cloud is not None:
            parts.append(site_samples(survey.active.frame_points(survey.dataset.cloud.xyz)))
    return np.vstack(parts) if parts else np.empty((0, 3))


@kind("reproject", "cloudlabeller.photogrammetry.reproject_cli", "reconstruction",
      "Move the Project frame into a projected CRS")
def _prepare_reproject(engine, project, options):
    from cloudlabeller.engine.geo import _origin

    geo = project.frame
    if not geo:
        raise Problem("job.reproject.not_georeferenced",
                      "Reprojection needs a georeferenced model.")
    samples = _aligned_samples(project)
    if not len(samples):
        raise Problem("job.no_reconstruction", "No aligned Survey to reproject.")
    epsg = _opt(options, "epsg", int, None)
    if epsg is None:
        raise InvalidParams("option 'epsg' is required", option="epsg")
    logs = _logs(project)
    given = logs / "reproject_input.json"
    given.write_text(json.dumps({"frame": geo, "origin_lla": list(_origin(engine, geo).lla),
                                 "samples": samples.tolist()}), encoding="utf-8")
    args = [str(given), "--epsg", str(epsg)]
    if _opt(options, "orthometric", bool, False):
        args.append("--orthometric")
    return Plan(args, logs / "reproject.log")


@applies("reproject")
def _apply_reproject(engine, project, results, plan):
    from cloudlabeller.core.reconstruction import compose

    out = parse_result(results)
    if not {"s", "R", "t"} <= set(out):
        raise Problem("job.reproject.no_fit", "The fit printed no transform.")
    move = {"s": out["s"], "R": out["R"], "t": out["t"]}
    for survey in project.surveys.values():
        for rec in survey.reconstructions.values():
            if rec.alignment is not None:
                rec.alignment = compose(move, rec.alignment)
    project.frame["crs"] = {
        "epsg": out.get("epsg"), "name": out.get("name"),
        "orthometric": out.get("orthometric", False),
        "offset": out.get("offset"),
    }
    engine.cache.clear()
    engine.save_project()
    engine.changed("sparse", "dense", "mesh", "images", "settings")
    return out


@kind("mvs", "cloudlabeller.photogrammetry.mvs_cli", "reconstruction",
      "Dense multi-view stereo: the dense cloud")
def _prepare_mvs(engine, project, options):
    survey = _survey(engine, project, options)
    if survey.dataset.cloud is None:
        raise Problem("job.no_reconstruction", "Run SfM first.")
    binary = _colmap_binary()
    if binary is None and not _pycolmap_has_cuda():
        raise Problem("job.mvs.no_colmap",
                      "Dense MVS needs CUDA: the CUDA COLMAP executable or a CUDA "
                      "build of pycolmap.")
    out, context = _product_plan(survey, "mvs")
    args = [str(survey.sfm_dir), str(survey.images_dir), "--out", str(out),
            "--max-image-size", str(_opt(options, "max_image_size", int, 2000)),
            "--quality", _choice(options, "quality", ("standard", "draft"), "standard")]
    if binary:
        args += ["--colmap-binary", binary]
    return Plan(args, _logs(survey) / "mvs.log", gpu=True, context=context,
                stage=_stage(project, [survey.images_dir, survey.sfm_dir], [out]))


@applies("mvs")
def _apply_mvs(engine, project, results, plan):
    survey = _planned(project, plan)
    _record_product(survey, plan, ("dense_cloud_labels",))
    survey.load_products()
    engine.save_project()
    engine.changed("dense", survey=survey.id)
    dense = survey.dataset.dense_cloud
    return {"n_points": dense.n_points if dense is not None else 0}


@kind("mesh", "cloudlabeller.photogrammetry.mesh_cli", "reconstruction",
      "Build the triangle mesh from the dense cloud")
def _prepare_mesh(engine, project, options):
    survey = _survey(engine, project, options)
    dense_path = survey.dense_cloud_path
    if survey.dataset.dense_cloud is None or dense_path is None or not dense_path.exists():
        raise Problem("job.mesh.no_dense", "Meshing needs the dense cloud.")
    method = _choice(options, "method", ("poisson", "delaunay"), "poisson")
    binary = _colmap_binary()
    if method == "delaunay" and binary is None:
        # pycolmap binds Delaunay meshing only when built with CGAL, and the
        # 4.2 wheels (Windows CPU, Linux CUDA) are not.
        raise Problem("job.mesh.no_delaunay",
                      "Delaunay meshing needs the COLMAP executable; use Poisson.")
    out, context = _product_plan(survey, "mesh")
    args = [str(dense_path), str(out / "mesh.npz"), "--method", method]
    if method == "delaunay":
        args += ["--workspace", str(survey.mvs_dir / "dense")]
    else:
        depth = _opt(options, "depth", int, None)
        if depth is not None:
            args += ["--depth", str(depth)]
        args += ["--trim", str(float(_opt(options, "trim", (int, float), 10.0))),
                 "--point-weight", str(float(_opt(options, "point_weight", (int, float), 1.0)))]
    if binary:
        args += ["--colmap-binary", binary]
    return Plan(args, _logs(survey) / "mesh.log", context=context,
                stage=_stage(project, [survey.mvs_dir], [out]))


@applies("mesh")
def _apply_mesh(engine, project, results, plan):
    from cloudlabeller.engine.methods import sync_mesh_labels

    survey = _planned(project, plan)
    _record_product(survey, plan, ("mesh_labels",))
    survey.load_products()
    engine.save_project()
    engine.changed("mesh", survey=survey.id)
    sync_mesh_labels(engine, survey)               # labels follow the dense cloud
    mesh = survey.dataset.mesh
    return {"n_vertices": len(mesh.vertices) if mesh is not None else 0,
            "n_faces": len(mesh.faces) if mesh is not None else 0}


# -- ml ---------------------------------------------------------------------------
def _need_normals(project, surveys: list[str], code: str) -> None:
    """A model on the cloud's normals (experimental) needs each Survey's
    active Reconstruction to have a dense cloud with normals."""
    lacking = [s for s in surveys if not project.surveys[s].has_normals()]
    if lacking:
        raise Problem(code, "The model uses the cloud's normals (experimental), and "
                      f"{', '.join(lacking)} has no dense cloud: run MVS there first, or "
                      "leave it out.", surveys=lacking)


def _surveys_option(project, options) -> list[str]:
    """The Surveys an ml Job covers (option ``surveys``; default: all)."""
    given = options.get("surveys")
    if given is None:
        return list(project.surveys)
    if not isinstance(given, list) or not given or not all(isinstance(s, str) for s in given):
        raise InvalidParams("option 'surveys' is a list of Survey ids", option="surveys")
    unknown = [s for s in given if s not in project.surveys]
    if unknown:
        raise Problem("survey.not_found", "No Survey has that id.", survey=unknown[0])
    return [s for s in project.surveys if s in given]


@kind("train", "cloudlabeller.ml.train_cli", "ml",
      "Train the U-Net on the labelled photos of every Survey")
def _prepare_train(engine, project, options):
    if not project.schema.classes:
        raise Problem("job.train.no_classes", "Define some label classes first.")
    surveys = _surveys_option(project, options)
    if not any(project.surveys[s].peek_images() for s in surveys):
        raise Problem("job.train.no_images", "Add images to the Project first.")
    name = _opt(options, "name", str, "unet")
    resume = _opt(options, "resume", bool, False)
    if resume:
        from cloudlabeller.ml.model_store import list_models

        from cloudlabeller.ml.model_spec import is_tiled

        manifest = next((m for m in list_models(project.ml_dir) if m.get("name") == name),
                        None)
        if manifest is not None and (manifest.get("spec") or {}).get("normals"):
            _need_normals(project, surveys, "job.train.no_normals")
        if manifest is not None and not is_tiled(manifest):
            raise Problem("job.train.legacy_model",
                          "That model was trained on whole photos, before Tiles: train a "
                          "fresh one.", model=name)
        n_classes = len(project.schema.classes)
        if manifest is not None and manifest.get("n_classes") != n_classes:
            raise Problem("job.train.class_mismatch",
                          "The saved model has a different number of classes: "
                          "it cannot be continued. Train a fresh one instead.",
                          model=name, model_classes=manifest.get("n_classes"),
                          project_classes=n_classes)
    else:
        spec = project.model_spec()
        error = spec.validate()
        if error:
            raise Problem("job.train.bad_spec", error)
        if spec.normals:
            _need_normals(project, surveys, "job.train.no_normals")
    # TensorFlow runs in the child; it reads the Project from disk.
    engine.save_project()
    stop_file = project.ml_dir / "train.stop"
    stop_file.unlink(missing_ok=True)
    args = [str(project.root), "--name", name,
            "--steps", str(max(1, _opt(options, "steps", int, 50000))),
            "--patience", str(max(0, _opt(options, "patience", int, 10))),
            "--batch-size", str(_opt(options, "batch_size", int, 2)),
            "--val-fraction", str(float(_opt(options, "val_fraction", (int, float), 0.2))),
            "--augment-rolls", str(_opt(options, "augment_rolls", int, 5)),
            "--stop-file", str(stop_file)]
    if _opt(options, "include_auto", bool, False):
        args.append("--include-auto")
        if _opt(options, "weight_by_uncertainty", bool, False):
            args.append("--weight-by-uncertainty")
    if resume:
        args.append("--resume")
    if surveys != list(project.surveys):
        args += ["--surveys", ",".join(surveys)]
    return Plan(args, project.ml_dir / "train.log", gpu=True, stop_file=stop_file)


@applies("train")
def _apply_train(engine, project, results, plan):
    engine.changed("settings")                     # a new or updated model
    return parse_result(results)


@kind("predict", "cloudlabeller.ml.predict_cli", "ml",
      "Predict masks for every photo not labelled by the user or from the cloud")
def _prepare_predict(engine, project, options):
    from cloudlabeller.ml.model_store import list_models

    model = _opt(options, "model", str, None)
    if not model:
        raise InvalidParams("option 'model' is required", option="model")
    manifest = next((m for m in list_models(project.ml_dir) if m.get("name") == model), None)
    if manifest is None:
        raise Problem("job.predict.no_model", "No saved model has that name.", model=model)
    from cloudlabeller.ml.model_spec import is_tiled

    if not is_tiled(manifest):
        raise Problem("job.predict.legacy_model",
                      "That model was trained on whole photos, before Tiles: retrain it.",
                      model=model)
    surveys = _surveys_option(project, options)
    if (manifest.get("spec") or {}).get("normals"):
        _need_normals(project, surveys, "job.predict.no_normals")
    n_classes = len(project.schema.classes)
    if manifest.get("n_classes") != n_classes:
        raise Problem("job.predict.class_mismatch",
                      "The model was trained with a different number of classes.",
                      model=model, model_classes=manifest.get("n_classes"),
                      project_classes=n_classes)
    from cloudlabeller.ml.predict_cli import PROTECTED

    if not any(s.labels.status_of(r.name) not in PROTECTED
               for s in (project.survey(sid) for sid in surveys)
               for r in s.dataset.images):
        raise Problem("job.predict.nothing",
                      "Every photo is labelled already, by you or from the cloud.")
    project.save_manifest()                        # the child reads statuses from disk
    args = [str(project.root), "--model", model]
    if surveys != list(project.surveys):
        args += ["--surveys", ",".join(surveys)]
    return Plan(args, project.ml_dir / "predict.log", gpu=True)


@applies("predict")
def _apply_predict(engine, project, results, plan):
    predicted = parse_result(results).get("predicted", [])
    by_survey: dict[str, list[str]] = {}
    for item in predicted:
        if isinstance(item, dict) and item.get("survey") in project.surveys:
            by_survey.setdefault(item["survey"], []).append(item.get("name"))
            unsure = project.surveys[item["survey"]].settings.setdefault("uncertainty", {})
            if item.get("uncertainty") is not None:
                unsure[item["name"]] = float(item["uncertainty"])
    for sid, names in by_survey.items():
        survey = project.survey(sid)
        for name in names:
            survey.labels.mark_ml_labeled(name)
    project.save_manifest()
    for sid, names in by_survey.items():
        engine.changed("status", survey=sid, images=names)
    return {"predicted": predicted}


# -- downloads --------------------------------------------------------------------
COLMAP_VARIANTS = {"cuda": "cuda", "cpu": "nocuda"}    # option -> release asset


@kind("geoid", "cloudlabeller.photogrammetry.geoid_cli", "download",
      "Download the EGM96 geoid grid for sea-level heights (~3 MB, once)")
def _prepare_geoid(engine, project, options):
    epsg = _opt(options, "epsg", int, None)
    if epsg is None:
        raise InvalidParams("option 'epsg' is required", option="epsg")
    return Plan(["--epsg", str(epsg)], _logs(project) / "geoid.log")


@applies("geoid")
def _apply_geoid(engine, project, results, plan):
    return parse_result(results)


@kind("colmap", "cloudlabeller.photogrammetry.colmap_cli", "download",
      "Download the official COLMAP release (Windows)")
def _prepare_colmap(engine, project, options):
    import sys

    variant = _choice(options, "variant", tuple(COLMAP_VARIANTS), "cuda")
    if sys.platform != "win32":
        raise Problem("job.colmap.unsupported",
                      "The COLMAP download is the Windows release; this Environment "
                      "uses pycolmap instead.")
    stop_file = _logs(project) / "colmap_download.stop"
    stop_file.unlink(missing_ok=True)
    return Plan(["--variant", COLMAP_VARIANTS[variant], "--stop-file", str(stop_file)],
                _logs(project) / "colmap_download.log", stop_file=stop_file)


@applies("colmap")
def _apply_colmap(engine, project, results, plan):
    out = parse_result(results)
    variant = out.get("variant")
    return {"path": out.get("path"),
            "variant": next((k for k, v in COLMAP_VARIANTS.items() if v == variant),
                            variant)}


# -- facts -------------------------------------------------------------------------
def _gpu_available() -> bool:
    """Whether GPU SfM / MVS can run here: a CUDA colmap.exe or CUDA pycolmap."""
    return _colmap_binary() is not None or _pycolmap_has_cuda()


def facts(engine: "Engine", project: "Project", kind_name: str, survey=None):
    """What the window shows before planning a Job of ``kind_name`` (sfm, mvs
    and mesh are about ``survey``; train counts every Survey).

    Returns ``(work, finish)`` for a Deferred: ``work`` reads files and
    snapshots only, ``finish`` (loop thread) may cache.
    """
    if kind_name == "sfm":
        from cloudlabeller.core.images import list_image_files
        from cloudlabeller.photogrammetry.options import CAMERA_MODELS, MATCHERS

        images_dir = survey.images_dir

        def work(progress):
            from cloudlabeller.photogrammetry.georef import count_gps_images

            return {"gps_images": count_gps_images(list_image_files(images_dir)),
                    "camera_models": list(CAMERA_MODELS), "matchers": list(MATCHERS),
                    "gpu": _gpu_available()}
        return work, None
    if kind_name == "mvs":
        def work(progress):
            return {"colmap": _gpu_available()}
        return work, None
    if kind_name == "mesh":
        from cloudlabeller.photogrammetry.meshing import delaunay_available, total_ram_gb

        dense = survey.dataset.dense_cloud
        cached = engine.cache.get("poisson_depth")
        workspace = survey.mvs_dir / "dense" if survey.mvs_dir else None

        def work(progress):
            depth = None
            if dense is not None:
                if cached is not None and cached[0] is dense:
                    depth = cached[1]
                else:
                    from cloudlabeller.photogrammetry.meshing import suggest_poisson_depth

                    depth = suggest_poisson_depth(dense)
            return {"delaunay": _colmap_binary() is not None and workspace is not None
                    and delaunay_available(workspace),
                    "suggested_depth": depth, "ram_gb": total_ram_gb()}

        def finish(out):
            if dense is not None and survey.dataset.dense_cloud is dense:
                engine.cache["poisson_depth"] = (dense, out["suggested_depth"])
            return out
        return work, finish
    if kind_name == "train":
        from cloudlabeller.core.label_schema import UNLABELLED_ID

        user = auto = ml = 0
        for sid in project.surveys:
            sv = project.survey(sid)
            labels = sv.labels
            user += sum(1 for n, m in labels.image_masks.items()
                        if labels.status_of(n) == "user" and (m != UNLABELLED_ID).any())
            auto += sum(1 for r in sv.dataset.solved_images()
                        if labels.status_of(r.name) == "auto")
            ml += sum(1 for r in sv.dataset.images if labels.status_of(r.name) == "ml")
        out = {"user_images": user, "auto_images": auto, "ml_images": ml}
        return (lambda progress: out), None
    raise InvalidParams("jobs.facts covers sfm, mvs, mesh and train", kind=kind_name,
                   kinds=["sfm", "mvs", "mesh", "train"])


# -- plans ----------------------------------------------------------------------
@dataclass
class Outstanding:
    """A plan the core has (or will have) started and not yet finished."""

    token: int
    kind: JobKind
    plan: Plan
    project: "Project"

    def describe(self) -> dict:
        root = Path(self.project.root)
        return {"plan": self.token, "kind": self.kind.name, "group": self.kind.group,
                "gpu": self.plan.gpu, "module": self.kind.module, "args": self.plan.args,
                "log": str(self.plan.log_path),
                "stop_file": str(self.plan.stop_file) if self.plan.stop_file else None,
                "project": str(root), "stage": self.plan.stage}


class JobPlans:
    """The Engine's half of Jobs (ADR-0004): plan them, apply their results.

    The Tauri core starts the process the plan describes, holds the GPU
    Lease, streams the log and cancels; then it reports back with
    :meth:`finish`. Loop thread only.
    """

    def __init__(self, engine: "Engine") -> None:
        self._engine = engine
        self._tokens = itertools.count(1)
        self.outstanding: dict[int, Outstanding] = {}

    def plan(self, kind_name: str, options: dict) -> dict:
        engine = self._engine
        project = engine.require_project()
        kind_ = KINDS.get(kind_name)
        if kind_ is None:
            raise InvalidParams(f"no Job kind {kind_name!r}", kind=kind_name,
                                kinds=sorted(KINDS))
        busy = next((o for o in self.outstanding.values() if o.kind.group == kind_.group),
                    None)
        if busy is not None:
            raise Problem("job.busy", "Another Job of this group hasn't finished.",
                          group=kind_.group, plan=busy.token, kind=busy.kind.name)
        plan = kind_.prepare(engine, project, options)
        plan.log_path.parent.mkdir(parents=True, exist_ok=True)
        entry = Outstanding(next(self._tokens), kind_, plan, project)
        self.outstanding[entry.token] = entry
        return entry.describe()

    def finish(self, token: int, exit_code: int | None, results: dict,
               cancelled: bool) -> dict:
        """Apply a finished Job. ``exit_code`` None: it never ran or was lost."""
        engine = self._engine
        entry = self.outstanding.pop(token, None)
        if entry is None:
            raise Problem("job.unknown_plan", "No plan has that number.", plan=token)
        if entry.plan.stop_file is not None:
            entry.plan.stop_file.unlink(missing_ok=True)
        graceful = cancelled and entry.plan.stop_file is not None
        out: dict = {"plan": token, "kind": entry.kind.name}
        if exit_code != 0 or (cancelled and not graceful):
            if cancelled:
                code, english = "job.cancelled", "Cancelled."
            elif exit_code is None:
                code, english = "job.lost", "The Job didn't run to its end."
            else:
                code, english = "job.failed", f"The Job failed (exit {exit_code})."
            out.update(ok=False, error={"code": code, "english": english,
                                        "params": {"exit": exit_code}})
            return out
        if engine.project is not entry.project:
            out.update(ok=False, error={"code": "project.closed",
                                        "english": "The Project was closed.", "params": {}})
            return out
        try:
            result = entry.kind.apply(engine, entry.project, results, entry.plan)
        except Problem as problem:
            out.update(ok=False, error=problem.to_data())
            return out
        out.update(ok=True, result=result, stopped=cancelled)
        return out

    def forget_project(self) -> None:
        self.outstanding.clear()

    def list(self) -> list[dict]:
        return [o.describe() for o in self.outstanding.values()]
