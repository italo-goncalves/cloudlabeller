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

"""Jobs: the ``*_cli.py`` processes, started and watched by the Engine.

A Job kind knows three things: how to build its command line from the
window's options (refusing with a Problem when it can't run), whether it is
a GPU Job, and what to do to the open Project when it ends well (reload
products, record a new frame, mark predicted images). That last part is
what the Qt window's ``_on_*_done`` handlers did.

The child's protocol is the one ``ProcessJob`` reads: ``PROGRESS <frac>
<msg>`` lines on stdout, ``RESULT…`` lines with the outcome, everything else
is log. Every line also goes to the Job's log file in the Project.

Two Jobs may run at once only if they are in different groups:
``reconstruction`` Jobs rewrite the Project's products, ``ml`` Jobs read
labels and write models and predictions.

The GPU Lease (ADR-0003) is not taken here: it is held by the Tauri core,
which arrives in step 3. ``job.started`` says whether the Job wants the GPU.
"""

from __future__ import annotations

import itertools
import json
import logging
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

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


def parse_result(results: dict, key: str = "RESULT") -> dict:
    """The JSON payload of a ``RESULT <json>`` line, or ``{}``."""
    text = results.get(key, "")
    try:
        value = json.loads(text) if text else {}
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


# -- reconstruction -----------------------------------------------------------
@kind("ingest", "cloudlabeller.photogrammetry.run_cli", "reconstruction",
      "Copy a folder of photos into the image store")
def _prepare_ingest(engine, project, options):
    folder = _opt(options, "folder", str, None)
    if not folder:
        raise InvalidParams("option 'folder' is required", option="folder")
    if not Path(folder).is_dir():
        raise Problem("job.ingest.no_folder", "The folder does not exist.", folder=folder)
    policy = _choice(options, "on_conflict", ("skip", "overwrite"), "skip")
    args = [str(project.images_dir), str(project.reconstruction_dir),
            "--ingest-from", folder, "--ingest-only", "--on-conflict", policy]
    return Plan(args, project.reconstruction_dir / "ingest.log")


@applies("ingest")
def _apply_ingest(engine, project, results, plan):
    fields_ = dict(part.split("=", 1) for part in results.get("RESULT_INGEST", "").split()
                   if "=" in part)
    changed = fields_.get("changed") == "1"
    if changed and project.dataset.cloud is not None:
        project.mark_reconstruction_outdated()
    project.load_products()
    engine.save_project()
    engine.changed("images", "settings")
    return {"copied": int(fields_.get("copied", 0)),
            "overwritten": int(fields_.get("overwritten", 0)),
            "changed": changed,
            "n_images": len(project.dataset.images),
            "reconstruction": project.reconstruction_status()}


@kind("sfm", "cloudlabeller.photogrammetry.run_cli", "reconstruction",
      "Structure from motion: solve the cameras and the sparse cloud")
def _prepare_sfm(engine, project, options):
    from cloudlabeller.photogrammetry.options import CAMERA_MODELS, MATCHERS, SfmOptions

    if not project.dataset.images:
        raise Problem("job.sfm.no_images", "The Project's image store is empty.")
    opts = SfmOptions(
        matcher=_choice(options, "matcher", MATCHERS, "spatial"),
        use_gpu=_opt(options, "gpu", bool, False),
        single_camera=_opt(options, "single_camera", bool, True),
        camera_model=_choice(options, "camera_model", CAMERA_MODELS, "SIMPLE_RADIAL"),
        max_image_size=_opt(options, "max_image_size", int, 3200),
    )
    args = [str(project.images_dir), str(project.reconstruction_dir), *opts.to_cli_args()]
    binary = _colmap_binary()
    if binary:
        args += ["--colmap-binary", binary]
    return Plan(args, project.reconstruction_dir / "sfm.log", gpu=opts.use_gpu)


@applies("sfm")
def _apply_sfm(engine, project, results, plan):
    project.load_products()
    if project.dataset.cloud is None:
        raise Problem("job.sfm.no_cloud", "No point cloud was produced.")
    project.mark_reconstruction_current()
    # A fresh solve is a new local frame: an earlier alignment no longer fits.
    cleared = project.settings.pop("georeferenced", None) is not None
    shutil.rmtree(project.reconstruction_dir / "sparse_prealigned", ignore_errors=True)
    engine.save_project()
    engine.changed("sparse", "images", "labels", "settings")
    return {"n_points": project.dataset.cloud.n_points,
            "n_cameras": len(project.dataset.solved_images()),
            "georeferencing_cleared": cleared}


@kind("georef", "cloudlabeller.photogrammetry.georef_cli", "reconstruction",
      "Align the model to the photos' EXIF GPS (ENU, metres)")
def _prepare_georef(engine, project, options):
    if project.dataset.cloud is None:
        raise Problem("job.no_reconstruction", "Run SfM first.")
    max_error = _opt(options, "max_error", (int, float), 5.0)
    engine.save_project()                         # the child reads the Project from disk
    return Plan([str(project.root), "--max-error", str(float(max_error))],
                project.reconstruction_dir / "georef.log")


@applies("georef")
def _apply_georef(engine, project, results, plan):
    out = parse_result(results)
    project.load_products()
    project.settings["georeferenced"] = {
        "frame": "ENU", "origin_lla": out.get("origin_lla"),
        "origin_convention": out.get("origin_convention"),
        "scale_m_per_unit": out.get("scale_m_per_unit"),
        "n_gps": out.get("n_gps"),
    }
    engine.save_project()
    engine.changed("sparse", "dense", "mesh", "images", "settings")
    return out


@kind("reproject", "cloudlabeller.photogrammetry.reproject_cli", "reconstruction",
      "Move a georeferenced Project into a projected CRS")
def _prepare_reproject(engine, project, options):
    if project.dataset.cloud is None:
        raise Problem("job.no_reconstruction", "Run SfM first.")
    if not project.settings.get("georeferenced"):
        raise Problem("job.reproject.not_georeferenced",
                      "Reprojection needs a georeferenced model.")
    epsg = _opt(options, "epsg", int, None)
    if epsg is None:
        raise InvalidParams("option 'epsg' is required", option="epsg")
    args = [str(project.root), "--epsg", str(epsg)]
    if _opt(options, "orthometric", bool, False):
        args.append("--orthometric")
    engine.save_project()
    return Plan(args, project.reconstruction_dir / "reproject.log")


@applies("reproject")
def _apply_reproject(engine, project, results, plan):
    out = parse_result(results)
    project.load_products()
    project.settings.setdefault("georeferenced", {})["crs"] = {
        "epsg": out.get("epsg"), "name": out.get("name"),
        "orthometric": out.get("orthometric", False),
        "offset": out.get("offset"),
    }
    engine.save_project()
    engine.changed("sparse", "dense", "mesh", "images", "settings")
    return out


@kind("mvs", "cloudlabeller.photogrammetry.mvs_cli", "reconstruction",
      "Dense multi-view stereo: the dense cloud")
def _prepare_mvs(engine, project, options):
    if project.dataset.cloud is None:
        raise Problem("job.no_reconstruction", "Run SfM first.")
    binary = _colmap_binary()
    if binary is None:
        raise Problem("job.mvs.no_colmap", "Dense MVS needs the CUDA COLMAP executable.")
    args = [str(project.reconstruction_dir), str(project.images_dir),
            "--max-image-size", str(_opt(options, "max_image_size", int, 2000)),
            "--quality", _choice(options, "quality", ("standard", "draft"), "standard"),
            "--colmap-binary", binary]
    return Plan(args, project.reconstruction_dir / "mvs.log", gpu=True)


@applies("mvs")
def _apply_mvs(engine, project, results, plan):
    project.load_products()
    engine.changed("dense")
    dense = project.dataset.dense_cloud
    return {"n_points": dense.n_points if dense is not None else 0}


@kind("mesh", "cloudlabeller.photogrammetry.mesh_cli", "reconstruction",
      "Build the triangle mesh from the dense cloud")
def _prepare_mesh(engine, project, options):
    if project.dataset.dense_cloud is None or not project.dense_cloud_path.exists():
        raise Problem("job.mesh.no_dense", "Meshing needs the dense cloud.")
    method = _choice(options, "method", ("poisson", "delaunay"), "poisson")
    args = [str(project.dense_cloud_path), str(project.mesh_path), "--method", method]
    if method == "delaunay":
        args += ["--workspace", str(project.reconstruction_dir / "dense")]
    else:
        depth = _opt(options, "depth", int, None)
        if depth is not None:
            args += ["--depth", str(depth)]
        args += ["--trim", str(float(_opt(options, "trim", (int, float), 10.0))),
                 "--point-weight", str(float(_opt(options, "point_weight", (int, float), 1.0)))]
    binary = _colmap_binary()
    if binary:
        args += ["--colmap-binary", binary]
    return Plan(args, project.reconstruction_dir / "mesh.log")


@applies("mesh")
def _apply_mesh(engine, project, results, plan):
    from cloudlabeller.engine.methods import sync_mesh_labels

    project.load_products()
    engine.changed("mesh")
    sync_mesh_labels(engine)                       # labels follow the dense cloud
    mesh = project.dataset.mesh
    return {"n_vertices": len(mesh.vertices) if mesh is not None else 0,
            "n_faces": len(mesh.faces) if mesh is not None else 0}


# -- ml ---------------------------------------------------------------------------
@kind("train", "cloudlabeller.ml.train_cli", "ml", "Train the U-Net on the labelled photos")
def _prepare_train(engine, project, options):
    if not project.schema.classes:
        raise Problem("job.train.no_classes", "Define some label classes first.")
    source_res = project.source_resolution()
    if source_res is None:
        raise Problem("job.train.no_images", "Add images to the Project first.")
    name = _opt(options, "name", str, "unet")
    resume = _opt(options, "resume", bool, False)
    if not resume:
        error = project.model_spec().validate_size(*source_res)
        if error:
            raise Problem("job.train.bad_spec", error)
    # TensorFlow runs in the child; it reads the Project from disk.
    engine.save_project()
    stop_file = project.ml_dir / "train.stop"
    stop_file.unlink(missing_ok=True)
    args = [str(project.root), "--name", name,
            "--epochs", str(_opt(options, "epochs", int, 100)),
            "--batch-size", str(_opt(options, "batch_size", int, 2)),
            "--val-fraction", str(float(_opt(options, "val_fraction", (int, float), 0.2))),
            "--augment-rolls", str(_opt(options, "augment_rolls", int, 5)),
            "--stop-file", str(stop_file)]
    if _opt(options, "include_auto", bool, False):
        args.append("--include-auto")
    if resume:
        args.append("--resume")
    return Plan(args, project.ml_dir / "train.log", gpu=True, stop_file=stop_file)


@applies("train")
def _apply_train(engine, project, results, plan):
    engine.changed("settings")                     # a new or updated model
    return parse_result(results)


@kind("predict", "cloudlabeller.ml.predict_cli", "ml",
      "Predict masks for every photo the user hasn't labelled")
def _prepare_predict(engine, project, options):
    from cloudlabeller.ml.model_store import list_models

    model = _opt(options, "model", str, None)
    if not model:
        raise InvalidParams("option 'model' is required", option="model")
    manifest = next((m for m in list_models(project.ml_dir) if m.get("name") == model), None)
    if manifest is None:
        raise Problem("job.predict.no_model", "No saved model has that name.", model=model)
    n_classes = len(project.schema.classes)
    if manifest.get("n_classes") != n_classes:
        raise Problem("job.predict.class_mismatch",
                      "The model was trained with a different number of classes.",
                      model=model, model_classes=manifest.get("n_classes"),
                      project_classes=n_classes)
    if not any(project.labels.status_of(r.name) != "user" for r in project.dataset.images):
        raise Problem("job.predict.nothing", "All images are user-labelled.")
    project.save_manifest()                        # the child reads statuses from disk
    return Plan([str(project.root), "--model", model], project.ml_dir / "predict.log",
                gpu=True)


@applies("predict")
def _apply_predict(engine, project, results, plan):
    names = parse_result(results).get("predicted", [])
    for name in names:
        project.labels.mark_ml_labeled(name)
    project.save_manifest()
    engine.changed("status", images=names)
    return {"predicted": names}


# -- the runner -----------------------------------------------------------------
@dataclass
class RunningJob:
    id: int
    kind: JobKind
    plan: Plan
    proc: subprocess.Popen
    project: "Project"
    results: dict = field(default_factory=dict)
    stderr_tail: list[str] = field(default_factory=list)
    cancelled: bool = False

    def describe(self) -> dict:
        return {"job": self.id, "kind": self.kind.name, "gpu": self.plan.gpu,
                "log": str(self.plan.log_path)}


class JobRunner:
    """Starts Job processes and reports on them (loop thread, except the
    per-Job watcher threads, which only post back)."""

    def __init__(self, engine: "Engine") -> None:
        self._engine = engine
        self._ids = itertools.count(1)
        self.running: dict[int, RunningJob] = {}
        # How a Job's module is run: ``<command…> <module> <args…>``. The
        # Environment's own interpreter, unbuffered, so progress arrives live.
        self.command: list[str] = [sys.executable, "-u", "-m"]

    def start(self, kind_name: str, options: dict) -> dict:
        engine = self._engine
        project = engine.require_project()
        kind_ = KINDS.get(kind_name)
        if kind_ is None:
            raise InvalidParams(f"no Job kind {kind_name!r}", kind=kind_name,
                                kinds=sorted(KINDS))
        busy = next((j for j in self.running.values() if j.kind.group == kind_.group), None)
        if busy is not None:
            raise Problem("job.busy", "Another Job of this group is running.",
                          group=kind_.group, job=busy.id, kind=busy.kind.name)
        plan = kind_.prepare(engine, project, options)

        plan.log_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [*self.command, kind_.module, *plan.args]
        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NO_WINDOW
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, creationflags=creationflags)
        job = RunningJob(next(self._ids), kind_, plan, proc, project)
        self.running[job.id] = job
        engine.notify("job.started", job.describe())
        self._watch(job, "$ " + " ".join(cmd))
        return job.describe()

    def _watch(self, job: RunningJob, header: str) -> None:
        engine = self._engine
        log_fh = open(job.plan.log_path, "w", encoding="utf-8")
        lock = threading.Lock()

        def record(line: str) -> None:
            with lock:
                log_fh.write(line + "\n")
                log_fh.flush()
            engine.notify("job.log", {"job": job.id, "line": line})

        record(header)

        def read_stdout() -> None:
            for raw in iter(job.proc.stdout.readline, b""):
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                if line.startswith("PROGRESS "):
                    parts = line.split(" ", 2)
                    try:
                        frac = float(parts[1])
                    except (IndexError, ValueError):
                        frac = None
                    if frac is not None:
                        engine.notify("job.progress", {
                            "job": job.id, "fraction": frac,
                            "message": parts[2] if len(parts) > 2 else ""})
                elif line.startswith("RESULT"):
                    key, _, rest = line.partition(" ")
                    job.results[key] = rest.strip()
                record(line)

        def read_stderr() -> None:
            for raw in iter(job.proc.stderr.readline, b""):
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                if line.strip():
                    job.stderr_tail.append(line)
                    del job.stderr_tail[:-200]
                    record(line)

        def wait() -> None:
            readers = [threading.Thread(target=read_stdout, daemon=True),
                       threading.Thread(target=read_stderr, daemon=True)]
            for t in readers:
                t.start()
            code = job.proc.wait()
            for t in readers:
                t.join()
            with lock:
                log_fh.close()
            engine.post(lambda: self._finished(job, code))

        threading.Thread(target=wait, name=f"job-{job.id}", daemon=True).start()

    def _finished(self, job: RunningJob, code: int) -> None:
        engine = self._engine
        self.running.pop(job.id, None)
        if job.plan.stop_file is not None:
            job.plan.stop_file.unlink(missing_ok=True)
        params: dict = {"job": job.id, "kind": job.kind.name}
        graceful = job.cancelled and job.plan.stop_file is not None
        if code == 0 and (not job.cancelled or graceful):
            if engine.project is not job.project:
                params.update(ok=False, error={"code": "project.closed",
                                               "english": "The Project was closed.",
                                               "params": {}})
            else:
                try:
                    result = job.kind.apply(engine, job.project, job.results, job.plan)
                    params.update(ok=True, result=result, stopped=job.cancelled)
                except Problem as problem:
                    params.update(ok=False, error=problem.to_data())
                except Exception as exc:
                    log.exception("applying Job %s failed", job.id)
                    params.update(ok=False, error={
                        "code": "job.apply_failed", "english": f"{type(exc).__name__}: {exc}",
                        "params": {}})
        elif job.cancelled:
            params.update(ok=False, error={"code": "job.cancelled", "english": "Cancelled.",
                                           "params": {}})
        else:
            params.update(ok=False, error={
                "code": "job.failed", "english": f"The Job failed (exit {code}).",
                "params": {"exit": code, "stderr": job.stderr_tail[-15:]}})
        engine.notify("job.finished", params)

    def cancel(self, job_id: int) -> bool:
        """Stop a Job: training finishes its epoch and still saves (stop
        file); every other Job is killed."""
        job = self.running.get(job_id)
        if job is None:
            return False
        job.cancelled = True
        if job.plan.stop_file is not None:
            job.plan.stop_file.touch(exist_ok=True)
        else:
            _kill(job.proc)
        return True

    def kill_all(self) -> None:
        for job in list(self.running.values()):
            job.cancelled = True
            _kill(job.proc)

    def list(self) -> list[dict]:
        return [j.describe() for j in self.running.values()]


def _kill(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        try:
            proc.kill()
        except OSError:
            pass
