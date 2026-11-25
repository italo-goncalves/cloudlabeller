"""Contract tests for the Engine (ADR-0002, docs/engine.md).

Two harnesses:

- :class:`Pipe` runs the real ``python -m cloudlabeller.engine`` process and
  talks to it over stdin/stdout, exactly as the window will. Every line the
  Engine writes must be a JSON-RPC message.
- :class:`InProcess` runs :class:`Engine` on a thread, for what the pipe
  can't reach: cancellation timing and Job processes driven by a stand-in
  CLI (``tests/engine_fake_cli.py``).
"""

from __future__ import annotations

import json
import queue
import re
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

import cloudlabeller.engine.methods  # noqa: F401  (registers the methods)
from cloudlabeller.core.dataset import Camera, ImageRecord, PointCloud
from cloudlabeller.core.project import Project
from cloudlabeller.engine import arrays, protocol
from cloudlabeller.engine.server import METHODS, Deferred, Engine

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs" / "engine.md"

# Children run the working tree's package whatever the interpreter's own
# path setup (an embedded Python ignores PYTHONPATH).
BOOT = ("import runpy, sys; sys.path[:0] = [{repo!r}, {tests!r}]; mod = sys.argv[1]; "
        "sys.argv = [mod] + sys.argv[2:]; "
        "runpy.run_module(mod, run_name='__main__', alter_sys=True)").format(
            repo=str(REPO), tests=str(REPO / "tests"))
CHILD = [sys.executable, "-u", "-c", BOOT]


class RpcError(Exception):
    def __init__(self, error: dict) -> None:
        super().__init__(error.get("message"))
        self.code = error["code"]
        self.data = error.get("data") or {}


class _Client:
    """Shared request/response bookkeeping over a queue of incoming messages."""

    def __init__(self) -> None:
        self.incoming: queue.Queue = queue.Queue()
        self.notifications: list[dict] = []
        self._next = 0

    def send_raw(self, text: str) -> None:
        raise NotImplementedError

    def request(self, method: str, params: dict | None = None, *, req_id=None,
                timeout: float = 60) -> dict:
        if req_id is None:
            self._next += 1
            req_id = self._next
        msg = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params is not None:
            msg["params"] = params
        self.send_raw(json.dumps(msg))
        return self.wait_response(req_id, timeout)

    def wait_response(self, req_id, timeout: float = 60) -> dict:
        while True:
            msg = self.incoming.get(timeout=timeout)
            if "method" in msg:
                self.notifications.append(msg)
                continue
            if msg.get("id") == req_id:
                return msg
            raise AssertionError(f"unexpected response {msg}")

    def call(self, method: str, params: dict | None = None, **kw):
        msg = self.request(method, params, **kw)
        if "error" in msg:
            raise RpcError(msg["error"])
        return msg["result"]

    def wait_notification(self, method: str, timeout: float = 60, **match) -> dict:
        for n in self.notifications:
            if n["method"] == method and all(n["params"].get(k) == v for k, v in match.items()):
                self.notifications.remove(n)
                return n
        while True:
            msg = self.incoming.get(timeout=timeout)
            assert "method" in msg, f"unexpected response {msg}"
            if msg["method"] == method and all(msg["params"].get(k) == v
                                               for k, v in match.items()):
                return msg
            self.notifications.append(msg)

    def drained(self, method: str) -> list[dict]:
        """Notifications named ``method`` received so far (and removed)."""
        while True:
            try:
                self.notifications.append(self.incoming.get_nowait())
            except queue.Empty:
                break
        hits = [n for n in self.notifications if n.get("method") == method]
        self.notifications = [n for n in self.notifications if n.get("method") != method]
        return hits


class Pipe(_Client):
    def __init__(self, stderr_path: Path) -> None:
        super().__init__()
        self.stderr = open(stderr_path, "wb")
        self.proc = subprocess.Popen([*CHILD, "cloudlabeller.engine"],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self.stderr)
        self.bad_lines: list[bytes] = []
        threading.Thread(target=self._read, daemon=True).start()
        ready = self.wait_notification("engine.ready", timeout=120)
        self.hello = ready["params"]

    def _read(self) -> None:
        for line in iter(self.proc.stdout.readline, b""):
            try:
                msg = json.loads(line)
                assert msg.get("jsonrpc") == "2.0"
            except (json.JSONDecodeError, AssertionError):
                self.bad_lines.append(line)
                continue
            self.incoming.put(msg)

    def send_raw(self, text: str) -> None:
        self.proc.stdin.write(text.encode("utf-8") + b"\n")
        self.proc.stdin.flush()

    def close(self) -> int:
        try:
            self.call("engine.shutdown", timeout=30)
        except Exception:
            pass
        try:
            return self.proc.wait(timeout=30)
        finally:
            self.stderr.close()


class InProcess(_Client):
    def __init__(self, autosave_s: float = 0) -> None:
        super().__init__()
        self._lines: queue.Queue = queue.Queue()
        self.engine = Engine(lambda data: self.incoming.put(json.loads(data)),
                             autosave_s=autosave_s)
        self.engine.jobs.command = list(CHILD)
        self._thread = threading.Thread(
            target=self.engine.serve, args=(iter(self._lines.get, None),), daemon=True)
        self._thread.start()
        self.wait_notification("engine.ready")

    def send_raw(self, text: str) -> None:
        self._lines.put(text.encode("utf-8"))

    def close(self) -> None:
        self._lines.put(None)
        self._thread.join(timeout=30)


@pytest.fixture(scope="module")
def pipe(tmp_path_factory):
    client = Pipe(tmp_path_factory.mktemp("engine") / "stderr.log")
    yield client
    assert client.close() == 0
    assert client.bad_lines == [], "the Engine wrote non-protocol lines to stdout"


@pytest.fixture
def inproc():
    client = InProcess()
    yield client
    client.close()


# -- a small reconstructed Project ------------------------------------------------
W, H = 64, 48


def _camera(tx: float = 0.0) -> Camera:
    K = np.array([[80.0, 0, W / 2], [0, 80.0, H / 2], [0, 0, 1]])
    return Camera(K=K, R=np.eye(3), t=np.array([tx, 0.0, 0.0]), width=W, height=H)


def make_project(root: Path, n_classes: int = 2) -> Path:
    """Two photos (a.png, b.png) of a box of points, cameras solved."""
    from PIL import Image

    from cloudlabeller.photogrammetry.pipeline import ReconstructResult, save_result

    proj = Project.create(root)
    for _ in range(n_classes):
        proj.add_label()
    proj.save()
    for name in ("a.png", "b.png"):
        Image.new("RGB", (W, H), (120, 120, 120)).save(proj.images_dir / name)
    rng = np.random.default_rng(0)
    xyz = np.column_stack([rng.uniform(-1.5, 1.5, 600), rng.uniform(-1, 1, 600),
                           rng.uniform(4, 6, 600)]).astype(np.float32)
    cloud = PointCloud(xyz=xyz, rgb=np.full((600, 3), 200, np.uint8))
    images = [ImageRecord(0, proj.images_dir / "a.png", _camera(0.0)),
              ImageRecord(1, proj.images_dir / "b.png", _camera(0.2))]
    save_result(ReconstructResult(cloud, images), proj.reconstruction_dir)
    return root


def put_array(root: Path, name: str, array: np.ndarray) -> dict:
    """What the window does: write an inbound Array file."""
    return arrays.write(root, f"in/{name}", array)


def get_array(root: Path, ref: dict) -> np.ndarray:
    return arrays.read(root, ref, consume=False)


# -- the documented contract ------------------------------------------------------
def documented_methods() -> set[str]:
    return set(re.findall(r"^### `([a-z_.]+)`", DOC.read_text(encoding="utf-8"), re.M))


def test_every_method_is_documented_and_every_documented_method_exists(pipe):
    assert set(pipe.hello["methods"]) == documented_methods()
    assert set(METHODS) == documented_methods()


def test_hello_reports_protocol_and_version(pipe):
    from cloudlabeller import __version__
    from cloudlabeller.engine import PROTOCOL_VERSION

    hello = pipe.call("engine.hello")
    assert hello["protocol"] == PROTOCOL_VERSION
    assert hello["version"] == __version__


# -- protocol faults ---------------------------------------------------------------
def test_parse_error(pipe):
    pipe.send_raw("{not json")
    msg = pipe.incoming.get(timeout=30)
    assert msg["id"] is None and msg["error"]["code"] == protocol.PARSE_ERROR


def test_invalid_request(pipe):
    pipe.send_raw(json.dumps({"jsonrpc": "1.0", "id": 99, "method": "engine.hello"}))
    msg = pipe.wait_response(99)
    assert msg["error"]["code"] == protocol.INVALID_REQUEST


def test_unknown_method(pipe):
    with pytest.raises(RpcError) as err:
        pipe.call("no.such.method")
    assert err.value.code == protocol.METHOD_NOT_FOUND


def test_invalid_params(pipe):
    with pytest.raises(RpcError) as err:
        pipe.call("project.open", {"root": 3})
    assert err.value.code == protocol.INVALID_PARAMS
    assert err.value.data["code"] == "request.invalid_params"
    assert err.value.data["params"] == {"param": "root"}


def test_notifications_get_no_response(pipe):
    pipe.send_raw(json.dumps({"jsonrpc": "2.0", "method": "engine.hello"}))
    assert pipe.call("engine.hello")["protocol"] >= 1      # next answer is ours


def test_methods_needing_a_project_say_so(pipe):
    try:
        pipe.call("project.close")
    except RpcError as err:
        assert err.data["code"] == "project.none_open"
    with pytest.raises(RpcError) as err:
        pipe.call("project.state")
    assert err.value.code == protocol.PROBLEM
    assert err.value.data == {"code": "project.none_open",
                              "english": "No Project is open.", "params": {}}


def test_open_missing_project(pipe, tmp_path):
    with pytest.raises(RpcError) as err:
        pipe.call("project.open", {"root": str(tmp_path / "nowhere")})
    assert err.value.data["code"] == "project.not_found"


# -- project and schema -------------------------------------------------------------
def test_create_then_schema_edits(pipe, tmp_path):
    root = tmp_path / "new.clproj"
    state = pipe.call("project.create", {"root": str(root),
                                         "classes": [{"name": "rock"}, {"name": "soil"}]})
    assert [c["name"] for c in state["schema"]] == ["rock", "soil"]
    assert state["images"] == [] and state["reconstruction"] == "none"

    schema = pipe.call("schema.add", {"name": "water", "color": "#0000ff"})["schema"]
    assert [c["id"] for c in schema] == [0, 1, 2]
    pipe.call("schema.rename", {"id": 0, "name": "granite"})
    pipe.call("schema.set_color", {"id": 1, "color": "#123456"})
    schema = pipe.call("schema.delete", {"id": 0})["schema"]
    assert [(c["id"], c["name"], c["color"]) for c in schema] == [
        (0, "soil", "#123456"), (1, "water", "#0000ff")]
    assert any("schema" in n["params"]["parts"] for n in pipe.drained("project.changed"))

    with pytest.raises(RpcError) as err:
        pipe.call("schema.rename", {"id": 7, "name": "x"})
    assert err.value.data == {"code": "schema.no_class", "english": "No class has that id.",
                              "params": {"id": 7}}
    on_disk = Project.open(root)                    # the manifest was written at once
    assert [c.name for c in on_disk.schema.classes] == ["soil", "water"]

    with pytest.raises(RpcError) as err:
        pipe.call("project.create", {"root": str(root)})
    assert err.value.data["code"] == "project.exists"


def test_open_reports_state(pipe, tmp_path):
    root = make_project(tmp_path / "p.clproj")
    state = pipe.call("project.open", {"root": str(root)})
    assert [(i["name"], i["status"], i["solved"]) for i in state["images"]] == [
        ("a.png", "none", True), ("b.png", "none", True)]
    assert state["sparse"] == 600 and state["dense"] is None and state["mesh"] is None
    assert state["reconstruction"] == "current"
    assert state["source_resolution"] == [W, H]
    rows = dict(map(tuple, pipe.call("project.summary")["rows"]))
    assert rows["Sparse cloud"] == "600 points"


# -- arrays ------------------------------------------------------------------------
def test_cloud_points_arrive_as_array_files(pipe, tmp_path):
    root = make_project(tmp_path / "p.clproj")
    pipe.call("project.open", {"root": str(root)})
    out = pipe.call("cloud.points", {"cloud": "sparse"})
    assert out["xyz"]["dtype"] == "float32" and out["xyz"]["shape"] == [600, 3]
    assert out["xyz"]["path"].startswith(".arrays/")
    raw = np.fromfile(root / out["xyz"]["path"], dtype="<f4").reshape(600, 3)
    assert np.allclose(raw, Project.open(root).dataset.cloud.xyz)
    assert get_array(root, out["rgb"]).dtype == np.uint8

    with pytest.raises(RpcError) as err:
        pipe.call("cloud.points", {"cloud": "dense"})
    assert err.value.data["code"] == "cloud.none"


def test_inbound_arrays_are_checked(pipe, tmp_path):
    root = make_project(tmp_path / "p.clproj")
    pipe.call("project.open", {"root": str(root)})
    outside = {"path": "labels/cloud_labels.npy", "dtype": "int32", "shape": [1]}
    with pytest.raises(RpcError) as err:
        pipe.call("cloud.paint", {"cloud": "sparse", "class_id": 0, "indices": outside})
    assert err.value.code == protocol.INVALID_PARAMS

    ref = put_array(root, "idx", np.arange(5, dtype=np.int32))
    ref["shape"] = [6]                                     # lies about its size
    with pytest.raises(RpcError) as err:
        pipe.call("cloud.paint", {"cloud": "sparse", "class_id": 0, "indices": ref})
    assert err.value.code == protocol.INVALID_PARAMS
    assert "size" in err.value.data["english"]


# -- labelling -----------------------------------------------------------------------
def test_save_mask_is_undoable_and_persisted(pipe, tmp_path):
    root = make_project(tmp_path / "p.clproj")
    pipe.call("project.open", {"root": str(root)})
    assert pipe.call("image.mask", {"name": "a.png"}) == {"source": None, "mask": None}

    mask = np.full((H, W), -1, np.int32)
    mask[:, : W // 2] = 0
    mask[:, W // 2:] = 1
    ref = put_array(root, "a", mask)
    out = pipe.call("image.save_mask", {"name": "a.png", "mask": ref})
    assert out == {"changed_pixels": H * W, "status": "user"}
    assert not (root / ref["path"]).exists()               # inbound files are consumed
    assert (root / "labels" / "images" / "a.png.npy").exists()
    assert Project.open(root).labels.status_of("a.png") == "user"
    changed = pipe.drained("project.changed")
    assert any(n["params"].get("images") == ["a.png"] for n in changed)

    shown = pipe.call("image.mask", {"name": "a.png"})
    assert shown["source"] == "user"
    assert np.array_equal(get_array(root, shown["mask"]), mask)

    undo = pipe.call("labels.undo")
    assert undo == {"changed": {"modality": "image", "image": "a.png"}}
    shown = pipe.call("image.mask", {"name": "a.png"})
    assert (get_array(root, shown["mask"]) == -1).all()
    assert pipe.call("labels.redo")["changed"]["image"] == "a.png"
    assert pipe.call("project.state")["can_redo"] is False


def test_save_mask_refuses_wrong_shape_and_ids(pipe, tmp_path):
    root = make_project(tmp_path / "p.clproj")
    pipe.call("project.open", {"root": str(root)})
    with pytest.raises(RpcError) as err:
        pipe.call("image.save_mask", {"name": "a.png",
                                      "mask": put_array(root, "a", np.zeros((3, 3), np.int32))})
    assert err.value.data["code"] == "image.mask_shape"
    with pytest.raises(RpcError) as err:
        pipe.call("image.save_mask", {"name": "a.png",
                                      "mask": put_array(root, "a", np.full((H, W), 5, np.int32))})
    assert err.value.data["code"] == "image.mask_values"
    with pytest.raises(RpcError) as err:
        pipe.call("image.mask", {"name": "zzz.png"})
    assert err.value.data["code"] == "image.not_found"


def test_lasso_paint_undo_and_save_on_close(pipe, tmp_path):
    root = make_project(tmp_path / "p.clproj")
    pipe.call("project.open", {"root": str(root)})
    idx = put_array(root, "idx", np.arange(0, 600, 3, dtype=np.int32))
    assert pipe.call("cloud.paint", {"cloud": "sparse", "class_id": 1, "indices": idx}) == {
        "painted": 200}
    labels = get_array(root, pipe.call("cloud.labels", {"cloud": "sparse"})["labels"])
    assert (labels[::3] == 1).all() and (np.delete(labels, np.s_[::3]) == -1).all()
    assert pipe.call("project.state")["dirty"] is True

    pipe.call("labels.undo")
    pipe.call("labels.redo")
    pipe.call("project.close")
    reopened = Project.open(root)
    assert (reopened.labels.cloud_labels[::3] == 1).all()
    assert not (root / ".arrays").exists()                 # emptied on close


# -- transfers -----------------------------------------------------------------------
def test_images_to_cloud_then_cloud_to_images(pipe, tmp_path):
    root = make_project(tmp_path / "p.clproj")
    pipe.call("project.open", {"root": str(root)})
    with pytest.raises(RpcError) as err:
        pipe.call("transfer.images_to_cloud")
    assert err.value.data["code"] == "transfer.no_labelled_images"

    mask = np.zeros((H, W), np.int32)
    mask[:, W // 2:] = 1
    pipe.call("image.save_mask", {"name": "a.png", "mask": put_array(root, "a", mask)})

    msg = pipe.request("transfer.images_to_cloud", {"include_auto": True}, req_id="i2c")
    out = msg["result"]
    assert out["images"] == 1 and out["sparse"] > 0
    progress = pipe.drained("request.progress")
    assert progress and all(p["params"]["id"] == "i2c" for p in progress)
    assert progress[-1]["params"]["fraction"] == 1.0
    cloud_labels = get_array(root, pipe.call("cloud.labels")["labels"])
    assert set(np.unique(cloud_labels)) <= {-1, 0, 1} and (cloud_labels == 1).any()

    out = pipe.call("transfer.cloud_to_images")
    assert out == {"cloud": "sparse", "auto_labelled": ["b.png"]}
    statuses = {i["name"]: i["status"] for i in pipe.call("project.state")["images"]}
    assert statuses == {"a.png": "user", "b.png": "auto"}
    shown = pipe.call("image.mask", {"name": "b.png"})
    assert shown["source"] == "auto" and shown["mask"]["shape"] == [H, W]

    out = pipe.call("transfer.propagate", {"name": "a.png"})
    assert out["cloud"] == "sparse"
    with pytest.raises(RpcError) as err:
        pipe.call("transfer.propagate", {"name": "b.png"})
    assert err.value.data["code"] == "transfer.image_unlabelled"


# -- model and jobs over the pipe ------------------------------------------------------
def test_model_spec_roundtrip(pipe, tmp_path):
    root = make_project(tmp_path / "p.clproj")
    pipe.call("project.open", {"root": str(root)})
    spec = pipe.call("model.spec")["spec"]
    spec["channels"] = spec["channels"] * 2
    assert pipe.call("model.set_spec", {"spec": spec})["spec"]["channels"] == spec["channels"]
    assert Project.open(root).model_spec().channels == spec["channels"]
    assert pipe.call("models.list") == {"models": []}


def test_jobs_refuse_what_cannot_run(pipe, tmp_path):
    from cloudlabeller.engine.jobs import KINDS

    kinds = {k["kind"] for k in pipe.call("jobs.kinds")["kinds"]}
    assert kinds == set(KINDS) >= {"ingest", "sfm", "georef", "reproject", "mvs", "mesh",
                                   "train", "predict"}
    root = make_project(tmp_path / "p.clproj")
    pipe.call("project.open", {"root": str(root)})
    with pytest.raises(RpcError) as err:
        pipe.call("jobs.start", {"kind": "predict", "options": {"model": "nope"}})
    assert err.value.data["code"] == "job.predict.no_model"
    with pytest.raises(RpcError) as err:
        pipe.call("jobs.start", {"kind": "reproject", "options": {"epsg": 32723}})
    assert err.value.data["code"] == "job.reproject.not_georeferenced"
    with pytest.raises(RpcError) as err:
        pipe.call("jobs.start", {"kind": "mesh"})
    assert err.value.data["code"] == "job.mesh.no_dense"
    with pytest.raises(RpcError) as err:
        pipe.call("jobs.start", {"kind": "warp"})
    assert err.value.code == protocol.INVALID_PARAMS
    with pytest.raises(RpcError) as err:
        pipe.call("jobs.start", {"kind": "sfm", "options": {"matcher": "psychic"}})
    assert err.value.code == protocol.INVALID_PARAMS
    assert pipe.call("jobs.list") == {"jobs": []}


def test_job_arguments_match_the_clis(tmp_path):
    """Each kind's command line parses with its CLI's own argument parser."""
    import importlib

    from cloudlabeller.engine.jobs import KINDS

    root = make_project(tmp_path / "p.clproj")
    engine = Engine(lambda data: None)
    engine.project = Project.open(root)
    engine.project.settings["georeferenced"] = {"frame": "ENU"}
    options = {"ingest": {"folder": str(tmp_path)}, "sfm": {"gpu": True},
               "georef": {}, "reproject": {"epsg": 32723, "orthometric": True},
               # resume: the tiny test photos are too small for a fresh spec
               "train": {"name": "m1", "epochs": 3, "include_auto": True, "resume": True}}
    parsed = {}
    for name, opts in options.items():
        kind = KINDS[name]
        plan = kind.prepare(engine, engine.project, opts)
        cli = importlib.import_module(kind.module)
        parsed[name] = cli._parse([kind.module, *plan.args])   # exits on a bad flag
    assert parsed["ingest"].ingest_only and parsed["ingest"].on_conflict == "skip"
    assert parsed["sfm"].gpu and parsed["sfm"].matcher == "spatial"
    assert parsed["reproject"].epsg == 32723 and parsed["reproject"].orthometric
    assert parsed["train"].epochs == 3 and parsed["train"].include_auto
    assert parsed["train"].resume and parsed["train"].stop_file.endswith("train.stop")
    assert KINDS["sfm"].prepare(engine, engine.project, {"gpu": True}).gpu is True
    assert KINDS["sfm"].prepare(engine, engine.project, {}).gpu is False
    assert KINDS["train"].prepare(engine, engine.project, {"resume": True}).gpu is True
    with pytest.raises(Exception) as err:                  # a fresh spec is checked
        KINDS["train"].prepare(engine, engine.project, {})
    assert getattr(err.value, "code", None) == "job.train.bad_spec"


def test_native_libraries_load_on_worker_threads(tmp_path):
    """Regression: SciPy's Fortran BLAS, first loaded on a worker thread
    (here by the overlap graph in ``transfer.propagate``) while the reader
    thread sat in a read on stdin, hung the Engine on Windows. The Engine
    now reads the protocol from a private copy of stdin. A fresh Engine, so
    nothing has loaded SciPy's linalg yet."""
    root = make_project(tmp_path / "p.clproj")
    client = Pipe(tmp_path / "stderr.log")
    try:
        client.call("project.open", {"root": str(root)})
        mask = np.zeros((H, W), np.int32)
        client.call("image.save_mask", {"name": "a.png", "mask": put_array(root, "a", mask)})
        out = client.call("transfer.propagate", {"name": "a.png"}, timeout=60)
        assert out["cloud"] == "sparse"
    finally:
        assert client.close() == 0


# -- in-process: autosave, cancellation and Job processes -----------------------------
def test_autosave_runs_while_requests_keep_coming(tmp_path):
    """Regression: autosave fired only after an idle stretch, so an Engine
    kept busy by the window never saved dirty labels."""
    import time

    root = make_project(tmp_path / "p.clproj")
    client = InProcess(autosave_s=0.3)
    try:
        client.call("project.open", {"root": str(root)})
        idx = put_array(root, "idx", np.arange(10, dtype=np.int32))
        client.call("cloud.paint", {"cloud": "sparse", "class_id": 0, "indices": idx})
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline:                 # never idle for 0.3 s
            client.call("engine.hello")
            time.sleep(0.05)
        assert client.call("project.state")["dirty"] is False
        assert (np.load(root / "labels" / "cloud_labels.npy")[:10] == 0).all()
    finally:
        client.close()



def test_cancel_a_pending_request(inproc, monkeypatch):
    import time

    from cloudlabeller.engine.server import Method

    started = threading.Event()

    def slow(engine, params):
        def work(progress):
            started.set()
            for i in range(400):
                progress(i / 400)
                time.sleep(0.01)
            return "finished"
        return Deferred(work)

    monkeypatch.setitem(METHODS, "test.slow", Method("test.slow", slow, False))
    inproc.send_raw(json.dumps({"jsonrpc": "2.0", "id": "s", "method": "test.slow"}))
    assert started.wait(10)
    assert inproc.call("engine.cancel", {"id": "s"}) == {"cancelled": True}
    msg = inproc.wait_response("s")
    assert msg["error"]["data"]["code"] == "request.cancelled"
    assert inproc.call("engine.cancel", {"id": "s"}) == {"cancelled": False}


def _fake_kind(monkeypatch, tmp_path, args, stop_file=None):
    from cloudlabeller.engine import jobs

    def prepare(engine, project, options):
        return jobs.Plan(list(args), tmp_path / "fake.log", gpu=True, stop_file=stop_file)

    def apply(engine, project, results, plan):
        engine.changed("settings")
        return jobs.parse_result(results)

    monkeypatch.setitem(jobs.KINDS, "fake", jobs.JobKind(
        "fake", "engine_fake_cli", "test", prepare, apply, "stand-in"))


def test_job_reports_progress_log_and_result(inproc, monkeypatch, tmp_path):
    root = make_project(tmp_path / "p.clproj")
    inproc.call("project.open", {"root": str(root)})
    _fake_kind(monkeypatch, tmp_path, ["0"])
    started = inproc.call("jobs.start", {"kind": "fake"})
    assert started["kind"] == "fake" and started["gpu"] is True
    done = inproc.wait_notification("job.finished", job=started["job"])["params"]
    assert done["ok"] is True and done["result"] == {"answer": 42}
    assert done["stopped"] is False
    assert inproc.wait_notification("job.started", job=started["job"])
    prog = inproc.wait_notification("job.progress", job=started["job"])["params"]
    assert prog["fraction"] == 0.5 and prog["message"] == "halfway"
    lines = [n["params"]["line"] for n in inproc.drained("job.log")]
    assert "a plain log line" in lines and "warning on stderr" in lines
    log_text = (tmp_path / "fake.log").read_text(encoding="utf-8")
    assert "a plain log line" in log_text and log_text.startswith("$ ")
    assert inproc.call("jobs.list") == {"jobs": []}


def test_failed_job_carries_stderr(inproc, monkeypatch, tmp_path):
    root = make_project(tmp_path / "p.clproj")
    inproc.call("project.open", {"root": str(root)})
    _fake_kind(monkeypatch, tmp_path, ["3"])
    job = inproc.call("jobs.start", {"kind": "fake"})["job"]
    done = inproc.wait_notification("job.finished", job=job)["params"]
    assert done["ok"] is False and done["error"]["code"] == "job.failed"
    assert done["error"]["params"]["exit"] == 3
    assert "warning on stderr" in done["error"]["params"]["stderr"]


def test_one_job_per_group_and_kill_on_cancel(inproc, monkeypatch, tmp_path):
    root = make_project(tmp_path / "p.clproj")
    inproc.call("project.open", {"root": str(root)})
    _fake_kind(monkeypatch, tmp_path, ["0", "30"])
    job = inproc.call("jobs.start", {"kind": "fake"})["job"]
    with pytest.raises(RpcError) as err:
        inproc.call("jobs.start", {"kind": "fake"})
    assert err.value.data["code"] == "job.busy"
    with pytest.raises(RpcError) as err:
        inproc.call("project.close")
    assert err.value.data["code"] == "project.jobs_running"
    assert inproc.call("jobs.cancel", {"job": job}) == {"cancelled": True}
    done = inproc.wait_notification("job.finished", job=job, timeout=20)["params"]
    assert done["ok"] is False and done["error"]["code"] == "job.cancelled"


def test_stop_file_job_stops_gracefully_and_keeps_its_result(inproc, monkeypatch, tmp_path):
    root = make_project(tmp_path / "p.clproj")
    inproc.call("project.open", {"root": str(root)})
    stop = tmp_path / "train.stop"
    _fake_kind(monkeypatch, tmp_path, ["0", "30", str(stop)], stop_file=stop)
    job = inproc.call("jobs.start", {"kind": "fake"})["job"]
    inproc.wait_notification("job.progress", job=job)
    inproc.call("jobs.cancel", {"job": job})
    done = inproc.wait_notification("job.finished", job=job, timeout=20)["params"]
    assert done["ok"] is True and done["stopped"] is True
    assert done["result"] == {"answer": 42}
    assert not stop.exists()
