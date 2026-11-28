"""Contract tests for the Engine (ADR-0002, docs/engine.md).

Two harnesses:

- :class:`Pipe` runs the real ``python -m cloudlabeller.engine`` process and
  talks to it over stdin/stdout, exactly as the window will. Every line the
  Engine writes must be a JSON-RPC message.
- :class:`InProcess` runs :class:`Engine` on a thread, for what the pipe
  can't reach: cancellation timing and autosave.
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
from conftest import product_dir
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
BOOT = ("import runpy, sys; sys.path.insert(0, {repo!r}); mod = sys.argv[1]; "
        "sys.argv = [mod] + sys.argv[2:]; "
        "runpy.run_module(mod, run_name='__main__', alter_sys=True)").format(repo=str(REPO))
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
        Image.new("RGB", (W, H), (120, 120, 120)).save(proj.survey("s1").images_dir / name)
    rng = np.random.default_rng(0)
    xyz = np.column_stack([rng.uniform(-1.5, 1.5, 600), rng.uniform(-1, 1, 600),
                           rng.uniform(4, 6, 600)]).astype(np.float32)
    cloud = PointCloud(xyz=xyz, rgb=np.full((600, 3), 200, np.uint8))
    images = [ImageRecord(0, proj.survey("s1").images_dir / "a.png", _camera(0.0)),
              ImageRecord(1, proj.survey("s1").images_dir / "b.png", _camera(0.2))]
    save_result(ReconstructResult(cloud, images), product_dir(proj, "sfm"))
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
    assert state["surveys"][0]["images"] == [] and state["surveys"][0]["reconstruction"] == "none"

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
    assert [(i["name"], i["status"], i["solved"]) for i in state["surveys"][0]["images"]] == [
        ("a.png", "none", True), ("b.png", "none", True)]
    assert state["surveys"][0]["sparse"] == 600 and state["surveys"][0]["dense"] is None and state["surveys"][0]["mesh"] is None
    assert state["surveys"][0]["reconstruction"] == "current"
    assert state["surveys"][0]["source_resolution"] == [W, H]
    rows = dict(map(tuple, pipe.call("project.summary")["rows"]))
    assert rows["  Sparse cloud"] == "600 points"


# -- arrays ------------------------------------------------------------------------
def test_cloud_points_arrive_as_array_files(pipe, tmp_path):
    root = make_project(tmp_path / "p.clproj")
    pipe.call("project.open", {"root": str(root)})
    out = pipe.call("cloud.points", {"survey": "s1", "cloud": "sparse"})
    assert out["xyz"]["dtype"] == "float32" and out["xyz"]["shape"] == [600, 3]
    assert out["xyz"]["path"].startswith(".arrays/")
    raw = np.fromfile(root / out["xyz"]["path"], dtype="<f4").reshape(600, 3)
    assert np.allclose(raw, Project.open(root).survey("s1").dataset.cloud.xyz)
    assert get_array(root, out["rgb"]).dtype == np.uint8

    with pytest.raises(RpcError) as err:
        pipe.call("cloud.points", {"survey": "s1", "cloud": "dense"})
    assert err.value.data["code"] == "cloud.none"


def test_inbound_arrays_are_checked(pipe, tmp_path):
    root = make_project(tmp_path / "p.clproj")
    pipe.call("project.open", {"root": str(root)})
    outside = {"path": "labels/cloud_labels.npy", "dtype": "int32", "shape": [1]}
    with pytest.raises(RpcError) as err:
        pipe.call("cloud.paint", {"survey": "s1", "cloud": "sparse", "class_id": 0, "indices": outside})
    assert err.value.code == protocol.INVALID_PARAMS

    ref = put_array(root, "idx", np.arange(5, dtype=np.int32))
    ref["shape"] = [6]                                     # lies about its size
    with pytest.raises(RpcError) as err:
        pipe.call("cloud.paint", {"survey": "s1", "cloud": "sparse", "class_id": 0, "indices": ref})
    assert err.value.code == protocol.INVALID_PARAMS
    assert "size" in err.value.data["english"]


# -- labelling -----------------------------------------------------------------------
def test_save_mask_is_undoable_and_persisted(pipe, tmp_path):
    root = make_project(tmp_path / "p.clproj")
    pipe.call("project.open", {"root": str(root)})
    assert pipe.call("image.mask", {"survey": "s1", "name": "a.png"}) == {
        "source": None, "mask": None, "sources": None}

    mask = np.full((H, W), -1, np.int32)
    mask[:, : W // 2] = 0
    mask[:, W // 2:] = 1
    ref = put_array(root, "a", mask)
    out = pipe.call("image.save_mask", {"survey": "s1", "name": "a.png", "mask": ref})
    assert out == {"changed_pixels": H * W, "status": "user"}
    assert not (root / ref["path"]).exists()               # inbound files are consumed
    assert (root / "surveys" / "s1" / "masks" / "a.png.npy").exists()
    assert Project.open(root).survey("s1").labels.status_of("a.png") == "user"
    changed = pipe.drained("project.changed")
    assert any(n["params"].get("images") == ["a.png"] for n in changed)

    shown = pipe.call("image.mask", {"survey": "s1", "name": "a.png"})
    assert shown["source"] == "user"
    assert np.array_equal(get_array(root, shown["mask"]), mask)

    undo = pipe.call("labels.undo", {"survey": "s1"})
    assert undo == {"changed": {"modality": "image", "image": "a.png"}}
    shown = pipe.call("image.mask", {"survey": "s1", "name": "a.png"})
    assert (get_array(root, shown["mask"]) == -1).all()
    assert pipe.call("labels.redo", {"survey": "s1"})["changed"]["image"] == "a.png"
    assert pipe.call("project.state")["surveys"][0]["can_redo"] is False


def test_save_mask_refuses_wrong_shape_and_ids(pipe, tmp_path):
    root = make_project(tmp_path / "p.clproj")
    pipe.call("project.open", {"root": str(root)})
    with pytest.raises(RpcError) as err:
        pipe.call("image.save_mask", {"survey": "s1", "name": "a.png",
                                      "mask": put_array(root, "a", np.zeros((3, 3), np.int32))})
    assert err.value.data["code"] == "image.mask_shape"
    with pytest.raises(RpcError) as err:
        pipe.call("image.save_mask", {"survey": "s1", "name": "a.png",
                                      "mask": put_array(root, "a", np.full((H, W), 5, np.int32))})
    assert err.value.data["code"] == "image.mask_values"
    with pytest.raises(RpcError) as err:
        pipe.call("image.mask", {"survey": "s1", "name": "zzz.png"})
    assert err.value.data["code"] == "image.not_found"


def test_lasso_paint_undo_and_save_on_close(pipe, tmp_path):
    root = make_project(tmp_path / "p.clproj")
    pipe.call("project.open", {"root": str(root)})
    idx = put_array(root, "idx", np.arange(0, 600, 3, dtype=np.int32))
    assert pipe.call("cloud.paint", {"survey": "s1", "cloud": "sparse", "class_id": 1, "indices": idx}) == {
        "painted": 200}
    labels = get_array(root, pipe.call("cloud.labels", {"survey": "s1", "cloud": "sparse"})["labels"])
    assert (labels[::3] == 1).all() and (np.delete(labels, np.s_[::3]) == -1).all()
    assert pipe.call("project.state")["dirty"] is True

    pipe.call("labels.undo", {"survey": "s1"})
    pipe.call("labels.redo", {"survey": "s1"})
    pipe.call("project.close")
    reopened = Project.open(root)
    assert (reopened.survey("s1").labels.cloud_labels[::3] == 1).all()
    assert not (root / ".arrays").exists()                 # emptied on close


def _cloud_labels_b(pipe, root):
    """a.png painted half 0, half 1; the cloud labelled from it; b.png then
    labelled from the cloud (status auto). Returns b.png's overlay."""
    mask = np.zeros((H, W), np.int32)
    mask[:, W // 2:] = 1
    pipe.call("image.save_mask", {"survey": "s1", "name": "a.png", "mask": put_array(root, "a", mask)})
    pipe.call("transfer.images_to_cloud", {"survey": "s1"})
    pipe.call("transfer.cloud_to_images", {"survey": "s1"})
    pipe.drained("request.progress")                 # the pipe is shared between tests
    shown = pipe.call("image.mask", {"survey": "s1", "name": "b.png"})
    assert shown["source"] == "auto"
    return get_array(root, shown["mask"]).astype(np.int32)


def test_painting_over_cloud_labels_keeps_only_the_strokes(pipe, tmp_path):
    """ADR-0006: a few strokes on a photo labelled from the cloud must not turn
    the whole projected overlay into hand-drawn labels."""
    root = make_project(tmp_path / "p.clproj")
    pipe.call("project.open", {"root": str(root)})
    overlay = _cloud_labels_b(pipe, root)
    labelled = overlay != -1
    assert labelled.sum() > 50

    edited = overlay.copy()                          # what the editor sends back
    stroke = np.zeros((H, W), bool)
    stroke[:4, :4] = True                            # one small stroke of class 0
    edited[stroke] = 0
    pipe.call("image.save_mask", {"survey": "s1", "name": "b.png", "mask": put_array(root, "b", edited)})

    stored = Project.open(root).survey("s1").labels.image_masks["b.png"]
    changed = stroke & (overlay != 0)
    assert (stored[changed] == 0).all()
    assert (stored[~stroke] == -1).all()             # the overlay was not copied in
    shown = pipe.call("image.mask", {"survey": "s1", "name": "b.png"})
    assert shown["source"] == "user"
    composed = get_array(root, shown["mask"])
    assert np.array_equal(composed[~stroke], overlay[~stroke])   # still shown under the strokes


def test_touched_pixels_become_hand_drawn_whatever_their_class(pipe, tmp_path):
    """ADR-0006: painting the class a pixel already shows (or accepting the
    prediction there) makes it hand-drawn; erasing shows what is under."""
    root = make_project(tmp_path / "p.clproj")
    pipe.call("project.open", {"root": str(root)})
    overlay = _cloud_labels_b(pipe, root)
    labelled = np.argwhere(overlay != -1)
    (r, c) = labelled[0]
    touched = np.zeros((H, W), np.uint8)
    touched[r, c] = 1                                # accepted as shown
    pipe.call("image.save_mask", {"survey": "s1", "name": "b.png",
                                  "mask": put_array(root, "b", overlay),
                                  "touched": put_array(root, "t", touched)})
    stored = Project.open(root).survey("s1").labels.image_masks["b.png"]
    assert stored[r, c] == overlay[r, c] and (stored != -1).sum() == 1
    shown = pipe.call("image.mask", {"survey": "s1", "name": "b.png"})
    sources = get_array(root, shown["sources"])
    assert sources.dtype == np.uint8 and sources[r, c] == 2          # hand-drawn
    assert set(np.unique(sources[overlay != -1])) <= {2, 3}          # the rest from the cloud

    erased = overlay.copy()
    erased[r, c] = -1
    pipe.call("image.save_mask", {"survey": "s1", "name": "b.png",
                                  "mask": put_array(root, "b", erased),
                                  "touched": put_array(root, "t", touched)})
    shown = pipe.call("image.mask", {"survey": "s1", "name": "b.png"})
    assert get_array(root, shown["mask"])[r, c] == overlay[r, c]     # the cloud's again
    assert get_array(root, shown["sources"])[r, c] == 3


def test_images_to_cloud_leaves_predictions_out_unless_asked(pipe, tmp_path):
    """ADR-0006: a prediction voting on the cloud would come back as cloud-made
    labels that the network trains on."""
    root = make_project(tmp_path / "p.clproj")
    proj = Project.open(root)
    survey = proj.survey("s1")
    np.savez(survey.predictions_dir / "b.png.npz", labels=np.ones((H, W), np.int32))
    survey.labels.mark_ml_labeled("b.png")
    survey.save_manifest()
    pipe.call("project.open", {"root": str(root)})
    with pytest.raises(RpcError) as err:
        pipe.call("transfer.images_to_cloud", {"survey": "s1"})
    assert err.value.data["code"] == "transfer.no_labelled_images"
    out = pipe.call("transfer.images_to_cloud", {"survey": "s1", "include_predictions": True})
    assert out["images"] == 1 and out["sparse"] > 0
    pipe.drained("request.progress")                 # the pipe is shared between tests


def test_cloud_sources_say_where_labels_came_from(pipe, tmp_path):
    """ADR-0006: hand-drawn votes decide a point; a stroke on the cloud is
    painted, with no doubt; a projection doesn't vote on its own cloud."""
    root = make_project(tmp_path / "p.clproj")
    pipe.call("project.open", {"root": str(root)})
    _cloud_labels_b(pipe, root)                      # a.png -> cloud -> b.png
    out = pipe.call("cloud.sources", {"survey": "s1"})
    source = get_array(root, out["source"])
    unc = get_array(root, out["uncertainty"])
    labels = get_array(root, pipe.call("cloud.labels", {"survey": "s1"})["labels"])
    assert source.dtype == np.uint8 and len(source) == len(labels)
    assert set(np.unique(source[labels != -1])) == {2}           # hand-drawn
    assert ((unc[labels != -1] > 0) & (unc[labels != -1] <= 1)).all()
    assert np.isnan(unc[labels == -1]).all()

    # b.png's labels came from this sparse cloud: voting again with it adds nothing.
    again = pipe.call("transfer.images_to_cloud", {"survey": "s1", "include_auto": True})
    assert again["images"] == 2
    after = get_array(root, pipe.call("cloud.sources", {"survey": "s1"})["source"])
    assert np.array_equal(after, source)

    idx = put_array(root, "idx", np.array([0, 1], np.int32))
    pipe.call("cloud.paint", {"survey": "s1", "cloud": "sparse", "class_id": 1, "indices": idx})
    out = pipe.call("cloud.sources", {"survey": "s1"})
    assert get_array(root, out["source"])[:2].tolist() == [1, 1]
    assert get_array(root, out["uncertainty"])[:2].tolist() == [0.0, 0.0]
    pipe.drained("request.progress")


def test_uncertainty_shows_in_photos_and_on_the_cloud(pipe, tmp_path):
    """ADR-0006: a cloud-made photo shows its points' Uncertainty, a
    hand-drawn pixel is sure, and the cloud can be coloured by it."""
    from cloudlabeller.engine.scene import UNKNOWN_GREY, uncertainty_colours

    root = make_project(tmp_path / "p.clproj")
    pipe.call("project.open", {"root": str(root)})
    overlay = _cloud_labels_b(pipe, root)
    rows = {i["name"]: i for i in pipe.call("project.state")["surveys"][0]["images"]}
    assert 0 < rows["b.png"]["uncertainty"] <= 1                      # cloud-made: a mean
    assert rows["a.png"]["uncertainty"] is None                       # hand-drawn

    out = pipe.call("image.uncertainty", {"survey": "s1", "name": "b.png"})
    u = get_array(root, out["uncertainty"])
    assert out["step"] == 1 and u.shape == overlay.shape and u.dtype == np.float32
    assert np.isnan(u[overlay == -1]).all()
    assert ((u[overlay != -1] > 0) & (u[overlay != -1] <= 1)).all()
    assert out["mean"] == pytest.approx(float(u[overlay != -1].mean()), rel=1e-4)
    a = pipe.call("image.uncertainty", {"survey": "s1", "name": "a.png"})
    ua = get_array(root, a["uncertainty"])
    assert (ua == 0).all() and a["mean"] == 0.0                       # all hand-drawn

    view = pipe.call("cloud.view", {"survey": "s1", "colour": "uncertainty"})
    rgb = get_array(root, view["rgb"])
    labels = get_array(root, pipe.call("cloud.labels", {"survey": "s1"})["labels"])
    unc = get_array(root, pipe.call("cloud.sources", {"survey": "s1"})["uncertainty"])
    assert np.array_equal(rgb, uncertainty_colours(unc))
    assert (rgb[labels == -1] == UNKNOWN_GREY).all()
    ramp = uncertainty_colours(np.array([0.0, 0.5, 1.0, np.nan]))
    assert ramp[0].tolist() == [40, 110, 210] and ramp[2].tolist() == [215, 45, 40]
    assert ramp[3].tolist() == list(UNKNOWN_GREY)


def test_predictions_only_fill_the_gaps_better_labels_leave(pipe, tmp_path):
    """ADR-0006: Predict runs on every photo, but a prediction never hides the
    user's labels or the cloud's, and a cloud-made photo stays cloud-made."""
    root = make_project(tmp_path / "p.clproj")
    _fake_model(root)
    pipe.call("project.open", {"root": str(root)})
    overlay = _cloud_labels_b(pipe, root)            # a.png user, b.png auto
    plan = pipe.call("jobs.plan", {"kind": "predict", "options": {"model": "m1"}})
    assert plan["args"] == [str(root), "--model", "m1"]
    before = {i["name"]: i["uncertainty"]
              for i in pipe.call("project.state")["surveys"][0]["images"]}

    survey_dir = root / "surveys" / "s1" / "predictions"
    survey_dir.mkdir(parents=True, exist_ok=True)
    for name in ("a.png", "b.png"):                  # what predict_cli would write
        np.savez(survey_dir / f"{name}.npz", labels=np.zeros((H, W), np.int16),
                 uncertainty=np.full((H, W), 0.9, np.float16))
    predicted = [{"survey": "s1", "name": n, "uncertainty": 0.9} for n in ("a.png", "b.png")]
    pipe.call("jobs.finish", {"plan": plan["plan"], "exit": 0,
                              "results": {"RESULT": json.dumps({"predicted": predicted})}})
    rows = {i["name"]: i for i in pipe.call("project.state")["surveys"][0]["images"]}
    assert rows["a.png"]["status"] == "user" and rows["b.png"]["status"] == "auto"
    assert rows["b.png"]["uncertainty"] == before["b.png"]         # the cloud's mean stays

    shown = pipe.call("image.mask", {"survey": "s1", "name": "b.png"})
    mask = get_array(root, shown["mask"])
    sources = get_array(root, shown["sources"])
    reached = overlay != -1
    assert np.array_equal(mask[reached], overlay[reached])          # the cloud's, untouched
    assert (sources[reached] == 3).all()
    assert (mask[~reached] == 0).all() and (sources[~reached] == 4).all()   # gaps predicted
    pipe.drained("request.progress")


def test_remove_copied_labels_counts_then_removes_them(pipe, tmp_path):
    """Masks saved before ADR-0006 copied the overlay in as hand-drawn."""
    root = make_project(tmp_path / "p.clproj")
    pipe.call("project.open", {"root": str(root)})
    overlay = _cloud_labels_b(pipe, root)
    # an old save: the whole overlay, plus one real stroke, stored as the user's
    proj_mask = overlay.copy()
    r, c = np.argwhere(overlay != -1)[0]
    proj_mask[r, c] = 1 - overlay[r, c]
    pipe.call("project.close")
    survey = Project.open(root).survey("s1")
    survey.labels.image_masks["b.png"] = proj_mask
    survey.labels.mark_user_labeled("b.png")
    survey.save_mask("b.png")
    survey.save_manifest()
    pipe.call("project.open", {"root": str(root)})

    labelled = int((proj_mask != -1).sum())
    out = pipe.call("image.remove_copied", {"survey": "s1", "name": "b.png"})
    assert out == {"labelled": labelled, "copied": labelled - 1, "removed": 0}
    out = pipe.call("image.remove_copied", {"survey": "s1", "name": "b.png", "apply": True})
    assert out["removed"] == labelled - 1
    stored = Project.open(root).survey("s1").labels.image_masks["b.png"]
    assert (stored != -1).sum() == 1 and stored[r, c] == proj_mask[r, c]   # the stroke stays
    shown = pipe.call("image.mask", {"survey": "s1", "name": "b.png"})
    assert np.array_equal(get_array(root, shown["mask"]), proj_mask)        # still all shown
    pipe.call("labels.undo", {"survey": "s1"})
    again = pipe.call("image.remove_copied", {"survey": "s1", "name": "b.png"})
    assert again["copied"] == labelled - 1
    pipe.drained("request.progress")

def test_images_to_cloud_then_cloud_to_images(pipe, tmp_path):
    root = make_project(tmp_path / "p.clproj")
    pipe.call("project.open", {"root": str(root)})
    with pytest.raises(RpcError) as err:
        pipe.call("transfer.images_to_cloud", {"survey": "s1"})
    assert err.value.data["code"] == "transfer.no_labelled_images"

    mask = np.zeros((H, W), np.int32)
    mask[:, W // 2:] = 1
    pipe.call("image.save_mask", {"survey": "s1", "name": "a.png", "mask": put_array(root, "a", mask)})

    msg = pipe.request("transfer.images_to_cloud", {"survey": "s1", "include_auto": True}, req_id="i2c")
    out = msg["result"]
    assert out["images"] == 1 and out["sparse"] > 0
    progress = pipe.drained("request.progress")
    assert progress and all(p["params"]["id"] == "i2c" for p in progress)
    assert progress[-1]["params"]["fraction"] == 1.0
    cloud_labels = get_array(root, pipe.call("cloud.labels", {"survey": "s1"})["labels"])
    assert set(np.unique(cloud_labels)) <= {-1, 0, 1} and (cloud_labels == 1).any()

    out = pipe.call("transfer.cloud_to_images", {"survey": "s1"})
    assert out == {"cloud": "sparse", "auto_labelled": ["b.png"]}
    statuses = {i["name"]: i["status"] for i in pipe.call("project.state")["surveys"][0]["images"]}
    assert statuses == {"a.png": "user", "b.png": "auto"}
    shown = pipe.call("image.mask", {"survey": "s1", "name": "b.png"})
    assert shown["source"] == "auto" and shown["mask"]["shape"] == [H, W]

    out = pipe.call("transfer.propagate", {"survey": "s1", "name": "a.png"})
    assert out["cloud"] == "sparse"
    with pytest.raises(RpcError) as err:
        pipe.call("transfer.propagate", {"survey": "s1", "name": "b.png"})
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
        pipe.call("jobs.plan", {"kind": "predict", "options": {"model": "nope"}})
    assert err.value.data["code"] == "job.predict.no_model"
    with pytest.raises(RpcError) as err:
        pipe.call("jobs.plan", {"kind": "reproject", "options": {"epsg": 32723}})
    assert err.value.data["code"] == "job.reproject.not_georeferenced"
    with pytest.raises(RpcError) as err:
        pipe.call("jobs.plan", {"kind": "mesh", "options": {"survey": "s1"}})
    assert err.value.data["code"] == "job.mesh.no_dense"
    with pytest.raises(RpcError) as err:
        pipe.call("jobs.plan", {"kind": "warp"})
    assert err.value.code == protocol.INVALID_PARAMS
    with pytest.raises(RpcError) as err:
        pipe.call("jobs.plan", {"kind": "sfm", "options": {"survey": "s1", "matcher": "psychic"}})
    assert err.value.code == protocol.INVALID_PARAMS
    assert pipe.call("jobs.list") == {"jobs": []}


def test_job_arguments_match_the_clis(tmp_path):
    """Each kind's command line parses with its CLI's own argument parser."""
    import importlib

    from cloudlabeller.engine.jobs import KINDS

    root = make_project(tmp_path / "p.clproj")
    engine = Engine(lambda data: None)
    engine.project = Project.open(root)
    options = {"ingest": {"survey": "s1", "folder": str(tmp_path)},
               "sfm": {"survey": "s1", "gpu": True},
               "georef": {"survey": "s1"}, "reproject": {"epsg": 32723, "orthometric": True},
               "train": {"name": "m1", "steps": 300, "patience": 4, "include_auto": True,
                         "weight_by_uncertainty": True,
                         "resume": True,
                         "surveys": ["s1"]},
               "geoid": {"epsg": 31982}}
    if sys.platform == "win32":
        options["colmap"] = {"variant": "cpu"}
    parsed = {}
    for name, opts in options.items():
        kind = KINDS[name]
        if name == "reproject":                     # it needs a frame and an aligned Survey
            engine.project.settings["georeferenced"] = {
                "frame": "ENU", "origin_lla": [-30.0, -53.0, 100.0],
                "origin_convention": "first_gps_image"}
            engine.project.survey("s1").active.alignment = {
                "s": 1.0, "R": np.eye(3).tolist(), "t": [0, 0, 0], "source": "gps"}
        plan = kind.prepare(engine, engine.project, opts)
        cli = importlib.import_module(kind.module)
        parsed[name] = cli._parse([kind.module, *plan.args])   # exits on a bad flag
    assert parsed["ingest"].ingest_only and parsed["ingest"].on_conflict == "skip"
    assert parsed["sfm"].gpu and parsed["sfm"].matcher == "spatial"
    assert parsed["reproject"].epsg == 32723 and parsed["reproject"].orthometric
    assert parsed["train"].steps == 300 and parsed["train"].patience == 4
    assert parsed["train"].epochs is None and parsed["train"].include_auto
    assert parsed["train"].weight_by_uncertainty
    assert parsed["train"].resume and parsed["train"].stop_file.endswith("train.stop")
    assert parsed["geoid"].epsg == 31982
    if "colmap" in parsed:
        assert parsed["colmap"].variant == "nocuda"
        assert parsed["colmap"].stop_file.endswith("colmap_download.stop")
    assert KINDS["sfm"].prepare(engine, engine.project,
                                {"survey": "s1", "gpu": True}).gpu is True
    assert KINDS["sfm"].prepare(engine, engine.project, {"survey": "s1"}).gpu is False
    assert KINDS["train"].prepare(engine, engine.project, {"resume": True}).gpu is True
    from cloudlabeller.ml.model_spec import ModelSpec

    engine.project.set_model_spec(ModelSpec(tile_size=1000, shrink=2))
    with pytest.raises(Exception) as err:                  # a fresh spec is checked
        KINDS["train"].prepare(engine, engine.project, {})
    assert getattr(err.value, "code", None) == "job.train.bad_spec"
    with pytest.raises(Exception) as err:
        KINDS["train"].prepare(engine, engine.project, {"surveys": ["s9"]})
    assert getattr(err.value, "code", None) == "survey.not_found"


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
        client.call("image.save_mask", {"survey": "s1", "name": "a.png", "mask": put_array(root, "a", mask)})
        out = client.call("transfer.propagate", {"survey": "s1", "name": "a.png"}, timeout=60)
        assert out["cloud"] == "sparse"
    finally:
        assert client.close() == 0


# -- in-process: autosave and cancellation -----------------------------
def test_autosave_runs_while_requests_keep_coming(tmp_path):
    """Regression: autosave fired only after an idle stretch, so an Engine
    kept busy by the window never saved dirty labels."""
    import time

    root = make_project(tmp_path / "p.clproj")
    client = InProcess(autosave_s=0.3)
    try:
        client.call("project.open", {"root": str(root)})
        idx = put_array(root, "idx", np.arange(10, dtype=np.int32))
        client.call("cloud.paint", {"survey": "s1", "cloud": "sparse", "class_id": 0, "indices": idx})
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline:                 # never idle for 0.3 s
            client.call("engine.hello")
            time.sleep(0.05)
        assert client.call("project.state")["dirty"] is False
        labels = root / "surveys" / "s1" / "reconstructions" / "r1" / "labels" / "sparse.npy"
        assert (np.load(labels)[:10] == 0).all()
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


def _fake_model(root: Path, name: str = "m1", n_classes: int = 2, tiled: bool = True) -> None:
    """A saved model as ``models.list`` sees one (no real weights needed)."""
    d = root / "ml" / "models" / name
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps(
        {"name": name, "n_classes": n_classes, "created": "2026-11-25",
         "spec": {"tile_size": 1024, "shrink": 2} if tiled else {"resolution_divisor": 10}}),
        encoding="utf-8")
    (d / "model.weights.h5").write_bytes(b"")


def test_plan_says_how_to_run_and_finish_applies(pipe, tmp_path):
    """ADR-0004: the Engine plans, the core runs, the Engine applies."""
    root = make_project(tmp_path / "p.clproj")
    _fake_model(root)
    pipe.call("project.open", {"root": str(root)})
    plan = pipe.call("jobs.plan", {"kind": "predict", "options": {"model": "m1"}})
    assert plan["kind"] == "predict" and plan["group"] == "ml" and plan["gpu"] is True
    assert plan["module"] == "cloudlabeller.ml.predict_cli"
    assert plan["args"] == [str(root), "--model", "m1"]
    assert plan["log"].endswith("predict.log") and plan["stop_file"] is None
    assert pipe.call("jobs.list") == {"jobs": [plan]}

    predicted = [{"survey": "s1", "name": "b.png"}]
    result_line = json.dumps({"predicted": predicted})
    done = pipe.call("jobs.finish", {"plan": plan["plan"], "exit": 0,
                                     "results": {"RESULT": result_line}})
    assert done == {"plan": plan["plan"], "kind": "predict", "ok": True,
                    "result": {"predicted": predicted}, "stopped": False}
    statuses = {i["name"]: i["status"] for i in pipe.call("project.state")["surveys"][0]["images"]}
    assert statuses["b.png"] == "ml"
    assert any(n["params"].get("images") == ["b.png"] for n in pipe.drained("project.changed"))
    assert Project.open(root).survey("s1").labels.status_of("b.png") == "ml"   # persisted
    assert pipe.call("jobs.list") == {"jobs": []}


def _core_results(text: str) -> dict:
    """``RESULT…`` lines as the core hands them to ``jobs.finish``."""
    out = {}
    for line in text.splitlines():
        if line.startswith("RESULT"):
            key, _, rest = line.partition(" ")
            out[key] = rest.strip()
    return out


def test_a_long_result_arrives_in_pieces_and_is_applied(pipe, tmp_path):
    """Regression: predicting 1,399 photos printed a 65 KB RESULT line, the
    core told it in two pieces (lines are capped at 64 KiB), the JSON failed
    to parse as nothing, and no photo was marked predicted."""
    import io

    from cloudlabeller.workers.result import print_result

    root = make_project(tmp_path / "p.clproj")
    _fake_model(root)
    pipe.call("project.open", {"root": str(root)})
    plan = pipe.call("jobs.plan", {"kind": "predict", "options": {"model": "m1"}})
    predicted = [{"survey": "s1", "name": "b.png", "uncertainty": 0.25}] + [
        {"survey": "s9", "name": f"photo {k:04d}.JPG", "uncertainty": k / 7} for k in range(400)]
    out = io.StringIO()
    print_result({"predicted": predicted}, file=out, size=1000)
    results = _core_results(out.getvalue())
    assert len(results) > 10 and "RESULT" not in results
    done = pipe.call("jobs.finish", {"plan": plan["plan"], "exit": 0, "results": results})
    assert done["ok"] is True and done["result"] == {"predicted": predicted}
    statuses = {i["name"]: i["status"] for i in pipe.call("project.state")["surveys"][0]["images"]}
    assert statuses["b.png"] == "ml"


def test_a_result_that_is_not_json_fails_the_job(pipe, tmp_path):
    root = make_project(tmp_path / "p.clproj")
    _fake_model(root)
    pipe.call("project.open", {"root": str(root)})
    plan = pipe.call("jobs.plan", {"kind": "predict", "options": {"model": "m1"}})
    cut = json.dumps({"predicted": [{"survey": "s1", "name": "b.png"}]})[:30]
    done = pipe.call("jobs.finish", {"plan": plan["plan"], "exit": 0,
                                     "results": {"RESULT": cut}})
    assert done["ok"] is False and done["error"]["code"] == "job.bad_result"
    assert pipe.call("project.state")["surveys"][0]["images"][1]["status"] == "none"


def test_outstanding_plans_guard_the_project(pipe, tmp_path):
    root = make_project(tmp_path / "p.clproj")
    _fake_model(root)
    pipe.call("project.open", {"root": str(root)})
    plan = pipe.call("jobs.plan", {"kind": "predict", "options": {"model": "m1"}})["plan"]
    with pytest.raises(RpcError) as err:                    # same group: ml
        pipe.call("jobs.plan", {"kind": "train", "options": {"resume": True}})
    assert err.value.data["code"] == "job.busy"
    with pytest.raises(RpcError) as err:
        pipe.call("project.close")
    assert err.value.data["code"] == "project.jobs_running"
    other = pipe.call("jobs.plan", {"kind": "georef", "options": {"survey": "s1"}})       # another group is fine
    pipe.call("jobs.finish", {"plan": other["plan"], "exit": None})

    done = pipe.call("jobs.finish", {"plan": plan, "exit": 3})
    assert done["ok"] is False and done["error"]["code"] == "job.failed"
    assert done["error"]["params"]["exit"] == 3
    assert pipe.call("project.state")["surveys"][0]["images"][1]["status"] == "none"   # nothing applied
    with pytest.raises(RpcError) as err:
        pipe.call("jobs.finish", {"plan": plan, "exit": 0})
    assert err.value.data["code"] == "job.unknown_plan"
    pipe.call("project.close")                               # nothing outstanding now


def test_finish_reports_lost_and_cancelled_jobs(pipe, tmp_path):
    root = make_project(tmp_path / "p.clproj")
    pipe.call("project.open", {"root": str(root)})
    plan = pipe.call("jobs.plan", {"kind": "georef", "options": {"survey": "s1"}})["plan"]
    lost = pipe.call("jobs.finish", {"plan": plan, "exit": None})
    assert lost["error"]["code"] == "job.lost"
    plan = pipe.call("jobs.plan", {"kind": "georef", "options": {"survey": "s1"}})["plan"]
    killed = pipe.call("jobs.finish", {"plan": plan, "exit": 1, "cancelled": True})
    assert killed["error"]["code"] == "job.cancelled"
    with pytest.raises(RpcError) as err:
        pipe.call("jobs.finish", {"plan": 1, "exit": "0"})
    assert err.value.code == protocol.INVALID_PARAMS


def test_stopped_training_still_counts(pipe, tmp_path):
    """Cancelling through the stop file ends training gracefully: the model
    is saved, so the result is applied and the stop file cleared."""
    root = make_project(tmp_path / "p.clproj")
    pipe.call("project.open", {"root": str(root)})
    plan = pipe.call("jobs.plan", {"kind": "train", "options": {"resume": True}})
    stop = Path(plan["stop_file"])
    assert plan["gpu"] is True and stop.name == "train.stop"
    stop.touch()                                             # what the core does to cancel
    summary = json.dumps({"name": "unet", "epochs_run": 2})
    done = pipe.call("jobs.finish", {"plan": plan["plan"], "exit": 0, "cancelled": True,
                                     "results": {"RESULT": summary}})
    assert done["ok"] is True and done["stopped"] is True
    assert done["result"] == {"name": "unet", "epochs_run": 2}
    assert not stop.exists()


def _staged_engine(tmp_path, monkeypatch, *, binary, cuda):
    """An in-process Engine on a Project with a dense cloud, with the COLMAP
    executable and pycolmap's CUDA support faked as given."""
    from cloudlabeller.engine import jobs
    from cloudlabeller.io.geometry import save_cloud_npz

    monkeypatch.setattr(jobs, "_colmap_binary", lambda: binary)
    monkeypatch.setattr(jobs, "_pycolmap_has_cuda", lambda: cuda)
    root = make_project(tmp_path / "p.clproj")
    proj = Project.open(root)
    save_cloud_npz(PointCloud(xyz=np.zeros((10, 3), np.float32)), (product_dir(proj, "mvs") / "dense.npz"))
    engine = Engine(lambda data: None)
    engine.project = Project.open(root)
    return engine


def test_plans_name_the_dirs_to_stage(tmp_path, monkeypatch):
    """sfm, mvs and mesh say which Project dirs they read and write; every
    Project path in their args lies inside one, so the core can rewrite the
    root prefix to a staged copy. The other kinds have no stage."""
    import importlib

    from cloudlabeller.engine.jobs import KINDS, Outstanding

    engine = _staged_engine(tmp_path, monkeypatch, binary=None, cuda=True)
    project = engine.project
    root = Path(project.root)
    s = "surveys/s1"
    # every step writes a new product folder (ADR-0005)
    expected = {"sfm": {"in": [f"{s}/images"], "out": [f"{s}/products/sfm-2"]},
                "mvs": {"in": [f"{s}/images", f"{s}/products/sfm-1"],
                        "out": [f"{s}/products/mvs-2"]},
                "mesh": {"in": [f"{s}/products/mvs-1"], "out": [f"{s}/products/mesh-1"]}}
    for name, stage in expected.items():
        kind = KINDS[name]
        plan = kind.prepare(engine, project, {"survey": "s1"})
        described = Outstanding(1, kind, plan, project).describe()
        assert described["stage"] == stage, name
        dirs = [root / d for d in stage["in"] + stage["out"]]
        for arg in plan.args:
            if arg.startswith(str(root)):
                assert any(Path(arg).is_relative_to(d) for d in dirs), (name, arg)
        importlib.import_module(kind.module)._parse([kind.module, *plan.args])
    for name, opts in {"georef": {"survey": "s1"},
                       "ingest": {"survey": "s1", "folder": str(tmp_path)}}.items():
        plan = KINDS[name].prepare(engine, project, opts)
        assert Outstanding(1, KINDS[name], plan, project).describe()["stage"] is None


def test_plan_over_the_pipe_carries_stage(pipe, tmp_path):
    root = make_project(tmp_path / "p.clproj")
    pipe.call("project.open", {"root": str(root)})
    plan = pipe.call("jobs.plan", {"kind": "sfm", "options": {"survey": "s1"}})
    assert plan["stage"] == {"in": ["surveys/s1/images"],
                             "out": ["surveys/s1/products/sfm-2"]}
    pipe.call("jobs.finish", {"plan": plan["plan"], "exit": 1})
    _fake_model(root)
    plan = pipe.call("jobs.plan", {"kind": "predict", "options": {"model": "m1"}})
    assert plan["stage"] is None


def test_mvs_runs_on_a_cuda_pycolmap_without_colmap_exe(tmp_path, monkeypatch):
    from cloudlabeller.engine.jobs import KINDS
    from cloudlabeller.engine.protocol import Problem

    engine = _staged_engine(tmp_path, monkeypatch, binary=None, cuda=True)
    plan = KINDS["mvs"].prepare(engine, engine.project, {"survey": "s1"})
    assert plan.gpu and "--colmap-binary" not in plan.args

    engine = _staged_engine(tmp_path / "cpu", monkeypatch, binary=None, cuda=False)
    with pytest.raises(Problem) as err:
        KINDS["mvs"].prepare(engine, engine.project, {"survey": "s1"})
    assert err.value.code == "job.mvs.no_colmap"


def test_delaunay_mesh_needs_colmap_exe(tmp_path, monkeypatch):
    from cloudlabeller.engine.jobs import KINDS
    from cloudlabeller.engine.protocol import Problem

    engine = _staged_engine(tmp_path, monkeypatch, binary=None, cuda=True)
    with pytest.raises(Problem) as err:
        KINDS["mesh"].prepare(engine, engine.project, {"survey": "s1", "method": "delaunay"})
    assert err.value.code == "job.mesh.no_delaunay"
    assert KINDS["mesh"].prepare(engine, engine.project, {"survey": "s1"}).args[2:4] == [
        "--method", "poisson"]

    engine = _staged_engine(tmp_path / "exe", monkeypatch, binary="colmap.exe", cuda=False)
    plan = KINDS["mesh"].prepare(engine, engine.project, {"survey": "s1", "method": "delaunay"})
    assert plan.args[-2:] == ["--colmap-binary", "colmap.exe"]


# -- normals (experimental) ------------------------------------------------------------
def _add_dense(root: Path) -> None:
    """A dense cloud facing the cameras: every normal points back at them (-z)."""
    from cloudlabeller.io.geometry import save_cloud_npz

    proj = Project.open(root)
    rng = np.random.default_rng(1)
    xyz = np.column_stack([rng.uniform(-1.5, 1.5, 4000), rng.uniform(-1, 1, 4000),
                           np.full(4000, 5.0)]).astype(np.float32)
    normals = np.tile(np.array([0, 0, -1], np.float32), (4000, 1))
    save_cloud_npz(PointCloud(xyz=xyz, rgb=np.full((4000, 3), 90, np.uint8), normals=normals),
                   product_dir(proj, "mvs") / "dense.npz")


def _settle(pipe) -> None:
    """End the plans an earlier test on this module's Engine left outstanding."""
    for job in pipe.call("jobs.list")["jobs"]:
        pipe.call("jobs.finish", {"plan": job["plan"], "exit": None})


def test_the_cloud_and_photos_can_be_coloured_by_normals(pipe, tmp_path):
    _settle(pipe)
    root = make_project(tmp_path / "p.clproj")
    pipe.call("project.open", {"root": str(root)})
    assert pipe.call("project.state")["surveys"][0]["normals"] is False
    with pytest.raises(RpcError) as err:
        pipe.call("cloud.view", {"survey": "s1", "cloud": "sparse", "colour": "normals"})
    assert err.value.data["code"] == "cloud.no_normals"
    assert pipe.call("image.normals", {"survey": "s1", "name": "a.png"}) == {
        "normals": None, "step": None}
    pipe.call("project.close")

    _add_dense(root)
    pipe.call("project.open", {"root": str(root)})
    assert pipe.call("project.state")["surveys"][0]["normals"] is True
    shown = pipe.call("cloud.view", {"survey": "s1", "cloud": "dense", "colour": "normals",
                                     "max_points": 500})
    rgb = get_array(root, shown["rgb"])
    assert rgb.shape == (500, 3) and (rgb == [128, 128, 0]).all()      # -z: no blue
    photo = get_array(root, pipe.call("cloud.view", {"survey": "s1", "cloud": "dense"})["rgb"])
    assert (photo == 90).all()                                            # the default
    with pytest.raises(RpcError):
        pipe.call("cloud.view", {"survey": "s1", "cloud": "dense", "colour": "rainbow"})

    out = pipe.call("image.normals", {"survey": "s1", "name": "a.png"})
    assert out["step"] == 1
    nmap = get_array(root, out["normals"])
    assert nmap.shape == (H, W, 3) and nmap.dtype == np.uint8
    seen = (nmap != 128).any(axis=2)
    assert seen.mean() > 0.5 and (nmap[seen] == [128, 128, 0]).all()


def test_a_model_on_normals_needs_a_dense_cloud(pipe, tmp_path):
    from cloudlabeller.ml.model_spec import ModelSpec

    _settle(pipe)
    root = make_project(tmp_path / "p.clproj")
    pipe.call("project.open", {"root": str(root)})
    spec = ModelSpec(tile_size=32, shrink=2, blocks=1, normals=True).to_dict()
    pipe.call("model.set_spec", {"spec": spec})
    assert pipe.call("model.spec")["spec"]["normals"] is True
    with pytest.raises(RpcError) as err:
        pipe.call("jobs.plan", {"kind": "train", "options": {}})
    assert err.value.data["code"] == "job.train.no_normals"
    assert err.value.data["params"]["surveys"] == ["s1"]
    pipe.call("project.close")

    _add_dense(root)
    pipe.call("project.open", {"root": str(root)})
    plan = pipe.call("jobs.plan", {"kind": "train", "options": {}})
    pipe.call("jobs.finish", {"plan": plan["plan"], "exit": None})

    d = root / "ml" / "models" / "nm"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps(
        {"name": "nm", "n_classes": 2, "spec": spec, "in_channels": 6}), encoding="utf-8")
    (d / "model.weights.h5").write_bytes(b"")
    plan = pipe.call("jobs.plan", {"kind": "predict", "options": {"model": "nm"}})
    pipe.call("jobs.finish", {"plan": plan["plan"], "exit": None})
