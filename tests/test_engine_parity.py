"""Contract tests for the Engine methods the web window needs for parity
with the Qt app (docs/engine.md): the 3D scene, cleaning, Job facts, CRS
and export, and the refusals the Qt dialogs made.

Same harnesses as ``test_engine_contract.py``: the real Engine process over
a pipe where possible, an in-process Engine where a test fakes a library.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from conftest import product_dir
import pytest
from test_engine_contract import (
    H,
    InProcess,
    Pipe,
    RpcError,
    W,
    _fake_model,
    get_array,
    make_project,
    put_array,
)

from cloudlabeller.core.dataset import Mesh, PointCloud
from cloudlabeller.core.project import Project
from cloudlabeller.engine import protocol

ORIGIN = (-30.05, -51.2, 10.0)          # southern Brazil: SIRGAS 2000 / UTM 22S
UTM_SIRGAS_22S = 31982


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


def _georeference(root: Path, crs: dict | None = None) -> None:
    """Give the Project a frame (ENU, exact origin), optionally reprojected,
    and align Survey s1 into it as it stands (identity Alignment)."""
    proj = Project.open(root)
    geo = {"frame": "ENU", "origin_lla": list(ORIGIN),
           "origin_convention": "first_gps_image"}
    if crs is not None:
        geo["crs"] = crs
    proj.settings["georeferenced"] = geo
    proj.survey("s1").active.alignment = {"s": 1.0, "R": np.eye(3).tolist(),
                                          "t": [0.0, 0.0, 0.0], "source": "gps"}
    proj.save_manifest()


def _with_dense(root: Path, n: int = 400) -> None:
    from cloudlabeller.io.geometry import save_cloud_npz

    rng = np.random.default_rng(1)
    xyz = rng.uniform(0, 1, (n, 3)).astype(np.float32)
    save_cloud_npz(PointCloud(xyz=xyz), product_dir(Project.open(root), "mvs") / "dense.npz")


def _three_view_projection(fov_deg: float = 60.0, aspect: float = W / H,
                           near: float = 0.1, far: float = 100.0) -> np.ndarray:
    """A three.js-style perspective camera at the origin looking down +Z
    (world), as ``projectionMatrix × matrixWorldInverse``."""
    f = 1.0 / np.tan(np.radians(fov_deg) / 2)
    proj = np.array([[f / aspect, 0, 0, 0], [0, f, 0, 0],
                     [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)],
                     [0, 0, -1, 0]])
    view = np.diag([1.0, -1.0, -1.0, 1.0])     # three.js cameras look down -Z
    return proj @ view


def _elements(m: np.ndarray) -> list[float]:
    """three.js's column-major ``Matrix4.elements``."""
    return m.T.reshape(16).tolist()


# -- the 3D view ----------------------------------------------------------------
def test_cloud_view_subsamples_with_its_full_cloud_index(pipe, tmp_path):
    root = make_project(tmp_path / "p.clproj")
    pipe.call("project.open", {"root": str(root)})
    full = Project.open(root).survey("s1").dataset.cloud

    out = pipe.call("cloud.view", {"survey": "s1", "cloud": "sparse", "max_points": 100})
    assert out["n_points"] == 600 and out["shown"] == 100
    index = get_array(root, out["index"])
    assert out["index"]["dtype"] == "int32" and index.shape == (100,)
    assert (np.diff(index) > 0).all() and index.max() < 600
    assert np.allclose(get_array(root, out["xyz"]), full.xyz[index])
    assert np.array_equal(get_array(root, out["rgb"]), full.rgb[index])
    again = pipe.call("cloud.view", {"survey": "s1", "cloud": "sparse", "max_points": 100})
    assert np.array_equal(get_array(root, again["index"]), index)     # seeded

    for max_points in (0, 600, 5000):
        out = pipe.call("cloud.view", {"survey": "s1", "max_points": max_points})
        assert out["index"] is None and out["shown"] == 600
    with pytest.raises(RpcError) as err:
        pipe.call("cloud.view", {"survey": "s1", "max_points": -1})
    assert err.value.code == protocol.INVALID_PARAMS


def test_cloud_select_takes_a_threejs_lasso(pipe, tmp_path):
    root = make_project(tmp_path / "p.clproj")
    pipe.call("project.open", {"root": str(root)})
    xyz = Project.open(root).survey("s1").dataset.cloud.xyz.astype(np.float64)
    m = _three_view_projection()
    left_half = [[0, 0], [W / 2, 0], [W / 2, H], [0, H]]
    out = pipe.call("cloud.select", {"survey": "s1", "cloud": "sparse", "view_projection": _elements(m),
                                     "polygon": left_half, "width": W, "height": H})
    indices = get_array(root, out["indices"])
    assert out["indices"]["dtype"] == "uint32" and out["count"] == len(indices)

    f = 1.0 / np.tan(np.radians(30.0))
    ndc_x, ndc_y = f / (W / H) * xyz[:, 0] / xyz[:, 2], f * xyz[:, 1] / xyz[:, 2]
    expected = np.flatnonzero((ndc_x > -1) & (ndc_x < 0) & (np.abs(ndc_y) < 1))
    assert len(expected) > 50
    assert np.array_equal(indices, expected)

    # CSS y points down. This view flips world y, so the screen's top half
    # shows world y < 0.
    top_half = [[0, 0], [W, 0], [W, H / 2], [0, H / 2]]
    out = pipe.call("cloud.select", {"survey": "s1", "view_projection": _elements(m), "polygon": top_half,
                                     "width": W, "height": H})
    top = get_array(root, out["indices"])
    assert len(top) and (xyz[top, 1] < 0).all()

    assert pipe.call("cloud.select", {"survey": "s1", "view_projection": _elements(m),
                                      "polygon": [[0, 0], [1, 1]], "width": W,
                                      "height": H})["count"] == 0
    with pytest.raises(RpcError) as err:
        pipe.call("cloud.select", {"survey": "s1", "view_projection": [1, 2, 3], "polygon": left_half,
                                   "width": W, "height": H})
    assert err.value.code == protocol.INVALID_PARAMS


def test_cloud_delete_keeps_the_complement_and_labels_follow(pipe, tmp_path):
    """As the Qt window's lasso delete: the cloud file is rewritten, labels
    follow the surviving points, and undo is cleared."""
    root = make_project(tmp_path / "p.clproj")
    pipe.call("project.open", {"root": str(root)})
    pipe.call("cloud.paint", {"survey": "s1", "class_id": 1,
                              "indices": put_array(root, "p", np.arange(0, 600, 3,
                                                                        dtype=np.int32))})
    before = get_array(root, pipe.call("cloud.labels", {"survey": "s1"})["labels"])
    full = Project.open(root).survey("s1").dataset.cloud.xyz
    assert pipe.call("project.state")["surveys"][0]["can_undo"] is True

    gone = np.arange(100, dtype=np.uint32)
    out = pipe.call("cloud.delete", {"survey": "s1", "cloud": "sparse",
                                     "indices": put_array(root, "d", gone)})
    assert out == {"removed": 100, "n_points": 500}
    assert any("sparse" in n["params"]["parts"] for n in pipe.drained("project.changed"))
    state = pipe.call("project.state")
    assert state["surveys"][0]["sparse"] == 500 and state["surveys"][0]["can_undo"] is False
    labels = get_array(root, pipe.call("cloud.labels", {"survey": "s1"})["labels"])
    assert np.array_equal(labels, before[100:])
    reopened = Project.open(root)                       # rewritten and saved
    assert np.allclose(reopened.survey("s1").dataset.cloud.xyz, full[100:])
    assert np.array_equal(reopened.survey("s1").labels.cloud_labels, before[100:])

    with pytest.raises(RpcError) as err:
        pipe.call("cloud.delete", {"survey": "s1", "indices": put_array(root, "d",
                                                         np.array([500], np.int32))})
    assert err.value.data["code"] == "cloud.bad_indices"
    with pytest.raises(RpcError) as err:
        pipe.call("cloud.delete", {"survey": "s1", "cloud": "dense",
                                   "indices": put_array(root, "d", gone)})
    assert err.value.code == protocol.INVALID_PARAMS


def test_cloud_stats_need_the_colmap_model(tmp_path):
    """Also a regression: pycolmap, first loaded on a worker thread, hung
    the Engine on Windows. The protocol's copy of stdin had reused the
    startup stdout's handle value, which pycolmap's runtime probes as it
    loads. A fresh Engine, so nothing has loaded pycolmap yet."""
    root = make_project(tmp_path / "p.clproj")             # no reconstruction/sparse
    client = Pipe(tmp_path / "stderr.log")
    try:
        client.call("project.open", {"root": str(root)})
        with pytest.raises(RpcError) as err:
            client.call("cloud.stats", {"survey": "s1"}, timeout=60)
        assert err.value.data["code"] == "cloud.no_model"
    finally:
        assert client.close() == 0


def _fake_stats(monkeypatch, views, errors):
    from cloudlabeller.photogrammetry import extract

    calls = []
    monkeypatch.setattr(extract, "load_best_model", lambda root: calls.append(root) or "rec")
    monkeypatch.setattr(extract, "sparse_point_stats",
                        lambda rec, xyz: (views.copy(), errors.copy()))
    return calls


def test_cloud_stats_refuse_a_mismatched_model(inproc, tmp_path, monkeypatch):
    """Regression for the Qt check: under 90% of the points in the saved
    model means it is from another solve, so its stats can't clean this cloud."""
    root = make_project(tmp_path / "p.clproj")
    views = np.full(600, 4, np.int32)
    views[:120] = 0                                       # 80% matched
    _fake_stats(monkeypatch, views, np.ones(600))
    inproc.call("project.open", {"root": str(root)})
    with pytest.raises(RpcError) as err:
        inproc.call("cloud.stats", {"survey": "s1"})
    assert err.value.data["code"] == "cloud.model_mismatch"
    assert err.value.data["params"]["matched"] == pytest.approx(0.8)
    with pytest.raises(RpcError) as err:
        inproc.call("cloud.clean", {"survey": "s1", "min_images": 3, "max_error": 2.0})
    assert err.value.data["code"] == "cloud.model_mismatch"
    assert inproc.call("project.state")["surveys"][0]["sparse"] == 600


def test_clean_preview_then_clean(inproc, tmp_path, monkeypatch):
    root = make_project(tmp_path / "p.clproj")
    views = np.full(600, 5, np.int32)
    views[:100] = 2                                       # too few photos
    errors = np.ones(600)
    errors[599] = 9.0                                     # too large an error
    calls = _fake_stats(monkeypatch, views, errors)
    inproc.call("project.open", {"root": str(root)})

    stats = inproc.call("cloud.stats", {"survey": "s1"})
    assert stats["n_points"] == 600 and stats["matched"] == 1.0
    assert np.array_equal(get_array(root, stats["views"]), views)
    assert get_array(root, stats["errors"]).dtype == np.float32

    preview = inproc.call("cloud.clean_preview", {"survey": "s1", "min_images": 3, "max_error": 2.0})
    assert preview["keep"] == 499 and preview["total"] == 600
    removed = get_array(root, preview["removed"])
    assert preview["removed"]["dtype"] == "uint32"
    assert removed.tolist() == list(range(100)) + [599]
    assert len(calls) == 1                                # cached for this cloud
    assert inproc.call("project.state")["surveys"][0]["sparse"] == 600  # a preview changes nothing

    out = inproc.call("cloud.clean", {"survey": "s1", "min_images": 3, "max_error": 2.0})
    assert out == {"removed": 101, "n_points": 499}
    assert Project.open(root).survey("s1").dataset.cloud.n_points == 499
    with pytest.raises(RpcError) as err:
        inproc.call("cloud.clean", {"survey": "s1", "min_images": 3.5, "max_error": 2.0})
    assert err.value.code == protocol.INVALID_PARAMS


def test_cameras_list_with_up_and_crs_positions(pipe, tmp_path):
    root = make_project(tmp_path / "p.clproj")
    pipe.call("project.open", {"root": str(root)})
    out = pipe.call("cameras.list", {"survey": "s1"})
    assert [c["name"] for c in out["cameras"]] == ["a.png", "b.png"]
    a, b = out["cameras"]
    assert (a["width"], a["height"]) == (W, H)
    assert np.allclose(a["K"], [[80, 0, W / 2], [0, 80, H / 2], [0, 0, 1]])
    assert np.allclose(b["R"], np.eye(3)) and np.allclose(b["t"], [0.2, 0, 0])
    assert np.allclose(b["position"], [-0.2, 0, 0])
    assert b["position_crs"] == b["position"]
    assert np.allclose(out["up"], [0, -1, 0])            # -mean(R[1])

    offset = [500000.0, 6600000.0, 0.0]
    _georeference(root, {"epsg": UTM_SIRGAS_22S, "name": "SIRGAS 2000 / UTM zone 22S",
                         "orthometric": False, "offset": offset})
    pipe.call("project.open", {"root": str(root)})
    out = pipe.call("cameras.list", {"survey": "s1"})
    assert out["up"] == [0.0, 0.0, 1.0]
    b = out["cameras"][1]
    assert np.allclose(b["position_crs"], np.array(b["position"]) + offset)


def test_image_info(pipe, tmp_path):
    root = make_project(tmp_path / "p.clproj")
    pipe.call("project.open", {"root": str(root)})
    assert pipe.call("image.info", {"survey": "s1", "name": "b.png"}) == {
        "path": "surveys/s1/images/b.png", "width": W, "height": H}
    with pytest.raises(RpcError) as err:
        pipe.call("image.info", {"survey": "s1", "name": "zzz.png"})
    assert err.value.data["code"] == "image.not_found"


# -- Job facts and estimates ----------------------------------------------------
def test_jobs_facts(pipe, tmp_path):
    from cloudlabeller.photogrammetry.options import CAMERA_MODELS, MATCHERS

    root = make_project(tmp_path / "p.clproj")
    pipe.call("project.open", {"root": str(root)})
    sfm = pipe.call("jobs.facts", {"kind": "sfm", "survey": "s1"})
    assert sfm["gps_images"] == 0                         # PNGs without EXIF
    assert sfm["camera_models"] == list(CAMERA_MODELS)
    assert sfm["matchers"] == list(MATCHERS)
    assert isinstance(sfm["gpu"], bool)
    assert isinstance(pipe.call("jobs.facts", {"kind": "mvs", "survey": "s1"})["colmap"], bool)
    mesh = pipe.call("jobs.facts", {"kind": "mesh", "survey": "s1"})
    assert mesh["suggested_depth"] is None and mesh["delaunay"] is False
    assert mesh["ram_gb"] is None or mesh["ram_gb"] > 0

    assert pipe.call("jobs.facts", {"kind": "train"}) == {
        "user_images": 0, "auto_images": 0, "ml_images": 0}
    mask = np.zeros((H, W), np.int32)
    pipe.call("image.save_mask", {"survey": "s1", "name": "a.png", "mask": put_array(root, "a", mask)})
    pipe.call("transfer.images_to_cloud", {"survey": "s1"})
    pipe.call("transfer.cloud_to_images", {"survey": "s1"})                 # b.png becomes auto
    assert pipe.call("jobs.facts", {"kind": "train"}) == {
        "user_images": 1, "auto_images": 1, "ml_images": 0}
    with pytest.raises(RpcError) as err:
        pipe.call("jobs.facts", {"kind": "predict"})
    assert err.value.code == protocol.INVALID_PARAMS


def test_mesh_estimate_and_facts_with_a_dense_cloud(pipe, tmp_path):
    root = make_project(tmp_path / "p.clproj")
    pipe.call("project.open", {"root": str(root)})
    with pytest.raises(RpcError) as err:
        pipe.call("mesh.estimate", {"survey": "s1"})
    assert err.value.data["code"] == "job.mesh.no_dense"

    _with_dense(root)
    pipe.call("project.open", {"root": str(root)})
    from cloudlabeller.photogrammetry.meshing import estimate_mesh_cost, suggest_poisson_depth

    suggested = suggest_poisson_depth(Project.open(root).survey("s1").dataset.dense_cloud)
    assert pipe.call("jobs.facts", {"kind": "mesh", "survey": "s1"})["suggested_depth"] == suggested
    out = pipe.call("mesh.estimate", {"survey": "s1", "method": "poisson"})
    assert out["suggested_depth"] == suggested
    expected = estimate_mesh_cost(400, suggested, suggested)
    assert out["minutes"] == pytest.approx(expected["minutes"])
    deeper = pipe.call("mesh.estimate", {"survey": "s1", "method": "poisson", "depth": suggested + 1})
    assert deeper["ram_gb"] > out["ram_gb"]
    delaunay = pipe.call("mesh.estimate", {"survey": "s1", "method": "delaunay"})
    assert delaunay["vertices"] == 400.0
    with pytest.raises(RpcError) as err:
        pipe.call("mesh.estimate", {"survey": "s1", "method": "marching"})
    assert err.value.code == protocol.INVALID_PARAMS


def test_ingest_conflicts(pipe, tmp_path):
    from PIL import Image

    root = make_project(tmp_path / "p.clproj")
    pipe.call("project.open", {"root": str(root)})
    folder = tmp_path / "new"
    folder.mkdir()
    for name in ("a.png", "c.png"):
        Image.new("RGB", (4, 4)).save(folder / name)
    assert pipe.call("ingest.conflicts", {"survey": "s1", "folder": str(folder)}) == {"conflicts": ["a.png"]}
    with pytest.raises(RpcError) as err:
        pipe.call("ingest.conflicts", {"survey": "s1", "folder": str(tmp_path / "nowhere")})
    assert err.value.data["code"] == "job.ingest.no_folder"


# -- refusals the Qt dialogs made ---------------------------------------------------
def test_georef_stores_the_alignment_and_keeps_the_files(pipe, tmp_path):
    """ADR-0005: aligning again replaces the stored Alignment; no product is
    rewritten (0.3.0 transformed them in place and refused a second time)."""
    root = make_project(tmp_path / "p.clproj")
    _georeference(root)
    cloud = root / "surveys" / "s1" / "products" / "sfm-1" / "cloud.npz"
    before = cloud.read_bytes()
    pipe.call("project.open", {"root": str(root)})
    plan = pipe.call("jobs.plan", {"kind": "georef", "options": {"survey": "s1"}})
    i = plan["args"].index("--origin")                # into the Project's own frame
    assert [float(v) for v in plan["args"][i + 1:i + 4]] == list(ORIGIN)
    fit = {"s": 2.0, "R": np.eye(3).tolist(), "t": [10.0, 0.0, 0.0],
           "origin_lla": list(ORIGIN), "n_gps": 2, "n_inliers": 2, "fit_rms_m": 0.5}
    done = pipe.call("jobs.finish", {"plan": plan["plan"], "exit": 0,
                                     "results": {"RESULT": json.dumps(fit)}})
    assert done["ok"], done
    assert cloud.read_bytes() == before
    xyz = Project.open(root).survey("s1").dataset.cloud.xyz
    view = pipe.call("cloud.points", {"survey": "s1", "cloud": "sparse"})
    shown = np.fromfile(root / view["xyz"]["path"], "<f4").reshape(-1, 3)
    assert np.allclose(shown, 2.0 * xyz + [10.0, 0.0, 0.0], atol=1e-4)
    a = Project.open(root).survey("s1").active.alignment
    assert a["source"] == "gps" and a["s"] == 2.0 and a["rms"] == 0.5
    assert pipe.call("jobs.list") == {"jobs": []}


def test_train_refuses_to_continue_a_model_with_other_classes(pipe, tmp_path):
    """Regression: continuing a model trained on 3 classes in a 2-class
    Project (the Qt train dialog refused; the Engine planned it)."""
    root = make_project(tmp_path / "p.clproj")
    _fake_model(root, "m3", n_classes=3)
    _fake_model(root, "m2", n_classes=2)
    pipe.call("project.open", {"root": str(root)})
    with pytest.raises(RpcError) as err:
        pipe.call("jobs.plan", {"kind": "train", "options": {"name": "m3", "resume": True}})
    assert err.value.data["code"] == "job.train.class_mismatch"
    assert err.value.data["params"] == {"model": "m3", "model_classes": 3,
                                        "project_classes": 2}
    plan = pipe.call("jobs.plan", {"kind": "train", "options": {"name": "m2",
                                                                 "resume": True}})
    pipe.call("jobs.finish", {"plan": plan["plan"], "exit": None})


# -- models -------------------------------------------------------------------------
def test_models_delete(pipe, tmp_path):
    root = make_project(tmp_path / "p.clproj")
    _fake_model(root, "m1")
    _fake_model(root, "m2")
    pipe.call("project.open", {"root": str(root)})
    plan = pipe.call("jobs.plan", {"kind": "predict", "options": {"model": "m1"}})
    with pytest.raises(RpcError) as err:
        pipe.call("models.delete", {"name": "m2"})
    assert err.value.data["code"] == "models.in_use"
    pipe.call("jobs.finish", {"plan": plan["plan"], "exit": None})

    out = pipe.call("models.delete", {"name": "m2"})
    assert [m["name"] for m in out["models"]] == ["m1"]
    assert not (root / "ml" / "models" / "m2").exists()
    with pytest.raises(RpcError) as err:
        pipe.call("models.delete", {"name": "m2"})
    assert err.value.data["code"] == "models.not_found"


# -- the download Jobs ------------------------------------------------------------
def test_download_jobs_plan_and_finish(pipe, tmp_path):
    root = make_project(tmp_path / "p.clproj")
    pipe.call("project.open", {"root": str(root)})
    plan = pipe.call("jobs.plan", {"kind": "geoid", "options": {"epsg": UTM_SIRGAS_22S}})
    assert plan["group"] == "download" and plan["gpu"] is False and plan["stage"] is None
    assert plan["module"] == "cloudlabeller.photogrammetry.geoid_cli"
    assert plan["args"] == ["--epsg", str(UTM_SIRGAS_22S)]
    sfm = pipe.call("jobs.plan", {"kind": "sfm", "options": {"survey": "s1"}})        # downloads don't block it
    pipe.call("jobs.finish", {"plan": sfm["plan"], "exit": None})
    done = pipe.call("jobs.finish", {"plan": plan["plan"], "exit": 0,
                                     "results": {"RESULT": json.dumps({"epsg": 31982})}})
    assert done["ok"] and done["result"] == {"epsg": 31982}
    with pytest.raises(RpcError) as err:
        pipe.call("jobs.plan", {"kind": "geoid"})
    assert err.value.code == protocol.INVALID_PARAMS

    if sys.platform != "win32":
        with pytest.raises(RpcError) as err:
            pipe.call("jobs.plan", {"kind": "colmap"})
        assert err.value.data["code"] == "job.colmap.unsupported"
        return
    plan = pipe.call("jobs.plan", {"kind": "colmap", "options": {"variant": "cpu"}})
    assert plan["args"][:2] == ["--variant", "nocuda"]
    assert plan["stop_file"].endswith("colmap_download.stop")
    exe = str(tmp_path / "colmap" / "bin" / "colmap.exe")
    done = pipe.call("jobs.finish", {"plan": plan["plan"], "exit": 0, "results": {
        "RESULT": json.dumps({"path": exe, "variant": "nocuda"})}})
    assert done["result"] == {"path": exe, "variant": "cpu"}
    with pytest.raises(RpcError) as err:
        pipe.call("jobs.plan", {"kind": "colmap", "options": {"variant": "opencl"}})
    assert err.value.code == protocol.INVALID_PARAMS


# -- CRS --------------------------------------------------------------------------
def test_crs_search_is_a_word_contains_match(pipe):
    out = pipe.call("crs.search", {"query": "sirgas 22s"})
    assert {"epsg": UTM_SIRGAS_22S, "name": "SIRGAS 2000 / UTM zone 22S"} in out["results"]
    assert all("sirgas" in r["name"].lower() and "22s" in r["name"].lower()
               for r in out["results"])
    by_code = pipe.call("crs.search", {"query": str(UTM_SIRGAS_22S)})["results"]
    assert by_code[0]["epsg"] == UTM_SIRGAS_22S
    assert len(pipe.call("crs.search", {"query": "", "limit": 7})["results"]) == 7
    assert pipe.call("crs.search", {"query": "no such crs anywhere"}) == {"results": []}


def test_crs_suggest_and_frame(pipe, tmp_path):
    root = make_project(tmp_path / "p.clproj")
    pipe.call("project.open", {"root": str(root)})
    with pytest.raises(RpcError) as err:
        pipe.call("crs.suggest")
    assert err.value.data["code"] == "crs.not_georeferenced"
    assert pipe.call("crs.frame")["short"] == "no CRS"

    _georeference(root)
    pipe.call("project.open", {"root": str(root)})
    out = pipe.call("crs.suggest")
    assert out == {"epsg": UTM_SIRGAS_22S, "name": "SIRGAS 2000 / UTM zone 22S",
                   "origin_lla": list(ORIGIN), "exact": True}
    assert pipe.call("crs.frame")["short"] == "local ENU"


def test_heal_enu_origin_writes_back_a_recovered_origin(monkeypatch):
    from cloudlabeller.photogrammetry import crs

    recovered = crs.EnuOrigin((1.0, 2.0, 3.0), True)
    monkeypatch.setattr(crs, "resolve_enu_origin", lambda geo, ws, images=None: recovered)
    geo = {"frame": "ENU", "origin_lla": [9.0, 9.0, 9.0]}       # legacy: the mean GPS
    origin, healed = crs.heal_enu_origin(geo, "ws")
    assert healed and origin is recovered
    assert geo["origin_lla"] == [1.0, 2.0, 3.0]
    assert geo["origin_convention"] == crs.ORIGIN_CONVENTION_FIRST_GPS
    assert crs.heal_enu_origin(geo, "ws")[1] is False


# -- export -----------------------------------------------------------------------
def test_export_settings_roundtrip(pipe, tmp_path):
    root = make_project(tmp_path / "p.clproj")
    pipe.call("project.open", {"root": str(root)})
    assert pipe.call("export.settings") == {
        "frame": "local", "crs": None, "export_mode": None, "export_epsg": None,
        "export_orthometric": False}
    with pytest.raises(RpcError) as err:
        pipe.call("export.set_settings", {"export_mode": "projected", "export_epsg": 1})
    assert err.value.data["code"] == "export.not_georeferenced"

    _georeference(root)
    pipe.call("project.open", {"root": str(root)})
    with pytest.raises(RpcError) as err:
        pipe.call("export.set_settings", {"export_mode": "projected"})
    assert err.value.code == protocol.INVALID_PARAMS
    out = pipe.call("export.set_settings", {"export_mode": "projected",
                                            "export_epsg": UTM_SIRGAS_22S,
                                            "export_orthometric": True})
    assert out["frame"] == "enu" and out["export_epsg"] == UTM_SIRGAS_22S
    geo = Project.open(root).settings["georeferenced"]
    assert (geo["export_mode"], geo["export_epsg"], geo["export_orthometric"]) == (
        "projected", UTM_SIRGAS_22S, True)
    out = pipe.call("export.set_settings", {"export_mode": "local"})
    assert out["export_mode"] == "local" and out["export_epsg"] == UTM_SIRGAS_22S


def test_export_cloud_and_mesh_in_the_local_frame(pipe, tmp_path):
    from cloudlabeller.io.geometry import save_mesh_npz

    root = make_project(tmp_path / "p.clproj")
    pipe.call("project.open", {"root": str(root)})
    with pytest.raises(RpcError) as err:
        pipe.call("export.mesh", {"survey": "s1", "path": str(tmp_path / "m"), "format": "ply"})
    assert err.value.data["code"] == "mesh.none"

    out = pipe.call("export.cloud", {"survey": "s1", "path": str(tmp_path / "cloud"), "format": "csv"})
    assert out == {"path": str(tmp_path / "cloud.csv"), "cloud": "sparse",
                   "n_points": 600, "crs": None}
    assert len((tmp_path / "cloud.csv").read_text().splitlines()) == 601
    assert not (tmp_path / "cloud.prj").exists()
    pipe.call("export.cloud", {"survey": "s1", "path": str(tmp_path / "cloud.las"), "format": "las"})
    assert (tmp_path / "cloud.las").stat().st_size > 0
    with pytest.raises(RpcError) as err:
        pipe.call("export.cloud", {"survey": "s1", "path": str(tmp_path / "no" / "c.las"), "format": "las"})
    assert err.value.data["code"] == "export.no_folder"
    with pytest.raises(RpcError) as err:
        pipe.call("export.cloud", {"survey": "s1", "path": str(tmp_path / "c.xyz"), "format": "xyz"})
    assert err.value.code == protocol.INVALID_PARAMS

    mesh = Mesh(vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32),
                faces=np.array([[0, 1, 2]], np.int32))
    save_mesh_npz(mesh, product_dir(Project.open(root), "mesh") / "mesh.npz")
    pipe.call("project.open", {"root": str(root)})
    out = pipe.call("export.mesh", {"survey": "s1", "path": str(tmp_path / "mesh.obj"), "format": "obj"})
    assert out == {"path": str(tmp_path / "mesh.obj"), "vertices": 3, "faces": 1,
                   "crs": None}
    assert (tmp_path / "mesh.obj").exists()


def test_export_cloud_into_a_projected_crs(pipe, tmp_path):
    import laspy

    from cloudlabeller.photogrammetry.crs import enu_to_crs_transform

    root = make_project(tmp_path / "p.clproj")
    _georeference(root)
    pipe.call("project.open", {"root": str(root)})
    out = pipe.call("export.cloud", {"survey": "s1", "path": str(tmp_path / "utm.las"), "format": "las",
                                     "crs": {"mode": "projected", "epsg": UTM_SIRGAS_22S}})
    assert out["crs"].startswith(f"EPSG:{UTM_SIRGAS_22S}")
    las = laspy.read(tmp_path / "utm.las")
    fn, _ = enu_to_crs_transform(ORIGIN, UTM_SIRGAS_22S)
    expected = fn(Project.open(root).survey("s1").dataset.cloud.xyz.astype(np.float64))
    assert np.allclose(np.column_stack([las.x, las.y, las.z]), expected, atol=0.01)
    assert pipe.call("export.settings")["export_mode"] == "projected"   # kept

    out = pipe.call("export.cloud", {"survey": "s1", "path": str(tmp_path / "local.csv"), "format": "csv",
                                     "crs": {"mode": "local"}})
    assert out["crs"] is None


def test_export_from_a_reprojected_project_adds_the_offset(pipe, tmp_path):
    root = make_project(tmp_path / "p.clproj")
    offset = [500000.0, 6600000.0, 0.0]
    _georeference(root, {"epsg": UTM_SIRGAS_22S, "name": "SIRGAS 2000 / UTM zone 22S",
                         "orthometric": False, "offset": offset})
    pipe.call("project.open", {"root": str(root)})
    out = pipe.call("export.cloud", {"survey": "s1", "path": str(tmp_path / "c.csv"), "format": "csv",
                                     "crs": {"mode": "local"}})          # ignored
    assert out["crs"].startswith(f"EPSG:{UTM_SIRGAS_22S}")
    assert (tmp_path / "c.prj").exists()
    first = (tmp_path / "c.csv").read_text().splitlines()[1].split(",")
    xyz = Project.open(root).survey("s1").dataset.cloud.xyz[0].astype(np.float64)
    assert float(first[0]) == pytest.approx(xyz[0] + offset[0], abs=1e-3)
    assert float(first[1]) == pytest.approx(xyz[1] + offset[1], abs=1e-3)


def test_export_asks_for_the_geoid_job_first(inproc, tmp_path, monkeypatch):
    from cloudlabeller.photogrammetry import crs

    monkeypatch.setattr(crs, "geoid_ready", lambda epsg: False)
    downloads = []
    monkeypatch.setattr(crs, "download_geoid_grids", downloads.append)
    root = make_project(tmp_path / "p.clproj")
    _georeference(root)
    inproc.call("project.open", {"root": str(root)})
    with pytest.raises(RpcError) as err:
        inproc.call("export.cloud", {"survey": "s1", "path": str(tmp_path / "c.las"), "format": "las",
                                     "crs": {"mode": "projected", "epsg": UTM_SIRGAS_22S,
                                             "orthometric": True}})
    assert err.value.data["code"] == "export.needs_geoid"
    assert err.value.data["params"] == {"epsg": UTM_SIRGAS_22S}
    assert downloads == [] and not (tmp_path / "c.las").exists()
