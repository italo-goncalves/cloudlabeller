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

"""Layout v2: Surveys, Reconstructions, the stored Alignment, and moving a
0.3.0 Project into it (0.4.0 step 1, ADR-0005)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cloudlabeller.core.dataset import Camera, ImageRecord, Mesh, PointCloud
from cloudlabeller.core.migrate import JOURNAL, V1_BACKUP, migrate_if_needed
from cloudlabeller.core.project import Project, ProjectFormatError
from cloudlabeller.core.reconstruction import apply_camera, apply_points, compose

W, H = 64, 48


def _camera(tx: float) -> Camera:
    K = np.array([[80.0, 0, W / 2], [0, 80.0, H / 2], [0, 0, 1]])
    return Camera(K=K, R=np.eye(3), t=np.array([tx, 0.0, 0.0]), width=W, height=H)


def _v1_project(root: Path, *, georeferenced: bool = False, dense: bool = True,
                mesh: bool = True) -> Path:
    """A Project as 0.3.0 left it: one image store, one reconstruction."""
    from PIL import Image

    from cloudlabeller.io.geometry import save_cloud_npz, save_mesh_npz
    from cloudlabeller.photogrammetry.pipeline import ReconstructResult, save_result

    (root / "images").mkdir(parents=True)
    for name in ("a.png", "b.png"):
        Image.new("RGB", (W, H), (90, 90, 90)).save(root / "images" / name)
    rec = root / "reconstruction"
    xyz = (np.random.default_rng(0).uniform(-1, 1, (50, 3)) + [0, 0, 5]).astype(np.float32)
    images = [ImageRecord(0, root / "images" / "a.png", _camera(0.0)),
              ImageRecord(1, root / "images" / "b.png", _camera(0.2))]
    save_result(ReconstructResult(PointCloud(xyz=xyz), images), rec)
    (rec / "sparse" / "0").mkdir(parents=True)
    (rec / "sparse" / "0" / "points3D.bin").write_bytes(b"model")
    (rec / "pmap").mkdir()
    (rec / "pmap" / "a.png.12345678.npz").write_bytes(b"cache")
    labels = root / "labels"
    (labels / "images").mkdir(parents=True)
    np.save(labels / "cloud_labels.npy", np.arange(50, dtype=np.int32) % 2)
    np.save(labels / "images" / "a.png.npy", np.zeros((H, W), np.int32))
    if dense:
        (rec / "dense" / "stereo").mkdir(parents=True)
        save_cloud_npz(PointCloud(xyz=xyz[:20]), rec / "dense.npz")
        np.save(labels / "dense_cloud_labels.npy", np.ones(20, np.int32))
    if mesh:
        (root / "products").mkdir()
        save_mesh_npz(Mesh(vertices=xyz[:3], faces=np.array([[0, 1, 2]], np.int32)),
                      root / "products" / "mesh.npz")
        np.save(labels / "mesh_labels.npy", np.ones(3, np.int32))
    (root / "ml" / "predictions").mkdir(parents=True)
    np.save(root / "ml" / "predictions" / "b.png.npy", np.zeros((4, 4), np.int32))
    settings: dict = {"reconstruction_outdated": False, "source_resolution": [W, H],
                      "model_spec": {"resolution_divisor": 8}}
    if georeferenced:
        settings["georeferenced"] = {"frame": "ENU", "origin_lla": [-30.0, -53.0, 100.0],
                                     "origin_convention": "first_gps_image",
                                     "scale_m_per_unit": 3.5, "n_gps": 2}
        (rec / "sparse_prealigned" / "0").mkdir(parents=True)
    manifest = {"format_version": 1,
                "schema": {"classes": [{"id": 0, "name": "rock", "color": "#ff0000"},
                                       {"id": 1, "name": "soil", "color": "#00ff00"}]},
                "settings": settings, "images": ["a.png", "b.png"],
                "image_status": {"a.png": "user", "b.png": "ml"}}
    (root / "project.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


# -- migration ---------------------------------------------------------------------
def test_a_030_project_opens_as_one_survey(tmp_path):
    root = _v1_project(tmp_path / "p")
    proj = Project.open(root)
    assert list(proj.surveys) == ["s1"]
    sv = proj.survey("s1")
    assert [r.name for r in sv.dataset.images] == ["a.png", "b.png"]
    assert sv.dataset.cloud.n_points == 50 and sv.dataset.dense_cloud.n_points == 20
    assert len(sv.dataset.mesh.vertices) == 3
    assert list(sv.labels.cloud_labels[:4]) == [0, 1, 0, 1]
    assert (sv.labels.dense_cloud_labels == 1).all() and (sv.labels.mesh_labels == 1).all()
    assert sv.labels.status_of("a.png") == "user" and "a.png" in sv.labels.image_masks
    assert sv.prediction_mask("b.png") is not None
    rec = sv.active
    assert rec.id == "r1" and rec.products == {"sfm": "sfm-1", "mvs": "mvs-1", "mesh": "mesh-1"}
    assert rec.alignment is None and not sv.aligned
    assert sv.settings == {"reconstruction_outdated": False, "source_resolution": [W, H]}
    assert proj.settings == {"model_spec": {"resolution_divisor": 8}}

    s1 = root / "surveys" / "s1"
    assert (s1 / "products" / "sfm-1" / "sparse" / "0" / "points3D.bin").exists()
    assert (s1 / "products" / "mvs-1" / "dense" / "stereo").is_dir()
    assert (s1 / "cache" / "pmap" / "a.png.12345678.npz").exists()
    for gone in ("images", "reconstruction", "labels", "products", "ml/predictions"):
        assert not (root / gone).exists(), gone
    assert json.loads((root / V1_BACKUP).read_text())["format_version"] == 1
    assert json.loads((root / "project.json").read_text())["format_version"] == 2
    assert not (root / JOURNAL).exists()
    assert not migrate_if_needed(root)                     # once only


def test_a_project_georeferenced_in_place_keeps_its_files_and_frame(tmp_path):
    root = _v1_project(tmp_path / "p", georeferenced=True)
    before = (root / "reconstruction" / "cloud.npz").read_bytes()
    proj = Project.open(root)
    sv = proj.survey("s1")
    a = sv.active.alignment
    assert a["source"] == "legacy" and a["s"] == 1.0 and a["scale_m_per_unit"] == 3.5
    assert sv.aligned and proj.frame["origin_lla"] == [-30.0, -53.0, 100.0]
    assert (root / "surveys" / "s1" / "products" / "sfm-1" / "cloud.npz").read_bytes() == before
    assert (root / "surveys" / "s1" / "products" / "sfm-1" / "sparse_prealigned").is_dir()
    assert np.array_equal(sv.active.frame_points(sv.dataset.cloud.xyz), sv.dataset.cloud.xyz)


def test_a_project_without_dense_or_mesh(tmp_path):
    root = _v1_project(tmp_path / "p", dense=False, mesh=False)
    sv = Project.open(root).survey("s1")
    assert sv.active.products == {"sfm": "sfm-1", "mvs": None, "mesh": None}
    assert sv.dataset.dense_cloud is None and sv.dataset.mesh is None


def test_an_interrupted_migration_resumes(tmp_path, monkeypatch):
    from cloudlabeller.core import migrate

    root = _v1_project(tmp_path / "p")
    real = migrate._rename
    calls = []

    def flaky(r, src, dst):
        calls.append(src)
        if len(calls) == 4:
            raise OSError("the disk went away")
        real(r, src, dst)

    monkeypatch.setattr(migrate, "_rename", flaky)
    with pytest.raises(OSError):
        Project.open(root)
    assert (root / JOURNAL).exists()
    assert json.loads((root / "project.json").read_text())["format_version"] == 1
    assert (root / "surveys" / "s1" / "images").is_dir()     # some moves done

    monkeypatch.setattr(migrate, "_rename", real)
    sv = Project.open(root).survey("s1")
    assert sv.dataset.cloud.n_points == 50 and sv.dataset.dense_cloud.n_points == 20
    assert list(sv.labels.cloud_labels[:2]) == [0, 1]
    assert not (root / JOURNAL).exists()


def test_a_newer_format_is_refused(tmp_path):
    root = tmp_path / "p"
    Project.create(root)
    data = json.loads((root / "project.json").read_text())
    data["format_version"] = 3
    (root / "project.json").write_text(json.dumps(data))
    with pytest.raises(ProjectFormatError):
        Project.open(root)


# -- products are not edited -------------------------------------------------------
def test_deleting_points_keeps_the_product(tmp_path):
    root = _v1_project(tmp_path / "p")
    sv = Project.open(root).survey("s1")
    product = sv.sfm_dir / "cloud.npz"
    before = product.read_bytes()
    keep = np.ones(50, bool)
    keep[:10] = False
    assert sv.filter_sparse_cloud(keep) == 10
    keep2 = np.ones(40, bool)
    keep2[0] = False                                   # index 0 now is product point 10
    assert sv.filter_sparse_cloud(keep2) == 1
    sv.save()
    assert product.read_bytes() == before
    again = Project.open(root).survey("s1")
    assert again.dataset.cloud.n_points == 39
    xyz = np.load(product)["xyz"]
    assert np.array_equal(again.dataset.cloud.xyz, xyz[11:])
    assert len(again.labels.cloud_labels) == 39


# -- the frame -----------------------------------------------------------------------
def test_alignment_helpers_compose_like_the_transforms():
    rng = np.random.default_rng(3)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.linalg.det(q))
    inner = {"s": 2.0, "R": q.tolist(), "t": [1.0, 2.0, 3.0], "source": "gps", "rms": 0.1}
    outer = {"s": 0.5, "R": np.eye(3)[[1, 0, 2]].tolist(), "t": [10.0, 0.0, 0.0]}
    xyz = rng.normal(size=(5, 3))
    both = compose(outer, inner)
    assert both["source"] == "gps" and both["rms"] == 0.1
    assert np.allclose(apply_points(both, xyz), apply_points(outer, apply_points(inner, xyz)),
                       atol=1e-5)
    cam = Camera(K=np.eye(3), R=np.eye(3), t=np.array([0.0, 0.0, 4.0]), width=2, height=2)
    moved = apply_camera(inner, cam)
    assert np.allclose(moved.position, apply_points(inner, cam.position[None])[0], atol=1e-5)


def _engine_with(root: Path):
    import cloudlabeller.engine.methods  # noqa: F401  (registers the methods)
    from cloudlabeller.engine.server import Engine

    engine = Engine(lambda data: None)
    engine.project = Project.open(root)
    return engine


def test_reprojecting_composes_into_every_alignment(tmp_path):
    from cloudlabeller.engine.jobs import KINDS

    root = _v1_project(tmp_path / "p", georeferenced=True)
    engine = _engine_with(root)
    proj = engine.project
    sv = proj.survey("s1")
    before = (sv.sfm_dir / "cloud.npz").read_bytes()
    plan = KINDS["reproject"].prepare(engine, proj, {"epsg": 32722})
    given = json.loads(Path(plan.args[0]).read_text())
    assert given["frame"]["frame"] == "ENU" and len(given["samples"]) == 9

    move = {"s": 1.0001, "R": np.eye(3).tolist(), "t": [5.0, 6.0, 7.0],
            "epsg": 32722, "name": "UTM 22S", "orthometric": False,
            "offset": [500000.0, 6600000.0, 0.0]}
    KINDS["reproject"].apply(engine, proj, {"RESULT": json.dumps(move)}, plan)
    assert proj.frame["crs"]["epsg"] == 32722
    framed = sv.active.frame_points(sv.dataset.cloud.xyz)
    assert np.allclose(framed, 1.0001 * sv.dataset.cloud.xyz + [5, 6, 7], atol=1e-4)
    assert (sv.sfm_dir / "cloud.npz").read_bytes() == before
    reopened = Project.open(root)
    assert reopened.survey("s1").active.alignment["s"] == pytest.approx(1.0001)


def test_a_new_sfm_starts_unaligned_and_georef_goes_into_a_crs_frame(tmp_path):
    from cloudlabeller.engine.jobs import KINDS

    root = _v1_project(tmp_path / "p", georeferenced=True)
    engine = _engine_with(root)
    proj = engine.project
    proj.frame["crs"] = {"epsg": 32722, "name": "UTM 22S", "orthometric": False,
                         "offset": [700000.0, 6600000.0, 0.0]}
    plan = KINDS["sfm"].prepare(engine, proj, {"survey": "s1"})
    assert plan.args[1].endswith("sfm-1")              # step 1: a re-run replaces it
    out = KINDS["sfm"].apply(engine, proj, {}, plan)
    assert out["georeferencing_cleared"] and not proj.survey("s1").aligned

    plan = KINDS["georef"].prepare(engine, proj, {"survey": "s1"})
    i = plan.args.index("--origin")
    assert [float(v) for v in plan.args[i + 1:i + 4]] == [-30.0, -53.0, 100.0]
    fit = {"s": 1.0, "R": np.eye(3).tolist(), "t": [0.0, 0.0, 0.0],
           "origin_lla": [-30.0, -53.0, 100.0], "n_gps": 2, "n_inliers": 2}
    KINDS["georef"].apply(engine, proj, {"RESULT": json.dumps(fit)}, plan)
    sv = proj.survey("s1")
    assert sv.aligned
    # ENU (about the origin) carried on into UTM minus the offset.
    from cloudlabeller.photogrammetry.crs import enu_to_crs_transform

    exact, _ = enu_to_crs_transform([-30.0, -53.0, 100.0], 32722)
    xyz = sv.dataset.cloud.xyz.astype(np.float64)
    want = exact(xyz) - [700000.0, 6600000.0, 0.0]
    assert np.allclose(sv.active.frame_points(xyz), want, atol=0.05)


def test_methods_name_their_survey(tmp_path):
    from cloudlabeller.engine.methods import _survey
    from cloudlabeller.engine.protocol import InvalidParams, Problem

    root = _v1_project(tmp_path / "p")
    engine = _engine_with(root)
    assert _survey(engine, {"survey": "s1"}).id == "s1"
    with pytest.raises(InvalidParams):
        _survey(engine, {})
    with pytest.raises(Problem) as err:
        _survey(engine, {"survey": "s9"})
    assert err.value.code == "survey.not_found"
