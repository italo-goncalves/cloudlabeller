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

"""Reprojection to a chosen CRS: frame model, CLI, export and UI plumbing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from test_georef_native import ORIGIN, _centers, _make_project

from cloudlabeller.photogrammetry import crs as crsmod

UTM_SIRGAS_22S = 31982
UTM_WGS84_22S = 32722


# -- frame helpers ------------------------------------------------------------
def test_choose_offset_km_rounded_xy_only():
    coords = np.array([[480_819.7, 6_675_857.5, 15.0],
                       [480_920.1, 6_675_930.2, 40.0]])
    off = crsmod.choose_offset(coords)
    assert tuple(off) == (480_000.0, 6_675_000.0, 0.0)
    assert np.all(off[:2] % 1000 == 0)


def test_frame_labels_states():
    short, full = crsmod.frame_labels(None)
    assert short == "no CRS" and "not georeferenced" in full
    short, full = crsmod.frame_labels({"frame": "ENU"})
    assert short == "local ENU" and "East-North-Up" in full
    geo = {"frame": "ENU",
           "crs": {"epsg": 31982, "name": "SIRGAS 2000 / UTM zone 22S",
                   "orthometric": True, "offset": [480000.0, 6675000.0, 0.0]}}
    short, full = crsmod.frame_labels(geo)
    assert short == "EPSG:31982 (offset)"
    assert "SIRGAS 2000" in full and "EGM96" in full and "480000" in full


def test_project_frame_transform_from_offset_frame():
    """Stored CRS-minus-offset coordinates -> another CRS must equal the
    direct pyproj path on the absolute coordinates."""
    import pyproj

    geo = {"crs": {"epsg": UTM_SIRGAS_22S, "orthometric": False,
                   "offset": [480_000.0, 6_675_000.0, 0.0]}}
    stored = np.array([[819.7, 857.5, 15.0], [920.1, 930.2, 40.0]])
    fn, crs = crsmod.project_frame_transform(geo, None, UTM_WGS84_22S)
    got = fn(stored)
    direct = pyproj.Transformer.from_crs(f"EPSG:{UTM_SIRGAS_22S}",
                                         f"EPSG:{UTM_WGS84_22S}",
                                         always_xy=True)
    absolute = stored + np.array([480_000.0, 6_675_000.0, 0.0])
    e, n, h = direct.transform(absolute[:, 0], absolute[:, 1], absolute[:, 2])
    assert np.allclose(got, np.column_stack([e, n, h]), atol=1e-6)
    assert crs.to_epsg() == UTM_WGS84_22S


# -- reproject_cli end-to-end ---------------------------------------------------
def _write_manifest(root: Path, geo: dict) -> None:
    (root / "project.json").write_text(
        json.dumps({"settings": {"georeferenced": geo}}), encoding="utf-8")


def test_reproject_cli_enu_to_utm_and_back(tmp_path, capsys):
    from cloudlabeller.core.dataset import PointCloud
    from cloudlabeller.io.geometry import load_cloud_npz, save_cloud_npz
    from cloudlabeller.photogrammetry import reproject_cli
    from cloudlabeller.photogrammetry.pipeline import load_result

    root, images, cloud = _make_project(tmp_path)
    rec_dir = root / "reconstruction"
    geo = {"frame": "ENU", "origin_lla": list(ORIGIN),
           "origin_convention": crsmod.ORIGIN_CONVENTION_FIRST_GPS}
    _write_manifest(root, geo)
    centers = _centers(images)                 # play the role of ENU metres
    save_cloud_npz(PointCloud(xyz=centers.astype(np.float32)),
                   rec_dir / "dense.npz")

    # --- ENU -> WGS84 UTM 22S
    assert reproject_cli.main(["reproject_cli", str(root),
                               "--epsg", str(UTM_WGS84_22S)]) == 0
    out = json.loads(capsys.readouterr().out.partition("RESULT ")[2]
                     .splitlines()[0])
    assert out["epsg"] == UTM_WGS84_22S and "WGS 84" in out["name"]
    offset = np.array(out["offset"])
    assert np.all(offset[:2] % 1000 == 0) and offset[2] == 0
    assert out["fit_rms_m"] < 0.01             # tiny site: near-exact fit

    exact = crsmod.enu_to_crs_transform(ORIGIN, UTM_WGS84_22S)[0]
    expected_abs = exact(centers)
    aligned = load_result(rec_dir, root / "images")
    assert np.allclose(_centers(aligned.images) + offset, expected_abs,
                       atol=0.01)
    dense_after = load_cloud_npz(rec_dir / "dense.npz")
    assert np.allclose(dense_after.xyz + offset, expected_abs, atol=0.1)

    # --- WGS84 UTM -> SIRGAS 2000 UTM (offset-frame source branch)
    geo["crs"] = {"epsg": out["epsg"], "name": out["name"],
                  "orthometric": False, "offset": out["offset"]}
    _write_manifest(root, geo)
    assert reproject_cli.main(["reproject_cli", str(root),
                               "--epsg", str(UTM_SIRGAS_22S)]) == 0
    out2 = json.loads(capsys.readouterr().out.partition("RESULT ")[2]
                      .splitlines()[0])
    exact2 = crsmod.project_frame_transform(geo, None, UTM_SIRGAS_22S)[0]
    stored1 = _centers(aligned.images)
    expected2 = exact2(stored1)
    aligned2 = load_result(rec_dir, root / "images")
    assert np.allclose(_centers(aligned2.images) + np.array(out2["offset"]),
                       expected2, atol=0.01)


def test_reproject_cli_requires_georeferencing(tmp_path, capsys):
    from cloudlabeller.photogrammetry import reproject_cli

    root, _images, _cloud = _make_project(tmp_path)
    _write_manifest(root, None)                # settings without georeferenced
    (root / "project.json").write_text(json.dumps({"settings": {}}),
                                       encoding="utf-8")
    assert reproject_cli.main(["reproject_cli", str(root),
                               "--epsg", str(UTM_WGS84_22S)]) == 2


# -- export from an offset frame ------------------------------------------------
def test_export_transform_offset_kind():
    from cloudlabeller.ui.main_window import MainWindow

    spec = {"kind": "offset", "epsg": UTM_SIRGAS_22S, "orthometric": False,
            "offset": [480_000.0, 6_675_000.0, 0.0]}
    transform, crs = MainWindow._build_export_transform(
        spec, lambda f, m="": None)
    got = transform(np.array([[819.7, 857.5, 15.0]], np.float32))
    assert got.dtype == np.float64
    assert np.allclose(got[0], [480_819.7, 6_675_857.5, 15.0], atol=1e-4)
    assert crs.to_epsg() == UTM_SIRGAS_22S

    transform, crs = MainWindow._build_export_transform(
        {"kind": "local"}, lambda f, m="": None)
    assert transform is None and crs is None


# -- viewer up direction --------------------------------------------------------
def test_scene_up_georeferenced_is_plus_z_else_camera_derived():
    """Regression (2026-07-16): the empirical +R[1] up-heuristic showed
    georeferenced models upside down. Georeferenced frames have +Z up by
    construction; the unreferenced heuristic uses -R[1] (COLMAP's image
    y-axis points down)."""
    from types import SimpleNamespace as NS

    from cloudlabeller.ui.viewer3d import Viewer3D

    geo = NS(project=NS(settings={"georeferenced": {"frame": "ENU"}}),
             _frustums={})
    assert np.allclose(Viewer3D._scene_up(geo), [0.0, 0.0, 1.0])

    # Oblique cameras with image-down = world -Z  =>  up must come out +Z.
    R = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    local = NS(project=None,
               _frustums={"a": NS(camera=NS(R=R)), "b": NS(camera=NS(R=R))})
    assert np.allclose(Viewer3D._scene_up(local), [0.0, 0.0, 1.0])


# -- dataset pane coordinates ----------------------------------------------------
def test_dataset_panel_shows_camera_coordinates():
    from pathlib import Path as P
    from types import SimpleNamespace as NS

    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from cloudlabeller.core.dataset import Camera, Dataset, ImageRecord
    from cloudlabeller.core.events import EventBus
    from cloudlabeller.core.labels import LabelStore
    from cloudlabeller.ui.dataset_panel import DatasetPanel

    QApplication.instance() or QApplication([])
    cam = Camera(K=np.eye(3), R=np.eye(3), t=np.array([-819.7, -857.5, -15.0]),
                 width=100, height=100)        # centre = -R.T @ t
    project = NS(
        dataset=Dataset(images=[ImageRecord(0, P("a.jpg"), camera=cam),
                                ImageRecord(1, P("b.jpg"))]),
        labels=LabelStore(),
        settings={"georeferenced": {"crs": {
            "epsg": 31982, "offset": [480_000.0, 6_675_000.0, 0.0]}}})
    bus = EventBus()
    panel = DatasetPanel(bus)
    bus.project_opened.emit(project)

    solved = panel.image_list.item(0)
    assert solved.data(Qt.UserRole) == "a.jpg"
    assert solved.text() == "a.jpg    (480819.7, 6675857.5, 15.0)"
    unsolved = panel.image_list.item(1)
    assert unsolved.text() == "b.jpg"          # no camera -> name only

    # Selection still round-trips through the UserRole name.
    emitted: list[str] = []
    bus.image_selected.connect(emitted.append)
    panel.image_list.setCurrentRow(0)
    assert emitted[-1] == "a.jpg"
    bus.image_selected.emit("b.jpg")
    assert panel.image_list.currentItem().data(Qt.UserRole) == "b.jpg"
