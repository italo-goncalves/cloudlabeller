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

"""CRS-aware export: ENU -> projected transforms, CRS metadata, dialog."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from cloudlabeller.core.dataset import Mesh, PointCloud
from cloudlabeller.photogrammetry import crs as crsmod

ORIGIN = (-30.05, -51.2, 10.0)          # (lat, lon, alt) — southern Brazil
UTM_SIRGAS_22S = 31982
UTM_WGS84_22S = 32722


# -- transform correctness -------------------------------------------------
def test_enu_origin_maps_to_projected_origin():
    """ENU (0,0,0) must land exactly where pyproj puts the origin itself."""
    import pyproj

    fn, crs = crsmod.enu_to_crs_transform(ORIGIN, UTM_WGS84_22S)
    out = fn(np.zeros((1, 3)))
    ref = pyproj.Transformer.from_crs("EPSG:4979", "EPSG:32722",
                                      always_xy=True)
    e0, n0, h0 = ref.transform([ORIGIN[1]], [ORIGIN[0]], [ORIGIN[2]])
    assert np.allclose(out[0], [e0[0], n0[0], h0[0]], atol=1e-6)
    assert crs.to_epsg() == UTM_WGS84_22S


def test_enu_axes_are_metric_east_north_up():
    """100 m east/north/up in ENU stay ~100 m in UTM (scale factor ~0.9996)."""
    fn, _ = crsmod.enu_to_crs_transform(ORIGIN, UTM_WGS84_22S)
    pts = fn(np.array([[0.0, 0.0, 0.0],
                       [100.0, 0.0, 0.0],       # east
                       [0.0, 100.0, 0.0],       # north
                       [0.0, 0.0, 100.0]]))     # up
    east = pts[1] - pts[0]
    north = pts[2] - pts[0]
    up = pts[3] - pts[0]
    assert abs(np.hypot(east[0], east[1]) - 100.0) < 0.2
    assert abs(east[0]) > 99.0                   # mostly +E (small convergence)
    assert abs(np.hypot(north[0], north[1]) - 100.0) < 0.2
    assert abs(north[1]) > 99.0                  # mostly +N
    assert abs(up[2] - 100.0) < 0.01             # ellipsoidal height, exact
    assert np.hypot(up[0], up[1]) < 0.1


def test_transform_output_is_float64():
    fn, _ = crsmod.enu_to_crs_transform(ORIGIN, UTM_WGS84_22S)
    out = fn(np.zeros((2, 3), np.float32))
    assert out.dtype == np.float64


def test_orthometric_requires_geoid_grid():
    if crsmod.geoid_ready(UTM_WGS84_22S):
        pytest.skip("geoid grid installed on this machine")
    with pytest.raises(RuntimeError, match="geoid"):
        crsmod.enu_to_crs_transform(ORIGIN, UTM_WGS84_22S, orthometric=True)


def test_refresh_grid_cache_reveals_new_grids():
    """Regression (2026-07-16): a grid downloaded in-process stayed invisible
    to the downloading thread (stale PROJ context), so download_geoid_grids
    reported "offline" after a successful download. Simulate by stashing the
    grid file away and back — the stale context must miss it until refreshed."""
    import shutil

    import pyproj.datadir

    grid = (Path(pyproj.datadir.get_user_data_dir()) / "us_nga_egm96_15.tif")
    if not grid.exists() or not crsmod.geoid_ready(UTM_SIRGAS_22S):
        pytest.skip("EGM96 grid not installed in the user PROJ dir")
    stash = grid.with_suffix(".tif.stash")
    shutil.move(grid, stash)
    try:
        crsmod.refresh_grid_cache()
        assert not crsmod.geoid_ready(UTM_SIRGAS_22S)      # gone once refreshed
        shutil.move(stash, grid)
        assert not crsmod.geoid_ready(UTM_SIRGAS_22S)      # stale context...
        crsmod.refresh_grid_cache()
        assert crsmod.geoid_ready(UTM_SIRGAS_22S)          # ...until refreshed
    finally:
        if stash.exists():
            shutil.move(stash, grid)


@pytest.mark.skipif(os.environ.get("CLOUDLABELLER_NET_TESTS") != "1",
                    reason="network test — set CLOUDLABELLER_NET_TESTS=1")
def test_geoid_download_and_offset():
    crsmod.download_geoid_grids(UTM_WGS84_22S)
    assert crsmod.geoid_ready(UTM_WGS84_22S)
    fn_ell, _ = crsmod.enu_to_crs_transform(ORIGIN, UTM_WGS84_22S)
    fn_ort, _ = crsmod.enu_to_crs_transform(ORIGIN, UTM_WGS84_22S,
                                            orthometric=True)
    dh = fn_ell(np.zeros((1, 3)))[0, 2] - fn_ort(np.zeros((1, 3)))[0, 2]
    assert 1.0 < abs(dh) < 60.0                  # geoid undulation, not zero


# -- suggestion + catalogue -------------------------------------------------
def test_suggests_sirgas_utm_for_southern_brazil():
    assert crsmod.suggest_projected_epsg(ORIGIN) == UTM_SIRGAS_22S


def test_catalogue_is_large_and_contains_suggestion():
    cat = crsmod.projected_crs_catalogue()
    assert len(cat) > 3000
    assert (UTM_SIRGAS_22S, "SIRGAS 2000 / UTM zone 22S") in cat


# -- ENU origin resolution ---------------------------------------------------
def test_resolve_origin_new_convention_is_direct(tmp_path):
    geo = {"origin_lla": list(ORIGIN),
           "origin_convention": crsmod.ORIGIN_CONVENTION_FIRST_GPS}
    origin = crsmod.resolve_enu_origin(geo, tmp_path)
    assert origin.exact and origin.lla == ORIGIN


def test_resolve_origin_legacy_scans_cameras_json_order(tmp_path, monkeypatch):
    """Legacy projects stored the mean GPS; the true COLMAP ENU origin is the
    first cameras.json image with GPS EXIF (= first ref_images.txt line)."""
    ws = tmp_path / "reconstruction"
    ws.mkdir()
    (ws / "cameras.json").write_text(json.dumps(
        [{"name": "a.jpg"}, {"name": "b.jpg"}, {"name": "c.jpg"}]))
    first_gps = (-30.06, -51.21, 12.0)
    from cloudlabeller.photogrammetry import georef

    monkeypatch.setattr(georef, "exif_gps",
                        lambda p: None if "a.jpg" in str(p) else first_gps)
    geo = {"origin_lla": list(ORIGIN)}           # stored mean — must be ignored
    origin = crsmod.resolve_enu_origin(geo, ws, tmp_path / "images")
    assert origin.exact and origin.lla == first_gps


def test_resolve_origin_falls_back_to_stored_mean(tmp_path):
    geo = {"origin_lla": list(ORIGIN)}
    origin = crsmod.resolve_enu_origin(geo, tmp_path / "nowhere")
    assert not origin.exact and origin.lla == ORIGIN


# -- exports carry the CRS ----------------------------------------------------
def _small_cloud() -> tuple[PointCloud, np.ndarray]:
    xyz = np.array([[0, 0, 0], [10, 0, 1], [0, 10, 2], [5, 5, 3]], np.float32)
    rgb = np.arange(12, dtype=np.uint8).reshape(4, 3)
    return PointCloud(xyz=xyz, rgb=rgb), np.array([0, 1, -1, 2], np.int32)


def test_las_export_embeds_crs_and_survives_roundtrip(tmp_path):
    import laspy

    from cloudlabeller.io.export import export_cloud

    cloud, labels = _small_cloud()
    fn, crs = crsmod.enu_to_crs_transform(ORIGIN, UTM_SIRGAS_22S)
    path = tmp_path / "out.las"
    export_cloud(cloud, labels, path, transform=fn, crs=crs)
    las = laspy.read(str(path))
    assert las.header.parse_crs().to_epsg() == UTM_SIRGAS_22S
    got = np.column_stack([las.x, las.y, las.z])
    assert np.allclose(got, fn(cloud.xyz), atol=0.01)   # within LAS int scale
    assert np.array_equal(np.asarray(las["label"]), labels)
    assert not (tmp_path / "out.prj").exists()          # CRS is in the header


def test_ply_export_writes_f8_and_prj_sidecar(tmp_path):
    from plyfile import PlyData

    from cloudlabeller.io.export import export_cloud

    cloud, labels = _small_cloud()
    fn, crs = crsmod.enu_to_crs_transform(ORIGIN, UTM_SIRGAS_22S)
    path = tmp_path / "out.ply"
    export_cloud(cloud, labels, path, transform=fn, crs=crs)
    v = PlyData.read(str(path))["vertex"]
    assert v.data.dtype["x"] == np.float64      # f4 would round ~0.5 m at UTM
    got = np.column_stack([v["x"], v["y"], v["z"]])
    assert np.allclose(got, fn(cloud.xyz), atol=1e-6)
    _check_prj(tmp_path / "out.prj", UTM_SIRGAS_22S)


def test_csv_export_transforms_and_writes_prj(tmp_path):
    from cloudlabeller.io.export import export_cloud

    cloud, labels = _small_cloud()
    fn, crs = crsmod.enu_to_crs_transform(ORIGIN, UTM_SIRGAS_22S)
    path = tmp_path / "out.csv"
    export_cloud(cloud, labels, path, transform=fn, crs=crs)
    got = np.loadtxt(path, delimiter=",", skiprows=1)[:, :3]
    assert np.allclose(got, fn(cloud.xyz), atol=1e-5)
    _check_prj(tmp_path / "out.prj", UTM_SIRGAS_22S)


def test_local_export_unchanged_writes_no_prj(tmp_path):
    from cloudlabeller.io.export import export_cloud

    cloud, labels = _small_cloud()
    path = tmp_path / "out.csv"
    export_cloud(cloud, labels, path)                    # old signature
    got = np.loadtxt(path, delimiter=",", skiprows=1)[:, :3]
    assert np.allclose(got, cloud.xyz, atol=1e-5)
    assert not (tmp_path / "out.prj").exists()


def test_mesh_ply_export_is_f8_with_prj(tmp_path):
    from plyfile import PlyData

    from cloudlabeller.io.export import export_mesh

    mesh = Mesh(vertices=np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0]],
                                  np.float32),
                faces=np.array([[0, 1, 2]], np.int32),
                vertex_colors=np.array([[255, 0, 0]] * 3, np.uint8))
    fn, crs = crsmod.enu_to_crs_transform(ORIGIN, UTM_SIRGAS_22S)
    path = tmp_path / "mesh.ply"
    export_mesh(mesh, path, transform=fn, crs=crs)
    ply = PlyData.read(str(path))
    v = ply["vertex"]
    assert v.data.dtype["x"] == np.float64
    got = np.column_stack([v["x"], v["y"], v["z"]])
    assert np.allclose(got, fn(mesh.vertices), atol=1e-6)
    assert np.array_equal(ply["face"]["vertex_indices"][0], [0, 1, 2])
    assert tuple(v["red"][:1]) == (255,)
    _check_prj(tmp_path / "mesh.prj", UTM_SIRGAS_22S)


def _check_prj(prj_path, epsg) -> None:
    import pyproj

    assert prj_path.exists()
    parsed = pyproj.CRS.from_wkt(prj_path.read_text(encoding="utf-8"))
    assert parsed.to_epsg() == epsg


# -- dialog -------------------------------------------------------------------
def test_export_crs_dialog_search_and_preselection():
    from PySide6.QtWidgets import QApplication

    from cloudlabeller.ui.export_crs_dialog import ExportCrsDialog

    QApplication.instance() or QApplication([])
    dlg = ExportCrsDialog(ORIGIN, geo_settings={})
    # Suggested projection pre-selected, projected mode by default.
    assert dlg.rb_proj.isChecked()
    assert dlg.cmb_crs.currentData() == UTM_SIRGAS_22S
    assert dlg.cmb_crs.count() > 3000
    # The completer searches anywhere in the entry, case-insensitively.
    from PySide6.QtCore import Qt

    completer = dlg.cmb_crs.completer()
    assert completer.filterMode() == Qt.MatchContains
    completer.setCompletionPrefix("sirgas 2000 / utm zone 21")
    assert completer.completionCount() > 0
    choice = dlg.choice()
    assert choice.mode == "projected" and choice.epsg == UTM_SIRGAS_22S


def test_export_crs_dialog_remembers_last_choice():
    from PySide6.QtWidgets import QApplication

    from cloudlabeller.ui.export_crs_dialog import ExportCrsDialog

    QApplication.instance() or QApplication([])
    geo = {"export_mode": "projected", "export_epsg": UTM_WGS84_22S,
           "export_orthometric": False}
    dlg = ExportCrsDialog(ORIGIN, geo_settings=geo)
    assert dlg.cmb_crs.currentData() == UTM_WGS84_22S

    geo_local = {"export_mode": "local"}
    dlg2 = ExportCrsDialog(ORIGIN, geo_settings=geo_local)
    assert dlg2.rb_local.isChecked()
    assert dlg2.choice().mode == "local"
