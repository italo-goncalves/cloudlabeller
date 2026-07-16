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

"""Native georeferencing: RANSAC fit, lla->ENU, end-to-end georef_cli."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cloudlabeller.photogrammetry import crs as crsmod
from cloudlabeller.photogrammetry import georef

ORIGIN = (-30.05, -51.2, 10.0)


def _rot_z(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# -- RANSAC fit ---------------------------------------------------------------
def test_ransac_recovers_similarity_despite_outliers():
    rng = np.random.default_rng(7)
    src = rng.uniform(-50, 50, (20, 3))
    s_true, R_true, t_true = 2.5, _rot_z(35.0), np.array([120.0, -40.0, 6.0])
    dst = src @ (s_true * R_true).T + t_true
    dst += rng.normal(0, 0.5, dst.shape)          # GPS-like noise
    dst[[3, 8, 15]] += [[60, 0, 0], [0, -90, 20], [40, 40, 40]]   # outliers

    s, R, t, inliers, rms = georef.ransac_similarity(src, dst, max_error=5.0)
    assert abs(s - s_true) < 0.05
    assert np.allclose(R, R_true, atol=0.02)
    assert np.allclose(t, t_true, atol=2.0)
    assert not inliers[[3, 8, 15]].any()
    assert inliers.sum() == 17
    assert rms < 1.5


def test_ransac_raises_without_consensus():
    rng = np.random.default_rng(1)
    src = rng.uniform(-50, 50, (10, 3))
    dst = rng.uniform(-5000, 5000, (10, 3))       # unrelated positions
    with pytest.raises(RuntimeError, match="GPS"):
        georef.ransac_similarity(src, dst, max_error=1e-3)


def test_ransac_needs_three_points():
    with pytest.raises(RuntimeError, match=">= 3"):
        georef.ransac_similarity(np.zeros((2, 3)), np.zeros((2, 3)))


# -- lla -> ENU ----------------------------------------------------------------
def test_lla_to_enu_origin_and_axes():
    fn = crsmod.lla_to_enu_transform(ORIGIN)
    lat, lon, alt = ORIGIN
    out = fn(np.array([[lat, lon, alt],
                       [lat + 0.001, lon, alt],       # ~111 m north
                       [lat, lon, alt + 100.0]]))     # 100 m up
    assert np.allclose(out[0], 0.0, atol=1e-9)
    assert abs(out[1][1] - 110.9) < 0.5 and abs(out[1][0]) < 0.5
    assert abs(out[2][2] - 100.0) < 1e-6


def test_lla_to_enu_inverts_export_direction():
    """lla->ENU->projected must equal the direct geodetic->projected path."""
    import pyproj

    lla = np.array([[-30.049, -51.201, 25.0], [-30.052, -51.198, 12.0]])
    enu = crsmod.lla_to_enu_transform(ORIGIN)(lla)
    via_enu = crsmod.enu_to_crs_transform(ORIGIN, 32722)[0](enu)
    direct = pyproj.Transformer.from_crs("EPSG:4979", "EPSG:32722",
                                         always_xy=True)
    e, n, h = direct.transform(lla[:, 1], lla[:, 0], lla[:, 2])
    assert np.allclose(via_enu, np.column_stack([e, n, h]), atol=1e-6)


# -- capped GPS scan ----------------------------------------------------------
def test_count_gps_images_stops_early(monkeypatch):
    calls = []

    def fake_exif(path):
        calls.append(path)
        return ORIGIN                             # every image has GPS

    monkeypatch.setattr(georef, "exif_gps", fake_exif)
    found = georef.count_gps_images([f"im{i}.jpg" for i in range(100)])
    assert found == 3
    assert len(calls) == 3                        # stopped at `need`

    monkeypatch.setattr(georef, "exif_gps", lambda p: None)
    assert georef.count_gps_images([f"im{i}.jpg" for i in range(100)]) == 0


# -- end-to-end: georef_cli on a synthetic project ----------------------------
def _make_project(tmp_path: Path):
    """Synthetic solved project: COLMAP model + cameras.json + cloud.npz."""
    import pycolmap

    from cloudlabeller.photogrammetry.cameras import colmap_to_cameras
    from cloudlabeller.photogrammetry.extract import reconstruction_to_cloud
    from cloudlabeller.photogrammetry.pipeline import ReconstructResult, save_result

    opts = pycolmap.SyntheticDatasetOptions()
    opts.num_rigs, opts.num_frames_per_rig, opts.num_points3D = 6, 1, 40
    rec = pycolmap.synthesize_dataset(opts)
    root = tmp_path / "proj"
    model_dir = root / "reconstruction" / "sparse" / "0"
    model_dir.mkdir(parents=True)
    rec.write(str(model_dir))
    images = colmap_to_cameras(rec, root / "images")
    cloud = reconstruction_to_cloud(rec)
    save_result(ReconstructResult(cloud=cloud, images=images),
                root / "reconstruction")
    return root, images, cloud


def _centers(images) -> np.ndarray:
    return np.array([-(r.camera.R.T @ np.asarray(r.camera.t, np.float64))
                     for r in images])


def test_georef_cli_aligns_project_natively(tmp_path, monkeypatch, capsys):
    from cloudlabeller.io.geometry import load_cloud_npz, save_cloud_npz
    from cloudlabeller.photogrammetry import georef_cli
    from cloudlabeller.photogrammetry.pipeline import load_result

    root, images, cloud = _make_project(tmp_path)
    rec_dir = root / "reconstruction"

    # Ground truth: cameras sit at s·R@C + t metres (ENU about ORIGIN); the
    # EXIF GPS of each image is that position converted to geodetic.
    centers = _centers(images)
    s_true, R_true, t_true = 2.0, _rot_z(30.0), np.array([120.0, -40.0, 6.0])
    enu_true = centers @ (s_true * R_true).T + t_true
    pipeline = crsmod._enu_lla_pipeline(ORIGIN)
    lon, lat, h = pipeline.transform(enu_true[:, 0], enu_true[:, 1],
                                     enu_true[:, 2])
    gps_by_name = {Path(r.path).name: (float(la), float(lo), float(al))
                   for r, la, lo, al in zip(images, lat, lon, h)}
    monkeypatch.setattr(georef, "exif_gps",
                        lambda p: gps_by_name.get(Path(p).name))

    # A dense cloud whose points coincide with the camera centres — after
    # alignment they must land on the cameras' GPS ENU positions.
    from cloudlabeller.core.dataset import PointCloud

    save_cloud_npz(PointCloud(xyz=centers.astype(np.float32)),
                   rec_dir / "dense.npz")

    rc = georef_cli.main(["georef_cli", str(root)])
    assert rc == 0

    out = capsys.readouterr().out
    result = json.loads(out.partition("RESULT ")[2].splitlines()[0])
    assert result["origin_convention"] == crsmod.ORIGIN_CONVENTION_FIRST_GPS
    assert result["n_inliers"] == len(images)
    assert result["fit_rms_m"] < 0.01
    assert abs(result["scale_m_per_unit"] - s_true) < 0.01

    # The CLI anchors ENU at the FIRST image's GPS: expected positions are
    # the GPS coordinates re-expressed in that frame.
    first_name = Path(images[0].path).name
    assert tuple(result["origin_lla"]) == gps_by_name[first_name]
    expected = crsmod.lla_to_enu_transform(result["origin_lla"])(
        np.array([gps_by_name[Path(r.path).name] for r in images]))

    aligned = load_result(rec_dir, root / "images")
    assert np.allclose(_centers(aligned.images), expected, atol=1e-3)

    dense_after = load_cloud_npz(rec_dir / "dense.npz")
    assert np.allclose(dense_after.xyz, expected, atol=0.1)   # float32 storage

    # Sparse cloud scales rigidly (pairwise distances × s_true).
    d0 = np.linalg.norm(cloud.xyz[0] - cloud.xyz[1])
    d1 = np.linalg.norm(aligned.cloud.xyz[0] - aligned.cloud.xyz[1])
    assert abs(d1 - s_true * d0) < 0.01

    # The COLMAP model on disk was transformed too (float64 — tight).
    import pycolmap

    rec2 = pycolmap.Reconstruction(str(rec_dir / "sparse" / "0"))
    got = np.array([im.projection_center()
                    for im in sorted(rec2.images.values(), key=lambda i: i.name)])
    want_by_name = {Path(r.path).name: e for r, e in zip(images, expected)}
    want = np.array([want_by_name[im.name]
                     for im in sorted(rec2.images.values(), key=lambda i: i.name)])
    assert np.allclose(got, want, atol=1e-6)
    assert (rec_dir / "sparse_prealigned" / "0").exists()

    # Aligning twice would double-transform — must refuse.
    assert georef_cli.main(["georef_cli", str(root)]) == 2


# -- dialogs ------------------------------------------------------------------
def test_sfm_dialog_georef_checkbox():
    from PySide6.QtWidgets import QApplication

    from cloudlabeller.ui.run_sfm_dialog import RunSfmDialog

    QApplication.instance() or QApplication([])
    dlg = RunSfmDialog(10, gps_found=3)
    assert dlg.chk_georef.isEnabled() and dlg.georeference()
    dlg_no = RunSfmDialog(10, gps_found=1)
    assert not dlg_no.chk_georef.isEnabled() and not dlg_no.georeference()


def test_pipeline_dialog_georef_checkbox():
    from PySide6.QtWidgets import QApplication

    from cloudlabeller.ui.run_pipeline_dialog import RunPipelineDialog

    QApplication.instance() or QApplication([])
    dlg = RunPipelineDialog(10, gps_found=3)
    assert dlg.chk_georef.isEnabled() and dlg.georeference()
    dlg_no = RunPipelineDialog(10, gps_found=0)
    assert not dlg_no.chk_georef.isEnabled() and not dlg_no.georeference()
