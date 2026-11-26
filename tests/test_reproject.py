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
def _write_input(path: Path, geo, samples: np.ndarray) -> Path:
    path.write_text(json.dumps({"frame": geo, "origin_lla": list(ORIGIN),
                                "samples": np.asarray(samples).tolist()}),
                    encoding="utf-8")
    return path


def _fit(out: dict):
    return out["s"], np.asarray(out["R"]), np.asarray(out["t"])


def test_reproject_cli_fits_enu_to_utm_and_on(tmp_path, capsys):
    """The CLI fits the move of the Project frame; the Engine composes it
    into each Alignment (ADR-0005), so nothing is read but the input."""
    from cloudlabeller.photogrammetry import reproject_cli

    _root, images, _cloud = _make_project(tmp_path)
    geo = {"frame": "ENU", "origin_lla": list(ORIGIN),
           "origin_convention": crsmod.ORIGIN_CONVENTION_FIRST_GPS}
    centers = _centers(images)                 # play the role of ENU metres
    given = _write_input(tmp_path / "in.json", geo, crsmod.site_samples(centers))

    # --- ENU -> WGS84 UTM 22S
    assert reproject_cli.main(["reproject_cli", str(given),
                               "--epsg", str(UTM_WGS84_22S)]) == 0
    out = json.loads(capsys.readouterr().out.partition("RESULT ")[2]
                     .splitlines()[0])
    assert out["epsg"] == UTM_WGS84_22S and "WGS 84" in out["name"]
    offset = np.array(out["offset"])
    assert np.all(offset[:2] % 1000 == 0) and offset[2] == 0
    assert out["fit_rms_m"] < 0.01             # tiny site: near-exact fit
    s, R, t = _fit(out)
    stored1 = centers @ (s * R).T + t
    exact = crsmod.enu_to_crs_transform(ORIGIN, UTM_WGS84_22S)[0]
    assert np.allclose(stored1 + offset, exact(centers), atol=0.01)

    # --- WGS84 UTM -> SIRGAS 2000 UTM (offset-frame source branch)
    geo["crs"] = {"epsg": out["epsg"], "name": out["name"],
                  "orthometric": False, "offset": out["offset"]}
    given = _write_input(tmp_path / "in2.json", geo, crsmod.site_samples(stored1))
    assert reproject_cli.main(["reproject_cli", str(given),
                               "--epsg", str(UTM_SIRGAS_22S)]) == 0
    out2 = json.loads(capsys.readouterr().out.partition("RESULT ")[2]
                      .splitlines()[0])
    s2, R2, t2 = _fit(out2)
    exact2 = crsmod.project_frame_transform(geo, None, UTM_SIRGAS_22S)[0]
    assert np.allclose(stored1 @ (s2 * R2).T + t2 + np.array(out2["offset"]),
                       exact2(stored1), atol=0.01)


def test_reproject_cli_requires_a_frame(tmp_path, capsys):
    from cloudlabeller.photogrammetry import reproject_cli

    given = _write_input(tmp_path / "in.json", None, np.zeros((9, 3)))
    assert reproject_cli.main(["reproject_cli", str(given),
                               "--epsg", str(UTM_WGS84_22S)]) == 2


# -- export from an offset frame ------------------------------------------------
def test_export_transform_offset_kind():
    spec = {"kind": "offset", "epsg": UTM_SIRGAS_22S, "orthometric": False,
            "offset": [480_000.0, 6_675_000.0, 0.0]}
    transform, crs = crsmod.build_export_transform(spec, lambda f, m="": None)
    got = transform(np.array([[819.7, 857.5, 15.0]], np.float32))
    assert got.dtype == np.float64
    assert np.allclose(got[0], [480_819.7, 6_675_857.5, 15.0], atol=1e-4)
    assert crs.to_epsg() == UTM_SIRGAS_22S

    transform, crs = crsmod.build_export_transform({"kind": "local"},
                                                   lambda f, m="": None)
    assert transform is None and crs is None
# -- dataset pane coordinates ----------------------------------------------------


def test_scene_up_georeferenced_is_plus_z_else_camera_derived():
    """Regression (2026-07-16): the empirical +R[1] up-heuristic showed
    georeferenced models upside down. Georeferenced frames have +Z up by
    construction; the unreferenced heuristic uses -R[1] (COLMAP's image
    y-axis points down)."""
    from cloudlabeller.photogrammetry.cameras import scene_up

    # Oblique cameras with image-down = world -Z  =>  up must come out +Z.
    R = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    assert np.allclose(scene_up([R, R], False), [0.0, 0.0, 1.0])
    assert np.allclose(scene_up([R, R], True), [0.0, 0.0, 1.0])
    assert np.allclose(scene_up([np.eye(3)], True), [0.0, 0.0, 1.0])
