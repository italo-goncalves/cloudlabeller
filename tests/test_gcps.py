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

"""GCPs and Marks (0.4.0 step 3): triangulating photo Marks, reading
coordinates files, fitting Surveys to GCPs and to each other, and the
snapped pick on the cloud."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cloudlabeller.core.dataset import Camera
from cloudlabeller.core.gcp import GCP, CsvError, parse_csv
from cloudlabeller.core.reconstruction import apply_points
from cloudlabeller.photogrammetry.align import fit, inverse_points
from cloudlabeller.photogrammetry.triangulate import triangulate, undistort
from cloudlabeller.transfer.projection import project_points


def _look_at(centre, target) -> Camera:
    """A camera at ``centre`` looking at ``target`` (world -> camera)."""
    z = np.asarray(target, float) - centre
    z /= np.linalg.norm(z)
    x = np.cross([0.0, 0.0, 1.0], z)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    K = np.array([[1200.0, 0, 800], [0, 1200.0, 600], [0, 0, 1]])
    return Camera(K=K, R=R, t=-R @ centre, width=1600, height=1200,
                  distortion=np.array([-0.08, 0.01]), model="RADIAL")


# -- triangulation ------------------------------------------------------------------
def test_undistort_inverts_the_camera_model():
    cam = _look_at(np.array([0.0, -10.0, 5.0]), [0, 0, 0])
    pts = np.random.default_rng(1).uniform(-2, 2, (20, 3))
    p, ok = project_points(cam, pts)
    xy = undistort(cam, p[ok, :2])
    cam_pts = pts[ok] @ cam.R.T + cam.t
    assert np.allclose(xy, cam_pts[:, :2] / cam_pts[:, 2:], atol=1e-9)


def test_triangulating_photo_marks_finds_the_point():
    X = np.array([0.3, -0.2, 0.5])
    cams = [_look_at(np.array(c), X) for c in ([0, -10, 5], [8, -6, 4], [-7, -7, 6])]
    uvs = [project_points(c, X[None])[0][0, :2] for c in cams]
    got, rms = triangulate(cams, uvs)
    assert np.allclose(got, X, atol=1e-6) and rms < 1e-6
    noisy = [uv + [0.7, -0.4] for uv in uvs]
    got, rms = triangulate(cams, noisy)
    assert np.linalg.norm(got - X) < 0.02 and 0.1 < rms < 2.0
    with pytest.raises(ValueError):
        triangulate(cams[:1], uvs[:1])


# -- coordinates files --------------------------------------------------------------------
def test_coordinates_files_with_a_header_semicolons_and_decimal_commas():
    rows = parse_csv("name,E,N,H,acc\nG1,500100.5,6600200.25,101.5,0.02\nG2,1,2,3\n")
    assert rows == [("G1", [500100.5, 6600200.25, 101.5], 0.02), ("G2", [1.0, 2.0, 3.0], None)]
    rows = parse_csv("A;10,5;20,25;3\nB;1;2;3;0,05\n")
    assert rows == [("A", [10.5, 20.25, 3.0], None), ("B", [1.0, 2.0, 3.0], 0.05)]
    assert parse_csv("P1\t1\t2\t3\n")[0][1] == [1.0, 2.0, 3.0]
    with pytest.raises(CsvError) as err:
        parse_csv("a,1,2,3\nb,1,x,3\n")
    assert err.value.line == 2
    with pytest.raises(CsvError):
        parse_csv("only,two\n")


# -- the fit ---------------------------------------------------------------------------
def _rot(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def _sim(s, deg, t) -> dict:
    return {"s": s, "R": _rot(deg).tolist(), "t": list(t)}


def test_surveys_fit_to_surveyed_gcps_and_to_each_other_through_ties():
    rng = np.random.default_rng(4)
    world = {f"g{i}": rng.uniform(-50, 50, 3) for i in range(1, 8)}
    gcps = {gid: GCP(gid, gid) for gid in world}
    gcps["g7"].use = "check"
    true = {"s1": _sim(0.1, 20, [5, -3, 1]), "s2": _sim(2.5, -60, [-40, 10, 2])}
    # each Survey's own frame: the world through the inverse of its Alignment
    seen = {"s1": ["g1", "g2", "g3", "g4", "g5", "g7"], "s2": ["g3", "g4", "g5", "g6", "g7"]}
    positions = {sid: {gid: {"xyz": inverse_points(true[sid], world[gid])[0], "from": "cloud"}
                       for gid in seen[sid]} for sid in true}
    # g1..g3 and g7 are surveyed; s2 sees one of them (g3), so it needs the
    # ties g4 and g5, which only s1's fit can place
    targets = {gid: world[gid] for gid in ("g1", "g2", "g3", "g7")}
    out = fit(gcps, positions, targets, {"s1": None, "s2": None}, None)
    assert out["surveys"]["s1"]["how"] == "gcp" and out["surveys"]["s1"]["n_gcps"] == 3
    assert out["surveys"]["s2"]["how"] == "gcp" and out["surveys"]["s2"]["n_gcps"] == 3
    for sid in true:
        a = out["alignments"][sid]
        pts = np.array([positions[sid][g]["xyz"] for g in seen[sid]])
        assert np.allclose(apply_points(a, pts), [world[g] for g in seen[sid]], atol=1e-3)
    # the check GCP took no part but is measured, in both Surveys
    assert set(out["residuals"]["g7"]) == {"s1", "s2"}
    assert np.linalg.norm(out["residuals"]["g7"]["s1"]) < 1e-3


def test_a_survey_seeing_two_ties_keeps_what_it_had():
    world = {f"g{i}": np.array(v, float) for i, v in
             enumerate([[0, 0, 0], [10, 0, 0], [0, 10, 0], [5, 5, 3]], 1)}
    gcps = {gid: GCP(gid, gid) for gid in world}
    positions = {"s1": {g: {"xyz": world[g], "from": "cloud"} for g in world},
                 "s2": {g: {"xyz": world[g] + 1, "from": "cloud"} for g in ("g1", "g2")}}
    gps = _sim(1.0, 0, [0, 0, 0])
    out = fit(gcps, positions, {g: world[g] for g in world}, {"s1": None, "s2": gps}, None)
    assert "s2" not in out["alignments"]
    assert out["surveys"]["s2"] == {"aligned": True, "how": "kept", "n_gcps": 2, "rms": None}


def test_without_coordinates_the_reference_survey_keeps_its_frame():
    world = {f"g{i}": np.array(v, float) for i, v in
             enumerate([[0, 0, 0], [10, 0, 0], [0, 10, 0], [5, 5, 3]], 1)}
    gcps = {gid: GCP(gid, gid) for gid in world}
    other = _sim(3.0, 45, [1, 2, 3])
    positions = {"s1": {g: {"xyz": world[g], "from": "cloud"} for g in world},
                 "s2": {g: {"xyz": inverse_points(other, world[g])[0], "from": "photos"}
                        for g in world}}
    out = fit(gcps, positions, {}, {"s1": None, "s2": None}, "s1")
    assert out["surveys"]["s1"]["how"] == "reference"
    assert out["alignments"]["s1"]["source"] == "reference"   # it had none: identity
    a = out["alignments"]["s2"]
    assert np.isclose(a["s"], 3.0) and np.allclose(a["t"], [1, 2, 3], atol=1e-6)


# -- the snapped pick ----------------------------------------------------------------
def test_a_click_snaps_to_the_front_point_near_it():
    from cloudlabeller.core.selection import pick_nearest

    # three.js-style camera at the origin looking down +Z (as test_engine_parity)
    f = 1.0 / np.tan(np.radians(30))
    proj = np.array([[f, 0, 0, 0], [0, f, 0, 0],
                     [0, 0, -100.1 / 99.9, -20 / 99.9], [0, 0, -1, 0]])
    vp = proj @ np.diag([1.0, -1.0, -1.0, 1.0])
    xyz = np.array([[0.0, 0.0, 10.0], [0.0, 0.0, 5.0], [0.01, 0.0, 5.0], [3.0, 0.0, 5.0]])
    i = pick_nearest(xyz, vp.T.reshape(-1).tolist(), 100.0, 100.0, 200.0, 200.0, radius=6)
    assert i == 1                                     # front, and nearest the click
    assert pick_nearest(xyz, vp, 5.0, 5.0, 200.0, 200.0) is None


# -- over the Engine -------------------------------------------------------------------
UTM = 31982                                           # SIRGAS 2000 / UTM zone 22S


def test_gcps_marks_and_the_fit_over_the_engine(tmp_path):
    from test_surveys import _engine_with, _v1_project

    import cloudlabeller.engine.methods  # noqa: F401
    from cloudlabeller.core.project import Project
    from cloudlabeller.engine.server import METHODS

    root = _v1_project(tmp_path / "p")
    engine = _engine_with(root)
    call = lambda m, **p: METHODS[m].fn(engine, p)       # noqa: E731
    sv = engine.project.survey("s1")
    xyz = sv.dataset.cloud.xyz.astype(np.float64)
    # Survey points 0..3 as GCPs: their UTM coordinates are a known similarity away.
    true = _sim(4.0, 30, [100.0, 200.0, 50.0])
    utm = apply_points(true, xyz[:4]).astype(np.float64) + [600000.0, 6650000.0, 0.0]
    csv = tmp_path / "gcps.csv"
    csv.write_text("name,E,N,H\n" + "".join(f"P{i},{e:.4f},{n:.4f},{h:.4f}\n"
                                            for i, (e, n, h) in enumerate(utm)))
    got = call("gcp.import_csv", path=str(csv), epsg=UTM)
    assert got["added"] == 4 and [g["name"] for g in got["gcps"]] == ["P0", "P1", "P2", "P3"]
    ids = [g["id"] for g in got["gcps"]]
    for i, gid in enumerate(ids):
        call("mark.set", survey="s1", gcp=gid, cloud="sparse", index=i)
    # a photo Mark too: it is kept, and the hints skip the GCP it marks
    call("mark.set", survey="s1", gcp=ids[0], name="a.png", uv=[10.0, 12.0])

    deferred = call("align.fit")
    summary = deferred.finish(deferred.work(lambda f, m="": None))
    assert summary["frame_created"] and summary["surveys"]["s1"]["how"] == "gcp"
    geo = engine.project.frame
    assert geo["crs"]["epsg"] == UTM and geo["crs"]["offset"][0] % 1000 == 0
    framed = sv.active.frame_points(xyz[:4]).astype(np.float64)
    assert np.allclose(framed + geo["crs"]["offset"], utm, atol=1e-3)
    assert max(np.linalg.norm(d) for g in summary["residuals"].values()
               for d in g.values()) < 1e-3
    assert sv.active.alignment["source"] == "gcp"

    shown = call("marks.photo", survey="s1", name="a.png")
    assert shown["marks"] == [{"gcp": ids[0], "uv": [10.0, 12.0]}]
    assert {h["gcp"] for h in shown["hints"]} <= set(ids[1:])
    cloud = call("marks.cloud", survey="s1")["marks"]
    assert np.allclose(cloud[0]["xyz"], framed[0], atol=1e-3)

    again = Project.open(root)
    assert set(again.gcps) == set(ids) and again.survey("s1").marks[ids[0]] == {"a.png": [10.0, 12.0]}
    assert again.survey("s1").active.marks[ids[1]] == pytest.approx(xyz[1].tolist(), abs=1e-6)

    call("gcp.delete", id=ids[0])
    assert ids[0] not in engine.project.survey("s1").marks
    assert ids[0] not in engine.project.survey("s1").active.marks
    assert json.loads((root / "gcps.json").read_text())["gcps"][0]["name"] == "P1"
