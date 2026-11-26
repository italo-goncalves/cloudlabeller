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

"""Aligning Surveys to their GCPs, and to each other through shared GCPs.

Each Survey's active Reconstruction locates each GCP in its own frame (a
cloud Mark, else its photo Marks triangulated). The fit then works outward:

1. GCPs with surveyed coordinates (and use ``control``) are targets in the
   Project frame; with none at all, the reference Survey keeps its frame
   and its GCPs become the targets.
2. Any Survey that sees three or more targets gets a similarity fitted to
   them (Umeyama).
3. A GCP without coordinates, once seen by aligned Surveys, becomes a target
   at the mean of where they put it; step 2 repeats until no Survey gains.

A Survey with fewer than three targets keeps the Alignment it had (EXIF
GPS, say). Residuals are reported for every GCP in every aligned Survey,
check GCPs included; they never take part in a fit.
"""

from __future__ import annotations

import numpy as np

from cloudlabeller.core.reconstruction import alignment_parts, apply_points

MIN_GCPS = 3


def inverse_points(alignment: dict | None, xyz) -> np.ndarray:
    """Project-frame points back into the Reconstruction's own frame."""
    xyz = np.asarray(xyz, np.float64).reshape(-1, 3)
    if alignment is None:
        return xyz
    s, R, t = alignment_parts(alignment)
    return (xyz - t) @ R / s


def gcp_positions(survey, gcp_ids) -> dict[str, dict]:
    """Where the active Reconstruction puts each GCP, in its own frame:
    ``{gid: {"xyz", "from": "cloud"|"photos", "rms_px"?, "photos"}}``. A
    cloud Mark wins; photo Marks need two or more solved photos."""
    from cloudlabeller.photogrammetry.triangulate import triangulate

    rec = survey.active
    if rec is None:
        return {}
    cams = {r.name: r.camera for r in survey.dataset.images if r.camera is not None}
    out: dict[str, dict] = {}
    for gid in gcp_ids:
        photo = {n: uv for n, uv in survey.marks.get(gid, {}).items() if n in cams}
        if gid in rec.marks:
            out[gid] = {"xyz": np.asarray(rec.marks[gid], np.float64), "from": "cloud",
                        "photos": len(photo)}
        if len(photo) >= 2:
            X, rms = triangulate([cams[n] for n in photo], list(photo.values()))
            if gid in out:
                out[gid]["triangulated"] = X
                out[gid]["rms_px"] = rms
            else:
                out[gid] = {"xyz": X, "from": "photos", "rms_px": rms, "photos": len(photo)}
    return out


def _umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[dict, float]:
    from cloudlabeller.photogrammetry.georef import umeyama_similarity

    s, R, t = umeyama_similarity(src, dst)
    a = {"s": float(s), "R": np.asarray(R).tolist(), "t": np.asarray(t).tolist()}
    res = apply_points(a, src).astype(np.float64) - dst
    return a, float(np.sqrt((res ** 2).sum(axis=1).mean()))


def fit(gcps: dict, positions: dict[str, dict[str, dict]], targets: dict[str, np.ndarray],
        current: dict[str, dict | None], reference: str | None) -> dict:
    """The fit described in the module docstring.

    ``positions[sid][gid]``: each Survey's own-frame GCP positions.
    ``targets[gid]``: surveyed GCPs in the Project frame. ``current[sid]``:
    each Survey's Alignment now. Returns ``{"alignments": {sid: alignment},
    "surveys": {sid: {aligned, how, n_gcps, rms}}, "residuals": {gid: {sid:
    [dx, dy, dz]}}}``; only Surveys in ``alignments`` change.
    """
    control = {gid for gid, g in gcps.items() if g.use == "control"}
    known = {gid: np.asarray(t, np.float64) for gid, t in targets.items() if gid in control}
    aligned: dict[str, dict] = {}
    summary: dict[str, dict] = {}
    fixed: set[str] = set()                       # Surveys that keep their own frame
    if not known and positions:
        ref = reference if reference in positions else next(iter(positions))
        a = current.get(ref) or {"s": 1.0, "R": np.eye(3).tolist(), "t": [0.0, 0.0, 0.0],
                                 "source": "reference"}
        aligned[ref] = a
        if current.get(ref) is not None:
            fixed.add(ref)
        summary[ref] = {"aligned": True, "how": "reference", "n_gcps": 0, "rms": None}

    def ties():
        """Targets for GCPs without coordinates: the mean of where the
        aligned Surveys put them."""
        for gid in control - set(targets):
            seen = [apply_points(aligned[sid], positions[sid][gid]["xyz"][None])[0]
                    for sid in aligned if gid in positions[sid]]
            if seen:
                known[gid] = np.mean(np.asarray(seen, np.float64), axis=0)

    ties()
    progress = True
    while progress:
        progress = False
        for sid, pos in positions.items():
            if sid in aligned:
                continue
            use = [gid for gid in pos if gid in known and gid in control]
            if len(use) < MIN_GCPS:
                continue
            src = np.array([pos[g]["xyz"] for g in use])
            dst = np.array([known[g] for g in use])
            a, rms = _umeyama(src, dst)
            a.update(source="gcp", n_gcps=len(use), rms=rms)
            aligned[sid] = a
            summary[sid] = {"aligned": True, "how": "gcp", "n_gcps": len(use), "rms": rms}
            ties()
            progress = True
    for sid, pos in positions.items():
        if sid not in summary:
            n = sum(1 for g in pos if g in known and g in control)
            summary[sid] = {"aligned": current.get(sid) is not None, "how": "kept",
                            "n_gcps": n, "rms": None}

    residuals: dict[str, dict[str, list[float]]] = {}
    for gid in gcps:
        target = targets.get(gid)
        for sid, pos in positions.items():
            a = aligned.get(sid, current.get(sid))
            if gid not in pos or a is None:
                continue
            here = apply_points(a, pos[gid]["xyz"][None])[0].astype(np.float64)
            if target is not None:
                ref_point = np.asarray(target, np.float64)
            else:                                   # a tie: against the other Surveys
                others = [apply_points(aligned[o], positions[o][gid]["xyz"][None])[0]
                          for o in aligned if o != sid and gid in positions[o]]
                if not others:
                    continue
                ref_point = np.mean(np.asarray(others, np.float64), axis=0)
            residuals.setdefault(gid, {})[sid] = (here - ref_point).tolist()
    return {"alignments": {sid: a for sid, a in aligned.items() if sid not in fixed},
            "surveys": summary, "residuals": residuals}
