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

"""Move a 0.3.0 Project (layout v1) into layout v2, one Survey ``s1``.

Only renames, inside the Project folder, so it is quick and nothing is
copied. The list of renames is written to ``.migration.json`` first and
each rename is skipped once done, so a migration that was interrupted
resumes the next time the Project is opened. ``project.json`` is rewritten
last; the v1 one is kept as ``project.v1.json``.

A Project georeferenced in place by 0.3.0 keeps its files as they are, with
an identity Alignment ("legacy") and its frame in the Project settings.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

JOURNAL = ".migration.json"
V1_BACKUP = "project.v1.json"
SID, RID = "s1", "r1"

# Settings that belonged to the one reconstruction / image set in v1.
_SURVEY_SETTINGS = ("reconstruction_outdated", "source_resolution")


def needs_migration(root: Path) -> bool:
    root = Path(root)
    if (root / JOURNAL).exists():
        return True
    try:
        data = json.loads((root / "project.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return int(data.get("format_version", 1)) < 2


def _plan(root: Path) -> list[list[str]]:
    """The renames, as root-relative [from, to] pairs, for what exists."""
    s = f"surveys/{SID}"
    moves = [
        ["images", f"{s}/images"],
        ["reconstruction/dense", f"{s}/products/mvs-1/dense"],
        ["reconstruction/dense.npz", f"{s}/products/mvs-1/dense.npz"],
        ["reconstruction/pmap", f"{s}/cache/pmap"],
        ["reconstruction", f"{s}/products/sfm-1"],
        ["products/mesh.npz", f"{s}/products/mesh-1/mesh.npz"],
        ["labels/images", f"{s}/masks"],
        ["labels/cloud_labels.npy", f"{s}/reconstructions/{RID}/labels/sparse.npy"],
        ["labels/dense_cloud_labels.npy", f"{s}/reconstructions/{RID}/labels/dense.npy"],
        ["labels/mesh_labels.npy", f"{s}/reconstructions/{RID}/labels/mesh.npy"],
        ["labels/mesh_nn.npz", f"{s}/reconstructions/{RID}/labels/mesh_nn.npz"],
        ["ml/predictions", f"{s}/predictions"],
    ]
    return [m for m in moves if (root / m[0]).exists()]


def _rename(root: Path, src: str, dst: str) -> None:
    a, b = root / src, root / dst
    if not a.exists() or b.exists():
        return                                   # done already (or nothing to move)
    b.parent.mkdir(parents=True, exist_ok=True)
    os.replace(a, b)


def migrate_if_needed(root: str | Path) -> bool:
    """Migrate ``root`` when it is a v1 Project (or a migration was cut
    short). Returns whether anything was done."""
    root = Path(root)
    if not needs_migration(root):
        return False
    journal = root / JOURNAL
    if journal.exists():
        state = json.loads(journal.read_text(encoding="utf-8"))
    else:
        v1 = json.loads((root / "project.json").read_text(encoding="utf-8"))
        state = {"v1": v1, "moves": _plan(root)}
        shutil.copyfile(root / "project.json", root / V1_BACKUP)
        journal.write_text(json.dumps(state, indent=2), encoding="utf-8")

    for src, dst in state["moves"]:
        _rename(root, src, dst)
    _write_v2(root, state["v1"])
    for d in ("products", "labels", "ml/checkpoints"):
        try:
            (root / d).rmdir()                   # only when left empty
        except OSError:
            pass
    journal.unlink()
    return True


def _write_v2(root: Path, v1: dict) -> None:
    from cloudlabeller.core.reconstruction import Reconstruction, identity_alignment
    from cloudlabeller.core.survey import Survey

    settings = dict(v1.get("settings") or {})
    survey_settings = {k: settings.pop(k) for k in _SURVEY_SETTINGS if k in settings}
    sdir = root / "surveys" / SID
    survey = Survey(id=SID, dir=sdir, name="Survey 1", settings=survey_settings)
    survey.labels.image_status = dict(v1.get("image_status") or {})

    sfm = sdir / "products" / "sfm-1"
    has = {"sfm": (sfm / "cloud.npz").exists(),
           "mvs": (sdir / "products" / "mvs-1" / "dense.npz").exists(),
           "mesh": (sdir / "products" / "mesh-1" / "mesh.npz").exists()}
    if any(has.values()):
        geo = settings.get("georeferenced")
        alignment = None
        if geo:
            alignment = identity_alignment("legacy")
            for k_from, k_to in (("scale_m_per_unit", "scale_m_per_unit"),
                                 ("n_gps", "n_gps")):
                if geo.get(k_from) is not None:
                    alignment[k_to] = geo[k_from]
        rec = Reconstruction(
            id=RID, dir=sdir / "reconstructions" / RID, name="Reconstruction 1",
            products={"sfm": "sfm-1" if has["sfm"] else None,
                      "mvs": "mvs-1" if has["mvs"] else None,
                      "mesh": "mesh-1" if has["mesh"] else None},
            alignment=alignment)
        survey.reconstructions[RID] = rec
        survey.active_id = RID
    for d in (survey.images_dir, survey.masks_dir, survey.predictions_dir,
              survey.products_dir, survey.reconstructions_dir):
        d.mkdir(parents=True, exist_ok=True)
    survey.save_manifest()

    manifest = {"format_version": 2, "schema": v1.get("schema", {}),
                "settings": settings, "surveys": [SID]}
    (root / "ml" / "models").mkdir(parents=True, exist_ok=True)
    (root / "project.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
