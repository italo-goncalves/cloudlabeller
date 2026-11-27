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

"""Project = a folder on disk holding one site's Surveys (layout v2).

Layout::

    project.json          format_version 2, schema, settings, survey order
    surveys/<sid>/        each Survey (see :mod:`cloudlabeller.core.survey`)
    ml/models/            trained models

``settings["georeferenced"]`` is the Project frame: the ENU origin, and
``crs`` once reprojected. Each Reconstruction's Alignment carries its
products into that frame (ADR-0005). A 0.3.0 Project (layout v1) is moved
into this layout when it is opened (:mod:`cloudlabeller.core.migrate`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from cloudlabeller.core.label_schema import UNLABELLED_ID, LabelSchema
from cloudlabeller.core.labels import load_label_array, save_label_array
from cloudlabeller.core.survey import Survey, next_id

MANIFEST = "project.json"
FORMAT_VERSION = 2


class ProjectFormatError(RuntimeError):
    """The folder holds a Project this version can't read."""


@dataclass
class Project:
    root: Path
    schema: LabelSchema = field(default_factory=LabelSchema)
    settings: dict = field(default_factory=dict)
    surveys: dict[str, Survey] = field(default_factory=dict)
    gcps: dict = field(default_factory=dict)          # id -> GCP (core.gcp)

    # -- paths ------------------------------------------------------------------
    @property
    def surveys_dir(self) -> Path:
        return self.root / "surveys"

    @property
    def ml_dir(self) -> Path:
        return self.root / "ml"

    # -- surveys ------------------------------------------------------------------
    def survey(self, sid: str) -> Survey:
        """The Survey ``sid``, its photos, geometry and labels loaded."""
        return self.surveys[sid].ensure_loaded()

    def add_survey(self, name: str = "", **meta) -> Survey:
        sid = next_id("s", self.surveys)
        survey = Survey.create(self.surveys_dir / sid, sid, name, **meta)
        self.surveys[sid] = survey
        self.save_manifest()
        return survey

    @property
    def frame(self) -> dict | None:
        """The Project frame (``settings["georeferenced"]``), or None."""
        return self.settings.get("georeferenced")

    # -- classes --------------------------------------------------------------------
    def add_label(self, name: str | None = None, color: str | None = None):
        return self.schema.add(name=name, color=color)

    def rename_label(self, class_id: int, name: str) -> None:
        self.schema.rename(class_id, name)

    def set_label_color(self, class_id: int, color: str) -> None:
        self.schema.set_color(class_id, color)

    def delete_label(self, class_id: int) -> None:
        """Delete a Class, remapping every Survey's labels (also those of
        Reconstructions not active) to stay contiguous."""
        for sid in self.surveys:
            survey = self.survey(sid)
            survey.labels.remap_after_delete(class_id)
            for rec in survey.reconstructions.values():
                if rec is survey.active or not rec.labels_dir.exists():
                    continue
                for f in rec.labels_dir.glob("*.npy"):
                    arr = load_label_array(f)
                    arr[arr == class_id] = UNLABELLED_ID
                    arr[arr > class_id] -= 1
                    save_label_array(f, arr)
        self.schema.remove(class_id)

    # -- model settings ----------------------------------------------------------------
    def model_spec(self):
        """The persisted U-Net :class:`ModelSpec`; never set, the defaults
        with the Tile size the photos call for (:meth:`default_tile_size`)."""
        from cloudlabeller.ml.model_spec import ModelSpec

        saved = self.settings.get("model_spec")
        if saved:
            return ModelSpec.from_dict(saved)
        tile = self.default_tile_size()
        return ModelSpec(tile_size=tile) if tile else ModelSpec()

    def default_tile_size(self) -> int | None:
        """:func:`~cloudlabeller.ml.model_spec.default_tile_size` of the
        largest photos in any Survey, or None while no photo's size is known."""
        from cloudlabeller.ml.model_spec import default_tile_size

        sizes = [wh for s in self.surveys.values() if (wh := s.peek_resolution())]
        if not sizes:
            return None
        return default_tile_size(*max(sizes, key=lambda wh: max(wh)))

    def set_model_spec(self, spec) -> None:
        self.settings["model_spec"] = spec.to_dict()

    # -- summary ---------------------------------------------------------------------
    def summary_info(self) -> list[tuple[str, str]]:
        """(label, value) rows for the Project Info dialog."""
        rows: list[tuple[str, str]] = [("Project", str(self.root))]
        geo = self.frame
        if geo:
            crs_info = geo.get("crs")
            if crs_info:
                off = crs_info.get("offset") or [0, 0, 0]
                rows.append(("Coordinates", f"EPSG:{crs_info.get('epsg')} — "
                                            f"{crs_info.get('name', '?')}"))
                rows.append(("Heights", "EGM96 (sea level)"
                             if crs_info.get("orthometric") else "ellipsoidal"))
                if any(off):
                    rows.append(("Stored offset", f"{off[0]:.0f}, {off[1]:.0f}, "
                                                  f"{off[2]:.0f} (exports add it back)"))
            else:
                rows.append(("Coordinates",
                             f"{geo.get('frame', 'ENU')} — metres, true north"))
            lat, lon, alt = (geo.get("origin_lla") or [0, 0, 0])
            rows.append(("Origin (lat, lon, alt)", f"{lat:.7f}°, {lon:.7f}°, {alt:.1f} m"))
        else:
            rows.append(("Coordinates", "arbitrary local frame (not georeferenced)"))
        if self.schema.classes:
            rows.append(("Classes", ", ".join(f"{c.id}: {c.name}"
                                              for c in self.schema.classes)))
        spec = self.model_spec()
        rows.append(("U-Net spec", f"{spec.tile_size} px Tiles shrunk ×{spec.shrink} "
                                   f"({spec.net_size} px), {spec.channels} ch × "
                                   f"{spec.blocks} blocks, filter {spec.filter_size}"))
        from cloudlabeller.ml.model_store import list_models

        models = list_models(self.ml_dir)
        if models:
            rows.append(("Saved models", ", ".join(m["name"] for m in models)))
        for sid in self.surveys:
            survey = self.survey(sid)
            rows.append((f"Survey {sid}", survey.name))
            rows.extend((f"  {k}", v) for k, v in survey.summary_rows())
        return rows

    # -- lifecycle ---------------------------------------------------------------------
    @classmethod
    def create(cls, root: str | Path, schema: LabelSchema | None = None) -> "Project":
        """A new Project with one empty Survey."""
        root = Path(root)
        proj = cls(root=root, schema=schema or LabelSchema())
        (proj.ml_dir / "models").mkdir(parents=True, exist_ok=True)
        proj.surveys_dir.mkdir(parents=True, exist_ok=True)
        sid = next_id("s", ())
        proj.surveys[sid] = Survey.create(proj.surveys_dir / sid, sid)
        proj.save()
        return proj

    @classmethod
    def open(cls, root: str | Path) -> "Project":
        """Open a Project (moving a 0.3.0 one into layout v2 first). Surveys'
        photos, geometry and labels load when first asked for."""
        from cloudlabeller.core.migrate import migrate_if_needed

        root = Path(root)
        migrate_if_needed(root)
        data = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
        version = data.get("format_version", 1)
        if version != FORMAT_VERSION:
            raise ProjectFormatError(
                f"Project format {version} is not one this version reads "
                f"(it reads {FORMAT_VERSION}).")
        proj = cls(root=root, schema=LabelSchema.from_dict(data.get("schema", {})),
                   settings=data.get("settings", {}))
        order = list(data.get("surveys") or [])
        if proj.surveys_dir.exists():
            order += sorted(p.name for p in proj.surveys_dir.iterdir()
                            if (p / "survey.json").exists() and p.name not in order)
        for sid in order:
            if (proj.surveys_dir / sid / "survey.json").exists():
                proj.surveys[sid] = Survey.load(proj.surveys_dir / sid)
        from cloudlabeller.core.gcp import load_gcps

        proj.gcps = load_gcps(root)
        return proj

    def _write_manifest(self) -> None:
        manifest = {"format_version": FORMAT_VERSION, "schema": self.schema.to_dict(),
                    "settings": self.settings, "surveys": list(self.surveys)}
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        from cloudlabeller.core.gcp import save_gcps

        save_gcps(self.root, self.gcps)

    def save_manifest(self) -> None:
        """``project.json`` and every Survey's manifest: cheap, used for
        status changes."""
        self._write_manifest()
        for survey in self.surveys.values():
            survey.save_manifest()

    def save(self) -> None:
        """Manifests plus every loaded Survey's label arrays."""
        self._write_manifest()
        for survey in self.surveys.values():
            survey.save()
