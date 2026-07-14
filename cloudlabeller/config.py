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

"""Application-level configuration (separate from per-project settings)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


def _config_dir() -> Path:
    return Path.home() / ".cloudlabeller"


@dataclass
class AppConfig:
    """User preferences that persist across sessions."""

    recent_projects: list[str] = field(default_factory=list)
    point_budget: int = 5_000_000          # LOD threshold for the 3D viewer
    default_brush_radius: float = 0.05      # in world units
    background_color: str = "#1e1e1e"
    colmap_binary: str | None = None        # optional path override

    @classmethod
    def path(cls) -> Path:
        return _config_dir() / "config.json"

    @classmethod
    def load(cls) -> "AppConfig":
        p = cls.path()
        if p.exists():
            try:
                return cls(**json.loads(p.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, TypeError):
                pass  # fall back to defaults on a corrupt/legacy file
        return cls()

    def save(self) -> None:
        p = self.path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    def add_recent(self, project_path: str, keep: int = 10) -> None:
        self.recent_projects = [project_path] + [
            p for p in self.recent_projects if p != project_path
        ]
        del self.recent_projects[keep:]
