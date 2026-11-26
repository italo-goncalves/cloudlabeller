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

"""GCPs: ground points recognised in the field and in the photos.

A GCP belongs to the Project (``gcps.json``). It may carry surveyed
coordinates, in a stated CRS with heights ellipsoidal or orthometric, and an
accuracy. Its *use* says whether it fits Alignments (``control``) or is only
measured against them (``check``). Where it was identified is a Mark: on a
Photo (kept by the Survey) or on a Reconstruction's cloud (kept by the
Reconstruction, in its own frame).
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path

GCP_FILE = "gcps.json"
USES = ("control", "check")


@dataclass
class GCP:
    id: str
    name: str
    xyz: list[float] | None = None        # surveyed coordinates, in ``epsg``
    epsg: int | None = None
    orthometric: bool = False             # heights above the geoid (EGM96)
    accuracy: float | None = None         # metres
    use: str = "control"

    @property
    def surveyed(self) -> bool:
        return self.xyz is not None and self.epsg is not None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GCP":
        xyz = d.get("xyz")
        return cls(id=str(d["id"]), name=str(d.get("name") or d["id"]),
                   xyz=[float(v) for v in xyz] if xyz is not None else None,
                   epsg=int(d["epsg"]) if d.get("epsg") is not None else None,
                   orthometric=bool(d.get("orthometric", False)),
                   accuracy=float(d["accuracy"]) if d.get("accuracy") is not None else None,
                   use=d.get("use") if d.get("use") in USES else "control")


def load_gcps(root: Path) -> dict[str, GCP]:
    path = Path(root) / GCP_FILE
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {g.id: g for g in (GCP.from_dict(d) for d in data.get("gcps", []))}


def save_gcps(root: Path, gcps: dict[str, GCP]) -> None:
    path = Path(root) / GCP_FILE
    if not gcps and not path.exists():
        return
    path.write_text(json.dumps({"gcps": [g.to_dict() for g in gcps.values()]}, indent=2),
                    encoding="utf-8")


def next_gcp_id(gcps) -> str:
    numbers = [int(k[1:]) for k in gcps if k[1:].isdigit()]
    return f"g{max(numbers, default=0) + 1}"


class CsvError(ValueError):
    """A coordinates file that can't be read; ``line`` says where."""

    def __init__(self, message: str, line: int | None = None) -> None:
        super().__init__(message)
        self.line = line


def parse_csv(text: str) -> list[tuple[str, list[float], float | None]]:
    """Rows ``name, x, y, z[, accuracy]`` from a coordinates file.

    Commas, semicolons or tabs separate the fields; a first line whose
    coordinates aren't numbers is a header. For a geographic CRS, x is the
    longitude and y the latitude.
    """
    first = next((line for line in text.splitlines() if line.strip()), "")
    delimiter = ";" if ";" in first else "	" if "	" in first else ","
    rows = []
    for i, row in enumerate(csv.reader(io.StringIO(text), delimiter=delimiter), 1):
        cells = [c.strip() for c in row]
        if not any(cells):
            continue
        if len(cells) < 4:
            raise CsvError(f"line {i} has {len(cells)} fields; name, x, y, z are needed", i)
        try:
            xyz = [float(c.replace(",", ".")) if delimiter != "," else float(c)
                   for c in cells[1:4]]
        except ValueError:
            if not rows and i == 1:
                continue                                    # a header
            raise CsvError(f"line {i}: the coordinates aren't numbers", i) from None
        acc = None
        if len(cells) > 4 and cells[4]:
            try:
                acc = float(cells[4].replace(",", "."))
            except ValueError:
                raise CsvError(f"line {i}: the accuracy isn't a number", i) from None
        if not cells[0]:
            raise CsvError(f"line {i} has no name", i)
        rows.append((cells[0], xyz, acc))
    if not rows:
        raise CsvError("no coordinates in the file")
    return rows
