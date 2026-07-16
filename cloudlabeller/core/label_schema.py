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

"""Label classes: the shared vocabulary used by both 2D and 3D labelling.

Model:
  * **Unlabelled is ``-1``** — the default for every point/pixel; it is not a
    user class and never appears in the schema list.
  * User classes are stored **contiguously numbered 0..N-1**; a class's ``id`` is
    its position in :attr:`LabelSchema.classes`. Deleting a class renumbers the
    rest so the range stays 0..N-1 (the label *data* is remapped separately by
    :meth:`LabelStore.remap_after_delete`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Sentinel for "no label". Stored in the label arrays; never a user class.
UNLABELLED_ID = -1

# Default colours handed out to new classes, cycled by index.
DEFAULT_PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
    "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080", "#e6beff",
    "#9a6324", "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd8b1",
]


@dataclass
class LabelClass:
    """A single semantic class. ``id`` equals its index in the schema."""

    id: int
    name: str
    color: str = "#cccccc"   # hex RGB, used for both 2D mask overlay and 3D LUT
    hotkey: str | None = None

    def rgb(self) -> tuple[int, int, int]:
        h = self.color.lstrip("#")
        return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


@dataclass
class LabelSchema:
    """Ordered, contiguously-numbered set of user label classes."""

    classes: list[LabelClass] = field(default_factory=list)

    # -- lookups -----------------------------------------------------------
    def by_id(self, class_id: int) -> LabelClass:
        return self.classes[class_id]

    def __len__(self) -> int:
        return len(self.classes)

    # -- editing -----------------------------------------------------------
    def add(self, name: str | None = None, color: str | None = None,
            hotkey: str | None = None) -> LabelClass:
        """Append a class with the next contiguous id; name/colour default
        to "label <id>" and the palette colour for that id."""
        cid = len(self.classes)
        cls = LabelClass(
            id=cid,
            name=name if name is not None else f"label {cid}",
            color=color if color is not None else DEFAULT_PALETTE[cid % len(DEFAULT_PALETTE)],
            hotkey=hotkey,
        )
        self.classes.append(cls)
        return cls

    def rename(self, class_id: int, name: str) -> None:
        self.classes[class_id].name = name

    def set_color(self, class_id: int, color: str) -> None:
        self.classes[class_id].color = color

    def remove(self, class_id: int) -> None:
        """Remove a class and renumber the rest to stay contiguous (0..N-1)."""
        del self.classes[class_id]
        for index, cls in enumerate(self.classes):
            cls.id = index

    # -- serialisation -----------------------------------------------------
    def to_dict(self) -> dict:
        return {"classes": [vars(c) for c in self.classes]}

    @classmethod
    def from_dict(cls, data: dict) -> "LabelSchema":
        schema = cls(classes=[LabelClass(**c) for c in data.get("classes", [])])
        # Be forgiving of older/edited files: enforce contiguous 0..N-1 ids.
        for index, c in enumerate(schema.classes):
            c.id = index
        return schema

    def lookup_table(self) -> dict[int, tuple[int, int, int]]:
        """id -> RGB, for building VTK/Qt colour maps (user classes only)."""
        return {c.id: c.rgb() for c in self.classes}
