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

"""Label storage with an undo/redo command stack.

The same store backs both modalities so that a transfer (image->cloud or
cloud->image) and a manual paint are recorded uniformly and can be undone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import numpy as np

from cloudlabeller.core.label_schema import UNLABELLED_ID


class Modality(str, Enum):
    CLOUD = "cloud"
    IMAGE = "image"


@dataclass
class _PaintCommand:
    """A reversible assignment of ``values`` to ``indices`` of one array.

    ``values`` may be a scalar class id (painting) or an array of per-index
    labels (bulk replace, e.g. a transfer result) — both replay correctly on
    redo. ``modality``/``key`` identify what changed so the UI can refresh the
    right view after undo/redo.
    """

    target: np.ndarray
    indices: np.ndarray
    values: int | np.ndarray
    modality: "Modality" = None
    key: str | None = None
    _previous: np.ndarray | None = None

    def apply(self) -> None:
        self._previous = self.target[self.indices].copy()
        self.target[self.indices] = self.values

    def undo(self) -> None:
        if self._previous is not None:
            self.target[self.indices] = self._previous


@dataclass
class LabelStore:
    """Owns per-point cloud labels and per-image pixel masks.

    Notifications are delivered through :meth:`subscribe`; the UI layer wires
    these to the Qt event bus (keeping this module Qt-free).
    """

    cloud_labels: np.ndarray | None = None          # (N,) int32 sparse-cloud labels
    dense_cloud_labels: np.ndarray | None = None    # (M,) int32 dense-cloud labels
    mesh_labels: np.ndarray | None = None           # (V,) int32 per-vertex (derived
    #                                                 from the dense cloud by NN)
    image_masks: dict[str, np.ndarray] = field(default_factory=dict)  # name -> (H,W) int32
    # Per-image label provenance: "user" (drawn polygons) > "auto" (hylite/ML).
    # Absent = "none" (unlabelled). Drives the frustum dot colour.
    image_status: dict[str, str] = field(default_factory=dict)

    _undo_stack: list[_PaintCommand] = field(default_factory=list)
    _redo_stack: list[_PaintCommand] = field(default_factory=list)
    _listeners: list[Callable[[Modality, str | None], None]] = field(default_factory=list)

    # -- allocation --------------------------------------------------------
    def init_cloud(self, n_points: int) -> None:
        self.cloud_labels = np.full(n_points, UNLABELLED_ID, dtype=np.int32)

    def init_image(self, name: str, height: int, width: int) -> None:
        self.image_masks[name] = np.full((height, width), UNLABELLED_ID, dtype=np.int32)

    def init_dense_cloud(self, n_points: int) -> None:
        self.dense_cloud_labels = np.full(n_points, UNLABELLED_ID, dtype=np.int32)

    def set_dense_cloud_labels(self, labels: np.ndarray) -> None:
        self.dense_cloud_labels = np.asarray(labels, dtype=np.int32)
        self._notify(Modality.CLOUD, None)

    def paint_dense_cloud(self, indices: np.ndarray, class_id: int | np.ndarray) -> None:
        """Undoable label assignment on the dense cloud (e.g. lasso selection)."""
        if self.dense_cloud_labels is None:
            raise RuntimeError("dense cloud labels not initialised")
        self._run(_PaintCommand(self.dense_cloud_labels, np.asarray(indices), class_id,
                                Modality.CLOUD, None))

    # -- label provenance --------------------------------------------------
    def mark_user_labeled(self, name: str) -> None:
        """User drew a polygon on this image (wins over auto)."""
        self.image_status[name] = "user"

    def mark_auto_labeled(self, name: str) -> None:
        """Image was labelled by cloud projection (hylite transfer), unless the
        user has already drawn on it. Overwrites 'ml': a newer cloud render
        supersedes an older U-Net prediction."""
        if self.image_status.get(name) != "user":
            self.image_status[name] = "auto"

    def mark_ml_labeled(self, name: str) -> None:
        """Image was labelled by a U-Net prediction (stored under
        ml/predictions/), unless the user has already drawn on it."""
        if self.image_status.get(name) != "user":
            self.image_status[name] = "ml"

    def status_of(self, name: str) -> str:
        """One of 'user', 'auto' (cloud projection), 'ml' (U-Net prediction),
        or 'none'. Both 'auto' and 'ml' render as yellow dots; they differ in
        where the overlay mask comes from."""
        return self.image_status.get(name, "none")

    def filter_cloud(self, keep: np.ndarray) -> None:
        """Structural edit: the sparse cloud was filtered to ``keep`` (bool
        mask). Labels follow the surviving points; the paint history refers to
        old indices, so it is cleared."""
        if self.cloud_labels is not None and len(self.cloud_labels) == len(keep):
            self.cloud_labels = self.cloud_labels[keep]
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._notify(Modality.CLOUD, None)

    # -- structural edits (label-class deletion) ---------------------------
    def remap_after_delete(self, class_id: int) -> None:
        """Update label data when a class is deleted: points of that class become
        unlabelled (-1); classes numbered above it shift down by one."""
        def remap(arr: np.ndarray | None) -> None:
            if arr is None:
                return
            arr[arr == class_id] = UNLABELLED_ID
            arr[arr > class_id] -= 1

        remap(self.cloud_labels)
        remap(self.dense_cloud_labels)
        remap(self.mesh_labels)
        for mask in self.image_masks.values():
            remap(mask)
        self._undo_stack.clear()   # structural change invalidates the paint history
        self._redo_stack.clear()
        self._notify_all()

    # -- notifications -----------------------------------------------------
    def subscribe(self, fn: Callable[[Modality, str | None], None]) -> None:
        self._listeners.append(fn)

    def _notify(self, modality: Modality, key: str | None = None) -> None:
        for fn in self._listeners:
            fn(modality, key)

    # -- editing -----------------------------------------------------------
    # ``class_id`` may be a scalar or a per-index array of labels.
    def paint_cloud(self, indices: np.ndarray, class_id: int | np.ndarray) -> None:
        if self.cloud_labels is None:
            raise RuntimeError("cloud labels not initialised")
        self._run(_PaintCommand(self.cloud_labels, np.asarray(indices), class_id,
                                Modality.CLOUD, None))

    def paint_image(self, name: str, indices: np.ndarray, class_id: int) -> None:
        mask = self.image_masks[name]
        # ``indices`` are flat indices into the (H*W) mask.
        self._run(_PaintCommand(mask.reshape(-1), np.asarray(indices), class_id,
                                Modality.IMAGE, name))

    def set_cloud_labels(self, labels: np.ndarray) -> None:
        """Bulk replace. Recorded as one undoable command whose redo restores
        exactly these labels. Prefer :meth:`merge_cloud_labels` for transfers —
        replacing wipes labels assigned by other tools (e.g. the 3D lasso)."""
        labels = np.asarray(labels, dtype=np.int32)
        if self.cloud_labels is None:
            self.init_cloud(len(labels))
        self._run(_PaintCommand(self.cloud_labels, np.arange(len(labels)),
                                labels.copy(), Modality.CLOUD, None))

    def merge_cloud_labels(self, labels: np.ndarray, dense: bool = False) -> int:
        """Merge transferred labels into the existing array (one undoable command).

        Only the points the transfer actually labelled are updated; every other
        point keeps its current label (lasso work, earlier transfers…). Returns
        the number of points updated.
        """
        labels = np.asarray(labels, dtype=np.int32)
        idx = np.flatnonzero(labels != UNLABELLED_ID)
        if dense:
            if (self.dense_cloud_labels is None
                    or len(self.dense_cloud_labels) != len(labels)):
                self.init_dense_cloud(len(labels))
            if idx.size:
                self.paint_dense_cloud(idx, labels[idx])
        else:
            if self.cloud_labels is None:
                self.init_cloud(len(labels))
            if idx.size:
                self.paint_cloud(idx, labels[idx])
        return int(idx.size)

    def _run(self, cmd: _PaintCommand) -> None:
        """Apply a paint command and record it for undo (clears redo)."""
        cmd.apply()
        self._undo_stack.append(cmd)
        self._redo_stack.clear()
        self._notify(cmd.modality, cmd.key)

    # -- undo / redo -------------------------------------------------------
    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    def can_redo(self) -> bool:
        return bool(self._redo_stack)

    def undo(self) -> tuple[Modality, str | None] | None:
        """Undo the last edit; returns (modality, key) of what changed, or None."""
        if not self._undo_stack:
            return None
        cmd = self._undo_stack.pop()
        cmd.undo()
        self._redo_stack.append(cmd)
        self._notify(cmd.modality, cmd.key)
        return cmd.modality, cmd.key

    def redo(self) -> tuple[Modality, str | None] | None:
        """Redo the last undone edit; returns (modality, key) or None."""
        if not self._redo_stack:
            return None
        cmd = self._redo_stack.pop()
        cmd.apply()
        self._undo_stack.append(cmd)
        self._notify(cmd.modality, cmd.key)
        return cmd.modality, cmd.key

    def _notify_all(self) -> None:
        self._notify(Modality.CLOUD, None)
        for name in self.image_masks:
            self._notify(Modality.IMAGE, name)
