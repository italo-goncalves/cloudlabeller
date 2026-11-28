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
from pathlib import Path
from typing import Callable

import numpy as np

from cloudlabeller.core import sources as src
from cloudlabeller.core.label_schema import UNLABELLED_ID


# -- label arrays on disk and on the wire --------------------------------------------
def narrow(labels: np.ndarray) -> np.ndarray:
    """``labels`` in the smallest signed integer type that holds them (int8
    for up to 127 classes): a photo's Mask is 20 MB as int8, 80 MB as int32,
    and the Engine reads the Project through WSL's slow ``/mnt/c``."""
    labels = np.asarray(labels)
    if labels.size == 0:
        return labels.astype(np.int8)
    lo, hi = int(labels.min()), int(labels.max())
    for dtype in (np.int8, np.int16):
        info = np.iinfo(dtype)
        if info.min <= lo and hi <= info.max:
            return labels.astype(dtype, copy=False)
    return labels.astype(np.int32, copy=False)


def save_label_array(path: Path, labels: np.ndarray) -> None:
    """A Mask or a cloud's labels as ``.npy``, narrowed (any older int32
    file reads back the same)."""
    np.save(path, narrow(labels))


def load_label_array(path: Path) -> np.ndarray:
    """A label array saved by :func:`save_label_array`, as int32 (what the
    LabelStore holds)."""
    return np.load(path).astype(np.int32, copy=False)


#: A Reconstruction's label files: ``(LabelStore attribute, file name)``.
LABEL_FILES = (("cloud_labels", "sparse.npy"), ("dense_cloud_labels", "dense.npy"),
               ("mesh_labels", "mesh.npy"))
#: Beside them, each cloud's sources and Uncertainty (ADR-0006).
PROVENANCE_FILES = (("cloud_source", "sparse.source.npy"),
                    ("cloud_uncertainty", "sparse.uncertainty.npy"),
                    ("dense_source", "dense.source.npy"),
                    ("dense_uncertainty", "dense.uncertainty.npy"))
#: The files that go with each product slot when a child shares it.
SLOT_FILES = {"sfm": ("sparse.npy", "sparse.source.npy", "sparse.uncertainty.npy"),
              "mvs": ("dense.npy", "dense.source.npy", "dense.uncertainty.npy")}


def read_label_files(store: "LabelStore", directory: Path) -> None:
    """Load a Reconstruction's label arrays (those present) into ``store``."""
    for attr, fname in LABEL_FILES:
        if (directory / fname).exists():
            setattr(store, attr, load_label_array(directory / fname))
    for attr, fname in PROVENANCE_FILES:
        setattr(store, attr, np.load(directory / fname) if (directory / fname).exists()
                else None)


def write_label_files(store: "LabelStore", directory: Path) -> None:
    """Save ``store``'s label arrays (those it has) into a Reconstruction."""
    directory.mkdir(parents=True, exist_ok=True)
    for attr, fname in LABEL_FILES:
        arr = getattr(store, attr)
        if arr is not None:
            save_label_array(directory / fname, arr)
    for attr, fname in PROVENANCE_FILES:
        arr = getattr(store, attr)
        if arr is not None:
            np.save(directory / fname, arr)


class Modality(str, Enum):
    CLOUD = "cloud"
    IMAGE = "image"


@dataclass
class _PaintCommand:
    """A reversible assignment of ``values`` to ``indices`` of one array.

    ``values`` may be a scalar class id (painting) or an array of per-index
    labels (bulk replace, e.g. a transfer result) — both replay correctly on
    redo. ``modality``/``key`` identify what changed so the UI can refresh the
    right view after undo/redo. ``also`` holds more ``(array, values)`` pairs
    changed at the same indices in the same step (a cloud's source and
    Uncertainty beside its labels).
    """

    target: np.ndarray
    indices: np.ndarray
    values: int | np.ndarray
    modality: "Modality" = None
    key: str | None = None
    also: list = field(default_factory=list)
    _previous: list | None = None

    def _pairs(self) -> list:
        return [(self.target, self.values), *self.also]

    def apply(self) -> None:
        self._previous = [a[self.indices].copy() for a, _ in self._pairs()]
        for a, v in self._pairs():
            a[self.indices] = v

    def undo(self) -> None:
        if self._previous is not None:
            for (a, _), before in zip(self._pairs(), self._previous):
                a[self.indices] = before


@dataclass
class LabelStore:
    """Owns per-point cloud labels and per-image pixel masks.

    Notifications are delivered through :meth:`subscribe`; the Engine turns
    them into its own notifications (this module knows no UI).
    """

    cloud_labels: np.ndarray | None = None          # (N,) int32 sparse-cloud labels
    dense_cloud_labels: np.ndarray | None = None    # (M,) int32 dense-cloud labels
    mesh_labels: np.ndarray | None = None           # (V,) int32 per-vertex (derived
    #                                                 from the dense cloud by NN)
    image_masks: dict[str, np.ndarray] = field(default_factory=dict)  # name -> (H,W) int32
    # Per-image label provenance: "user" (drawn polygons) > "auto" (transfer/ML).
    # Absent = "none" (unlabelled). Drives the frustum dot colour.
    image_status: dict[str, str] = field(default_factory=dict)
    # Each cloud's label sources (uint8, core.sources) and Uncertainty
    # (float16, NaN unknown), beside its labels (ADR-0006). None: not made
    # yet; :meth:`provenance` makes them from the labels.
    cloud_source: np.ndarray | None = None
    cloud_uncertainty: np.ndarray | None = None
    dense_source: np.ndarray | None = None
    dense_uncertainty: np.ndarray | None = None

    _undo_stack: list[_PaintCommand] = field(default_factory=list)
    _redo_stack: list[_PaintCommand] = field(default_factory=list)
    _listeners: list[Callable[[Modality, str | None], None]] = field(default_factory=list)

    # -- allocation --------------------------------------------------------
    def init_cloud(self, n_points: int) -> None:
        self.cloud_labels = np.full(n_points, UNLABELLED_ID, dtype=np.int32)
        self.cloud_source = self.cloud_uncertainty = None

    def init_image(self, name: str, height: int, width: int) -> None:
        self.image_masks[name] = np.full((height, width), UNLABELLED_ID, dtype=np.int32)

    def init_dense_cloud(self, n_points: int) -> None:
        self.dense_cloud_labels = np.full(n_points, UNLABELLED_ID, dtype=np.int32)
        self.dense_source = self.dense_uncertainty = None

    def set_dense_cloud_labels(self, labels: np.ndarray) -> None:
        self.dense_cloud_labels = np.asarray(labels, dtype=np.int32)
        self.dense_source = self.dense_uncertainty = None
        self._notify(Modality.CLOUD, None)

    def provenance(self, dense: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """A cloud's ``(source, uncertainty)`` arrays, made from its labels
        when missing or out of step (labels from before sources existed are
        hand-drawn, of unknown Uncertainty)."""
        labels = self.dense_cloud_labels if dense else self.cloud_labels
        if labels is None:
            raise RuntimeError("cloud labels not initialised")
        prefix = "dense" if dense else "cloud"
        source = getattr(self, f"{prefix}_source")
        unc = getattr(self, f"{prefix}_uncertainty")
        if source is None or len(source) != len(labels):
            source = src.legacy(labels)[0]
            setattr(self, f"{prefix}_source", source)
        if unc is None or len(unc) != len(labels):
            unc = src.legacy(labels)[1]
            setattr(self, f"{prefix}_uncertainty", unc)
        return source, unc

    def _paint_points(self, dense: bool, indices: np.ndarray,
                      class_id: int | np.ndarray) -> None:
        """A stroke on a cloud: painted where a Class is set (Uncertainty 0),
        no source where it is erased."""
        labels = self.dense_cloud_labels if dense else self.cloud_labels
        if labels is None:
            raise RuntimeError(("dense " if dense else "") + "cloud labels not initialised")
        source, unc = self.provenance(dense)
        erased = np.asarray(class_id) == UNLABELLED_ID
        new_source = np.where(erased, np.uint8(src.NONE), np.uint8(src.PAINTED))
        new_unc = np.where(erased, np.float16(np.nan), np.float16(0))
        self._run(_PaintCommand(labels, np.asarray(indices), class_id, Modality.CLOUD, None,
                                also=[(source, new_source), (unc, new_unc)]))

    def paint_dense_cloud(self, indices: np.ndarray, class_id: int | np.ndarray) -> None:
        """Undoable label assignment on the dense cloud (e.g. lasso selection)."""
        self._paint_points(True, indices, class_id)

    # -- label provenance --------------------------------------------------
    def mark_user_labeled(self, name: str) -> None:
        """User drew a polygon on this image (wins over auto)."""
        self.image_status[name] = "user"

    def mark_auto_labeled(self, name: str) -> None:
        """Image was labelled by cloud projection (label transfer), unless the
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
            for attr in ("cloud_source", "cloud_uncertainty"):
                arr = getattr(self, attr)
                if arr is not None and len(arr) == len(keep):
                    setattr(self, attr, arr[keep])
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
        for labels, source, unc in (
                (self.cloud_labels, self.cloud_source, self.cloud_uncertainty),
                (self.dense_cloud_labels, self.dense_source, self.dense_uncertainty)):
            gone = None if labels is None else labels == UNLABELLED_ID
            if source is not None and gone is not None and len(source) == len(gone):
                source[gone] = src.NONE
            if unc is not None and gone is not None and len(unc) == len(gone):
                unc[gone] = np.nan
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
        self._paint_points(False, indices, class_id)

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
        source, unc = self.provenance()
        new_source, new_unc = src.legacy(labels)
        self._run(_PaintCommand(self.cloud_labels, np.arange(len(labels)),
                                labels.copy(), Modality.CLOUD, None,
                                also=[(source, new_source), (unc, new_unc)]))

    def merge_cloud_labels(self, labels: np.ndarray, dense: bool = False,
                           source: np.ndarray | int | None = None,
                           uncertainty: np.ndarray | None = None) -> int:
        """Merge transferred labels into the existing array (one undoable command).

        Only the points the transfer actually labelled are updated; every other
        point keeps its current label (lasso work, earlier transfers…). A point
        is updated only when the new label's ``source`` (an array or one code;
        default hand-drawn) is as good as the one it has or better, and never
        when it was painted on the cloud (ADR-0006). Returns the number of
        points updated.
        """
        labels = np.asarray(labels, dtype=np.int32)
        n = len(labels)
        if dense:
            if self.dense_cloud_labels is None or len(self.dense_cloud_labels) != n:
                self.init_dense_cloud(n)
            target = self.dense_cloud_labels
        else:
            if self.cloud_labels is None:
                self.init_cloud(n)
            target = self.cloud_labels
        have_source, have_unc = self.provenance(dense)
        new_source = np.broadcast_to(
            np.asarray(src.HAND if source is None else source, np.uint8), (n,))
        new_unc = (np.full(n, np.nan, np.float16) if uncertainty is None
                   else np.asarray(uncertainty, np.float16))
        ok = ((labels != UNLABELLED_ID) & (have_source != src.PAINTED)
              & (src.rank(new_source) <= src.rank(have_source)))
        idx = np.flatnonzero(ok)
        if idx.size:
            self._run(_PaintCommand(target, idx, labels[idx], Modality.CLOUD, None,
                                    also=[(have_source, new_source[idx]),
                                          (have_unc, new_unc[idx])]))
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
