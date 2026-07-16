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

"""Image pane: show an image with its label overlay and draw polygon labels.

Click an image (Dataset pane or its 3D frustum) to load it here. With *Draw
polygon* active, click to drop vertices; *Done* closes the polygon and fills the
image's int32 label array with the active class code. A slider sets the overlay
transparency. Edits flow into the project's :class:`LabelStore` and are saved
automatically (autosave timer + on close).
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    qRgba,
)
from PySide6.QtWidgets import (
    QFileDialog,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from cloudlabeller.core.events import EventBus
from cloudlabeller.core.raster import mask_from_file, mask_to_indexed, rasterize_polygon


class _Canvas(QGraphicsView):
    """Pannable/zoomable image canvas.

    While drawing: left-click drops a vertex, right-click undoes the last one.
    Middle-drag pans in any mode. Wheel zooms.
    """

    clicked = Signal(QPointF)
    undo_requested = Signal()
    key_pressed = Signal(int)        # labelling shortcuts, handled by ImageView

    #: keys forwarded to ImageView when the canvas has focus
    SHORTCUT_KEYS = frozenset(
        [Qt.Key_D, Qt.Key_Return, Qt.Key_Enter, Qt.Key_Escape,
         Qt.Key_Left, Qt.Key_Right]
        + list(range(Qt.Key_0, Qt.Key_9 + 1)))

    def __init__(self, scene: QGraphicsScene) -> None:
        super().__init__(scene)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setContextMenuPolicy(Qt.NoContextMenu)      # right-click is "undo point"
        self.setFocusPolicy(Qt.StrongFocus)              # click to grab keyboard
        self._draw = False
        self._panning = False
        self._pan_last = QPointF()

    def keyPressEvent(self, event) -> None:
        if event.key() in self.SHORTCUT_KEYS:
            self.key_pressed.emit(event.key())
            return
        super().keyPressEvent(event)

    def set_draw_mode(self, on: bool) -> None:
        self._draw = on
        self.setDragMode(QGraphicsView.NoDrag if on else QGraphicsView.ScrollHandDrag)
        self.setCursor(Qt.CrossCursor if on else Qt.ArrowCursor)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MiddleButton:            # pan with middle-drag
            self._panning = True
            self._pan_last = event.position()
            self.setCursor(Qt.ClosedHandCursor)
            return
        if self._draw and event.button() == Qt.LeftButton:
            self.clicked.emit(self.mapToScene(event.position().toPoint()))
            return
        if self._draw and event.button() == Qt.RightButton:
            self.undo_requested.emit()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._panning:
            delta = event.position() - self._pan_last
            self._pan_last = event.position()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(delta.x()))
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(delta.y()))
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MiddleButton and self._panning:
            self._panning = False
            self.setCursor(Qt.CrossCursor if self._draw else Qt.ArrowCursor)
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)


def _load_payload(project, name: str, record, seq: int) -> dict:
    """Heavy per-image work, safe on a worker thread: decode the photo (QImage is
    thread-safe; QPixmap is not — conversion stays on the UI thread), render the
    auto mask from the cloud if needed, and colourise the label overlay."""
    image = QImage(str(record.path))
    payload = {"seq": seq, "name": name, "path": str(record.path),
               "image": image, "auto_mask": None, "overlay": None}
    if image.isNull():
        return payload
    labels = project.labels
    mask = labels.image_masks.get(name)
    if mask is None and labels.status_of(name) in ("auto", "ml"):
        mask = payload["auto_mask"] = project.render_auto_mask(name)
    if mask is not None:
        payload["overlay"] = mask_to_indexed(mask, project.schema.lookup_table())
    return payload


class ImageView(QWidget):
    """The Image pane: canvas + toolbar. Shows the selected image with its
    label overlay, drives polygon drawing (draw/finish/done/cancel), and
    offers per-image propagation to the cloud and mask regeneration."""

    propagate_requested = Signal(str)   # ask to spread this image's labels to neighbours

    def __init__(self, bus: EventBus) -> None:
        super().__init__()
        self.bus = bus
        self.project = None
        self.current_name: str | None = None
        self._active_class = -1
        self._w = self._h = 0

        self._base_item = None
        self._overlay_item = None
        self._overlay_buf: np.ndarray | None = None   # keep RGBA buffer alive
        self._auto_mask: np.ndarray | None = None     # transient, current image only
        self._poly_item: QGraphicsPathItem | None = None
        self._markers: list = []
        self._vertices: list[tuple[float, float]] = []
        self._load_seq = 0                            # supersedes stale async loads

        self.scene = QGraphicsScene(self)
        self.canvas = _Canvas(self.scene)

        # -- toolbar -------------------------------------------------------
        self.btn_draw = QToolButton()
        self.btn_draw.setText("Draw polygon")
        self.btn_draw.setCheckable(True)
        self.btn_draw.setToolTip("Toggle polygon drawing (D)")
        self.btn_finish = QPushButton("Finish")           # exit drawing mode
        self.btn_finish.setToolTip("Finish editing (leave polygon-drawing mode)")
        self.btn_done = QPushButton("Done")
        self.btn_done.setToolTip("Close the current polygon and apply the "
                                 "active label (Enter)")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setToolTip("Discard the current polygon (Esc)")
        self.btn_propagate = QPushButton("Propagate →")
        self.btn_propagate.setToolTip(
            "Project this image's labels via the 3D cloud onto all other images "
            "that see the same area (they become auto-labelled).")
        self.btn_mask = QPushButton("Import Mask…")
        self.btn_mask.setToolTip(
            "Label this image from an external mask file with the active class: "
            "a .jpg selects the highest-intensity pixels, a .png the pixels "
            "with the highest alpha. The mask must match the image size.")
        self.lbl_name = QLabel("")                    # current image's file name
        self.lbl_name.setStyleSheet("color: gray;")
        # Let the label clip instead of forcing the pane wide enough for a
        # long file name — the dock must stay shrinkable.
        self.lbl_name.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.lbl_name.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(50)
        self.slider.setFixedWidth(120)
        self.slider.setToolTip("Label overlay transparency")

        # Two button rows (edit row / action row) so the dock can be narrowed;
        # right column: file name with the opacity slider below it.
        row_edit = QHBoxLayout()
        for b in (self.btn_draw, self.btn_finish, self.btn_done, self.btn_cancel):
            row_edit.addWidget(b)
        row_edit.addStretch(1)
        row_act = QHBoxLayout()
        for b in (self.btn_propagate, self.btn_mask):
            row_act.addWidget(b)
        row_act.addStretch(1)

        opacity = QHBoxLayout()
        opacity.addStretch(1)
        opacity.addWidget(QLabel("Opacity"))
        opacity.addWidget(self.slider)

        bar = QGridLayout()
        bar.addLayout(row_edit, 0, 0)
        bar.addLayout(row_act, 1, 0)
        bar.addWidget(self.lbl_name, 0, 1)
        bar.addLayout(opacity, 1, 1)
        bar.setColumnStretch(0, 3)
        bar.setColumnStretch(1, 2)

        layout = QVBoxLayout(self)
        layout.addLayout(bar)
        layout.addWidget(self.canvas, 1)

        self.btn_draw.toggled.connect(self._on_draw_toggled)
        self.btn_finish.clicked.connect(lambda: self.btn_draw.setChecked(False))
        self.btn_done.clicked.connect(self._commit)
        self.btn_cancel.clicked.connect(self._cancel)
        self.btn_mask.clicked.connect(self._import_mask)
        self.btn_propagate.clicked.connect(
            lambda: self.current_name and self.propagate_requested.emit(self.current_name))
        self.slider.valueChanged.connect(self._set_opacity)
        self.canvas.clicked.connect(self._add_vertex)
        self.canvas.undo_requested.connect(self._undo_vertex)
        self.canvas.key_pressed.connect(self._on_key)

        bus.project_opened.connect(lambda p: setattr(self, "project", p))
        bus.image_selected.connect(self.show_image)
        bus.active_class_changed.connect(self._set_active_class)
        bus.image_labels_changed.connect(self._on_labels_changed)
        bus.schema_changed.connect(self._refresh_overlay)
        self._update_buttons()

    # -- loading -----------------------------------------------------------
    # Decoding a 21 Mpx JPEG (~360 ms) and colourising its label overlay used to
    # run on the UI thread, freezing the app for 1-2 s per click. The heavy work
    # (decode + auto-mask render + RGBA overlay) now runs on a worker thread; a
    # sequence number discards results superseded by a newer click.
    def show_image(self, name: str) -> None:
        """Load ``name`` on a worker thread and present it when ready (see
        the note above on why loading is asynchronous)."""
        if not self.project:
            return
        record = self.project.dataset.image_by_name(name)
        if record is None:
            return
        self._load_seq += 1
        seq = self._load_seq
        self._clear_scene()
        self.current_name = None                  # nothing editable until loaded
        self.scene.addText(f"Loading {name}…")
        self._update_buttons()

        from PySide6.QtCore import QThreadPool

        from cloudlabeller.workers.job import Job

        project = self.project
        job = Job(lambda progress=None: _load_payload(project, name, record, seq))
        job.signals.finished.connect(self._on_loaded)
        job.signals.failed.connect(lambda e, s=seq: self._on_load_failed(s, e))
        QThreadPool.globalInstance().start(job)

    def show_image_sync(self, name: str) -> None:
        """Synchronous variant (tests / scripting)."""
        record = self.project.dataset.image_by_name(name) if self.project else None
        if record is None:
            return
        self._load_seq += 1
        self._present(_load_payload(self.project, name, record, self._load_seq))

    def _on_loaded(self, payload: dict) -> None:
        if payload["seq"] == self._load_seq:      # else: a newer click won
            self._present(payload)

    def _on_load_failed(self, seq: int, error: str) -> None:
        if seq == self._load_seq:
            self._clear_scene()
            self.scene.addText(f"Failed to load image:\n{error}")

    def _clear_scene(self) -> None:
        self.scene.clear()
        self._base_item = self._overlay_item = self._poly_item = None
        self._markers.clear()
        self._vertices.clear()
        self._auto_mask = None

    def _present(self, payload: dict) -> None:
        """Main-thread part: build scene items from the preloaded payload."""
        self._clear_scene()
        image = payload["image"]
        if image.isNull():
            self.current_name = None
            self.lbl_name.setText("")
            self.scene.addText(f"Image not found in store:\n{payload['path']}")
            self._update_buttons()
            return

        self.current_name = payload["name"]
        self.lbl_name.setText(payload["name"])
        self._auto_mask = payload["auto_mask"]
        self._w, self._h = image.width(), image.height()
        self._base_item = self.scene.addPixmap(QPixmap.fromImage(image))
        self.scene.setSceneRect(QRectF(0, 0, self._w, self._h))

        self._overlay_item = self.scene.addPixmap(QPixmap())
        self._overlay_item.setZValue(1)
        self._overlay_item.setOpacity(self.slider.value() / 100.0)
        if payload["overlay"] is not None:
            self._set_overlay(*payload["overlay"])
        self.canvas.fitInView(self._base_item, Qt.KeepAspectRatio)
        self._update_buttons()

    def _set_overlay(self, index: np.ndarray, table: list) -> None:
        """Show a uint8 index image through a colour table (row 0 transparent)."""
        self._overlay_buf = index                          # keep the buffer alive
        h, w = index.shape
        qimg = QImage(index.data, w, h, w, QImage.Format_Indexed8)
        qimg.setColorTable([qRgba(*c) for c in table])
        self._overlay_item.setPixmap(QPixmap.fromImage(qimg))

    def _refresh_auto_mask(self) -> None:
        """(Re)render the transient auto overlay for the current image, if any."""
        self._auto_mask = None
        name = self.current_name
        if (name is not None and name not in self.project.labels.image_masks
                and self.project.labels.status_of(name) in ("auto", "ml")):
            self._auto_mask = self.project.render_auto_mask(name)

    def _current_mask(self) -> np.ndarray | None:
        """The stored (user) mask, else the transient auto-rendered one."""
        if self.current_name is None:
            return None
        stored = self.project.labels.image_masks.get(self.current_name)
        return stored if stored is not None else self._auto_mask

    def _on_labels_changed(self, name: str) -> None:
        if name == self.current_name:
            self._refresh_auto_mask()
            self._refresh_overlay()

    def _refresh_overlay(self) -> None:
        if self._overlay_item is None or self.current_name is None:
            return
        mask = self._current_mask()
        if mask is None:
            self._overlay_item.setPixmap(QPixmap())
            return
        self._set_overlay(*mask_to_indexed(mask, self.project.schema.lookup_table()))

    def _set_opacity(self, value: int) -> None:
        if self._overlay_item is not None:
            self._overlay_item.setOpacity(value / 100.0)

    def _set_active_class(self, class_id: int) -> None:
        self._active_class = class_id

    # -- polygon drawing ---------------------------------------------------
    def _on_draw_toggled(self, on: bool) -> None:
        self.canvas.set_draw_mode(on)
        if not on:
            self._cancel()
        self._update_buttons()

    def _add_vertex(self, point: QPointF) -> None:
        if self.current_name is None:
            return
        x = min(max(point.x(), 0.0), self._w - 1)
        y = min(max(point.y(), 0.0), self._h - 1)
        self._vertices.append((x, y))
        self._markers.append(self._make_dot(x, y, self._active_color()))
        self._redraw_polyline(self._active_color())
        self._update_buttons()

    def _make_dot(self, x: float, y: float, color: QColor) -> QGraphicsEllipseItem:
        """A vertex dot that stays a constant screen size at any zoom."""
        dot = QGraphicsEllipseItem(-4, -4, 8, 8)         # centred on its origin
        dot.setPen(QPen(Qt.white))
        dot.setBrush(QBrush(color))
        dot.setPos(x, y)
        dot.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        dot.setZValue(3)
        self.scene.addItem(dot)
        return dot

    def _undo_vertex(self) -> None:
        if not self._vertices:
            return
        self._vertices.pop()
        if self._markers:
            self.scene.removeItem(self._markers.pop())
        self._redraw_polyline(self._active_color())
        self._update_buttons()

    def _redraw_polyline(self, color: QColor) -> None:
        if self._poly_item is not None:
            self.scene.removeItem(self._poly_item)
            self._poly_item = None
        if len(self._vertices) < 2:
            return
        path = QPainterPath(QPointF(*self._vertices[0]))
        for x, y in self._vertices[1:]:
            path.lineTo(x, y)
        path.lineTo(QPointF(*self._vertices[0]))   # show the closing edge
        pen = QPen(color, 0)
        pen.setCosmetic(True)
        self._poly_item = self.scene.addPath(path, pen)
        self._poly_item.setZValue(2)

    def _commit(self) -> None:
        if self.current_name is not None and len(self._vertices) >= 3:
            labels = self.project.labels
            if self.current_name not in labels.image_masks:
                # Materialise the mask only now that the user actually draws on
                # this image — seeded from the auto overlay so hand edits refine
                # (not discard) the automatic labels.
                if self._auto_mask is not None:
                    labels.image_masks[self.current_name] = self._auto_mask.copy()
                else:
                    labels.init_image(self.current_name, self._h, self._w)
            mask = labels.image_masks[self.current_name]
            filled = rasterize_polygon(self._vertices, mask.shape[0], mask.shape[1])
            # Through the store so the polygon is undoable (Edit → Undo).
            labels.paint_image(self.current_name, np.flatnonzero(filled),
                               self._active_class)
            labels.mark_user_labeled(self.current_name)                # -> green dot
            self.bus.image_labels_changed.emit(self.current_name)
        self._cancel()

    # -- keyboard shortcuts (canvas focus) -----------------------------------
    def _on_key(self, key: int) -> None:
        """D = draw toggle, Enter = done, Esc = cancel, ←/→ = prev/next image,
        1-9 = active class, 0 = unlabelled (eraser)."""
        if key == Qt.Key_D:
            self.btn_draw.toggle()
        elif key in (Qt.Key_Return, Qt.Key_Enter):
            if self.btn_draw.isChecked() and len(self._vertices) >= 3:
                self._commit()
        elif key == Qt.Key_Escape:
            self._cancel()
        elif key in (Qt.Key_Left, Qt.Key_Right):
            self._navigate(1 if key == Qt.Key_Right else -1)
        elif Qt.Key_0 <= key <= Qt.Key_9:
            class_id = -1 if key == Qt.Key_0 else key - Qt.Key_1
            n = len(self.project.schema.classes) if self.project else 0
            if class_id < n:
                self.bus.active_class_changed.emit(class_id)

    def _navigate(self, step: int) -> None:
        """Show the previous/next image in dataset order."""
        if not self.project or not self.project.dataset.images:
            return
        names = [r.name for r in self.project.dataset.images]
        try:
            idx = names.index(self.current_name)
        except ValueError:
            idx = 0 if step > 0 else len(names) - 1
        else:
            idx = (idx + step) % len(names)
        self.bus.image_selected.emit(names[idx])

    def _import_mask(self) -> None:
        """Label the current image from an external mask file (active class).

        .jpg masks select the highest-intensity pixels; .png masks the pixels
        with the highest alpha value. The file must match the image size.
        """
        if self.current_name is None:
            return
        if self._active_class < 0:
            QMessageBox.information(self, "No active class",
                                    "Select a label class first — the mask is "
                                    "applied with the active label.")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Import mask", "",
            "Masks (*.jpg *.jpeg *.png);;All files (*)")
        if not path:
            return
        try:
            selected = mask_from_file(path, expected_shape=(self._h, self._w))
        except (ValueError, OSError) as exc:
            QMessageBox.critical(self, "Import mask failed", str(exc))
            return

        labels = self.project.labels
        if self.current_name not in labels.image_masks:
            # Same copy-on-write as polygon drawing: seed from the auto overlay
            # so the imported mask refines (not discards) automatic labels.
            if self._auto_mask is not None:
                labels.image_masks[self.current_name] = self._auto_mask.copy()
            else:
                labels.init_image(self.current_name, self._h, self._w)
        # Through the store so the import is undoable (Edit → Undo).
        labels.paint_image(self.current_name, np.flatnonzero(selected),
                           self._active_class)
        labels.mark_user_labeled(self.current_name)                    # -> green dot
        self.bus.image_labels_changed.emit(self.current_name)

    def _cancel(self) -> None:
        for item in self._markers:
            self.scene.removeItem(item)
        self._markers.clear()
        if self._poly_item is not None:
            self.scene.removeItem(self._poly_item)
            self._poly_item = None
        self._vertices.clear()
        self._update_buttons()

    # -- helpers -----------------------------------------------------------
    def _active_color(self) -> QColor:
        if self.project and self._active_class >= 0:
            try:
                return QColor(self.project.schema.by_id(self._active_class).color)
            except (IndexError, AttributeError):
                pass
        return QColor("#ffffff")

    def _update_buttons(self) -> None:
        drawing = self.btn_draw.isChecked() and self.current_name is not None
        self.btn_draw.setEnabled(self.current_name is not None)
        self.btn_finish.setEnabled(drawing)
        self.btn_done.setEnabled(drawing and len(self._vertices) >= 3)
        self.btn_cancel.setEnabled(drawing and len(self._vertices) > 0)
        self.btn_propagate.setEnabled(self.current_name is not None)
        self.btn_mask.setEnabled(self.current_name is not None)
