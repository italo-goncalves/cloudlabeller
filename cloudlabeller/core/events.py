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

"""Lightweight signal bus connecting the views.

This is the *only* core module permitted to import Qt, because Qt signals are the
cleanest cross-widget notification primitive. If you want core to stay strictly
Qt-free, swap this for a pure-Python observer — the public surface is just the
signals below.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal


class EventBus(QObject):
    """Global signals. One instance is shared by the main window and all docks."""

    # Project lifecycle
    project_opened = Signal(object)        # Project
    project_closed = Signal()

    # Data availability
    cloud_changed = Signal()               # new/replaced point cloud
    mesh_changed = Signal()
    images_changed = Signal()              # dataset image list changed

    # Selection / navigation
    image_selected = Signal(str)           # show this image (dataset click or frustum click)

    # Labelling
    cloud_labels_changed = Signal()
    image_labels_changed = Signal(str)     # image name — its mask changed
    active_class_changed = Signal(int)     # class id
    class_isolation_changed = Signal(object)  # class id to isolate in 3D, or None
    schema_changed = Signal()

    # Long jobs (SfM / MVS / train / infer)
    job_started = Signal(str)              # description
    job_progress = Signal(str, float)      # description, 0..1
    job_finished = Signal(str, bool)       # description, success
