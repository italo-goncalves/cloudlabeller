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

"""Job plumbing for the stage CLIs and the 0.2.x Qt app.

``resources`` (thread and memory limits) is Qt-free and imported by every
``*_cli`` subprocess. ``job`` and ``process_job`` wrap work in Qt for the Qt
app; import them by their own module path. This package root must not import
them: it would make every CLI, and the desktop app's Environment (which has no
PySide6), depend on Qt.
"""
