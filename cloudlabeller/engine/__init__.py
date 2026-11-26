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

"""The Engine: the Python side of an open Project (ADR-0002).

One long-running process, ``python -m cloudlabeller.engine``, started by the
window inside the active Environment. It speaks JSON-RPC 2.0 over
stdin/stdout, one JSON message per line. Large arrays travel as Array files
in the Project folder (:mod:`.arrays`), never through the pipe. Jobs stay
separate ``*_cli.py`` processes: the Engine plans them and applies their
results, the Tauri core runs them (ADR-0004, :mod:`.jobs`).

The contract — every method, notification and error code — is
``docs/engine.md``; ``tests/test_engine_contract.py`` holds the Engine to it.

GUI-free: nothing here may import PySide6, pyvista or vtk, and TensorFlow never runs in this
process (training and prediction are Jobs).
"""

PROTOCOL_VERSION = 1
