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

"""CloudLabeller — photogrammetry + bidirectional 2D/3D labelling.

The package is organised so that :mod:`cloudlabeller.core` is UI-agnostic and
free of Qt imports; the GUI in :mod:`cloudlabeller.ui` depends on core, never the
other way around.
"""

__version__ = "0.1.0"


def _preload_msvc_runtime() -> None:
    """Load the system MSVC runtime before anything else pins an older copy.

    The Anaconda base interpreter ships a 2020-era msvcp140.dll next to
    python310.dll, which the Windows loader resolves ahead of System32's.
    TensorFlow's native runtime needs a newer STL and fails with
    "DLL load failed … DllMain returned false" if the old copy is already
    loaded. Loading System32's by full path first makes it the copy every
    later by-name lookup (numpy, TF, …) binds to. Harmless when the DLLs
    are missing or already current.
    """
    import ctypes
    import os
    import sys

    if sys.platform != "win32":
        return
    sysdir = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
    for dll in ("msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll"):
        try:
            ctypes.WinDLL(os.path.join(sysdir, dll))
        except OSError:
            pass


_preload_msvc_runtime()
