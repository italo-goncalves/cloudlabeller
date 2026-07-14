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

"""Keep heavy background work from starving the OS and other applications.

Every ``*_cli`` subprocess calls :func:`limit_subprocess_resources` first:

1. **Thread budget** — :func:`worker_threads` returns ``cpu_count - 1`` so at
   least one core is always left for the operating system / UI. The CLIs pass
   it to pycolmap / colmap.exe stage options and export it via the common
   threadpool environment variables (OMP/MKL/TF) before those libraries
   initialise their pools.
2. **Process priority** — the subprocess drops itself to BELOW_NORMAL
   (Windows). Child processes it spawns (colmap.exe) inherit the priority
   class, so the scheduler favours interactive applications even when the
   pipeline is fully loaded.
"""

from __future__ import annotations

import logging
import os
import sys

log = logging.getLogger("cloudlabeller.resources")


def worker_threads() -> int:
    """CPU threads heavy work may use: all logical cores except one."""
    return max(1, (os.cpu_count() or 2) - 1)


def keep_system_awake(on: bool) -> None:
    """Stop Windows from sleeping while a long pipeline runs (the display may
    still turn off). Call with ``False`` when the work ends. Thread-affine:
    call both transitions from the same (UI) thread."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        es_continuous = 0x80000000
        es_system_required = 0x00000001
        flags = es_continuous | (es_system_required if on else 0)
        ctypes.windll.kernel32.SetThreadExecutionState(flags)
        log.info("keep-awake %s", "enabled" if on else "released")
    except Exception:
        log.debug("could not change execution state", exc_info=True)


def limit_subprocess_resources() -> None:
    """Cap threadpools and lower this process's priority (call first in CLIs)."""
    n = worker_threads()
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "TF_NUM_INTRAOP_THREADS"):
        os.environ.setdefault(var, str(n))
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

    if sys.platform == "win32":
        try:
            import ctypes
            import ctypes.wintypes as wt

            k32 = ctypes.windll.kernel32
            # Explicit prototypes: the default int restype truncates the
            # pseudo-handle on 64-bit and the call silently fails.
            k32.GetCurrentProcess.restype = wt.HANDLE
            k32.SetPriorityClass.argtypes = [wt.HANDLE, wt.DWORD]
            k32.SetPriorityClass.restype = wt.BOOL
            below_normal = 0x00004000                # BELOW_NORMAL_PRIORITY_CLASS
            k32.SetPriorityClass(k32.GetCurrentProcess(), below_normal)
        except Exception:                            # best-effort only
            log.debug("could not lower process priority", exc_info=True)
    log.info("resource limits: %d worker threads (of %d cores), "
             "below-normal priority", n, os.cpu_count() or 0)
