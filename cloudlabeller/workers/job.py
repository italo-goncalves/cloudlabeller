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

"""Generic cancellable background job runner built on QThreadPool.

Usage::

    job = Job(run_sfm, image_dir, workspace)
    job.signals.progress.connect(on_progress)
    job.signals.finished.connect(on_done)
    job.signals.failed.connect(on_error)
    QThreadPool.globalInstance().start(job)

The wrapped callable receives a ``progress`` kwarg (fraction, message) and may
poll ``job.is_cancelled`` via the injected ``should_stop`` kwarg if it accepts
one.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

from PySide6.QtCore import QObject, QRunnable, Signal, Slot


class JobSignals(QObject):
    """Qt signals for a :class:`Job` (QRunnable can't own signals itself)."""

    progress = Signal(float, str)
    finished = Signal(object)      # result
    failed = Signal(str)           # error message


class Job(QRunnable):
    """Run ``fn(*args, **kwargs)`` on the global thread pool (see module
    docstring for the progress/cancellation contract)."""

    def __init__(self, fn: Callable[..., Any], *args, **kwargs) -> None:
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = JobSignals()
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    @Slot()
    def run(self) -> None:
        try:
            params = inspect.signature(self.fn).parameters
            if "progress" in params:
                self.kwargs.setdefault("progress",
                                       lambda f, m="": self.signals.progress.emit(f, m))
            if "should_stop" in params:
                self.kwargs.setdefault("should_stop", lambda: self._cancelled)
            result = self.fn(*self.args, **self.kwargs)
            self.signals.finished.emit(result)
        except Exception as exc:  # surfaced to the UI, not swallowed
            self.signals.failed.emit(f"{type(exc).__name__}: {exc}")
