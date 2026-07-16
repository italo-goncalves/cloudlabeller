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

"""Run a Python module in a child process via QProcess, off the GUI thread.

Used for COLMAP reconstruction: native crashes (segfault / glog abort) end the
child with a non-zero exit code instead of killing the app, and the child's GPU/
OpenGL usage can't collide with the GUI's VTK context.

Output handling:
  * stdout ``PROGRESS <frac> <msg>`` lines drive the :attr:`progress` signal;
  * every output line is forwarded via :attr:`log_line` (for the Log panel) and
    appended to ``log_path`` if given;
  * stderr (incl. COLMAP's native glog) is captured; the tail is included in the
    :attr:`failed` message.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QTimer, Signal


class ProcessJob(QObject):
    """One ``python -m <module> <args>`` child process wired to Qt signals
    (see module docstring for the stdout/stderr protocol)."""

    progress = Signal(float, str)
    log_line = Signal(str)
    finished = Signal()              # exit code 0
    failed = Signal(str)             # message incl. captured stderr tail

    # Emit a "still running" line after this much output silence. Long native
    # stages (global bundle adjustment, stereo fusion, Poisson) print nothing
    # while solving and used to look frozen for 20+ minutes.
    HEARTBEAT_S = 60

    def __init__(self, module: str, args: list[str], log_path: str | Path | None = None,
                 parent: QObject | None = None,
                 heartbeat_s: int | None = None) -> None:
        super().__init__(parent)
        self._module = module
        self._args = args
        self._log_path = Path(log_path) if log_path else None
        self._log_fh = None
        self._stderr_tail: list[str] = []
        self._settled = False  # guard: emit finished/failed exactly once
        self.result_text = ""  # last "RESULT…" line the child printed
        self._heartbeat_s = heartbeat_s or self.HEARTBEAT_S
        self._started_at = 0.0
        self._last_output = 0.0
        self._pulse = QTimer(self)
        self._pulse.setInterval(max(1, self._heartbeat_s // 2) * 1000)
        self._pulse.timeout.connect(self._on_heartbeat)
        self.proc = QProcess(self)
        self.proc.setProcessChannelMode(QProcess.SeparateChannels)
        self.proc.readyReadStandardOutput.connect(self._on_stdout)
        self.proc.readyReadStandardError.connect(self._on_stderr)
        self.proc.finished.connect(self._on_finished)
        self.proc.errorOccurred.connect(self._on_error)

    def start(self) -> None:
        """Open the stage log and launch the child interpreter."""
        if self._log_path is not None:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_fh = open(self._log_path, "w", encoding="utf-8")
        # ``sys.executable`` is the venv interpreter, so the child has the same
        # environment (pycolmap, cloudlabeller importable).
        cmd = [self._module, *self._args]
        self._record(f"$ {sys.executable} -m {' '.join(cmd)}")
        self._started_at = self._last_output = time.monotonic()
        self._pulse.start()
        self.proc.start(sys.executable, ["-m", *cmd])

    def _on_heartbeat(self) -> None:
        """Reassure the user during long silent native stages (see HEARTBEAT_S)."""
        if self.proc.state() == QProcess.NotRunning:
            return
        now = time.monotonic()
        silent = now - self._last_output
        if silent >= self._heartbeat_s:
            elapsed = int((now - self._started_at) // 60)
            self.log_line.emit(
                f"… still running ({elapsed} min elapsed, no output for "
                f"{int(silent // 60)} min — long solves are silent)")

    def cancel(self) -> None:
        if self.proc.state() != QProcess.NotRunning:
            self.proc.kill()

    # -- stream handling ---------------------------------------------------
    def _record(self, line: str) -> None:
        """Forward one line to the Log panel and the run log file."""
        self._last_output = time.monotonic()      # real output resets the heartbeat
        self.log_line.emit(line)
        if self._log_fh is not None:
            self._log_fh.write(line + "\n")
            self._log_fh.flush()

    def _on_stdout(self) -> None:
        text = bytes(self.proc.readAllStandardOutput()).decode(errors="replace")
        for line in text.splitlines():
            if line.startswith("PROGRESS "):
                parts = line.split(" ", 2)
                try:
                    frac = float(parts[1])
                except (IndexError, ValueError):
                    frac = None
                if frac is not None:
                    self.progress.emit(frac, parts[2] if len(parts) > 2 else "")
            elif line.startswith("RESULT"):
                self.result_text = line
            self._record(line)

    def _on_stderr(self) -> None:
        text = bytes(self.proc.readAllStandardError()).decode(errors="replace")
        for line in text.splitlines():
            if line.strip():
                self._stderr_tail.append(line)
                self._record(line)
        del self._stderr_tail[:-200]  # bound memory

    def _on_error(self, _err) -> None:
        # errorOccurred can fire alongside finished(); _fail() de-dupes via _settled.
        if self.proc.state() == QProcess.NotRunning:
            self._fail("the reconstruction process could not run")

    def _on_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        if self._settled:
            return
        if exit_status == QProcess.NormalExit and exit_code == 0:
            self._settled = True
            self._close_log()
            self.finished.emit()
        elif exit_status == QProcess.CrashExit:
            self._fail("reconstruction crashed — it likely ran out of memory.\n"
                       "Try a smaller 'Max image size' (e.g. 3200) or enable GPU.")
        else:
            self._fail(f"reconstruction failed (exit {exit_code})")

    def _fail(self, headline: str) -> None:
        if self._settled:
            return
        self._settled = True
        tail = "\n".join(self._stderr_tail[-15:])
        self._close_log()
        self.failed.emit(f"{headline}\n\n{tail}" if tail else headline)

    def _close_log(self) -> None:
        self._pulse.stop()
        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None
