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

"""``python -m cloudlabeller.engine``: run the Engine on stdin/stdout.

stdin and stdout belong to the protocol alone. Before anything else is
imported, both are duplicated to private file descriptors for the protocol:

- fd 1 is pointed at stderr, so a stray ``print`` from a library (or native
  code writing to fd 1) lands in the log instead of corrupting the stream.
- fd 0 is pointed at NUL. On Windows a synchronous read on a pipe blocks
  every other call on that pipe, and native runtimes probe fd 0 when they
  load: SciPy's Fortran BLAS, loaded on a worker thread while the reader
  sat in ``readline``, hung the Engine forever.

Logs go to stderr.
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    proto_out = os.dup(sys.stdout.fileno())
    sys.stdout.flush()
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    out = os.fdopen(proto_out, "wb", buffering=0)

    proto_in = os.dup(sys.stdin.fileno())
    null = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null, sys.stdin.fileno())
    os.close(null)
    sys.stdin = open(os.devnull, encoding="utf-8")
    inp = os.fdopen(proto_in, "rb")

    import logging

    from cloudlabeller.logging_setup import setup_logging

    setup_logging(None, level=logging.INFO, console=True)

    from cloudlabeller.engine.server import Engine

    def write(data: bytes) -> None:
        out.write(data)

    engine = Engine(write)
    engine.serve(iter(inp.readline, b""))                    # saves and kills Jobs on exit
    # End here, without interpreter finalisation: the stdin reader thread may
    # still be blocked in readline (after engine.shutdown the window keeps
    # the pipe open), and finalising under it aborts the process
    # ("could not acquire lock for <stdin> at interpreter shutdown", which
    # the pipe fixture's exit-code check in the contract tests catches).
    logging.shutdown()
    sys.stderr.flush()
    out.close()
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
