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

"""How a Job reports its outcome: ``RESULT <json>`` on its standard output.

The Tauri core reads a Job's output a line at a time and tells a line longer
than 64 KiB in pieces (geokit's ``MAX_LINE``), without saying they belong
together. A long outcome (the names of 1,399 predicted photos, say) would
reach the Engine cut in two and fail to parse. So :func:`print_result`
splits a long payload over numbered lines, ``RESULT.0 …``, ``RESULT.1 …``,
each well under the limit, and the Engine's ``parse_result`` joins them.

The JSON is ASCII (``ensure_ascii``), so a piece's length in characters is
its length in bytes; no piece starts or ends with a space, since the core
trims each line.
"""

from __future__ import annotations

import json
import sys

#: The longest piece, in characters: a quarter of the core's line limit.
PIECE = 16 * 1024


def pieces(text: str, size: int = PIECE) -> list[str]:
    """``text`` cut into pieces of about ``size`` characters, never next to
    a space (searching back from each cut, else forward)."""
    def safe(i: int) -> bool:
        return not (text[i - 1].isspace() or text[i].isspace())

    out, start = [], 0
    while len(text) - start > size:
        cut = next((i for i in range(start + size, start, -1) if safe(i)), None)
        if cut is None:
            cut = next((i for i in range(start + size + 1, len(text)) if safe(i)), None)
        if cut is None:
            break
        out.append(text[start:cut])
        start = cut
    out.append(text[start:])
    return out


def print_result(payload, key: str = "RESULT", *, file=None, size: int = PIECE) -> None:
    """Print ``payload`` as JSON on ``key`` lines (one, or numbered pieces)."""
    file = file or sys.stdout
    text = json.dumps(payload, ensure_ascii=True)
    parts = pieces(text, size)
    if len(parts) == 1:
        print(f"{key} {text}", file=file, flush=True)
        return
    for i, part in enumerate(parts):
        print(f"{key}.{i} {part}", file=file, flush=True)
