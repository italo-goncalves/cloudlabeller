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


"""How a Job prints its outcome (``workers.result``)."""

import io
import json

from cloudlabeller.engine.jobs import parse_result
from cloudlabeller.workers.result import PIECE, pieces, print_result

#: The core's line limit (geokit's ``process::MAX_LINE``).
MAX_LINE = 64 * 1024


def _as_the_core_reads(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        assert len(line.encode()) + 1 < MAX_LINE
        key, _, rest = line.partition(" ")
        out[key] = rest.strip()
    return out


def test_a_short_result_is_one_line():
    out = io.StringIO()
    print_result({"epsg": 32723}, file=out)
    assert out.getvalue() == 'RESULT {"epsg": 32723}\n'
    assert parse_result(_as_the_core_reads(out.getvalue())) == {"epsg": 32723}


def test_a_long_result_round_trips_in_pieces_under_the_line_limit():
    payload = {"predicted": [{"survey": "s1", "name": f"DJI  {k:04d} ção.JPG", "u": k / 3}
                             for k in range(3000)]}
    assert len(json.dumps(payload)) > 2 * MAX_LINE
    out = io.StringIO()
    print_result(payload, file=out)
    results = _as_the_core_reads(out.getvalue())
    assert list(results)[:2] == ["RESULT.0", "RESULT.1"] and len(results) > 8
    assert parse_result(results) == payload


def test_pieces_never_start_or_end_with_a_space():
    """The core trims each line, so a space at a cut would be lost."""
    text = "a" + " " * 50 + "b" * 7 + "  c d"
    for size in range(1, 12):
        parts = pieces(text, size)
        assert "".join(parts) == text and len(parts) > 1
        assert all(p == p.strip() for p in parts)
    assert PIECE < MAX_LINE // 2


def test_pieces_are_joined_in_numeric_order():
    results = {f"RESULT.{i}": c for i, c in enumerate('{"k": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}')}
    assert parse_result(dict(reversed(list(results.items())))) == {"k": list(range(1, 12))}
