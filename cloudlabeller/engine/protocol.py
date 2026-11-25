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

"""JSON-RPC 2.0 framing and the Engine's one error type.

Framing: one JSON object per line, UTF-8, ``\\n``-terminated, in both
directions. Batches are not supported.

Errors: the JSON-RPC codes for protocol faults, and ``-32000`` for every
Problem the Engine reports. A Problem's ``data`` is ``{code, english,
params}`` — the same shape as geokit's ``Problem`` — so the window matches
on ``data.code`` and words the message itself.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
PROBLEM = -32000


class Problem(Exception):
    """Something the window should tell the user about, by ``code``.

    ``english`` is a fallback message; ``params`` carries the values the
    window needs to word it (names, counts).
    """

    def __init__(self, code: str, english: str, **params: Any) -> None:
        super().__init__(english)
        self.code = code
        self.english = english
        self.params = params

    def to_data(self) -> dict:
        return {"code": self.code, "english": self.english, "params": self.params}


class InvalidParams(Problem):
    """A request whose params don't fit the method (JSON-RPC -32602)."""

    def __init__(self, english: str, **params: Any) -> None:
        super().__init__("request.invalid_params", english, **params)


def _default(obj: Any) -> Any:
    """Let numpy scalars and small arrays through ``json.dumps``."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "__fspath__"):
        return str(obj)
    raise TypeError(f"{type(obj).__name__} is not JSON serialisable")


def encode(message: dict) -> bytes:
    """One message as a line of UTF-8 JSON (no embedded newlines)."""
    return (json.dumps(message, default=_default, ensure_ascii=False,
                       separators=(",", ":")) + "\n").encode("utf-8")


def response(req_id: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def error(req_id: Any, code: int, message: str, data: dict | None = None) -> dict:
    err: dict = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": req_id, "error": err}


def problem_error(req_id: Any, problem: Problem) -> dict:
    code = INVALID_PARAMS if isinstance(problem, InvalidParams) else PROBLEM
    return error(req_id, code, problem.english, problem.to_data())


def notification(method: str, params: dict) -> dict:
    return {"jsonrpc": "2.0", "method": method, "params": params}
