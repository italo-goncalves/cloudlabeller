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

"""The Engine's loop: read requests, dispatch them, answer, notify.

Threading model — one rule: **only the loop thread touches the Project.**

- A reader thread turns stdin lines into events on a queue.
- The loop thread takes events one at a time and runs the handler.
- A handler that would take long returns a :class:`Deferred`: its ``work``
  runs on a small thread pool against snapshots, reporting progress (which
  is also where cancellation lands), and its ``finish`` runs back on the
  loop thread to fold the result into the Project. The response is sent
  after ``finish``.
So requests can overlap (a cancel arrives while a transfer runs) without the
Project ever being touched from two threads.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from cloudlabeller.engine import protocol
from cloudlabeller.engine.protocol import InvalidParams, Problem

log = logging.getLogger("cloudlabeller.engine")

ProgressFn = Callable[..., None]


class Cancelled(Exception):
    """Raised inside a Deferred's work when its request was cancelled."""


@dataclass
class Deferred:
    """A handler's answer that takes a while (see module docstring).

    ``work(progress)`` runs off the loop thread and must not touch the
    Project; ``progress(fraction, message="")`` sends a progress notification
    and raises :class:`Cancelled` once the request is cancelled.
    ``finish(value)`` runs on the loop thread; its return is the response.
    """

    work: Callable[[ProgressFn], Any]
    finish: Callable[[Any], Any] | None = None


@dataclass
class Method:
    name: str
    fn: Callable[["Engine", dict], Any]
    needs_project: bool


METHODS: dict[str, Method] = {}


def method(name: str, *, needs_project: bool = True):
    """Register ``fn(engine, params)`` as the JSON-RPC method ``name``."""
    def register(fn):
        METHODS[name] = Method(name, fn, needs_project)
        return fn
    return register


class Engine:
    """One Engine: at most one open Project, its Jobs, its pending requests."""

    AUTOSAVE_S = 60.0

    def __init__(self, write: Callable[[bytes], None], *, autosave_s: float | None = None,
                 workers: int = 2) -> None:
        # Importing the handlers registers them (and keeps this module light).
        from cloudlabeller.engine import methods as _methods  # noqa: F401
        from cloudlabeller.engine.jobs import JobPlans

        self._write = write
        self._write_lock = threading.Lock()
        self._events: queue.Queue = queue.Queue()
        self._pool = ThreadPoolExecutor(max_workers=workers,
                                        thread_name_prefix="engine-work")
        self._cancel_flags: dict[Any, threading.Event] = {}
        self._autosave_s = self.AUTOSAVE_S if autosave_s is None else autosave_s
        self._running = True

        self.project = None          # cloudlabeller.core.project.Project | None
        self.dirty = False           # label arrays changed since the last full save
        self.cache: dict = {}        # derived values of the open Project (loop thread)
        self.jobs = JobPlans(self)

    # -- output (any thread) ----------------------------------------------
    def send(self, message: dict) -> None:
        data = protocol.encode(message)
        with self._write_lock:
            try:
                self._write(data)
            except (BrokenPipeError, OSError, ValueError):
                self._running = False          # the window is gone

    def notify(self, method_name: str, params: dict) -> None:
        self.send(protocol.notification(method_name, params))

    def post(self, fn: Callable[[], None]) -> None:
        """Run ``fn`` on the loop thread (from any thread)."""
        self._events.put(("call", fn))

    def changed(self, *parts: str, images: Iterable[str] = ()) -> None:
        """Tell the window which parts of the Project changed.

        Parts: ``schema``, ``images``, ``status``, ``sparse``, ``dense``,
        ``mesh``, ``labels``, ``settings``. ``images`` names the images whose
        masks or status changed.
        """
        params: dict = {"parts": sorted(set(parts))}
        names = sorted(set(images))
        if names:
            params["images"] = names
        self.notify("project.changed", params)

    # -- the loop -----------------------------------------------------------
    def serve(self, lines: Iterable[bytes]) -> None:
        """Answer requests read from ``lines`` until shutdown or end of input."""
        def reader() -> None:
            try:
                for line in lines:
                    self._events.put(("line", line))
            finally:
                self._events.put(("eof", None))

        threading.Thread(target=reader, name="engine-stdin", daemon=True).start()
        self.notify("engine.ready", self.hello())
        next_save = time.monotonic() + self._autosave_s
        while self._running:
            timeout = max(0.0, next_save - time.monotonic()) if self._autosave_s else None
            try:
                kind, payload = self._events.get(timeout=timeout)
            except queue.Empty:
                kind, payload = None, None
            if self._autosave_s and time.monotonic() >= next_save:
                self._autosave()                   # also while busy, not only when idle
                next_save = time.monotonic() + self._autosave_s
            if kind == "line":
                self._handle_line(payload)
            elif kind == "call":
                self._guarded(payload)
            elif kind == "eof":
                break
        self.shutdown()

    def hello(self) -> dict:
        from cloudlabeller import __version__
        from cloudlabeller.engine import PROTOCOL_VERSION

        return {"protocol": PROTOCOL_VERSION, "version": __version__,
                "methods": sorted(METHODS)}

    def stop(self) -> None:
        self._running = False

    def shutdown(self) -> None:
        """Save a dirty Project, release the pool. Running Jobs are the
        Tauri core's (ADR-0004) and keep running."""
        self._running = False
        self.jobs.forget_project()
        if self.project is not None:
            try:
                self.close_project()
            except Exception:
                log.exception("could not save the Project on shutdown")
        self._pool.shutdown(wait=False, cancel_futures=True)

    def _autosave(self) -> None:
        if self.project is not None and self.dirty:
            self._guarded(self.save_project)

    def _guarded(self, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception:
            log.exception("engine callback failed")
            self.notify("engine.log", {"level": "error",
                                       "message": traceback.format_exc(limit=5)})

    # -- project lifecycle helpers (loop thread) --------------------------
    def save_project(self) -> None:
        if self.project is not None:
            self.project.save()
            self.dirty = False

    def close_project(self) -> None:
        from cloudlabeller.engine import arrays

        project = self.project
        if project is None:
            return
        if self.dirty:
            self.save_project()
        self.project = None
        self.dirty = False
        self.cache.clear()
        arrays.clear(project.root)

    def require_project(self):
        if self.project is None:
            raise Problem("project.none_open", "No Project is open.")
        return self.project

    # -- dispatch -------------------------------------------------------------
    def _handle_line(self, raw: bytes | str) -> None:
        text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
        text = text.strip()
        if not text:
            return
        try:
            msg = json.loads(text)
        except json.JSONDecodeError as exc:
            self.send(protocol.error(None, protocol.PARSE_ERROR, f"parse error: {exc}"))
            return
        if (not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0"
                or not isinstance(msg.get("method"), str)):
            req_id = msg.get("id") if isinstance(msg, dict) else None
            self.send(protocol.error(req_id, protocol.INVALID_REQUEST,
                                     "not a JSON-RPC 2.0 request"))
            return
        self._dispatch(msg.get("id"), msg["method"], msg.get("params", {}),
                       is_notification="id" not in msg)

    def _dispatch(self, req_id: Any, name: str, params: Any, is_notification: bool) -> None:
        def reply(message: dict) -> None:
            if not is_notification:
                self.send(message)

        entry = METHODS.get(name)
        if entry is None:
            reply(protocol.error(req_id, protocol.METHOD_NOT_FOUND,
                                 f"no method {name!r}", {"method": name}))
            return
        if params is None:
            params = {}
        if not isinstance(params, dict):
            reply(protocol.error(req_id, protocol.INVALID_PARAMS,
                                 "params must be an object"))
            return
        try:
            if entry.needs_project:
                self.require_project()
            result = entry.fn(self, params)
        except Problem as problem:
            reply(protocol.problem_error(req_id, problem))
            return
        except Exception as exc:
            log.exception("method %s failed", name)
            reply(protocol.error(req_id, protocol.INTERNAL_ERROR,
                                 f"{type(exc).__name__}: {exc}",
                                 {"traceback": traceback.format_exc(limit=8)}))
            return

        if isinstance(result, Deferred):
            self._run_deferred(req_id, result, reply)
        else:
            reply(protocol.response(req_id, result))

    def _run_deferred(self, req_id: Any, deferred: Deferred,
                      reply: Callable[[dict], None]) -> None:
        flag = threading.Event()
        if req_id is not None:
            self._cancel_flags[req_id] = flag
        project = self.project

        def progress(fraction: float, message: str = "") -> None:
            if flag.is_set():
                raise Cancelled()
            if req_id is not None:
                self.notify("request.progress", {"id": req_id,
                                                 "fraction": float(fraction),
                                                 "message": message})

        def finish(ok: bool, value: Any) -> None:
            self._cancel_flags.pop(req_id, None)
            if not ok:
                reply(value)
                return
            if self.project is not project:
                reply(protocol.problem_error(req_id, Problem(
                    "project.closed", "The Project was closed before the work finished.")))
                return
            try:
                result = deferred.finish(value) if deferred.finish else value
            except Problem as problem:
                reply(protocol.problem_error(req_id, problem))
                return
            except Exception as exc:
                log.exception("finishing request %r failed", req_id)
                reply(protocol.error(req_id, protocol.INTERNAL_ERROR,
                                     f"{type(exc).__name__}: {exc}"))
                return
            reply(protocol.response(req_id, result))

        def work() -> None:
            try:
                value = deferred.work(progress)
            except Cancelled:
                err = protocol.problem_error(req_id, Problem(
                    "request.cancelled", "Cancelled."))
                self.post(lambda: finish(False, err))
                return
            except Problem as problem:
                err = protocol.problem_error(req_id, problem)
                self.post(lambda: finish(False, err))
                return
            except Exception as exc:
                log.exception("request %r failed", req_id)
                err = protocol.error(req_id, protocol.INTERNAL_ERROR,
                                     f"{type(exc).__name__}: {exc}",
                                     {"traceback": traceback.format_exc(limit=8)})
                self.post(lambda: finish(False, err))
                return
            self.post(lambda: finish(True, value))

        self._pool.submit(work)

    def cancel_request(self, req_id: Any) -> bool:
        flag = self._cancel_flags.get(req_id)
        if flag is None:
            return False
        flag.set()
        return True

    def run_background(self, work: Callable[[], Any],
                       finish: Callable[[Any], None]) -> None:
        """Internal Deferred with no request: ``finish`` runs on the loop
        thread only if the same Project is still open."""
        project = self.project

        def run() -> None:
            try:
                value = work()
            except Exception:
                log.exception("background work failed")
                return
            self.post(lambda: finish(value) if self.project is project else None)

        self._pool.submit(run)


def require(params: dict, key: str, kind: type | tuple[type, ...]) -> Any:
    """``params[key]``, checked to be of ``kind`` (bool is not an int)."""
    if key not in params:
        raise InvalidParams(f"missing param {key!r}", param=key)
    value = params[key]
    kinds = kind if isinstance(kind, tuple) else (kind,)
    if isinstance(value, bool) and bool not in kinds:      # bool is an int in Python
        raise InvalidParams(f"param {key!r} has the wrong type", param=key)
    if not isinstance(value, kinds):
        raise InvalidParams(f"param {key!r} has the wrong type", param=key)
    return value


def optional(params: dict, key: str, kind: type | tuple[type, ...], default: Any) -> Any:
    if params.get(key) is None:
        return default
    return require(params, key, kind)
