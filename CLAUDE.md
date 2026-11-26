# CloudLabeller — instructions for Claude

## Locked-code convention (MUST follow)

Some code is user-maintained and must never be modified by Claude.

- Block marker — never edit anything between these lines (not even formatting
  or imports). Reading, calling, and wrapping the code is fine:

  ```python
  # === LOCKED: <owner/reason> ===
  ...
  # === END LOCKED ===
  ```

- File marker — the whole file is user-maintained (placed at the top, after
  the module docstring):

  ```python
  # LOCKED FILE — user-maintained. Do not edit; propose changes instead.
  ```

Rules:
1. Before editing ANY file, check it for `LOCKED` markers.
2. If a locked region has a bug, or its interface doesn't fit, STOP and propose
   the change to the user (diff or description) — never apply it directly.
3. Renaming, moving, or deleting a locked file also requires asking first.
4. Unmarked code is normal — edit as usual.

## Notes-to-Claude convention

Comments starting with `# CLAUDE:` are notes addressed to Claude:

```python
# CLAUDE: this expects NHWC float32 in [0,1]; write the adapter accordingly.
# CLAUDE: is dropout=0.3 too high here? suggest, don't change.
```

Rules:
1. When the user says "check my notes" (or similar), grep the repo for
   `# CLAUDE:` and handle every hit.
2. Act on notes found while reading a file, even if not explicitly told.
3. After resolving a note, delete the comment (quote it in the reply so the
   resolution is traceable). Notes ending with `(keep)` stay in place.
4. A note inside a LOCKED region may be read and acted on, but the region
   itself still must not be edited — including removing the note; ask instead.

## Project facts

- The repo lives at `C:\Repos\RockAnalytica\cloudlabeller`, beside its
  siblings `geokit` and `geoscape`. Read `GLOSSARY.md` before naming
  anything, and `docs/adr/` before revisiting the 0.3.0 Tauri migration.

- The dev environment is `/c/Python_envs/cloudlabeller/python.exe`
  (Python 3.10, outside the repo). `cloudlabeller` is *not* installed in
  it: it imports the working tree only when run from the repo root (or with
  it on `PYTHONPATH`), so isolated mode (`-I`) can't see it. A debug build
  of the desktop app uses it when `CLOUDLABELLER_DEV_PYTHON` names it.
  There is no `.venv` in the repo. The `python` on PATH is a bare
  3.12 without the app's dependencies. Don't test with the installed app's
  embedded Python either (`C:\Programas\CloudLabeller`): it imports its own
  installed copy of the app.
- Run the app: `/c/Python_envs/cloudlabeller/python.exe -m cloudlabeller`.
- Tests: `QT_QPA_PLATFORM=offscreen PYVISTA_OFF_SCREEN=true
  /c/Python_envs/cloudlabeller/python.exe -m pytest tests/ -q` — keep the
  suite green; add a regression test with every bug fix.
- The desktop app (0.3.0) is in `apps/` (`apps/web` the page,
  `apps/desktop/src-tauri` the Rust core); see `docs/desktop.md`.
  `pnpm check` runs Biome, tsc, vitest, the node tests and `cargo test`;
  `pnpm desktop` runs it, and `pnpm bundle` builds the installer. Never copy
  code from GeoScape: it is closed and CloudLabeller is GPL. Shared code
  comes only from geokit, pinned by tag.
- `graft` (a context graph in `graft/`, hooks in `.claude/settings.json`):
  run `graft ask "<question>"` before grepping the tree, and
  `graft callers <symbol>` for what a change touches. `graft/` is a local,
  git-ignored cache; `graft build` rebuilds it.
- `cloudlabeller/core/` and `cloudlabeller/engine/` must stay Qt-free
  (except `core/events.py`).
- The Engine (`python -m cloudlabeller.engine`, step 2 of 0.3.0) is
  JSON-RPC over stdin/stdout, and its contract is `docs/engine.md`. A new or
  changed method is documented there in the same change, and
  `tests/test_engine_contract.py` fails when the two disagree. Only the
  Engine's loop thread touches the Project: slow work returns a `Deferred`.
  The Engine runs no processes: it plans Jobs and applies their results,
  and the Tauri core starts them (ADR-0004).
  The Engine reads and writes the protocol on private copies of stdin and
  stdout. Don't use `sys.stdin`/`sys.stdout` there: on Windows a native
  library probing fd 0 hung it (diagnosed 2026-11-25).
- Heavy native work (COLMAP, TensorFlow) runs in subprocesses (`*_cli.py` +
  `ProcessJob`); pure-numpy work runs in `QThreadPool` via `workers.job.Job`.
  NEVER run TensorFlow on an in-process thread — it deadlocks the GIL against
  Qt's event loop (frozen app, diagnosed 2026-07-08).
- Unlabelled label id is -1; user classes are contiguous 0..N-1.
- Dependencies: the base list in `pyproject.toml` is the desktop app's
  Environment (Engine + Jobs, no Qt); the 0.2.x Qt app needs the `qt` extra
  (the dev env has `[qt,dev]`). After changing any dependency, run
  `uv run --no-project tools/lock_environment.py` and commit both `uv.lock`
  and `apps/desktop/environment/requirements.lock` (`--check` says whether
  they are stale). No module that a `*_cli` or the Engine imports may import
  PySide6, pyvista or vtk (`tests/test_qt_free_imports.py`).
- PySide6 is pinned == 6.7.3 (6.8+ links ICU and fails to import on this
  machine — see requirements.txt).
- The user's U-Net code lives in `cloudlabeller/ml/` and is typically LOCKED.
