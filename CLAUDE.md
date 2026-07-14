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

- Run the app: `.venv/Scripts/python -m cloudlabeller` (venv Python 3.10;
  no bare `python` on PATH).
- Tests: `QT_QPA_PLATFORM=offscreen PYVISTA_OFF_SCREEN=true
  .venv/Scripts/python -m pytest tests/ -q` — keep the suite green; add a
  regression test with every bug fix.
- `cloudlabeller/core/` must stay Qt-free (except `core/events.py`).
- Heavy native work (COLMAP, TensorFlow) runs in subprocesses (`*_cli.py` +
  `ProcessJob`); pure-numpy work runs in `QThreadPool` via `workers.job.Job`.
  NEVER run TensorFlow on an in-process thread — it deadlocks the GIL against
  Qt's event loop (frozen app, diagnosed 2026-07-08).
- Unlabelled label id is -1; user classes are contiguous 0..N-1.
- PySide6 is pinned == 6.7.3 (6.8+ links ICU and fails to import on this
  machine — see requirements.txt).
- The user's U-Net code lives in `cloudlabeller/ml/` and is typically LOCKED.
