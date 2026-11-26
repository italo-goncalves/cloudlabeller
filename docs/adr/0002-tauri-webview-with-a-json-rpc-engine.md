# The UI moves to a Tauri webview, talking to a Python Engine over JSON-RPC

The PySide6/PyVista UI is replaced, in one release (0.3.0), by a React +
three.js page in a Tauri 2 shell, as in GeoScape. The Python core
(`core/`, `photogrammetry/`, `ml/`) stays and becomes the **Engine**: one
long-running process per open Project, started by the Tauri core inside the
active Environment. The Engine takes commands and sends events (progress,
results) as JSON-RPC over stdin/stdout. Large arrays (clouds, masks) travel as
files in the Project folder, not through the pipe. Jobs remain separate
processes (the `*_cli.py` entry points), as they are today. Until 0.3.0 reaches
feature parity, the Qt app keeps shipping as 0.2.x.

## Considered Options

- **GeoScape's file Exchange with one-shot Scripts only**: this fits long
  Jobs, but not CloudLabeller's interactive work (transfer a label → see the
  cloud change, predict one photo).
- **A local HTTP or WebSocket server**: opens a port, which goes against
  GeoScape's no-port posture. It also adds nothing that stdin/stdout lacks,
  even across the Windows/WSL boundary.
- **Restyling the Qt app with GeoScape's tokens**: no installer comes out of
  it, and nothing could be shared with GeoScape.

## Consequences

- Mask painting runs entirely in the page. Python is only called when a
  stroke is saved and for transfer, prediction and training. The Project
  format is unchanged.
- In 0.3.0 the viewer loads clouds as one flat binary buffer, thinned by the
  existing detail level. COPC waits until clouds outgrow the point budget.
  No permissive writer for it installs from pip alone.
- The installer is a per-user NSIS installer of tens of MB, updated through
  the Tauri updater. The Environment is installed on first run from a
  committed uv lockfile. The 941 MiB portable zip is dropped.
- 0.3.0 targets Windows only.
- The Tauri core starts the Engine itself, with `std::process` and the
  command line from geokit's `programs::python`, because geokit's process
  spawner closes a child's input and the Engine needs stdin. Nothing else
  needs this, so it stays in CloudLabeller (decided 2026-11-25). The
  Engine starts no processes, so ending it ends nothing else. Jobs go
  through geokit's `Processes` (ADR-0004).
