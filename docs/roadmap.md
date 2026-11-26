# Roadmap

*This file decides what comes next and in what order. It holds no design:
the design lives in [DESIGN.md](../DESIGN.md) and the decisions in
[adr/](adr/). **Now** is the next release, with what "done" means. **Next**
lists the themes after it, in order, each with the reason it comes where it
does. **Later** is unordered. **Dropped** records what was taken off, and
why, so it isn't proposed again. An item moves by editing this file, and
leaves the file when it is built.*

Last released: **0.2.1** (2026-07-21). It was built on 0.2.0: georeferencing
from EXIF GPS, CRS-aware export, reprojection to any projected CRS.

---

## Now — 0.3.0: the Desktop app

The PySide6/PyVista UI is replaced by a Tauri app that shares its UI system
with GeoScape, and the Environment installs itself
([ADR-0001](adr/0001-shared-permissive-kit-from-geoscape.md),
[ADR-0002](adr/0002-tauri-webview-with-a-json-rpc-engine.md),
[ADR-0003](adr/0003-wsl-environment-and-one-gpu-lease.md)). 0.2.x keeps
shipping the Qt app until this is done.

The steps, in order. Each step depends only on the ones before it, except
steps 1 and 2, which share no files and can run side by side. (The GPU SIFT
spike is done: all of COLMAP runs in WSL, see ADR-0003.)

1. **The kit.** Create `rockanalytica/geokit` (Apache-2.0, © RockAnalytica) by
   extracting GeoScape's tokens, its three.js-only viewer layers and its
   Environment manager, then add the GPU Lease. GeoScape switches to the kit
   and releases before CloudLabeller takes anything from it.
2. **The Engine.** JSON-RPC over stdin/stdout on top of `core/`,
   `photogrammetry/` and `ml/`, with contract tests. Large arrays go as files.
   Jobs stay `*_cli.py` processes.
   *Built 2026-11-25:* `cloudlabeller/engine/`, with its contract in
   [engine.md](engine.md). Left for step 4, added as the screens need them:
   export, cleaning the sparse cloud, the COLMAP download, the viewer's
   thinned flat buffer, and the multi-stage pipeline (run by the window).
3. **The Tauri shell.** Includes:
   - the Environment setup screen: WSL when a distro exists and `nvidia-smi`
     works there, otherwise native on the CPU;
   - uv with Python 3.12, from a committed lockfile, keeping TensorFlow 2.19;
   - pycolmap moved from 4.1 to 4.2;
   - *Under way (2026-11-25): the shell, the Environment setup, the Engine
     host, the Job runner and the release pipeline are in `apps/`
     ([desktop.md](desktop.md)), checked by `pnpm check`. Still to do: the
     lockfile and Python 3.12 (Part A), the signing key, and an
     end-to-end run on the installer.*
   - running the Jobs the Engine plans ([ADR-0004](adr/0004-the-tauri-core-starts-every-job.md)),
     the GPU Lease, Workspace staging for WSL COLMAP Jobs, the per-user NSIS
     installer and the updater.
4. **Screens to parity** with the Qt app: import, SfM/MVS, georeferencing,
   painting masks on photos in the page, labelling on the cloud, training,
   prediction, export. Along the way the kit gains camera frusta,
   single-point picking and lasso selection. Clouds reach the viewer as one
   flat buffer, thinned by the detail level.
5. **Release.** Qt and its tests are removed, the Windows installer is
   published, and Zenodo mints the new version's DOI.

**Done when:** a new user on Windows runs the installer, the app sets up its
Environment unattended (on the GPU when WSL allows it), and the whole workflow
works without the Qt app. COLMAP and the CNNs never hold the GPU at the same
time.

## Next

1. **Georeferencing Tier 2: ground control points picked on the cloud.**
   - Pick points on the dense cloud and enter their surveyed coordinates (a
     CSV import that knows its CRS).
   - Refit the similarity with the existing RANSAC fit.
   - Report a residual for each point, and support check points.
   - Re-align from the backed-up pre-alignment model.

   *Why here:* this needs point picking on the cloud, which 0.3.0 builds in
   the kit. Building it in Qt first would mean building it twice.
2. **Georeferencing Tier 3: the full treatment.**
   - Mark control points on the photos and triangulate them.
   - Bundle adjustment constrained by the control points (may need pycolmap
     work).
   - DJI RTK/PPK coordinates, weighted by each photo's accuracy.
   - Orthophoto and DEM export.

   *Why here:* it builds on Tier 2's control points.

## Later

- **Active learning**: the model proposes which uncertain photos to label
  next.
- **Out-of-core clouds**: COPC streaming once clouds outgrow the viewer's
  point budget. This needs a permissive COPC writer, such as copclib or PDAL.
- **Assisted 2D labelling**: polygon, superpixel or click-to-segment
  (SAM-style).
- **macOS and Linux installers**, when someone asks for them.
- **TensorFlow beyond 2.19.** This touches the locked U-Net, so it is a
  separate decision.

## Dropped

- **The portable zip** (941 MiB): replaced in 0.3.0 by an installer of tens
  of MB that downloads the Environment on first run (ADR-0002).
- **Restyling the Qt app** with GeoScape's tokens: it yields no installer and
  nothing shared with GeoScape (ADR-0002).
