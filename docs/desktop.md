# The desktop app

*How the 0.3.0 desktop app is laid out, run and released. The decisions
behind it are [ADR-0001](adr/0001-shared-permissive-kit-from-geoscape.md) to
[ADR-0004](adr/0004-the-tauri-core-starts-every-job.md). The Engine's
contract is [engine.md](engine.md).*

## Layout

| Path | What it is |
|---|---|
| `apps/web` | The page: React, Tailwind 4 and geokit's tokens (`@geokit/ui`) and viewer layers (`@geokit/viewer`). Its only way out is `src/desktop/bridge.ts`. |
| `apps/desktop/src-tauri` | The Tauri 2 core, in Rust, on `geokit-env`: `environment.rs` (the Environment), `engine.rs` and `project.rs` (the Engine host, folder picking, Array files), `files.rs` (photos, picked files, recent Projects, keeping awake), `jobs.rs` (Jobs, the GPU Lease, Workspaces), `updates.rs`. |
| `apps/desktop/scripts` | `resources.mjs` fills `apps/desktop/resources/` (git-ignored) with uv for Windows and Linux, the app's wheel and the lockfile. `latest.mjs` writes the updater's `latest.json`. |
| `apps/desktop/environment/requirements.lock` | The Environment's pinned packages (Part A of step 3, `tools/lock_environment.py`). |

No code comes from GeoScape. It is closed, and CloudLabeller is GPL. Shared
code reaches CloudLabeller only through geokit, pinned by tag. Something both
apps need is written in geokit first.

## The window

`apps/web/src`, by folder:

| Folder | What it holds |
|---|---|
| `shell/` | The window: the menu bar, the panes, moving between Surveys (`surveys.ts`: the window shows one at a time, and `engine/api.ts` sends every Survey-scoped call to it), the status bar (CRS frame, Project summary, the running Job with Stop and its place in the GPU queue), the keys that work everywhere (Ctrl+Z / Ctrl+Y, 0–9 for the classes), Welcome (New, Open, recent Projects), and the close guard. |
| `state/` | One reducer (`store.ts`, tested) and its context (`app.tsx`): running a Job or a long Engine request with the status bar following it, saying a Problem, asking a question. |
| `engine/api.ts` | Every Engine method the window calls, typed after [engine.md](engine.md), with Array files read into typed arrays. |
| `jobs/` | `flows.ts`: what a menu item sets going (georeferencing after SfM, a mesh after MVS, the questions around them). `pipeline.ts`: the full pipeline as a chain of Jobs, tested. |
| `viewer/` | The 3D view (`Viewer3D`, `rig.ts`) on geokit's `LayerHost`: the cloud as one node thinned by the detail level, the mesh, the camera frusta, the lasso, and the View pane. Pure helpers in `adapt.ts`, tested. |
| `image/` | The photo with its mask: polygon, brush and eraser, Import Mask, Propagate, the GCP tool (Marks and hints) and the Uncertainty overlay. Rasterising in `raster.ts`, tested. |
| `panes/` | Labels, Dataset (the Survey line, the photos, the GCPs tab in `GcpPanel.tsx`) and Log (each Job's output and every new progress message, `jobs/progress.ts`). |
| `dialogs/` | One per menu item that asks something: SfM, MVS, mesh, pipeline, COLMAP, cleaning, reprojecting, export, the model's settings, training, prediction (both with a Survey picker), a Survey's details, Reconstructions, a GCP, importing GCPs, Project Info, About. |
| `ui/controls.tsx` | The controls every screen builds from, on the kit's tokens. |

## Running it

```
pnpm install
pnpm check          # Biome, tsc, vitest, node tests, cargo test
pnpm desktop        # resources, then `tauri dev` (the page served on :5174)
```

On first start the app sets up its Environment. It goes into WSL when a
distribution reaches the GPU, and natively otherwise. That takes a while,
since TensorFlow is large. To skip it while developing, start a debug build
with your own Python:

```
CLOUDLABELLER_DEV_PYTHON='C:\Python_envs\cloudlabeller\python.exe' pnpm desktop
```

That Python is used as it is, with this checkout on `PYTHONPATH`. It runs
natively, so Jobs aren't staged into a Workspace.

## Releasing

1. Once only: `pnpm --filter @cloudlabeller/desktop tauri signer generate -w ~/.tauri/cloudlabeller.key`.
   - Put the public key in `tauri.conf.json` under `plugins.updater.pubkey`.
   - Store the private key as the GitHub secret `TAURI_SIGNING_PRIVATE_KEY`, with its password, if any, as `TAURI_SIGNING_PRIVATE_KEY_PASSWORD`.
   - Keep a copy of the key outside GitHub. If it's lost, installed copies can't update any more and have to be reinstalled by hand.
2. Set the version in `cloudlabeller/__init__.py`, `pyproject.toml`, the
   `package.json`s and `Cargo.toml`.
3. Push a `v<version>` tag. `.github/workflows/desktop.yml` builds the signed
   installer and attaches it, its `.sig` and `latest.json` to a draft
   Release. Publish the draft: the updater reads
   `releases/latest/download/latest.json`, which only a published Release
   serves.

The installer is NSIS, per user, with no Authenticode signature (the
updater's signature is separate). Pushes to `dev` build an unsigned
installer as a CI artefact. The updater never offers it.
