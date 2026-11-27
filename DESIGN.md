# CloudLabeller — Design Document

> A desktop framework for photogrammetric reconstruction and **bidirectional**
> labelling of 2D images ↔ 3D products (point clouds / meshes), with label
> transfer powered by [hylite](https://github.com/hifexplo/hylite) and a U-Net
> model that propagates a few hand-labelled images to the entire dataset.
>
> **0.3.0 notes:**
> - The PySide6/PyVista window is gone. The interface is a Tauri desktop app
>   ([docs/desktop.md](docs/desktop.md)) whose page talks to a Python Engine
>   over JSON-RPC ([docs/engine.md](docs/engine.md)); the reasons are in
>   [ADR-0002](docs/adr/0002-tauri-webview-with-a-json-rpc-engine.md). The
>   stack, the architecture and the module map below are current; the 3D
>   interaction table (§8) records the original VTK design.
> - hylite is no longer a dependency. The projection it inspired (full COLMAP
>   camera model, occlusion) lives in `transfer/projection.py`; the hylite
>   sections below are the original design.

---

## 1. Goals & non-goals

### Goals
- Reconstruct camera poses + sparse/dense geometry from an image set
  (Structure-from-Motion / Multi-View Stereo via **COLMAP/pycolmap**).
- Let the user label **either** modality:
  - paint/segment **images** (2D), or
  - select/paint the **3D cloud or mesh**,
  and *transfer the labels across modalities* using the camera geometry.
- Use a handful of labelled images to **train a U-Net** and run **inference on
  every image** in the dataset (user supplies the model code; we provide the
  data plumbing and placeholders).
- 3D interaction must be **fluid and precise**: GPU-accelerated rendering,
  responsive lasso/brush/box selection, sub-object picking, undo/redo.

### Non-goals (v1)
- We do **not** reimplement SfM/MVS — COLMAP does the heavy lifting; OpenCV is
  used for image I/O, feature/utility ops, and undistortion.
- No cloud/multi-user collaboration; single-machine desktop app.
- No georeferencing pipeline beyond importing COLMAP/known poses (can come
  later via control points).

---

## 2. Tech stack

| Concern              | Choice                                  | Why |
|----------------------|-----------------------------------------|-----|
| Language             | Python 3.12 (Engine, Jobs); TypeScript (window) | the U-Net and COLMAP bindings are Python |
| Desktop shell        | **Tauri 2** (Rust core, system webview) | Small installer, updater, starts every Job (ADR-0004) |
| Window               | **React** + Tailwind on geokit's tokens  | One page for every screen |
| 3D rendering         | **three.js** through `@geokit/viewer`   | Point clouds, meshes, frusta, picking, lasso |
| 2D image canvas      | `<img>` + canvas overlay in the page    | Pan/zoom, polygon and brush masks |
| Photogrammetry       | **pycolmap** (SfM/MVS), OpenCV (utils)  | De-facto reconstruction; exports the camera model hylite needs |
| Label transfer       | **hylite** (`HyScene`, `HyCloud`, `HyImage`, `Camera`) | Native image↔cloud projection w/ occlusion |
| ML                   | PyTorch (assumed) — **your code**       | Placeholders only; framework-agnostic interfaces |
| Mesh / cloud I/O     | `trimesh`, `laspy`, `io/ply.py`         | Broad format support |
| Persistence          | Project folder + JSON manifest + binary label arrays | Transparent, diffable, large arrays kept out of JSON |

> **0.2.x history — why VTK over Open3D:** VTK's picking and Qt embedding via
> `pyvistaqt` gave precise selection on millions of points without our own GL
> layer. 0.3.0 does the same in three.js: an id pass for picking, and lasso
> selection through the whole cloud in the Engine (`cloud.select`).

---

## 3. High-level architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                 Window (apps/web — React, @geokit/viewer)              │
│  Menus · Labels · View · Dataset · 3D viewer · Image view · Log        │
└───────────────▲───────────────────────────────────────────▲──────────┘
                │ Tauri commands / events                     │
┌───────────────┴───────────────────────────────────────────┴──────────┐
│        Tauri core (apps/desktop/src-tauri — Rust)                      │
│  Engine host · Job runner + GPU Lease · WSL staging · Environment      │
└───────▲──────────────────────────────────────────────────▲───────────┘
        │ JSON-RPC over stdin/stdout              starts    │
┌───────┴──────────────────────────────┐   ┌───────────────┴──────────┐
│ Engine (cloudlabeller/engine)        │   │ Jobs (*_cli.py processes)│
│  Project · LabelSchema · LabelStore  │   │  SfM/MVS/mesh (pycolmap) │
│  transfer · plans Jobs, applies them │   │  train/predict (U-Net)   │
└──────────────────────────────────────┘   └──────────────────────────┘
```

The **Core** layer (`core/`) is UI-agnostic and Qt-free. The Engine owns the
open Project; it tells the window what changed (`project.changed`), so
labelling in one view updates the other.

---

## 4. Domain model

### `Project`
A project is a **folder** on disk (layout v2, 0.4.0; a 0.3.0 Project is
moved into it when opened, `core/migrate.py`):

```
my_project.clproj/
├── project.json            # format 2: schema, settings (Project frame, model spec), Survey order
├── surveys/<sid>/          # one per Survey (s1, s2, …)
│   ├── survey.json         # name, timestamp, operator, camera, notes, settings,
│   │                       #   image status, active Reconstruction
│   ├── images/             # the photos
│   ├── masks/<name>.npy    # the user's Masks (int32, -1 = unlabelled)
│   ├── predictions/        # U-Net predictions, at model resolution
│   ├── products/
│   │   ├── sfm-<n>/        # COLMAP db, sparse/, cloud.npz, cameras.json, adjacency.json
│   │   ├── mvs-<n>/        # dense/ (COLMAP workspace), dense.npz
│   │   └── mesh-<n>/       # mesh.npz
│   ├── reconstructions/<rid>/
│   │   ├── recon.json      # its products per step, its Alignment
│   │   ├── keep.npy        # the sparse points it keeps (when some were deleted)
│   │   └── labels/         # sparse.npy, dense.npy, mesh.npy, mesh_nn.npz
│   └── cache/              # visibility (pmap/), derived adjacency
└── ml/models/<name>/       # manifest.json + weights
```

Running a step whose product the active Reconstruction already has makes
a child Reconstruction that shares the earlier products; products are
counted by the Reconstructions that use them and deleted with the last.
GCPs are the Project's (`gcps.json`); a photo Mark is the Survey's and a
cloud Mark the Reconstruction's (in its own frame).

Products are never edited (ADR-0005). Deleting sparse points records
`keep.npy`; georeferencing and reprojecting store a similarity, the
**Alignment**, in `recon.json`, applied when geometry is read for the
window or for export. MVS and meshing run in the SfM frame.

### Key classes (see `cloudlabeller/core/`)
- `LabelSchema` — ordered list of `LabelClass(id, name, color, hotkey)`,
  ids `0..N-1`; unlabelled is `-1`. One per Project.
- `LabelStore` — one per Survey: its Masks and statuses, and the active
  Reconstruction's cloud and mesh labels; undo/redo via a command stack.
- `Dataset` — a Survey's `ImageRecord`s (path, camera) and the active
  Reconstruction's sparse cloud, dense cloud and mesh, in its own frame.
- `Reconstruction` — product slots, kept points, Alignment and the
  `frame_*` functions every frame-dependent reader goes through.
- `Survey` — a Survey's metadata, photos, labels and Reconstructions.
- `Project` — the schema, the Project frame, the Surveys; load/save.

---

## 5. Photogrammetry pipeline (`photogrammetry/`)

`sfm.py` wraps pycolmap:
1. `extract_features` → `match_features` (exhaustive/sequential/vocab-tree).
2. `incremental_mapping` → sparse model (cameras + poses + sparse points).
3. (optional) `mvs.py`: image undistortion → patch-match stereo → fusion →
   dense cloud; Poisson/Delaunay meshing.
4. `cameras.py` converts COLMAP camera models (pinhole, OPENCV, etc.) into the
   intrinsics+extrinsics that **hylite's `Camera`** expects.

The output that matters for labelling: **per-image `Camera` (K, R, t,
distortion)** + a **point cloud** whose points can be projected back into each
image.

---

## 6. Label transfer (`transfer/`) — the hylite bridge

This is the heart of the app. hylite already models exactly what we need:

- `HyCloud` — point cloud with arbitrary per-point scalar/vector data.
- `HyImage` — image with arbitrary per-pixel bands.
- `Camera` — projection model (perspective / pushbroom).
- `HyScene` — binds a `HyCloud` + `HyImage` + `Camera`, computes the
  **visibility / depth buffer**, and the **pixel ↔ point index map**, handling
  occlusion.

### Image → Cloud
For each labelled image, build a `HyScene`; for every visible point, sample the
label of the pixel it projects into; accumulate votes across all images and
assign the **majority class** per point.

### Cloud → Image
Given cloud labels, project each visible point into an image and **splat** its
class id to the covered pixel(s), producing a per-image mask. Used both to
visualise cloud labels on photos and to **seed/clean U-Net training masks**.

### Cloud ↔ Mesh
Labels transfer by nearest-vertex / barycentric mapping (PyVista/`trimesh`
KD-tree). Mesh is mainly for visualisation; the cloud is the canonical label
carrier.

> ⚠️ **API note:** exact hylite call signatures (`HyScene` construction,
> `get_point_index` / `push_to_cloud`-style helpers) must be pinned to the
> installed hylite version — the bridge centralises every hylite call so there's
> one place to adapt. Stubs in `hylite_bridge.py` mark each touch-point.

---

## 7. ML pipeline (`ml/`) — the U-Net on Tiles

1. `dataset_builder.py` — training samples from the user's Masks **plus**
   masks synthesised by *cloud→image* projection (so labelling in 3D feeds
   2D training for free), from the chosen Surveys. `build_tile_set` cuts
   each photo into Tiles (uint8 images, int16 masks); the train/validation
   split holds out whole photos.
2. `model_spec.py` — the settings: Tile size in photo pixels, the shrink
   factor before the network, the overlap, the least-labelled fraction, and
   the architecture (`net_size = tile_size / shrink`, a multiple of
   `2**blocks`).
3. `tiles.py` — the grid, padding (unlabelled for training), feathered
   blending of overlapping Tiles at the network's resolution, and the
   Uncertainty (entropy over its maximum). Ported from `unet_tools`'
   `SlicedSegmentationProject` (same author, GPL-3.0).
4. `tile_dataset.py` — the input pipeline (`tf.data`): each epoch the Tiles
   are fed in a new order, mirrored, slid and gamma-shifted (`unet_tools`'
   augmentation) and batched inside the pipeline, so memory holds the Tiles
   once plus a few batches, however many augmentation rolls.
5. `unet.py` — the user's architecture (LOCKED) and `build_model`.
6. `trainer.py` / `inference.py` — the training loop and predicting a photo
   Tile by Tile; predictions go to each Survey's `predictions/` as `.npz`
   (classes and Uncertainty at the photo's size over the shrink factor).
7. Predicted masks can be **transferred back to the cloud** (§6), closing the
   loop: label a few → train → predict all → fuse into 3D.

Long-running ML/SfM work runs as Jobs: separate processes the Tauri core
starts, with progress and Stop, so the window stays fluid.

---

## 8. Fluid & precise 3D interaction (the original VTK design)

| Need | Implementation |
|------|----------------|
| Smooth nav on M+ points | VTK level-of-detail, point budget, optional octree decimation |
| Pick a single point/cell | `vtkCellPicker` / hardware picking |
| Region select | **Lasso** & **rubber-band box** → `vtkFrustumExtractor` / `vtkAreaPicker`; **brush** with adjustable 3D radius |
| Paint class | Apply active `LabelClass` to selection; live recolour via scalar→LUT |
| Precision | Snap-to-surface brush, depth-gated selection (front-facing only), adjustable falloff |
| Confidence | Undo/redo command stack; per-class visibility & opacity; isolate/lock |
| Feedback | Colour-by-label LUT, selection highlight overlay, status read-out of count/class |

Selection tools follow a small `Tool` strategy interface (`activate`,
`on_mouse_*`, `commit`) so new tools (e.g. region-grow, plane-cut) plug in
cleanly. The same painted result, whether produced in 2D or 3D, flows through
`LabelStore` and is mirrored to the other view via the `EventBus`.

---

## 9. Typical workflows

**A. 3D-first**
import images → run SfM/MVS → paint classes on the cloud → *cloud→image* to
generate masks → train U-Net → predict all images → *image→cloud* to refine.

**B. 2D-first**
import images → run SfM → label a few images → *image→cloud* to see 3D result →
train U-Net → predict all → fuse to cloud.

**C. Import-only** (skip reconstruction)
import existing cameras+cloud (Metashape/COLMAP export) → label & transfer.

---

## 10. Module map → files

| Module | Responsibility |
|--------|----------------|
| `core/project.py` | load/save project folder & manifest |
| `core/dataset.py` | images, cameras, cloud, mesh registry |
| `core/label_schema.py` | classes, colours, hotkeys |
| `core/labels.py` | label arrays + undo/redo command stack |
| `photogrammetry/sfm.py` | pycolmap SfM wrapper |
| `photogrammetry/mvs.py` | dense cloud + meshing |
| `photogrammetry/cameras.py` | COLMAP cameras → app `Camera` (intrinsics, distortion) |
| `photogrammetry/adjacency.py` | image-overlap (covisibility) graph from SfM point tracks; restricts per-image transfers to overlapping images |
| `photogrammetry/colmap_fetch.py` | one-click download of the official COLMAP release into `~/.cloudlabeller` (COLMAP is not shipped with the app: its bundle links GPL components) |
| `transfer/projection.py` | full COLMAP camera projection + occlusion (z-buffer, back-face culling) |
| `transfer/project_to_cloud.py` | image labels → cloud (voting) |
| `transfer/project_to_image.py` | cloud labels → image masks (splatting) |
| `transfer/mesh_cloud.py` | cloud ↔ mesh label mapping |
| `core/survey.py`, `core/reconstruction.py`, `core/gcp.py`, `core/migrate.py` | Surveys, Reconstructions (products, Alignment), GCPs, moving 0.3.0 Projects to layout v2 |
| `photogrammetry/triangulate.py`, `photogrammetry/align.py` | locating GCPs from photo Marks; fitting Surveys to GCPs and to each other |
| `ml/dataset_builder.py` | Tile samples from labels + projections, split by photo |
| `ml/model_spec.py`, `ml/tiles.py` | the Tile settings; cutting, blending, Uncertainty |
| `ml/tile_dataset.py` | the `tf.data` training pipeline: augmentation and batches on the fly |
| `ml/unet.py` | the user's U-Net (LOCKED) and `build_model` |
| `ml/trainer.py` | the training loop |
| `ml/inference.py` | predicting a photo Tile by Tile, predict-all |
| `engine/` | the Engine: JSON-RPC methods, Job plans and results ([docs/engine.md](docs/engine.md)) |
| `*_cli.py` | Job entry points (SfM, MVS, mesh, georef, train, predict, …) |
| `workers/resources.py` | thread and memory limits for Job processes |
| `apps/web/src/` | the window: shell, panes, dialogs, 3D viewer, Image view |
| `apps/desktop/src-tauri/` | the Tauri core: Engine host, Job runner, Environment, files |

---

## 11. Open decisions / future work
What comes next, and in what order, is kept in [docs/roadmap.md](docs/roadmap.md).
