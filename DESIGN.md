# CloudLabeller — Design Document

> A desktop framework for photogrammetric reconstruction and **bidirectional**
> labelling of 2D images ↔ 3D products (point clouds / meshes), with label
> transfer powered by [hylite](https://github.com/hifexplo/hylite) and a U-Net
> model that propagates a few hand-labelled images to the entire dataset.

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
| Language             | Python 3.11+                            | hylite + your U-Net code are Python |
| GUI shell            | **PySide6** (Qt 6)                       | Mature, native, dockable panels |
| 3D rendering         | **PyVista + `pyvistaqt`** (VTK backend) | GPU rendering, rich picking API, embeds in Qt |
| 2D image canvas      | `QGraphicsView` + NumPy/QImage          | Fast pan/zoom, overlay alpha masks |
| Photogrammetry       | **pycolmap** (SfM/MVS), OpenCV (utils)  | De-facto reconstruction; exports the camera model hylite needs |
| Label transfer       | **hylite** (`HyScene`, `HyCloud`, `HyImage`, `Camera`) | Native image↔cloud projection w/ occlusion |
| ML                   | PyTorch (assumed) — **your code**       | Placeholders only; framework-agnostic interfaces |
| Mesh / cloud I/O     | hylite, PyVista, `trimesh`, `laspy`     | Broad format support |
| Persistence          | Project folder + JSON manifest + binary label arrays | Transparent, diffable, large arrays kept out of JSON |

> **Why VTK over Open3D:** VTK's picking (`vtkCellPicker`,
> `vtkAreaPicker`/frustum, hardware selection) and Qt embedding via `pyvistaqt`
> give us precise, interactive selection on millions of points without writing
> our own GL layer.

---

## 3. High-level architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                              UI (PySide6)                              │
│  MainWindow ── docks ──┬─ Viewer3D (pyvistaqt)                         │
│                        ├─ ImageView (QGraphicsView)                    │
│                        ├─ LabelPanel (class schema, colors, visibility)│
│                        ├─ DatasetPanel (image list, products)          │
│                        └─ ToolBar (selection tools, transfer, train)   │
└───────────────▲───────────────────────────────────────────▲──────────┘
                │ Qt signals (EventBus)                       │
┌───────────────┴───────────────────────────────────────────┴──────────┐
│                              Core / Domain                             │
│  Project ── Dataset(images, cameras, cloud, mesh)                      │
│           ├─ LabelSchema (classes ↔ colors ↔ ids)                      │
│           └─ LabelStore (per-point labels, per-pixel masks, undo)      │
└───────┬───────────────────┬───────────────────┬───────────────────────┘
        │                   │                   │
┌───────▼───────┐  ┌────────▼────────┐  ┌───────▼─────────┐
│ photogrammetry │  │    transfer     │  │       ml        │
│  (pycolmap)    │  │   (hylite)      │  │  (your U-Net)   │
│  sfm/mvs/      │  │ img↔cloud↔mesh  │  │ build/train/    │
│  cameras       │  │  via HyScene    │  │ infer (stubs)   │
└────────────────┘  └─────────────────┘  └─────────────────┘
```

The **Core** layer is UI-agnostic and import-safe (no Qt). The UI subscribes to
a lightweight `EventBus` so that labelling in one view updates the other.

---

## 4. Domain model

### `Project`
A project is a **folder** on disk:

```
my_project.clproj/
├── project.json          # manifest: schema, paths, settings, product registry
├── images/               # source images (or symlinks/paths to them)
├── reconstruction/        # COLMAP db + sparse/ + dense/ outputs
├── products/
│   ├── cloud.ply         # point cloud (hylite HyCloud-compatible)
│   └── mesh.ply
├── labels/
│   ├── cloud_labels.npy  # int32 per-point class id  (-1 = unlabelled)
│   ├── images/<name>.npy # int32 per-pixel masks
│   └── history.json      # undo/redo journal (optional)
└── ml/
    ├── checkpoints/
    └── predictions/       # U-Net masks for all images
```

### Key classes (see `cloudlabeller/core/`)
- `LabelSchema` — ordered list of `LabelClass(id, name, color, hotkey)`.
  `id 0` reserved for *background/unlabelled*.
- `LabelStore` — owns the label arrays for the cloud and per image; emits
  `labels_changed(modality, ids)`; provides `undo()`/`redo()` via a command
  stack.
- `Dataset` — registry of `ImageRecord`s (path, COLMAP camera/pose), the
  `PointCloud`, and the `Mesh`.
- `Project` — load/save, glues the above to disk.

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

## 7. ML pipeline (`ml/`) — U-Net (your code goes here)

We provide the **data plumbing** and clear seams; you drop in your model/training
loop at the marked placeholders.

1. `dataset_builder.py` — assemble `(image, mask)` pairs from labelled images
   **plus** masks synthesised by *cloud→image* projection (so labelling in 3D
   feeds 2D training for free). Handles tiling, train/val split, augmentation
   hooks.
2. `unet.py` — `# PLACEHOLDER`: your architecture (`build_model()`).
3. `trainer.py` — `# PLACEHOLDER`: your training loop (`train(model, data)`),
   wrapped so the UI can run it off-thread with progress callbacks.
4. `inference.py` — `# PLACEHOLDER`: `predict(model, image) -> mask`; the
   orchestrator runs it over **all** images and writes `ml/predictions/`.
5. Predicted image masks can be **transferred back to the cloud** (§6), closing
   the loop: label a few → train → predict all → fuse into 3D.

Long-running ML/SfM jobs run in a `QThread`/worker with cancel + progress so the
UI stays fluid.

---

## 8. Fluid & precise 3D interaction (`ui/viewer3d.py`, `ui/tools/`)

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
| `core/events.py` | Qt signal bus connecting views |
| `photogrammetry/sfm.py` | pycolmap SfM wrapper |
| `photogrammetry/mvs.py` | dense cloud + meshing |
| `photogrammetry/cameras.py` | COLMAP ↔ hylite camera conversion |
| `photogrammetry/adjacency.py` | image-overlap (covisibility) graph from SfM point tracks; restricts per-image transfers to overlapping images |
| `photogrammetry/colmap_fetch.py` | one-click download of the official COLMAP release into `~/.cloudlabeller` (COLMAP is not shipped with the app: its bundle links GPL components) |
| `transfer/hylite_bridge.py` | all hylite calls (HyScene/HyCloud/HyImage) |
| `transfer/project_to_cloud.py` | image labels → cloud (voting) |
| `transfer/project_to_image.py` | cloud labels → image masks (splatting) |
| `transfer/mesh_cloud.py` | cloud ↔ mesh label mapping |
| `ml/dataset_builder.py` | build training pairs from labels + projections |
| `ml/unet.py` | **PLACEHOLDER** model |
| `ml/trainer.py` | **PLACEHOLDER** training loop |
| `ml/inference.py` | **PLACEHOLDER** predict-all orchestration |
| `ui/main_window.py` | dockable shell, menus, toolbars |
| `ui/viewer3d.py` | PyVista/VTK 3D view + picking |
| `ui/image_view.py` | 2D labelling canvas |
| `ui/label_panel.py` | schema editor + visibility |
| `ui/dataset_panel.py` | image/product browser |
| `ui/tools/` | selection/paint tool strategies |
| `workers/` | QThread wrappers for SfM/MVS/train/infer |

---

## 11. Open decisions / future work
- Georeferencing via GCPs / EXIF GPS.
- Active-learning loop (model proposes uncertain images to label next).
- Multi-resolution out-of-core cloud (Potree-style) for very large datasets.
- Polygon/superpixel-assisted 2D labelling (SAM-style click-to-segment).
