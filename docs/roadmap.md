# Roadmap

*This file decides what comes next and in what order. It holds no design:
the design lives in [DESIGN.md](../DESIGN.md) and the decisions in
[adr/](adr/). **Now** is the next release, with what "done" means. **Next**
lists the themes after it, in order, each with the reason it comes where it
does. **Later** is unordered. **Dropped** records what was taken off, and
why, so it isn't proposed again. An item moves by editing this file, and
leaves the file when it is built.*

Last released: **0.3.0** (2026-11-26), the Desktop app. The Qt window is
replaced by a Tauri app ([desktop.md](desktop.md)) over a JSON-RPC Engine
([engine.md](engine.md)); it shares geokit with GeoScape, installs per user,
updates itself, and sets up its Python Environment unattended, in WSL on the
GPU when it can ([ADR-0001](adr/0001-shared-permissive-kit-from-geoscape.md)
to [ADR-0004](adr/0004-the-tauri-core-starts-every-job.md)).

---

## Now

**0.4.0: Surveys.** A Project is divided into Surveys, each a separate
acquisition with its own Photos and Reconstructions, tied together by GCPs.
The terms are in [GLOSSARY.md](../GLOSSARY.md), and storage follows
[ADR-0005](adr/0005-reconstructions-are-kept-and-aligned-by-a-stored-transform.md).

Steps, in order (step 4 can run beside 2 and 3):

1. **Layout v2, Surveys under the hood, the stored Alignment.** Built:
   one Survey per Project for now, a 0.3.0 Project migrated on open,
   georeferencing and reprojection as stored transforms.
2. **Several Surveys and Reconstructions, in the window.** Built: a step
   run again makes a child Reconstruction sharing the earlier products;
   Surveys are made, described, switched and deleted in the Dataset pane.
3. **GCPs and Marks.** Built: GCPs imported from coordinates files or
   added by hand; Marks placed on photos and (snapped) on the cloud; photo
   Marks triangulated; Surveys fitted to surveyed GCPs and to each other
   through shared ones, with residuals and hints. A Mark is moved by
   placing it again rather than by dragging.
4. **Tiles and the Survey picker.** Built: Tiles ported from `unet_tools`,
   blended at network resolution with the Uncertainty kept; training and
   prediction take a Survey picker; validation holds out whole photos;
   0.3.0 models are listed as needing retraining.
5. **Docs and the release.** Docs and version 0.4.0 done; the release
   waits for the end-to-end test below.
6. **Stop predictions leaking into training** (the first step of
   [ADR-0006](adr/0006-labels-keep-their-source-and-predictions-only-fill-gaps.md),
   before the release, since 0.4.0 would otherwise train on its own output).
   Built: the overlay shows the user's labels over the cloud's over the
   model's, and a save keeps only what differs from it. A stroke of the class
   a pixel already shows doesn't make it hand-drawn yet (step 2 of Label
   sources, below).
   - Predict skips Photos labelled from the cloud, so it no longer hides
     their projection.
   - Images → cloud leaves predictions out unless asked.
   - Saving a painted Photo keeps only the pixels the user's strokes touched
     and the ones that were already hand-drawn, not the overlay it started
     from.
   - A regression test for each.

- **Surveys.**
  - A 0.3.0 Project opens as a single Survey without asking.
  - Each Survey has a name, a timestamp (by default the median of its
    photos' EXIF times), and optionally an operator, a drone or camera
    model, and notes.
  - A Photo is known by its Survey and file name. Each Survey has its own
    Photos; there is no moving Photos between Surveys.
  - There is one Class list per Project.
  - The window shows one current Survey through a Survey switcher in the
    Dataset pane, a single compact line while there is only one Survey.
    Import asks whether to start a new Survey or add to the current one.
  - Deleting a Survey removes its Photos, Masks, predictions,
    Reconstructions and Marks, after asking. GCPs and models stay.
- **Reconstructions.**
  - Running any step makes a new Reconstruction, which shares the earlier
    steps' products with its parent. Shared files are reference-counted.
  - One Reconstruction per Survey is active.
  - Masks belong to the Survey; cloud and mesh labels belong to the
    Reconstruction. A new transfer copies labels from another
    Reconstruction by nearest point.
  - The list shows each Reconstruction's parent and its size on disk.
- **GCPs and Alignment.**
  - A GCP has a name, optionally surveyed coordinates (with a CRS and an
    accuracy), and a use: control or check. Coordinates are entered in a
    table or imported from a CSV, whose dialog asks for the CRS and whether
    heights are ellipsoidal or orthometric.
  - Marks are placed on Photos in the Image view, or on the cloud in the 3D
    view, where they snap to the nearest point.
  - Photo Marks are triangulated with the Reconstruction's cameras. A cloud
    Mark wins over triangulated ones.
  - Each Reconstruction gets a similarity Alignment fitted to its GCPs with
    the existing RANSAC fit, falling back to EXIF GPS when it has fewer
    than 3 GCPs. The Alignment is stored and applied when products are
    read; MVS and meshing run in the SfM frame.
  - The Project frame comes from the GCPs' surveyed coordinates, or, when
    no GCP has any, from a chosen reference Survey.
  - The GCP table shows each GCP's residual per Survey.
  - Once a Survey is aligned, every GCP is projected into its other Photos
    as a hint.
- **Tiles**, ported from `unet_tools`' `SlicedSegmentationProject` (GPL-3.0,
  same author).
  - A model has a square Tile size in photo pixels, a shrink factor, and an
    overlap (35% by default).
  - Photos are padded to the grid. The padding is unlabelled, and Tiles
    with too few labelled pixels are skipped.
  - Predictions of overlapping Tiles are blended with feathered weights at
    the network's resolution, then scaled up.
  - Each prediction keeps its Uncertainty: the Image view shows it as an
    overlay and the Dataset pane shows its mean per Photo.
  - Training and prediction take a Survey picker; training uses every
    Survey by default.
  - A Photo keeps only its latest prediction, recording the model.
  - 0.3.0 models, trained on whole Photos, are listed as needing to be
    retrained.

**Done when:** a Project with two Surveys of the same site, flown on
different dates, is imported, reconstructed and tied by GCPs placed on both
cloud and photos. Its residuals are reported, and a Tile model trained on
both Surveys predicts a held-out third one.

---

## Next

1. **0.5.0: comparing Surveys.**
   - DEM GeoTIFF export, built from the mesh's topmost surface (or from the
     dense cloud), with an optional Class map.
     - The cell size defaults to about twice the ground pixel size.
     - Exported in the export frames (Local ENU, or a projected CRS with
       EGM96), as float32 with nodata −9999, written with rasterio.
   - The multi-survey view:
     - a signed distance from the compared Survey's mesh to the reference
       Survey's surface;
     - cut, fill and net volumes from the DEM difference inside a drawn
       region;
     - differences below the level of detection, about
       `1.96·√(rmsA² + rmsB²)`, shown as no change and left out of the
       volumes.
   - Label transfer between aligned Surveys, cloud to cloud by nearest
     point.

   *Why here:* everything here needs Surveys aligned into one Project frame.
2. **Label sources**
   ([ADR-0006](adr/0006-labels-keep-their-source-and-predictions-only-fill-gaps.md)):
   every label keeps its source (hand-drawn, from the cloud, predicted), and
   predictions only fill gaps. In order:
   1. **Voting by source, with an Uncertainty per point.**
      - `images_to_cloud` takes votes by source, best first, extending its
        current two passes to three.
      - While voting it keeps a 64-bit set of direction cells per point,
        the cameras' directions as seen from it, so the number of
        independent views is its popcount.
      - It returns the labels, a source per point (`uint8`) and an
        Uncertainty per point (`float16`); the vote counts are dropped.
      - The source and Uncertainty are stored beside each cloud's labels in
        the Reconstruction (`labels/dense.source.npy`,
        `labels/dense.uncertainty.npy`, the same for the sparse cloud).
      - Cloud paint sets the source to hand-drawn and the Uncertainty to 0.
      - Projected labels don't vote on the cloud they came from.
      - Tests: the table of worked cases in the ADR, the views counted once
        per cell, a better source never overruled by a worse one.
   2. **Composed labels per pixel.**
      - Cloud → images projects the source and the Uncertainty along with
        the labels.
      - `image.mask` returns the composed labels and a source per pixel.
      - `image.save_mask` takes only the user's strokes, as a mask with −1
        where nothing was painted.
      - The mask editor keeps a touched-pixel map, and gets an "Accept the
        prediction" fill.
      - `docs/engine.md` and `tests/test_engine_contract.py` change in the
        same commit.
   3. **Showing sources and Uncertainty.**
      - The Image view shows the source by opacity (solid hand-drawn, lighter
        from the cloud, hatched predicted) and with a "Show source" toggle.
      - The 3D view and the Image view get "Colour by Uncertainty".
      - The Dataset pane shows each Photo's sources and its mean
        Uncertainty.
   4. **Training by source.**
      - Per-pixel weights travel with each Tile: hand-drawn 1, from the
        cloud 0.5 or (1 − Uncertainty) as an option, predicted 0.
      - Validation scores hand-drawn pixels only.
      - The weights go through `dataset_builder`, `tile_dataset` and the
        trainer; the LOCKED `ml/unet.py` is untouched.
   5. **Predictions fill the gaps.**
      - Predict runs on every Photo, hand-drawn ones included.
      - Masks saved before this change get the per-Photo "Remove copied
        labels" action, with a preview.

   *Why here:* it fixes what the network learns from, so the ML track's
   losses and diagnostics are measured against honest validation. It can
   run beside 0.5.0, which touches the meshes and the alignment rather than
   the labels.
3. **The ML track**, which can run alongside 0.4.0 and 0.5.0 because it
   leaves the data model alone.
   - Losses in a new, unlocked `ml/losses.py`: weighted cross-entropy (with
     inverse-frequency class weights), focal, soft Dice or Tversky, and
     cross-entropy plus Dice (the default). All of them ignore unlabelled
     pixels.
   - Diagnostics saved with each model and shown in a Model pane:
     - curves per epoch, training against validation;
     - a confusion matrix plus IoU, precision and recall per Class;
     - the same metrics per Survey.
   - Validation by random Photos (the default) or by holding out whole
     Surveys.

   *Why here:* the Survey diagnostics and the held-out Survey need 0.4.0.
   The losses don't.
4. **Georeferencing Tier 3: the full treatment.**
   - Bundle adjustment constrained by GCPs (may need pycolmap work).
   - DJI RTK/PPK coordinates, weighted by each photo's accuracy.
   - Orthophoto export.

   *Why here:* it builds on 0.4.0's GCPs and Marks.

## Later

- **Active learning**: the model proposes which uncertain photos to label
  next, ranked by their Uncertainty.
- **Ground-scale Tiles**: a Tile size in metres from each photo's ground
  pixel size, so Surveys flown at different heights see the same ground.
- **Random-crop training**: Tiles drawn again every epoch through a
  streaming input pipeline, instead of a fixed grid held in memory.
- **Photo-to-photo label transfer by feature matching**, as in
  `unet_tools`' `transfer_labels`.
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
