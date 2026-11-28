# The Engine's contract

*The Python side of an open Project ([ADR-0002](adr/0002-tauri-webview-with-a-json-rpc-engine.md)).
This file is the contract between the Engine and the window.
`tests/test_engine_contract.py` holds the Engine to it, and checks that the
methods listed here are exactly the ones the Engine answers. A method is
added here in the same change that adds it to the Engine.*

## Running it

```
python -m cloudlabeller.engine
```

The window starts one Engine per open Project, inside the active
Environment. The Engine runs until `engine.shutdown` or until its stdin
closes. Either way it saves a dirty Project first. It runs no processes:
Jobs are started by the Tauri core ([ADR-0004](adr/0004-the-tauri-core-starts-every-job.md)),
and they keep running if the Engine stops.

## Framing

- JSON-RPC 2.0, one message per line, UTF-8, `\n`-terminated, both ways.
  Batches are not supported.
- stdout carries the protocol and nothing else. Logs, and anything a
  library prints, go to stderr.
- The first line the Engine writes is the `engine.ready` notification.
- Requests may overlap. A long request (a transfer) answers when it ends,
  and other requests are answered meanwhile. Responses can arrive out of
  order, so match them by `id`.

## Errors

| `error.code` | When |
|---|---|
| -32700 | The line is not JSON. |
| -32600 | The message is not a JSON-RPC 2.0 request. |
| -32601 | No such method. |
| -32602 | The params don't fit the method. `data.code` is `request.invalid_params`. |
| -32603 | A bug in the Engine. `data.traceback` has the details. |
| -32000 | A Problem: something to tell the user. |

A Problem's `data` is `{code, english, params}`, the same shape as geokit's
`Problem`. The window matches on `code` and words the message itself.
`english` is the fallback, and `params` holds the values the wording needs.
Two codes can come from any method:

- `project.none_open`: the method needs an open Project.
- `request.cancelled`: the request was cancelled with `engine.cancel`.

## Array files

Large arrays never go through the pipe. An Array file is raw little-endian,
C-ordered bytes with no header, in the Project's `.arrays/` folder.
Messages carry its descriptor:

```json
{"path": ".arrays/mask/s1/IMG_0001.JPG.bin", "dtype": "int32", "shape": [3000, 4000]}
```

- `path` is relative to the Project root and uses `/`, so it names the same
  file on both sides of the Windows/WSL boundary.
- `dtype` is one of `uint8`, `int8`, `uint16`, `int16`, `uint32`, `int32`,
  `float32` or `float64`.
- **Outbound** (the Engine writes): each purpose has a stable name. The file
  stays valid until the next request for the same array.
- **Inbound** (the window writes): the window puts the file anywhere under
  `.arrays/`. The Engine checks its size against dtype × shape, and deletes
  it once read.
- The Engine empties `.arrays/` when a Project opens and when it closes.

Label ids are `int32`. Unlabelled is `-1` and the classes are `0..N-1`.

## Surveys and frames

A Project holds one or more Surveys (layout v2, see `GLOSSARY.md` and
ADR-0005). Every method about photos, clouds or the mesh takes `survey`,
the Survey's id (`s1`, `s2`…), and works on that Survey's active
Reconstruction. Photo names are unique within a Survey only. A request
naming no Survey that exists fails with `survey.not_found` (`params.survey`).

Geometry sent to the window (`cloud.points`, `cloud.view`, `mesh.geometry`,
`cameras.list`), what the lasso reads (`cloud.select`) and what export
writes is in the **Project frame**: the Reconstruction's products stay in
their own SfM frame, and its Alignment carries them into the Project frame
as they are read. A Reconstruction with no Alignment is shown and exported
in its own frame. Point indices, labels and transfers don't depend on the
frame.

Opening a 0.3.0 Project (format 1) moves it into layout v2 first, as one
Survey `s1`; `project.json` from before is kept as `project.v1.json`.

## Notifications (Engine → window)

| Method | Params |
|---|---|
| `engine.ready` | Same as `engine.hello`. |
| `engine.log` | `{level, message}`: an internal failure the user didn't ask about. |
| `project.changed` | `{parts, survey?, images?}`. `parts` are drawn from `schema`, `surveys`, `gcps`, `images`, `status`, `sparse`, `dense`, `mesh`, `labels` and `settings`. `survey` names the Survey the change is in; absent, it is Project-wide (the schema, the frame, the models). `images` names that Survey's photos whose mask or status changed. |
| `request.progress` | `{id, fraction, message}` for a long request still running. |

A Job's progress and log reach the window from the Tauri core, which runs
the Job, not from the Engine.

## Methods

### `engine.hello`
→ `{protocol, version, methods}`. `protocol` is 1. It changes only when an
existing method changes incompatibly.

### `engine.shutdown`
→ `{}`, then the Engine saves, kills its Jobs and exits.

### `engine.cancel`
`{id}` → `{cancelled}`. Cancels a pending long request. That request then
answers with `request.cancelled`. `cancelled` is false when no such request
is pending.

### `project.create`
`{root, classes?: [{name?, color?}]}` → the Project state (as
`project.state`). Closes the open Project first. Problems: `project.exists`,
`project.jobs_running`.

### `project.open`
`{root}` → the Project state. Closes the open Project first; moves a
0.3.0 Project into layout v2 (see Surveys and frames). Problems:
`project.not_found`, `project.jobs_running`, `project.format` (a format this
version doesn't read).

### `project.close`
→ `{}`. Saves if dirty. Problem: `project.jobs_running`.

### `project.save`
→ `{}`. Writes the manifests and every label array.

### `project.state`
→ `{root, schema, surveys, georeferenced, dirty}`. `georeferenced` is the
Project frame (`{frame, origin_lla, origin_convention, crs?}`), or null.
Each Survey in `surveys`, in the Project's order, is `{id, name, timestamp,
operator, camera, notes, images: [{name, status, solved, user_mask,
uncertainty}],
reconstruction, active, reconstructions, aligned, sparse, dense, mesh, normals,
source_resolution, can_undo, can_redo}`.

- `active` is the active Reconstruction's id, null before SfM, and
  `reconstructions` how many the Survey has. `aligned`: the active one has
  an Alignment into the Project frame.
- `status` is `user`, `auto`, `ml` or `none`. `uncertainty` is an `ml` or `auto`
  photo's mean Uncertainty (0..1): its prediction's, or its projected labels', else null.
- `reconstruction` is `none`, `outdated` or `current`.
- `sparse` and `dense` are point counts, or null. `mesh` is
  `{vertices, faces}`, or null. `normals`: the dense cloud has normals
  (what `cloud.view`'s `colour: "normals"`, `image.normals` and a model
  on normals need).

### `project.summary`
→ `{rows: [[label, value], …]}`: the Project Info rows, as plain text,
the Project's first, then each Survey's (labels indented by two spaces).

### `schema.add`
`{name?, color?}` → `{schema}`. Each class in `schema` is
`{id, name, color, hotkey}`.

### `schema.rename`
`{id, name}` → `{schema}`. Problem: `schema.no_class`.

### `schema.set_color`
`{id, color}` → `{schema}`. Problem: `schema.no_class`.

### `schema.delete`
`{id}` → `{schema}`. The class's points and pixels become unlabelled, the
ids above it shift down by one, and the undo history is cleared.

### `survey.create`
`{name?, timestamp?, operator?, camera?, notes?}` → the new Survey, as
`project.state` lists it. It goes after the others; `name` defaults to
"Survey n". Its `timestamp`, when not given, is set from its photos' EXIF
once photos are added.

### `survey.update`
`{survey, name?, timestamp?, operator?, camera?, notes?}` → the Survey.
`timestamp` is `YYYY-MM-DDTHH:MM:SS`; empty clears it.

### `survey.delete`
`{survey}` → `{freed_bytes, surveys}`. Deletes the Survey's folder: its
photos, Masks, predictions, products and Reconstructions. GCPs and models
are the Project's and stay. Problems: `survey.last` (a Project keeps one),
`project.jobs_running`.

### `reconstruction.list`
`{survey}` → `{reconstructions: [{id, name, parent, products, aligned,
created, active, freed_bytes}], active}`, oldest first. `products` maps
`sfm`, `mvs` and `mesh` to product folders (null when not run); a child
shares its parent's earlier ones. `freed_bytes` is what deleting it would
free.

### `reconstruction.activate`
`{survey, reconstruction}` → as `reconstruction.list`. Shows, labels and
exports that one from now on; the labels of the one before are saved, and
the undo history is cleared. Problems: `reconstruction.not_found`,
`project.jobs_running`.

### `reconstruction.rename`
`{survey, reconstruction, name}` → as `reconstruction.list`.

### `reconstruction.delete`
`{survey, reconstruction}` → `{freed_bytes, reconstructions, active}`.
Deletes it and the products no other Reconstruction uses. Deleting the
active one activates its parent, else the newest left. Problems:
`reconstruction.not_found`, `project.jobs_running`.

### `labels.undo`
`{survey}` → `{changed: {modality: "cloud"|"image", image} | null}`. Each
Survey has its own history.

### `labels.redo`
`{survey}` → `{changed: {modality: "cloud"|"image", image} | null}`

### `image.mask`
`{survey, name}` → `{source, mask, sources}`. The overlay for a photo: the user's
own labels, over the labelled cloud's projection, over the U-Net's
prediction, so the better source wins at each pixel (ADR-0006).

- `source` is the best one present: `user` (the user's own mask), `auto`
  (projected from the labelled cloud) or `ml` (the U-Net's prediction, at
  the photo's size), or null when there is nothing to show.
- `mask` is an Array file of shape `[H, W]`, or null, in the narrowest of
  `int8`, `int16` and `int32` that holds its ids (`int8` for up to 127
  classes: a 20-megapixel mask is then 20 MB, not 80).
- `sources` is a `uint8` Array file `[H, W]`, or null: which source shows at
  each pixel, coded as `cloud.sources` does (0 none, 2 hand-drawn, 3 from
  the cloud, 4 predicted).

Problem: `image.not_found`.

### `image.save_mask`
`{survey, name, mask, touched?}` → `{changed_pixels, status}`. Stores the mask
the user painted, as one undoable edit, and marks the photo `user`. The
window sends the whole mask, seeded from the overlay `image.mask` showed,
and `touched`, a `uint8` Array file `[H, W]` that is nonzero where its
strokes went. Only those pixels are kept as the user's (ADR-0006), whatever
their class: painting the class a pixel shows, or accepting the prediction,
makes it hand-drawn, and erasing shows what is under the user's labels
again. Without `touched`, the strokes are the pixels that differ from the
overlay. The cloud's and the U-Net's labels are never copied in as
hand-drawn. Nothing is stored when a photo without a mask gets no stroke. `mask` is an inbound
`int8`, `int16` or `int32` Array file of the photo's size (the window sends
the narrowest). Problems: `image.not_found`,
`image.mask_shape`, `image.mask_values`.

### `image.info`
`{survey, name}` → `{path, width, height}`. `path` is the photo's file, relative to
the Project root with `/` separators. The size is the solved camera's, else
the file's. Problem: `image.not_found`.

### `image.normals`
`{survey, name}` → `{normals, step}`: the photo coloured by the dense
cloud's normals as `cloud.view` colours the cloud, each pixel the normal
of the point it sees (128, 128, 128 where it sees none). `normals` is a
`uint8 [h, w, 3]` Array file at the photo's size divided by `step` (its
longer side at most 2048 px). Both are null when the photo has no camera
or the Survey no dense cloud with normals.

### `image.uncertainty`
`{survey, name}` → `{uncertainty, mean, step}`: how unsure each pixel's
label is, 0 (sure) to 1, as `image.mask` composes them (ADR-0006): 0 where
hand-drawn; where from the cloud, the Uncertainty of the point it came from
(`cloud.sources`); where predicted, the U-Net's (the entropy of its class
probabilities over the most it can be); NaN where unlabelled or unknown. A
`float32` Array file `[h, w]`, the photo's size taken every `step` pixels
(its longest side at most 2048), and the mean of its known pixels. All are
null when the photo has no labels.

### `cloud.points`
`{survey, cloud?: "sparse"|"dense"}` → `{cloud, n_points, xyz, rgb}`. `xyz`
is a `float32` Array file `[N, 3]`, in the Project frame. `rgb` is a `uint8` one `[N, 3]`, or null.
Problem: `cloud.none`.

### `cloud.labels`
`{survey, cloud?}` → `{cloud, labels}`. `labels` is an Array file `[N]`, narrowed
as `image.mask`'s is.

### `cloud.sources`
`{survey, cloud?}` → `{cloud, source, uncertainty}`. Where each point's label
came from and how sure it is (ADR-0006). `source` is a `uint8` Array file
`[N]`: 0 none, 1 painted on the cloud, 2 hand-drawn (decided by photos the
user painted), 3 from the cloud (decided by projected labels), 4 predicted.
`uncertainty` is a `float32` Array file `[N]`, 0 (sure) to 1, from the votes
that decided the point: their disagreement and how many independent views
they came from. It is 0 where painted, and NaN where unlabelled or unknown
(labels from before sources were kept).

### `cloud.paint`
`{survey, cloud?, indices, class_id}` → `{painted}`. Gives the chosen points one
class (`-1` erases), as one undoable edit. `indices` is an inbound `int32`
or `uint32` Array file. Painting the dense cloud also updates the mesh's
labels. Problems: `schema.no_class`, `cloud.bad_indices`.

### `cloud.view`
`{survey, cloud?, max_points?: 0, colour?: "photo"}` → `{cloud, n_points, shown, xyz, rgb, index}`. A
display subsample: `max_points` points chosen uniformly at random with a
fixed seed (the same ones every call), in the cloud's own order. `xyz` is
`float32 [shown, 3]`, `rgb` is `uint8 [shown, 3]` or null, and `index` is
an `int32 [shown]` Array file holding each shown point's index in the full
cloud. When `max_points` is 0 or at least N, every point is shown and
`index` is null. With `colour: "normals"`, `rgb` is the points' normals in
the Project frame as a false colour, `round(127.5 (n + 1))` per component
(red east/+x, green north/+y, blue up/+z). With `colour: "uncertainty"`,
`rgb` is each point's Uncertainty (`cloud.sources`) on a ramp from blue
(sure) through yellow to red (unsure), grey where unlabelled or unknown.
Problems: `cloud.none`, `cloud.no_normals` (only a dense cloud has normals).

### `cloud.select`
`{survey, cloud?, view_projection, polygon, width, height}` → `{count, indices}`. A
long request. Selects the full cloud's points whose screen projection lies
inside the lasso, including points hidden behind others.

- `view_projection` is 16 numbers, column-major as three.js's
  `Matrix4.elements` gives them: `camera.projectionMatrix ×
  camera.matrixWorldInverse`, which maps world coordinates to clip space.
- `polygon` is `[[x, y], …]` in CSS pixels, origin top-left, y down, and
  closes implicitly. The test is even-odd, so a region circled twice is
  outside. A polygon of over 512 vertices is evenly thinned to 512.
- `width` and `height` are the canvas size in CSS pixels.
- Points behind the camera (clip w ≤ 0) are never selected.

`indices` is a `uint32` Array file, ascending. Fewer than 3 vertices select
nothing. Problem: `cloud.none`.

### `cloud.delete`
`{survey, cloud?: "sparse", indices}` → `{removed, n_points}`. Deletes the chosen
sparse points. The Reconstruction stops showing them (its SfM product is
not edited), the labels follow the surviving points, the undo history is
cleared, and the Project is saved.
Only the sparse cloud can lose points. `indices` is an inbound `int32` or
`uint32` Array file. Problem: `cloud.bad_indices`.

### `cloud.stats`
`{survey, cloud?: "sparse"}` → `{n_points, matched, views, errors}`. A long
request. The sparse points' SfM confidence, read from the saved COLMAP
model. `views` is an `int32 [N]` Array file: how many photos saw each
point, or 0 when the model lacks it. `errors` is a `float32 [N]` one: the
mean reprojection error in pixels, infinite when unmatched. `matched` is
the share of points the model holds. The statistics are cached until the
cloud changes. Problems: `cloud.none`, `cloud.no_model` (no COLMAP model on
disk), `cloud.model_mismatch` (`params.matched` is below 0.9: the model is
from another solve), `cloud.changed` (the cloud changed while they were
read).

### `cloud.clean_preview`
`{survey, min_images, max_error}` → `{keep, total, removed}`. A long request. What
a confidence clean would do. It keeps the points seen by at least
`min_images` photos with an error of at most `max_error` pixels. `removed`
is a `uint32` Array file of the points it would delete. Problems: as
`cloud.stats`.

### `cloud.clean`
`{survey, min_images, max_error}` → as `cloud.delete`. A long request. Applies the
clean that `cloud.clean_preview` describes, as `cloud.delete` does.
Problems: as `cloud.stats`.

### `cameras.list`
`{survey}` → `{cameras: [{name, width, height, K, R, t, position,
position_crs}], up}`. The solved cameras, posed in the Project frame.

- `K` and `R` are 3×3 lists of rows and `t` is 3 numbers. A world point
  `X` projects as `K (R X + t)`.
- `position` is the camera centre, `-Rᵀ t`, in the stored frame.
  `position_crs` adds the reprojected frame's offset, so it is the full CRS
  coordinate. It equals `position` when the Project isn't reprojected or
  the Survey isn't aligned.
- `up` is the scene's up for the home view: `[0, 0, 1]` when the Survey is
  aligned into a Project frame, otherwise the normalised mean of the cameras' `-R[1]`
  (COLMAP's image y points down). It is null when there are no cameras
  and no georeference.

### `mesh.geometry`
`{survey}` → `{vertices, faces, colors}`: Array files `float32 [V, 3]` (in
the Project frame),
`int32 [F, 3]` and `uint8 [V, 3]` (`colors` may be null). Problem: `mesh.none`.

### `mesh.labels`
`{survey}` → `{labels}`: `[V]`, narrowed as `image.mask`'s is, nearest-point labels from the dense cloud.

### `mesh.estimate`
`{survey, method?: "poisson"|"delaunay", depth?}` → `{minutes, ram_gb, vertices,
suggested_depth, machine_ram_gb}`. A long request: the first one measures
the dense cloud's point spacing. A very rough cost of meshing the dense
cloud. `depth` null means `suggested_depth`, the Poisson depth whose cells
match the point spacing. `machine_ram_gb` is this machine's RAM, or null.
Problem: `job.mesh.no_dense`.

### `transfer.images_to_cloud`
`{survey, include_auto?: true, include_predictions?: false}` → `{images, sparse, dense?}`.
A long request. Votes the photos' labels onto the sparse cloud and the dense
one, with user-drawn masks outranking automatic ones. `include_auto` adds
the cloud-projected photos that cover the surface; `include_predictions`
adds the U-Net's predictions, left out by default so the network's output
doesn't come back to it as training data (ADR-0006). Votes are taken per
source, best first: hand-drawn, then from the cloud, then predicted; a
point any better source reaches is decided by it alone. Projected labels
don't vote on the cloud they came from. Each point keeps the source that
decided it and an Uncertainty from those votes, counted by independent
views (`cloud.sources`); a point painted on the cloud is never changed.
The votes are merged into the existing labels, so points no photo reached
keep theirs. Returns the number of points updated. Problems:
`transfer.no_cloud`, `transfer.no_labelled_images`.

### `transfer.cloud_to_images`
`{survey}` → `{cloud, auto_labelled}`. A long request. Marks `auto` every solved photo
the user hasn't drawn on that the labelled cloud reaches, and records the
mean Uncertainty of its projected labels (the photo rows' `uncertainty`).
Their overlays come from `image.mask`. Problems: `transfer.no_cloud`,
`transfer.cloud_unlabelled`.

### `transfer.propagate`
`{survey, name}` → `{cloud, points, auto_labelled}`. A long request. Spreads one
photo's labels through the cloud to the photos that overlap it. A neighbour
needs 500 labelled pixels to be marked `auto`. Problems:
`transfer.image_unlabelled`, `transfer.image_unsolved`, `transfer.no_cloud`.

### `transfer.reconstruction`
`{survey, from}` → `{sparse, dense?}`. A long request. Copies labels from
Reconstruction `from` into the active one: each point takes the label of
the nearest labelled point of the same kind of cloud there (the other kind
when that has none), within three times its spacing. Merged as one
undoable edit; the counts are the points labelled. The two must share
their SfM product, or both be aligned. Problems: `reconstruction.not_found`,
`transfer.same_reconstruction`, `transfer.not_comparable`,
`transfer.cloud_unlabelled`.

### `gcp.list`
→ `{gcps, fit}`. Each GCP is `{id, name, xyz, epsg, orthometric, accuracy,
use, marks}`: `xyz` its surveyed coordinates in CRS `epsg` (null when not
surveyed; for a geographic CRS, x is the longitude), `use` `control` or
`check`, `marks` `{<survey>: {photos: [names], cloud}}` where it is marked.
`fit` is the last `align.fit` summary, or null.

### `gcp.create`
`{name?, xyz?, epsg?, orthometric?, accuracy?, use?}` → `{id, gcps, fit}`.

### `gcp.update`
`{id, name?, xyz?, epsg?, orthometric?, accuracy?, use?}` → `{gcps, fit}`.
`xyz` null clears the coordinates; coordinates need `epsg`. Problem:
`gcp.not_found`.

### `gcp.delete`
`{id}` → `{gcps, fit}`. Its Marks go too, in every Survey.

### `gcp.import_csv`
`{path, epsg, orthometric?, use?}` → `{added, updated, gcps, fit}`. Reads
`name, x, y, z[, accuracy]` lines (commas, semicolons or tabs; a header
line is skipped; with semicolons or tabs a decimal comma is fine). A name
already there takes the new coordinates. Problems: `gcp.no_file`,
`gcp.bad_csv` (`params.line`).

### `mark.set`
`{survey, gcp, name, uv}` places the GCP's Mark on photo `name` at pixel
`uv` (`[u, v]`, in the photo's pixels); `{survey, gcp, cloud?, index}`
places its cloud Mark on point `index` of the active Reconstruction's
cloud (as `cloud.pick` found it), kept in that Reconstruction's own
frame. Placing a Mark again moves it. → `{gcp}`. Problems:
`gcp.not_found`, `image.not_found`, `cloud.bad_indices`.

### `mark.delete`
`{survey, gcp, name?}` → `{gcp}`. Removes the Mark on photo `name`, or,
without `name`, the cloud Mark.

### `marks.photo`
`{survey, name}` → `{marks: [{gcp, uv}], hints: [{gcp, uv}]}`. The photo's
Marks, and where each other GCP falls in it: from where the active
Reconstruction locates it (its cloud Mark, else two or more photo Marks
triangulated) or, once the Survey is aligned, from its surveyed
coordinates.

### `marks.cloud`
`{survey}` → `{marks: [{gcp, xyz, from}]}`: where the active Reconstruction
locates each GCP, in the Project frame; `from` is `cloud` or `photos`.

### `cloud.pick`
`{survey, cloud?, view_projection, x, y, width, height, radius?: 8}` →
`{cloud, index, xyz}`: the point a click at CSS pixel `(x, y)` snaps to,
the front-most of the cloud's points within `radius` pixels (`index` and
`xyz` null when there is none). `view_projection` is as `cloud.select`'s.

### `align.fit`
`{reference?}` → `{surveys, residuals, located, frame_created}`. A long
request. Fits each Survey's active Reconstruction to the GCPs:

- Each Reconstruction locates each GCP (a cloud Mark, else its photo
  Marks triangulated). Surveyed control GCPs are targets in the Project
  frame. A Project with no frame takes one from them: their CRS minus a
  km offset when it is projected, else local ENU about them
  (`frame_created`).
- A Survey that sees three or more targets gets a similarity fitted to
  them. A GCP without coordinates becomes a target, once aligned Surveys
  see it, at the mean of where they put it, which can align further
  Surveys. With no coordinates at all, Survey `reference` (default: the
  first) keeps its frame.
- A Survey with fewer than three keeps its Alignment (EXIF GPS, say).
  Check GCPs never take part; they are measured.

`surveys` maps each Survey to `{aligned, how: "gcp"|"reference"|"kept",
n_gcps, rms}`; `residuals[gcp][survey]` is `[dx, dy, dz]` in metres from
its coordinates (or, for a GCP without them, from where the other
Surveys put it); `located[survey][gcp]` is `{from, photos, rms_px?}`.
The Alignment is stored with `source: "gcp"`. Problems: `align.no_gcps`,
`export.needs_geoid` (orthometric coordinates need the EGM96 grid),
`project.jobs_running`.

### `model.spec`
→ `{spec, training_size, problem, default_tile_size}`: the U-Net settings,
what the network sees of each Tile (`[n, n]`, the Tile size over the shrink
factor), why the settings can't build a network (or null), and the Tile
size the photos call for: the first power of two greater than a quarter of
the longest photo side in any Survey (null while no size is known). Until
the settings are first saved, `spec.tile_size` is that size. `spec` is `{tile_size,
shrink, overlap, min_labelled, channels, blocks, dropout, filter_size,
normals}`: Tiles of `tile_size` photo pixels on a grid whose neighbours
share `overlap` of a Tile, shrunk by `shrink` (1, 2, 4 or 8); training
skips Tiles less than `min_labelled` labelled. `normals` (experimental,
default false) gives the network the dense cloud's normals in each photo,
in the camera's frame, as three channels beside RGB: such a model trains
and predicts only photos with a camera in a Survey whose dense cloud has
normals.

### `model.set_spec`
`{spec}` → as `model.spec`.

### `models.list`
→ `{models}`: the saved models' manifests, newest first: `{name, created,
spec, n_classes, input_size, class_names, metrics, history, surveys,
best_epoch, monitor, tiled}`. `history` has each metric per epoch
(`accuracy` counts labelled pixels only, since 0.4.0); `surveys` the
Surveys it was trained on. The weights saved are those of epoch
`best_epoch` (0-based, in `history`), the one with the lowest `monitor`
(`val_loss`, or `loss` when nothing was held out), and `metrics` are that
epoch's; models saved before 0.4.0's release have neither field and kept
their last epoch. `tiled` is false for a model trained on whole photos (before
0.4.0): it can't predict or be continued.

### `models.delete`
`{name}` → `{models}`. Deletes a saved model, then lists the rest.
Problems: `models.not_found`, `models.in_use` (a `train` or `predict` Job
is outstanding).

### `crs.search`
`{query?: "", limit?: 50}` → `{results: [{epsg, name}]}`. Needs no Project.
Searches the EPSG projected CRS catalogue, in name order. Every word of
`query` must appear, case-insensitively, in `EPSG:<code> — <name>`, so
"SIRGAS 22S" finds "SIRGAS 2000 / UTM zone 22S" and "31983" finds it by
code.

### `crs.suggest`
→ `{epsg, name, origin_lla, exact}`. The UTM zone covering the site, in the
national datum where there is one (SIRGAS 2000 in Brazil), else WGS 84.
`origin_lla` is the ENU origin `[lat, lon, alt]`. An older Project that
stored the mean GPS position gets its exact origin recovered and saved.
`exact` is false when that wasn't possible because the photos are gone.
Problem: `crs.not_georeferenced`.

### `crs.frame`
→ `{short, full}`. The Project's coordinate frame, described for the status
bar (`short`, for example `EPSG:31983 (offset)`) and its tooltip (`full`).

### `export.settings`
→ `{frame, crs, export_mode, export_epsg, export_orthometric}`.

- `frame` is `local` (not georeferenced), `enu` (georeferenced) or `crs`
  (reprojected).
- `crs` is the reprojected frame `{epsg, name, orthometric, offset}`, or
  null. Stored coordinates are the CRS's minus `offset`.
- `export_mode` (`local`, `projected` or null), `export_epsg` and
  `export_orthometric` are an ENU Project's last export choice, kept in the
  manifest.

### `export.set_settings`
`{export_mode?, export_epsg?, export_orthometric?}` → as `export.settings`.
Params left out keep their values. `projected` needs `export_epsg`.
Problem: `export.not_georeferenced`.

### `export.cloud`
`{survey, path, format: "las"|"csv"|"ply", crs?, cloud?}` → `{path, cloud,
n_points, crs}`. A long request. Writes a cloud and its labels.

- `path` is absolute. `.<format>` is appended when it ends otherwise, and
  the result's `path` is the file written.
- `cloud` defaults to the dense cloud when there is one, else the sparse
  cloud.
- The coordinates follow the frame; a Survey that isn't aligned writes
  its own local frame. `local` writes the stored ones and
  `crs` the full CRS ones (stored plus offset); both ignore the `crs`
  param. For `enu`, `crs` is `{mode: "local"}` or `{mode: "projected",
  epsg, orthometric?}` and is kept as the export settings. Null reuses
  them.
- CSV and PLY get a `.prj` file beside them, and LAS carries the CRS
  itself. The result's `crs` names it, or is null for the local frame.

Problems: `cloud.none`, `export.no_folder`, `export.needs_geoid`
(`params.epsg`: sea-level heights need the EGM96 grid, so run the `geoid`
Job, then export again).

### `export.mesh`
`{survey, path, format: "ply"|"obj"|"stl", crs?}` → `{path, vertices, faces, crs}`.
A long request. Writes the mesh, with paths and coordinates as
`export.cloud`. Problems: `mesh.none`, `export.no_folder`,
`export.needs_geoid`.

### `ingest.conflicts`
`{survey, folder}` → `{conflicts}`. The sorted names of the photos in
`folder` that the Survey already holds. The window asks whether to skip or
overwrite them before it plans `ingest`. Problem: `job.ingest.no_folder`.

### `jobs.kinds`
→ `{kinds: [{kind, group, description}]}`

### `jobs.facts`
`{kind, survey?}` → what the kind's dialog shows before it is planned.
`survey` is needed for `sfm`, `mvs` and `mesh`; `train` counts every
Survey. A long
request: it reads EXIF and measures the dense cloud.

| `kind` | Result |
|---|---|
| `sfm` | `{gps_images, camera_models, matchers, gpu}`. `gps_images` counts the photos with GPS EXIF among the first 25, stopping at 3 (enough to offer georeferencing). `gpu`: a CUDA colmap.exe or a CUDA pycolmap is there. |
| `mvs` | `{colmap}`: dense MVS can run (what `job.mvs.no_colmap` checks). |
| `mesh` | `{delaunay, suggested_depth, ram_gb}`. `delaunay`: colmap.exe is there and the MVS workspace still has `fused.ply` and its visibility file. `suggested_depth` is null without a dense cloud. `ram_gb` is this machine's RAM, or null. |
| `train` | `{user_images, auto_images, ml_images}`: the photos with a user mask that holds labels, the solved photos marked `auto`, and the photos marked `ml`. |

Any other `kind` is invalid params.

### `jobs.plan`
`{kind, options?}` → `{plan, kind, group, gpu, module, args, log, stop_file,
project, stage}`. Checks that a Job can run, saves what it will read, and
says how to run it. The Tauri core then runs `<Environment python> -m
<module> <args…>`, with the Environment's own interpreter and paths as the
Engine sees them. While it runs, the core:

- takes the GPU Lease first when `gpu` is true;
- writes the Job's output to `log`;
- reads its `PROGRESS <fraction> <message>` lines;
- keeps its `RESULT…` lines for `jobs.finish`;
- cancels it by creating `stop_file` when there is one (training stops
  after its epoch and still saves the model), and kills it otherwise.

`stage` is `{in, out}` for the kinds with heavy file I/O (`sfm`, `mvs`,
`mesh`) and null for the others. `in` and `out` are lists of directories,
relative to the Project root with `/` separators: what the Job reads and
what it writes. In the WSL Environment the core copies the `in` dirs to
WSL's own disk, rewrites the Project-root prefix of `args` to the copy, runs
the Job, and copies the `out` dirs back before `jobs.finish`. Every path in
`args` under the Project root lies inside an `in` or an `out` dir. `log`
stays where it is: the core writes it.

With `S` = `surveys/<survey>`: each step writes a new product folder
(`<slot>-<n>`, the next free number). Finishing it fills the active
Reconstruction's slot when that is empty; otherwise it makes a child of
the active Reconstruction that shares the earlier steps' products (and
their labels and Alignment) and makes it active. Nothing is overwritten.

| Kind | `in` | `out` |
|---|---|---|
| `sfm` | `S/images` | `S/products/sfm-<new>` |
| `mvs` | `S/images`, the active `S/products/sfm-n` | `S/products/mvs-<new>` |
| `mesh` | the active `S/products/mvs-n` | `S/products/mesh-<new>` |

Until `jobs.finish`, a second plan of the same group is refused (Problem:
`job.busy`), and so is opening or closing a Project
(`project.jobs_running`). The kinds, their options, and the Problems
`jobs.plan` raises (or, marked †, the `error` of `jobs.finish`). The
kinds marked ‡ work on one Survey and need the option `survey`:

| Kind | Group | GPU | Options | Problems |
|---|---|---|---|---|
| `ingest` ‡ | reconstruction | no | `folder`, `on_conflict` (`skip` or `overwrite`) | `job.ingest.no_folder` |
| `sfm` ‡ | reconstruction | when `gpu` | `matcher`, `gpu`, `single_camera`, `camera_model`, `max_image_size` | `job.sfm.no_images`, `job.sfm.no_cloud` † |
| `georef` ‡ | reconstruction | no | `max_error` | `job.no_reconstruction`, `job.georef.no_fit` †, `export.needs_geoid` † |
| `reproject` | reconstruction | no | `epsg`, `orthometric` | `job.no_reconstruction`, `job.reproject.not_georeferenced`, `job.reproject.no_fit` † |
| `mvs` ‡ | reconstruction | yes | `max_image_size`, `quality` (`standard` or `draft`) | `job.no_reconstruction`, `job.mvs.no_colmap` |
| `mesh` ‡ | reconstruction | no | `method` (`poisson` or `delaunay`), `depth`, `trim`, `point_weight` | `job.mesh.no_dense`, `job.mesh.no_delaunay` |
| `train` | ml | yes | `name`, `steps` (batches, default 50000: the epochs are as many as that takes, rounded up), `patience` (stop after this many epochs without a better validation loss, default 10; 0 never stops early), `batch_size`, `val_fraction`, `augment_rolls`, `include_auto`, `resume`, `surveys` (default: every Survey; their photos are pooled, cut into Tiles, and validated on whole photos held out) | `job.train.no_classes`, `job.train.no_images`, `job.train.bad_spec`, `job.train.class_mismatch`, `job.train.legacy_model`, `job.train.no_normals` (`params.surveys`: those without a dense cloud), `survey.not_found` |
| `predict` | ml | yes | `model`, `surveys` (default: every Survey; the photos labelled neither by the user nor from the cloud; its result is `{predicted: [{survey, name, uncertainty}]}`, `uncertainty` the mean) | `job.predict.no_model`, `job.predict.class_mismatch`, `job.predict.nothing`, `job.predict.legacy_model`, `job.predict.no_normals`, `survey.not_found` |
| `geoid` | download | no | `epsg` | |
| `colmap` | download | no | `variant` (`cuda` or `cpu`) | `job.colmap.unsupported` |

`job.mvs.no_colmap`: there is neither a CUDA colmap.exe nor a CUDA pycolmap
(native Windows without COLMAP). `job.mesh.no_delaunay`: `delaunay` needs
colmap.exe, so the WSL Environment offers Poisson only.
`georef` fits the Survey's cameras to its EXIF GPS and stores the fit as
the active Reconstruction's Alignment, replacing an earlier one; no product
is rewritten (ADR-0005). The first one gives the Project its frame (ENU
about the first GPS photo); later ones fit into that frame's origin, and on
into its CRS when it is reprojected. `reproject` moves the Project frame
into a projected CRS by composing the fitted move into every Alignment;
it, too, rewrites no file. A new `sfm` leaves its Reconstruction without
an Alignment (`georeferencing_cleared` in its result says one was
dropped). `job.train.class_mismatch` (params
`model`, `model_classes`, `project_classes`): `resume` names a saved model
trained with a different number of classes.

The `download` Jobs fetch into the user's own folders, not the Project:

- `geoid` downloads the EGM96 grid (~3 MB) that sea-level heights in
  `epsg` need. Its result is `{epsg}`.
- `colmap` downloads the official COLMAP Windows release into
  `~/.cloudlabeller/colmap`, where the Jobs look for colmap.exe. `cuda` is
  ~307 MB and `cpu` ~114 MB. Its result is `{path, variant}`. It cancels
  through its stop file and leaves no partial install.
  `job.colmap.unsupported`: the Environment isn't native Windows (WSL uses
  pycolmap).

The Engine doesn't chain stages (SfM → georeferencing → MVS → mesh). The
window does.

### `jobs.finish`
`{plan, exit, results?, cancelled?}` → `{plan, kind, ok, result?, stopped?,
error?}`. The core reports that a planned Job has ended:

- `exit` is its exit code, or null if it never ran or was lost.
- `results` maps each `RESULT…` line's first word to the rest of the line,
  for example `{"RESULT": "{\"n_gps\": 12}"}`. The core reads at most
  64 KiB of a line at a time, so a Job prints a long JSON outcome in
  numbered pieces, `RESULT.0`, `RESULT.1`, … (each under 16 KiB, none
  starting or ending with a space), which the Engine joins in order. An
  outcome that isn't JSON fails the Job with `job.bad_result`.
- `cancelled` says the core cancelled it.

When the Job ended well, the Engine updates the open Project, sends
`project.changed`, and answers `ok: true` with the kind's summary. A
training run stopped through its stop file also ends well, with
`stopped: true`. Otherwise `error` is a Problem's data: `job.failed`
(`params.exit`), `job.cancelled`, `job.lost`, `project.closed`, or the
kind's own. Problem: `job.unknown_plan`, for example after the Engine
restarted. The core then reopens the Project to pick up what the Job wrote.

### `jobs.list`
→ `{jobs}`: the plans not yet finished, as `jobs.plan` describes them.
