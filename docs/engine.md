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
{"path": ".arrays/mask/IMG_0001.JPG.bin", "dtype": "int32", "shape": [3000, 4000]}
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

## Notifications (Engine → window)

| Method | Params |
|---|---|
| `engine.ready` | Same as `engine.hello`. |
| `engine.log` | `{level, message}`: an internal failure the user didn't ask about. |
| `project.changed` | `{parts, images?}`. `parts` are drawn from `schema`, `images`, `status`, `sparse`, `dense`, `mesh`, `labels` and `settings`. `images` names the photos whose mask or status changed. |
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
`{root}` → the Project state. Closes the open Project first. Problems:
`project.not_found`, `project.jobs_running`.

### `project.close`
→ `{}`. Saves if dirty. Problem: `project.jobs_running`.

### `project.save`
→ `{}`. Writes the manifest and every label array.

### `project.state`
→ `{root, schema, images: [{name, status, solved, user_mask}], reconstruction,
sparse, dense, mesh, georeferenced, source_resolution, can_undo, can_redo,
dirty}`.

- `status` is `user`, `auto`, `ml` or `none`.
- `reconstruction` is `none`, `outdated` or `current`.
- `sparse` and `dense` are point counts, or null. `mesh` is
  `{vertices, faces}`, or null.

### `project.summary`
→ `{rows: [[label, value], …]}`: the Project Info rows, as plain text.

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

### `labels.undo`
→ `{changed: {modality: "cloud"|"image", image} | null}`

### `labels.redo`
→ `{changed: {modality: "cloud"|"image", image} | null}`

### `image.mask`
`{name}` → `{source, mask}`. The overlay for a photo:

- `source` is `user` (the user's own mask), `ml` (the U-Net's prediction,
  at the photo's size) or `auto` (projected from the labelled cloud), or
  null when there is nothing to show.
- `mask` is an `int32` Array file of shape `[H, W]`, or null.

Problem: `image.not_found`.

### `image.save_mask`
`{name, mask}` → `{changed_pixels, status}`. Stores the mask the user
painted, as one undoable edit, and marks the photo `user`. The window sends
the whole mask, seeded from the overlay it showed. `mask` is an inbound
`int32` Array file of the photo's size. Problems: `image.not_found`,
`image.mask_shape`, `image.mask_values`.

### `cloud.points`
`{cloud?: "sparse"|"dense"}` → `{cloud, n_points, xyz, rgb}`. `xyz` is a
`float32` Array file `[N, 3]`. `rgb` is a `uint8` one `[N, 3]`, or null.
Problem: `cloud.none`.

### `cloud.labels`
`{cloud?}` → `{cloud, labels}`. `labels` is an `int32` Array file `[N]`.

### `cloud.paint`
`{cloud?, indices, class_id}` → `{painted}`. Gives the chosen points one
class (`-1` erases), as one undoable edit. `indices` is an inbound `int32`
or `uint32` Array file. Painting the dense cloud also updates the mesh's
labels. Problems: `schema.no_class`, `cloud.bad_indices`.

### `mesh.geometry`
→ `{vertices, faces, colors}`: Array files `float32 [V, 3]`,
`int32 [F, 3]` and `uint8 [V, 3]` (`colors` may be null). Problem: `mesh.none`.

### `mesh.labels`
→ `{labels}`: `int32 [V]`, nearest-point labels from the dense cloud.

### `transfer.images_to_cloud`
`{include_auto?: true}` → `{images, sparse, dense?}`. A long request.
Votes the photos' labels onto the sparse cloud and the dense one, with
user-drawn masks outranking automatic ones. `include_auto` adds the
U-Net's predictions and the cloud-projected photos that cover the surface.
The votes are merged into the existing labels, so points no photo reached
keep theirs. Returns the number of points updated. Problems:
`transfer.no_cloud`, `transfer.no_labelled_images`.

### `transfer.cloud_to_images`
→ `{cloud, auto_labelled}`. A long request. Marks `auto` every solved photo
the user hasn't drawn on that the labelled cloud reaches. Their overlays
come from `image.mask`. Problems: `transfer.no_cloud`,
`transfer.cloud_unlabelled`.

### `transfer.propagate`
`{name}` → `{cloud, points, auto_labelled}`. A long request. Spreads one
photo's labels through the cloud to the photos that overlap it. A neighbour
needs 500 labelled pixels to be marked `auto`. Problems:
`transfer.image_unlabelled`, `transfer.image_unsolved`, `transfer.no_cloud`.

### `model.spec`
→ `{spec, training_size, problem}`: the U-Net settings, the size it trains
at for this Project's photos, and why that size won't work (or null).

### `model.set_spec`
`{spec}` → as `model.spec`.

### `models.list`
→ `{models}`: the saved models' manifests, newest first.

### `jobs.kinds`
→ `{kinds: [{kind, group, description}]}`

### `jobs.plan`
`{kind, options?}` → `{plan, kind, group, gpu, module, args, log, stop_file,
project}`. Checks that a Job can run, saves what it will read, and says how
to run it. The Tauri core then runs `<Environment python> -m <module>
<args…>`, with the Environment's own interpreter and paths as the Engine
sees them. While it runs, the core:

- takes the GPU Lease first when `gpu` is true;
- writes the Job's output to `log`;
- reads its `PROGRESS <fraction> <message>` lines;
- keeps its `RESULT…` lines for `jobs.finish`;
- cancels it by creating `stop_file` when there is one (training stops
  after its epoch and still saves the model), and kills it otherwise.

Until `jobs.finish`, a second plan of the same group is refused (Problem:
`job.busy`), and so is opening or closing a Project
(`project.jobs_running`). The kinds, their options, and the Problems
`jobs.plan` raises (or, marked †, the `error` of `jobs.finish`):

| Kind | Group | GPU | Options | Problems |
|---|---|---|---|---|
| `ingest` | reconstruction | no | `folder`, `on_conflict` (`skip` or `overwrite`) | `job.ingest.no_folder` |
| `sfm` | reconstruction | when `gpu` | `matcher`, `gpu`, `single_camera`, `camera_model`, `max_image_size` | `job.sfm.no_images`, `job.sfm.no_cloud` † |
| `georef` | reconstruction | no | `max_error` | `job.no_reconstruction` |
| `reproject` | reconstruction | no | `epsg`, `orthometric` | `job.no_reconstruction`, `job.reproject.not_georeferenced` |
| `mvs` | reconstruction | yes | `max_image_size`, `quality` (`standard` or `draft`) | `job.no_reconstruction`, `job.mvs.no_colmap` |
| `mesh` | reconstruction | no | `method` (`poisson` or `delaunay`), `depth`, `trim`, `point_weight` | `job.mesh.no_dense` |
| `train` | ml | yes | `name`, `epochs`, `batch_size`, `val_fraction`, `augment_rolls`, `include_auto`, `resume` | `job.train.no_classes`, `job.train.no_images`, `job.train.bad_spec` |
| `predict` | ml | yes | `model` | `job.predict.no_model`, `job.predict.class_mismatch`, `job.predict.nothing` |

The Engine doesn't chain stages (SfM → georeferencing → MVS → mesh). The
window does.

### `jobs.finish`
`{plan, exit, results?, cancelled?}` → `{plan, kind, ok, result?, stopped?,
error?}`. The core reports that a planned Job has ended:

- `exit` is its exit code, or null if it never ran or was lost.
- `results` maps each `RESULT…` line's first word to the rest of the line,
  for example `{"RESULT": "{\"predicted\": [\"a.jpg\"]}"}`.
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
