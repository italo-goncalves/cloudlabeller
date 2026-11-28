// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

// The Engine's methods as the window calls them, typed after docs/engine.md.
// Every screen goes through these, not through `engineCall` with a string.

import { type ArrayRef, engineCall, readArray, writeArray } from "../desktop/bridge";
import { narrow, widen } from "./labels";

export type Status = "user" | "auto" | "ml" | "none";
export type Reconstruction = "none" | "outdated" | "current";
export type CloudName = "sparse" | "dense";

export interface SchemaClass {
  id: number;
  name: string;
  /** "#rrggbb" */
  color: string;
  hotkey: string | null;
}

export interface ImageRow {
  name: string;
  status: Status;
  solved: boolean;
  user_mask: boolean;
  /** The U-Net prediction's mean Uncertainty (0 sure .. 1 unsure), for `ml` photos. */
  uncertainty?: number | null;
}

/** One Survey as `project.state` reports it. */
export interface SurveyState {
  id: string;
  name: string;
  timestamp: string | null;
  operator: string;
  camera: string;
  notes: string;
  images: ImageRow[];
  reconstruction: Reconstruction;
  /** The active Reconstruction's id, null before SfM. */
  active: string | null;
  /** How many Reconstructions the Survey has. */
  reconstructions: number;
  /** Whether the active Reconstruction is aligned into the Project frame. */
  aligned: boolean;
  sparse: number | null;
  dense: number | null;
  mesh: { vertices: number; faces: number } | null;
  /** The dense cloud has normals (to colour by them, or train on them). */
  normals: boolean;
  source_resolution: [number, number] | null;
  can_undo: boolean;
  can_redo: boolean;
}

/** What `project.state` answers. */
export interface RawProjectState {
  root: string;
  schema: SchemaClass[];
  surveys: SurveyState[];
  /** The Project frame, null when it has none. */
  georeferenced: Record<string, unknown> | null;
  dirty: boolean;
}

/** The Project as the window shows it: the current Survey's fields at the
 * top level, next to every Survey. */
export interface ProjectState {
  root: string;
  schema: SchemaClass[];
  surveys: SurveyState[];
  /** The current Survey's id. */
  survey: string;
  images: ImageRow[];
  reconstruction: Reconstruction;
  sparse: number | null;
  dense: number | null;
  mesh: { vertices: number; faces: number } | null;
  /** The current Survey's dense cloud has normals. */
  normals: boolean;
  /** The Project has a frame and the current Survey is aligned into it. */
  georeferenced: boolean;
  source_resolution: [number, number] | null;
  can_undo: boolean;
  can_redo: boolean;
  dirty: boolean;
}

// -- the current Survey ---------------------------------------------------------
// The window shows one Survey at a time. Every Survey-scoped call below goes to
// the current one, which `view` sets from the Project's state.

let current: string | null = null;

/** The current Survey's id. */
export function currentSurvey(): string {
  if (current === null) throw new Error("no Survey is current");
  return current;
}

/** The Project's state seen from Survey `survey` (default: the current one,
 * else the first); makes it current. */
export function view(raw: RawProjectState, survey?: string): ProjectState {
  const want = survey ?? current;
  const s = raw.surveys.find((x) => x.id === want) ?? raw.surveys[0];
  if (!s) throw new Error("the Project has no Survey");
  current = s.id;
  return {
    root: raw.root,
    schema: raw.schema,
    surveys: raw.surveys,
    survey: s.id,
    images: s.images,
    reconstruction: s.reconstruction,
    sparse: s.sparse,
    dense: s.dense,
    mesh: s.mesh,
    normals: Boolean(s.normals),
    georeferenced: Boolean(raw.georeferenced) && s.aligned,
    source_resolution: s.source_resolution,
    can_undo: s.can_undo,
    can_redo: s.can_redo,
    dirty: raw.dirty,
  };
}

/** Make Survey `id` the current one (the next `state` shows it). */
export function setSurvey(id: string) {
  current = id;
}

/** The current Survey's id, or null when no Project is open. */
export function currentSurveyOrNull(): string | null {
  return current;
}

/** Forget the current Survey (the Project closed). */
export function clearSurvey() {
  current = null;
}

const sv = () => ({ survey: currentSurvey() });

export type Changed = { modality: "cloud" | "image"; image: string | null } | null;

// -- Array files ----------------------------------------------------------------

/** An Array file's bytes as the typed array its dtype names. */
export async function load(ref: ArrayRef): Promise<ArrayBufferView & ArrayLike<number>> {
  const buf = await readArray(ref);
  switch (ref.dtype) {
    case "uint8":
      return new Uint8Array(buf);
    case "int8":
      return new Int8Array(buf);
    case "uint16":
      return new Uint16Array(buf);
    case "int16":
      return new Int16Array(buf);
    case "uint32":
      return new Uint32Array(buf);
    case "int32":
      return new Int32Array(buf);
    case "float32":
      return new Float32Array(buf);
    case "float64":
      return new Float64Array(buf);
  }
}

let inbound = 0;
/** A fresh name for an inbound Array file. */
const fresh = (what: string) => `${what}-${Date.now()}-${inbound++}`;

// -- the Project --------------------------------------------------------------

export const state = async () => view(await engineCall<RawProjectState>("project.state"));
export const summary = () => engineCall<{ rows: [string, string][] }>("project.summary");
export const hello = () =>
  engineCall<{ protocol: number; version: string; methods: string[] }>("engine.hello");

export const schemaAdd = (name?: string, color?: string) =>
  engineCall<{ schema: SchemaClass[] }>("schema.add", {
    ...(name ? { name } : {}),
    ...(color ? { color } : {}),
  });
export const schemaRename = (id: number, name: string) =>
  engineCall<{ schema: SchemaClass[] }>("schema.rename", { id, name });
export const schemaSetColor = (id: number, color: string) =>
  engineCall<{ schema: SchemaClass[] }>("schema.set_color", { id, color });
export const schemaDelete = (id: number) =>
  engineCall<{ schema: SchemaClass[] }>("schema.delete", { id });

export const undo = () => engineCall<{ changed: Changed }>("labels.undo", sv());
export const redo = () => engineCall<{ changed: Changed }>("labels.redo", sv());

// -- Surveys and Reconstructions ------------------------------------------------

export interface SurveyMeta {
  name?: string;
  /** `YYYY-MM-DDTHH:MM:SS`; empty clears it. */
  timestamp?: string;
  operator?: string;
  camera?: string;
  notes?: string;
}

export const surveyCreate = (meta: SurveyMeta) =>
  engineCall<SurveyState>("survey.create", { ...meta });
export const surveyUpdate = (survey: string, meta: SurveyMeta) =>
  engineCall<SurveyState>("survey.update", { survey, ...meta });
export const surveyDelete = (survey: string) =>
  engineCall<{ freed_bytes: number; surveys: string[] }>("survey.delete", { survey });

export interface ReconstructionRow {
  id: string;
  name: string;
  parent: string | null;
  products: { sfm: string | null; mvs: string | null; mesh: string | null };
  aligned: boolean;
  created: string;
  active: boolean;
  /** What deleting it would free on disk. */
  freed_bytes: number;
}
export interface Reconstructions {
  reconstructions: ReconstructionRow[];
  active: string | null;
}
export const reconstructions = () => engineCall<Reconstructions>("reconstruction.list", sv());
export const reconstructionActivate = (id: string) =>
  engineCall<Reconstructions>("reconstruction.activate", { ...sv(), reconstruction: id });
export const reconstructionRename = (id: string, name: string) =>
  engineCall<Reconstructions>("reconstruction.rename", { ...sv(), reconstruction: id, name });
export const reconstructionDelete = (id: string) =>
  engineCall<Reconstructions & { freed_bytes: number }>("reconstruction.delete", {
    ...sv(),
    reconstruction: id,
  });
export const transferReconstruction = (from: string, ticket: string) =>
  engineCall<{ sparse: number; dense?: number }>(
    "transfer.reconstruction",
    { ...sv(), from },
    ticket,
  );

// -- GCPs and Marks ------------------------------------------------------------

export type GcpUse = "control" | "check";

export interface Gcp {
  id: string;
  name: string;
  /** Surveyed coordinates in `epsg` (x = longitude for a geographic CRS), or null. */
  xyz: [number, number, number] | null;
  epsg: number | null;
  orthometric: boolean;
  accuracy: number | null;
  use: GcpUse;
  /** Where it is marked: per Survey, the photos and whether the cloud. */
  marks: Record<string, { photos: string[]; cloud: boolean }>;
}

export interface GcpFit {
  surveys: Record<
    string,
    { aligned: boolean; how: "gcp" | "reference" | "kept"; n_gcps: number; rms: number | null }
  >;
  /** Per GCP, per Survey: [dx, dy, dz] in metres. */
  residuals: Record<string, Record<string, [number, number, number]>>;
  located: Record<
    string,
    Record<string, { from: "cloud" | "photos"; photos: number; rms_px?: number }>
  >;
  frame_created: boolean;
}

export interface Gcps {
  gcps: Gcp[];
  fit: GcpFit | null;
}

export type GcpFields = Partial<
  Pick<Gcp, "name" | "xyz" | "epsg" | "orthometric" | "accuracy" | "use">
>;

export const gcps = () => engineCall<Gcps>("gcp.list");
export const gcpCreate = (fields: GcpFields = {}) =>
  engineCall<Gcps & { id: string }>("gcp.create", { ...fields });
export const gcpUpdate = (id: string, fields: GcpFields) =>
  engineCall<Gcps>("gcp.update", { id, ...fields });
export const gcpDelete = (id: string) => engineCall<Gcps>("gcp.delete", { id });
export const gcpImportCsv = (path: string, epsg: number, orthometric: boolean, use: GcpUse) =>
  engineCall<Gcps & { added: number; updated: number }>("gcp.import_csv", {
    path,
    epsg,
    orthometric,
    use,
  });

export const markPhoto = (gcp: string, name: string, uv: [number, number]) =>
  engineCall<{ gcp: Gcp }>("mark.set", { ...sv(), gcp, name, uv });
export const markCloud = (gcp: string, cloud: CloudName, index: number) =>
  engineCall<{ gcp: Gcp }>("mark.set", { ...sv(), gcp, cloud, index });
export const markDelete = (gcp: string, name?: string) =>
  engineCall<{ gcp: Gcp }>("mark.delete", { ...sv(), gcp, ...(name ? { name } : {}) });

export interface PhotoMarks {
  marks: { gcp: string; uv: [number, number] }[];
  hints: { gcp: string; uv: [number, number] }[];
}
export const marksPhoto = (name: string) =>
  engineCall<PhotoMarks>("marks.photo", { ...sv(), name });
export const marksCloud = () =>
  engineCall<{ marks: { gcp: string; xyz: [number, number, number]; from: "cloud" | "photos" }[] }>(
    "marks.cloud",
    sv(),
  );

/** The cloud point a click snaps to (null: none near it). */
export const cloudPick = (
  cloud: CloudName,
  viewProjection: ArrayLike<number>,
  x: number,
  y: number,
  width: number,
  height: number,
) =>
  engineCall<{ cloud: CloudName; index: number | null; xyz: [number, number, number] | null }>(
    "cloud.pick",
    { ...sv(), cloud, view_projection: Array.from(viewProjection), x, y, width, height },
  );

export const alignFit = (ticket: string, reference?: string) =>
  engineCall<GcpFit>("align.fit", reference ? { reference } : {}, ticket);

/** Bytes as the window says them: "1.2 GB". */
export function sizeText(bytes: number): string {
  const units = ["B", "KB", "MB", "GB", "TB"];
  let v = bytes;
  let i = 0;
  while (v >= 1000 && i < units.length - 1) {
    v /= 1000;
    i++;
  }
  return `${i === 0 ? v : v.toFixed(v < 10 ? 1 : 0)} ${units[i]}`;
}

// -- photos -------------------------------------------------------------------

export interface ImageInfo {
  path: string;
  width: number;
  height: number;
}
export const imageInfo = (name: string) => engineCall<ImageInfo>("image.info", { ...sv(), name });

export interface Mask {
  source: "user" | "ml" | "auto" | null;
  /** H × W class ids (−1 unlabelled), row-major; null when there is none. */
  mask: Int32Array | null;
  width: number;
  height: number;
}

export async function imageMask(name: string): Promise<Mask> {
  const r = await engineCall<{ source: Mask["source"]; mask: ArrayRef | null }>("image.mask", {
    ...sv(),
    name,
  });
  if (!r.mask) return { source: r.source, mask: null, width: 0, height: 0 };
  const [h, w] = r.mask.shape as [number, number];
  return { source: r.source, mask: widen(await load(r.mask)), width: w, height: h };
}

/** Store a photo's mask; `survey` is the one it was painted in (default:
 * the current one), so a save queued before a switch lands where it belongs.
 * The Engine keeps only the pixels that differ from the overlay it showed:
 * the user's strokes. */
export async function saveMask(
  name: string,
  mask: Int32Array,
  width: number,
  height: number,
  survey: string = currentSurvey(),
) {
  const { data, dtype } = narrow(mask);
  const ref = await writeArray(fresh("mask"), data, dtype, [height, width]);
  return engineCall<{ changed_pixels: number; status: Status }>("image.save_mask", {
    survey,
    name,
    mask: ref,
  });
}

/** How unsure the prediction is per pixel (0..1), at the prediction's size; null: none. */
/** The photo coloured by the dense cloud's normals (false colour, Project
 * frame), at its size over `step`; null when it has none. */
export async function imageNormals(
  name: string,
): Promise<{ rgb: Uint8Array; width: number; height: number } | null> {
  const r = await engineCall<{ normals: ArrayRef | null; step: number | null }>("image.normals", {
    ...sv(),
    name,
  });
  if (!r.normals) return null;
  const [h, w] = r.normals.shape as [number, number, number];
  return { rgb: (await load(r.normals)) as Uint8Array, width: w, height: h };
}

export async function imageUncertainty(
  name: string,
): Promise<{ values: Float32Array; width: number; height: number; mean: number } | null> {
  const r = await engineCall<{ uncertainty: ArrayRef | null; mean: number | null }>(
    "image.uncertainty",
    { ...sv(), name },
  );
  if (!r.uncertainty) return null;
  const [h, w] = r.uncertainty.shape as [number, number];
  return {
    values: (await load(r.uncertainty)) as Float32Array,
    width: w,
    height: h,
    mean: r.mean ?? 0,
  };
}

// -- clouds and the mesh ------------------------------------------------------

export interface CloudView {
  cloud: CloudName;
  /** Points in the whole cloud. */
  n_points: number;
  xyz: Float32Array;
  rgb: Uint8Array | null;
  /** The full-cloud index of each point shown; null when every point is shown, in order. */
  index: Int32Array | null;
}

/** The cloud thinned to at most `maxPoints` (0: all of it). */
/** What the points are coloured by: the photos, or the normals as a false colour. */
export type CloudColour = "photo" | "normals";

export async function cloudView(
  cloud: CloudName,
  maxPoints: number,
  colour: CloudColour = "photo",
): Promise<CloudView> {
  const r = await engineCall<{
    cloud: CloudName;
    n_points: number;
    xyz: ArrayRef;
    rgb: ArrayRef | null;
    index: ArrayRef | null;
  }>("cloud.view", { ...sv(), cloud, max_points: maxPoints, colour });
  return {
    cloud: r.cloud,
    n_points: r.n_points,
    xyz: (await load(r.xyz)) as Float32Array,
    rgb: r.rgb ? ((await load(r.rgb)) as Uint8Array) : null,
    index: r.index ? ((await load(r.index)) as Int32Array) : null,
  };
}

/** Every point's class (−1 unlabelled), the whole cloud. */
export async function cloudLabels(cloud: CloudName): Promise<Int32Array> {
  const r = await engineCall<{ labels: ArrayRef }>("cloud.labels", { ...sv(), cloud });
  return widen(await load(r.labels));
}

export async function cloudPaint(cloud: CloudName, indices: Uint32Array, classId: number) {
  const ref = await writeArray(fresh("paint"), indices, "uint32", [indices.length]);
  return engineCall<{ painted: number }>("cloud.paint", {
    ...sv(),
    cloud,
    indices: ref,
    class_id: classId,
  });
}

/** The points whose projection is inside a screen loop, through the whole cloud. */
export async function cloudSelect(
  cloud: CloudName,
  viewProjection: ArrayLike<number>,
  polygon: [number, number][],
  width: number,
  height: number,
): Promise<Uint32Array> {
  const r = await engineCall<{ count: number; indices: ArrayRef }>("cloud.select", {
    ...sv(),
    cloud,
    view_projection: Array.from(viewProjection),
    polygon,
    width,
    height,
  });
  return r.count > 0 ? ((await load(r.indices)) as Uint32Array) : new Uint32Array(0);
}

export async function cloudDelete(cloud: CloudName, indices: Uint32Array) {
  const ref = await writeArray(fresh("delete"), indices, "uint32", [indices.length]);
  return engineCall<{ removed: number; n_points: number }>("cloud.delete", {
    ...sv(),
    cloud,
    indices: ref,
  });
}

export const cloudStats = () =>
  engineCall<{ n_points: number; matched: number; views: ArrayRef; errors: ArrayRef }>(
    "cloud.stats",
    sv(),
  );

export async function cleanPreview(minImages: number, maxError: number) {
  const r = await engineCall<{ keep: number; total: number; removed: ArrayRef }>(
    "cloud.clean_preview",
    { ...sv(), min_images: minImages, max_error: maxError },
  );
  return {
    keep: r.keep,
    total: r.total,
    removed: r.total - r.keep > 0 ? ((await load(r.removed)) as Uint32Array) : new Uint32Array(0),
  };
}

export const clean = (minImages: number, maxError: number) =>
  engineCall<{ removed: number; n_points: number }>("cloud.clean", {
    ...sv(),
    min_images: minImages,
    max_error: maxError,
  });

export interface MeshData {
  vertices: Float32Array;
  faces: Uint32Array;
  colors: Uint8Array | null;
}

export async function meshGeometry(): Promise<MeshData> {
  const r = await engineCall<{ vertices: ArrayRef; faces: ArrayRef; colors: ArrayRef | null }>(
    "mesh.geometry",
    sv(),
  );
  const faces = (await load(r.faces)) as Int32Array;
  return {
    vertices: (await load(r.vertices)) as Float32Array,
    faces: new Uint32Array(faces.buffer, faces.byteOffset, faces.length),
    colors: r.colors ? ((await load(r.colors)) as Uint8Array) : null,
  };
}

export async function meshLabels(): Promise<Int32Array> {
  const r = await engineCall<{ labels: ArrayRef }>("mesh.labels", sv());
  return widen(await load(r.labels));
}

// -- cameras ------------------------------------------------------------------

export interface Camera {
  name: string;
  width: number;
  height: number;
  /** 3×3 intrinsics, as rows. */
  K: number[][];
  /** World to camera rotation, as rows (COLMAP's: a point projects as K (R X + t)). */
  R: number[][];
  t: [number, number, number];
  /** The camera centre in the scene's frame. */
  position: [number, number, number];
  /** The camera centre in the Project's CRS, for the Dataset pane; null when it has none. */
  position_crs: [number, number, number] | null;
}

export const cameras = () =>
  engineCall<{ cameras: Camera[]; up: [number, number, number] | null }>("cameras.list", sv());

// -- transfers ----------------------------------------------------------------

export const imagesToCloud = (includeAuto: boolean, includePredictions: boolean, ticket: string) =>
  engineCall<{ images: number; sparse: number; dense?: number }>(
    "transfer.images_to_cloud",
    { ...sv(), include_auto: includeAuto, include_predictions: includePredictions },
    ticket,
  );
export const cloudToImages = (ticket: string) =>
  engineCall<{ cloud: CloudName; auto_labelled: number }>("transfer.cloud_to_images", sv(), ticket);
export const propagate = (name: string, ticket: string) =>
  engineCall<{ cloud: CloudName; points: number; auto_labelled: number }>(
    "transfer.propagate",
    { ...sv(), name },
    ticket,
  );

// -- the model ----------------------------------------------------------------

export interface ModelSpec {
  /** Tile side, in photo pixels. */
  tile_size: number;
  /** The Tile is shrunk by this before the network (1, 2, 4 or 8). */
  shrink: number;
  /** Of a Tile, between neighbours on the grid. */
  overlap: number;
  /** A training Tile needs at least this fraction labelled. */
  min_labelled: number;
  /** Channels in the first U-Net layer. */
  channels: number;
  blocks: number;
  dropout: number;
  filter_size: number;
  /** Experimental: the dense cloud's normals as three channels beside RGB. */
  normals?: boolean;
  [k: string]: unknown;
}

export const modelSpec = () =>
  engineCall<{
    spec: ModelSpec;
    training_size: [number, number] | null;
    problem: string | null;
    default_tile_size: number | null;
  }>("model.spec");
export const setModelSpec = (spec: ModelSpec) =>
  engineCall<{ spec: ModelSpec; training_size: [number, number] | null; problem: string | null }>(
    "model.set_spec",
    { spec },
  );

export type ModelManifest = Record<string, unknown> & {
  name: string;
  spec?: Partial<ModelSpec>;
  /** Trained on Tiles (0.4.0); older models must be retrained. */
  tiled: boolean;
  surveys?: string[];
};
export const models = () => engineCall<{ models: ModelManifest[] }>("models.list");
export const deleteModel = (name: string) =>
  engineCall<{ deleted: string }>("models.delete", { name });

// -- facts for the dialogs ------------------------------------------------------

export interface Facts {
  gps_images?: number;
  camera_models?: string[];
  matchers?: string[];
  gpu?: boolean;
  colmap?: boolean;
  delaunay?: boolean;
  suggested_depth?: number;
  ram_gb?: number;
  user_images?: number;
  auto_images?: number;
  ml_images?: number;
  [k: string]: unknown;
}

export const facts = (kind: string) => engineCall<Facts>("jobs.facts", { ...sv(), kind });
export const meshEstimate = (method: string, depth: number) =>
  engineCall<{
    minutes: number;
    ram_gb: number;
    vertices: number;
    suggested_depth: number | null;
    machine_ram_gb: number | null;
  }>("mesh.estimate", {
    ...sv(),
    method,
    depth,
  });
export const ingestConflicts = (folder: string) =>
  engineCall<{ conflicts: string[] }>("ingest.conflicts", { ...sv(), folder });

// -- CRS and export -----------------------------------------------------------

export interface Crs {
  epsg: number;
  name: string;
}
export const crsSearch = (query: string, limit = 50) =>
  engineCall<{ results: Crs[] }>("crs.search", { query, limit });
export const crsSuggest = () =>
  engineCall<{ epsg: number | null; name: string | null }>("crs.suggest");
export const crsFrame = () => engineCall<{ short: string; full: string }>("crs.frame");

export interface ExportSettings {
  export_mode: "local" | "projected" | null;
  export_epsg: number | null;
  export_orthometric: boolean;
}
export const exportSettings = () => engineCall<ExportSettings>("export.settings");
export const setExportSettings = (s: ExportSettings) =>
  engineCall<ExportSettings>("export.set_settings", { ...s });

export interface ExportCrs {
  mode: "local" | "projected";
  epsg?: number;
  orthometric?: boolean;
}
export const exportCloud = (
  path: string,
  format: "las" | "csv" | "ply",
  crs: ExportCrs,
  ticket: string,
) =>
  engineCall<{ path: string; points: number }>(
    "export.cloud",
    { ...sv(), path, format, crs },
    ticket,
  );
export const exportMesh = (
  path: string,
  format: "ply" | "obj" | "stl",
  crs: ExportCrs,
  ticket: string,
) =>
  engineCall<{ path: string; vertices: number }>(
    "export.mesh",
    { ...sv(), path, format, crs },
    ticket,
  );
