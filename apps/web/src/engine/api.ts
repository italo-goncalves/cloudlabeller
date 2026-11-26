// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

// The Engine's methods as the window calls them, typed after docs/engine.md.
// Every screen goes through these, not through `engineCall` with a string.

import { type ArrayRef, engineCall, readArray, writeArray } from "../desktop/bridge";

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
}

export interface ProjectState {
  root: string;
  schema: SchemaClass[];
  images: ImageRow[];
  reconstruction: Reconstruction;
  sparse: number | null;
  dense: number | null;
  mesh: { vertices: number; faces: number } | null;
  georeferenced: boolean;
  source_resolution: [number, number] | null;
  can_undo: boolean;
  can_redo: boolean;
  dirty: boolean;
}

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

export const state = () => engineCall<ProjectState>("project.state");
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

export const undo = () => engineCall<{ changed: Changed }>("labels.undo");
export const redo = () => engineCall<{ changed: Changed }>("labels.redo");

// -- photos -------------------------------------------------------------------

export interface ImageInfo {
  path: string;
  width: number;
  height: number;
}
export const imageInfo = (name: string) => engineCall<ImageInfo>("image.info", { name });

export interface Mask {
  source: "user" | "ml" | "auto" | null;
  /** H × W class ids (−1 unlabelled), row-major; null when there is none. */
  mask: Int32Array | null;
  width: number;
  height: number;
}

export async function imageMask(name: string): Promise<Mask> {
  const r = await engineCall<{ source: Mask["source"]; mask: ArrayRef | null }>("image.mask", {
    name,
  });
  if (!r.mask) return { source: r.source, mask: null, width: 0, height: 0 };
  const [h, w] = r.mask.shape as [number, number];
  return { source: r.source, mask: (await load(r.mask)) as Int32Array, width: w, height: h };
}

export async function saveMask(name: string, mask: Int32Array, width: number, height: number) {
  const ref = await writeArray(fresh("mask"), mask, "int32", [height, width]);
  return engineCall<{ changed_pixels: number; status: Status }>("image.save_mask", {
    name,
    mask: ref,
  });
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
export async function cloudView(cloud: CloudName, maxPoints: number): Promise<CloudView> {
  const r = await engineCall<{
    cloud: CloudName;
    n_points: number;
    xyz: ArrayRef;
    rgb: ArrayRef | null;
    index: ArrayRef | null;
  }>("cloud.view", { cloud, max_points: maxPoints });
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
  const r = await engineCall<{ labels: ArrayRef }>("cloud.labels", { cloud });
  return (await load(r.labels)) as Int32Array;
}

export async function cloudPaint(cloud: CloudName, indices: Uint32Array, classId: number) {
  const ref = await writeArray(fresh("paint"), indices, "uint32", [indices.length]);
  return engineCall<{ painted: number }>("cloud.paint", { cloud, indices: ref, class_id: classId });
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
  return engineCall<{ removed: number; n_points: number }>("cloud.delete", { cloud, indices: ref });
}

export const cloudStats = () =>
  engineCall<{ n_points: number; matched: number; views: ArrayRef; errors: ArrayRef }>(
    "cloud.stats",
  );

export async function cleanPreview(minImages: number, maxError: number) {
  const r = await engineCall<{ keep: number; total: number; removed: ArrayRef }>(
    "cloud.clean_preview",
    { min_images: minImages, max_error: maxError },
  );
  return {
    keep: r.keep,
    total: r.total,
    removed: r.total - r.keep > 0 ? ((await load(r.removed)) as Uint32Array) : new Uint32Array(0),
  };
}

export const clean = (minImages: number, maxError: number) =>
  engineCall<{ removed: number; n_points: number }>("cloud.clean", {
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
  );
  const faces = (await load(r.faces)) as Int32Array;
  return {
    vertices: (await load(r.vertices)) as Float32Array,
    faces: new Uint32Array(faces.buffer, faces.byteOffset, faces.length),
    colors: r.colors ? ((await load(r.colors)) as Uint8Array) : null,
  };
}

export async function meshLabels(): Promise<Int32Array> {
  const r = await engineCall<{ labels: ArrayRef }>("mesh.labels");
  return (await load(r.labels)) as Int32Array;
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
  engineCall<{ cameras: Camera[]; up: [number, number, number] | null }>("cameras.list");

// -- transfers ----------------------------------------------------------------

export const imagesToCloud = (includeAuto: boolean, ticket: string) =>
  engineCall<{ images: number; sparse: number; dense?: number }>(
    "transfer.images_to_cloud",
    { include_auto: includeAuto },
    ticket,
  );
export const cloudToImages = (ticket: string) =>
  engineCall<{ cloud: CloudName; auto_labelled: number }>("transfer.cloud_to_images", {}, ticket);
export const propagate = (name: string, ticket: string) =>
  engineCall<{ cloud: CloudName; points: number; auto_labelled: number }>(
    "transfer.propagate",
    { name },
    ticket,
  );

// -- the model ----------------------------------------------------------------

export interface ModelSpec {
  resolution_divisor: number;
  /** Channels in the first U-Net layer. */
  channels: number;
  blocks: number;
  dropout: number;
  filter_size: number;
  [k: string]: unknown;
}

export const modelSpec = () =>
  engineCall<{ spec: ModelSpec; training_size: [number, number] | null; problem: string | null }>(
    "model.spec",
  );
export const setModelSpec = (spec: ModelSpec) =>
  engineCall<{ spec: ModelSpec; training_size: [number, number] | null; problem: string | null }>(
    "model.set_spec",
    { spec },
  );

export type ModelManifest = Record<string, unknown> & { name: string };
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

export const facts = (kind: string) => engineCall<Facts>("jobs.facts", { kind });
export const meshEstimate = (method: string, depth: number) =>
  engineCall<{
    minutes: number;
    ram_gb: number;
    vertices: number;
    suggested_depth: number | null;
    machine_ram_gb: number | null;
  }>("mesh.estimate", {
    method,
    depth,
  });
export const ingestConflicts = (folder: string) =>
  engineCall<{ conflicts: string[] }>("ingest.conflicts", { folder });

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
) => engineCall<{ path: string; points: number }>("export.cloud", { path, format, crs }, ticket);
export const exportMesh = (
  path: string,
  format: "ply" | "obj" | "stl",
  crs: ExportCrs,
  ticket: string,
) => engineCall<{ path: string; vertices: number }>("export.mesh", { path, format, crs }, ticket);
