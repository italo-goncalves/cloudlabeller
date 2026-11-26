// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

// What lies between the Engine's arrays and geokit's viewer: pure functions,
// tested without a GPU. A cloud shown is one octree node; its classes are the
// labels plus one (0 is unlabelled); cameras become frusta; the mesh is
// coloured the way the Qt viewer coloured it (ui/viewer3d.py).

import { blendColour, type CloudSource, DIMMED, type Frustum } from "@geokit/viewer";
import type { Camera, SchemaClass, Status } from "../engine/api";

export type Vec3 = [number, number, number];
export type Rgb = [number, number, number];

/** The most points one node can hold and still be told apart point by point (geokit). */
export const NODE_LIMIT = 2 ** 24 - 1;

/** What to ask `cloud.view` for: 0 (every point) and anything too big for one node are capped. */
export function nodeLimit(maxPoints: number): number {
  return maxPoints <= 0 ? NODE_LIMIT : Math.min(maxPoints, NODE_LIMIT);
}

export interface Bounds {
  min: Vec3;
  max: Vec3;
}

/** The box round a flat xyz array, or null with no points. */
export function boundsOf(xyz: ArrayLike<number>): Bounds | null {
  const n = Math.floor(xyz.length / 3);
  if (n === 0) return null;
  const min: Vec3 = [Infinity, Infinity, Infinity];
  const max: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < n; i++)
    for (let k = 0; k < 3; k++) {
      const v = xyz[i * 3 + k] ?? 0;
      if (v < min[k as 0 | 1 | 2]) min[k as 0 | 1 | 2] = v;
      if (v > max[k as 0 | 1 | 2]) max[k as 0 | 1 | 2] = v;
    }
  return { min, max };
}

export const centreOf = (b: Bounds): Vec3 => [
  (b.min[0] + b.max[0]) / 2,
  (b.min[1] + b.max[1]) / 2,
  (b.min[2] + b.max[2]) / 2,
];

export const diagonalOf = (b: Bounds): number =>
  Math.hypot(b.max[0] - b.min[0], b.max[1] - b.min[1], b.max[2] - b.min[2]);

/** `xyz` moved so that `origin` is the zero: float32 keeps its precision near the origin. */
export function recentre(xyz: ArrayLike<number>, origin: Vec3): Float32Array {
  const out = new Float32Array(xyz.length);
  for (let i = 0; i + 2 < xyz.length; i += 3) {
    out[i] = (xyz[i] ?? 0) - origin[0];
    out[i + 1] = (xyz[i + 1] ?? 0) - origin[1];
    out[i + 2] = (xyz[i + 2] ?? 0) - origin[2];
  }
  return out;
}

/**
 * A geokit `CloudSource` of one node, "0-0-0-0", holding every point, its cube
 * enclosing them, positions recentred on `origin`. The node reads
 * `classification` when it loads, so a caller that changes the array in place
 * (and tells the layer) keeps a reloaded node right.
 */
export function singleNodeSource(
  xyz: Float32Array,
  rgb: Uint8Array | null,
  classification: Float32Array,
  origin: Vec3,
): CloudSource {
  const n = Math.floor(xyz.length / 3);
  const positions = recentre(xyz, origin);
  const b = boundsOf(positions);
  const centre: Vec3 = b ? centreOf(b) : [0, 0, 0];
  const side = b ? Math.max(b.max[0] - b.min[0], b.max[1] - b.min[1], b.max[2] - b.min[2]) : 0;
  const half = Math.max(side, 1e-3) / 2 + 1e-6 * Math.max(1, side);
  return {
    cube: [
      centre[0] - half,
      centre[1] - half,
      centre[2] - half,
      centre[0] + half,
      centre[1] + half,
      centre[2] + half,
    ],
    spacing: (2 * half) / Math.max(1, Math.cbrt(n)),
    nodes: [{ key: "0-0-0-0", depth: 0, x: 0, y: 0, z: 0, pointCount: n }],
    load: async () => ({
      count: n,
      positions,
      intensity: new Float32Array(n),
      classification,
      colour: rgb,
    }),
    zOrigin: origin[2],
  };
}

/**
 * The class number of each point shown: the label plus one, so 0 is
 * unlabelled (label −1). `index` maps each shown point to its full-cloud
 * index; null means the first `labels.length` points, in order.
 */
export function classificationFor(
  labels: ArrayLike<number>,
  index: ArrayLike<number> | null,
): Float32Array {
  const n = index ? index.length : labels.length;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const full = index ? (index[i] ?? -1) : i;
    const label = full >= 0 && full < labels.length ? (labels[full] ?? -1) : -1;
    out[i] = label + 1;
  }
  return out;
}

const UNLABELLED_GREY = 200;

/** "#rrggbb" as 0–255 channels; grey for anything else. */
export function parseHex(hex: string): Rgb {
  const m = /^#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(hex.trim());
  if (!m) return [UNLABELLED_GREY, UNLABELLED_GREY, UNLABELLED_GREY];
  return [
    Number.parseInt(m[1] ?? "c8", 16),
    Number.parseInt(m[2] ?? "c8", 16),
    Number.parseInt(m[3] ?? "c8", 16),
  ];
}

/** Class numbers up to the schema's highest: id + 1, plus 0 for unlabelled. */
function classCount(schema: readonly SchemaClass[]): number {
  let top = -1;
  for (const c of schema) top = Math.max(top, c.id);
  return top + 2;
}

/** Colours by class number: 0 is unlabelled grey, class `id` is at `id + 1`. */
export function classColours(schema: readonly SchemaClass[]): Rgb[] {
  const out: Rgb[] = Array.from({ length: classCount(schema) }, () => [
    UNLABELLED_GREY,
    UNLABELLED_GREY,
    UNLABELLED_GREY,
  ]);
  for (const c of schema) if (c.id >= 0) out[c.id + 1] = parseHex(c.color);
  return out;
}

/** The classes shown (all of them, so none is hidden), by class number. */
export const classMask = (schema: readonly SchemaClass[]): boolean[] =>
  Array.from({ length: classCount(schema) }, () => true);

/**
 * Which class numbers are drawn dimmed: with a class isolated, every class
 * but it, and unlabelled. Empty with none isolated.
 */
export function dimFor(schema: readonly SchemaClass[], isolate: number | null): boolean[] {
  if (isolate === null) return [];
  const out = Array.from({ length: classCount(schema) }, () => true);
  const k = isolate + 1;
  if (k >= 0 && k < out.length) out[k] = false;
  return out;
}

const inverses = new WeakMap<object, Int32Array>();

/** For each full-cloud index, its position among the points shown, or −1. */
function inverseOf(index: Int32Array): Int32Array {
  const known = inverses.get(index);
  if (known) return known;
  let top = -1;
  for (let i = 0; i < index.length; i++) top = Math.max(top, index[i] ?? -1);
  const inv = new Int32Array(top + 1).fill(-1);
  for (let i = 0; i < index.length; i++) {
    const full = index[i] ?? -1;
    if (full >= 0) inv[full] = i;
  }
  inverses.set(index, inv);
  return inv;
}

/**
 * Full-cloud indices as positions among the points shown, for `mark`: the
 * points that are not shown are left out. With `index` null every point is
 * shown, in order, `shown` of them (all, when not said).
 */
export function shownIndices(
  fullIndices: ArrayLike<number>,
  index: Int32Array | null,
  shown = Number.POSITIVE_INFINITY,
): Uint32Array {
  const out: number[] = [];
  if (!index) {
    for (let i = 0; i < fullIndices.length; i++) {
      const k = fullIndices[i] ?? -1;
      if (k >= 0 && k < shown) out.push(k);
    }
    return Uint32Array.from(out);
  }
  const inv = inverseOf(index);
  for (let i = 0; i < fullIndices.length; i++) {
    const k = fullIndices[i] ?? -1;
    const at = k >= 0 && k < inv.length ? (inv[k] ?? -1) : -1;
    if (at >= 0) out.push(at);
  }
  return Uint32Array.from(out);
}

/** A flat 3×3, whether the Engine sent it flat or as rows. */
function flat9(m: ArrayLike<number> | ArrayLike<number>[]): number[] {
  const out: number[] = [];
  for (const row of Array.from(m as ArrayLike<unknown>)) {
    if (typeof row === "number") out.push(row);
    else for (const v of Array.from(row as ArrayLike<number>)) out.push(v);
  }
  return out;
}

/**
 * A solved camera as a geokit frustum. COLMAP's `R` takes the world to the
 * camera; the frustum wants the world from the camera, so its transpose. The
 * position is moved to the scene's `origin`.
 */
export function frustumOf(
  camera: Pick<Camera, "width" | "height" | "K" | "R" | "position">,
  origin: Vec3 = [0, 0, 0],
): Frustum {
  const R = flat9(camera.R);
  const K = flat9(camera.K);
  const fy = K[4] ?? 1;
  return {
    position: [
      camera.position[0] - origin[0],
      camera.position[1] - origin[1],
      camera.position[2] - origin[2],
    ],
    rotation: [
      R[0] ?? 1,
      R[3] ?? 0,
      R[6] ?? 0,
      R[1] ?? 0,
      R[4] ?? 1,
      R[7] ?? 0,
      R[2] ?? 0,
      R[5] ?? 0,
      R[8] ?? 1,
    ],
    fovY: 2 * Math.atan(camera.height / (2 * fy)),
    aspect: camera.width / camera.height,
  };
}

/** How far a frustum reaches: 5% of the cloud's diagonal, times the user's scale. */
export const frustumSize = (bboxDiagonal: number, scale: number): number =>
  0.05 * bboxDiagonal * scale;

/** The status dots' colours, as in the Qt app's ui/status.py. */
export function statusColour(status: Status | string): Rgb {
  switch (status) {
    case "user":
      return [0x33, 0xcc, 0x33];
    case "auto":
    case "ml":
      return [0xff, 0xd4, 0x00];
    default:
      return [0xff, 0x33, 0x33];
  }
}

const MESH_GREY = 188;

/**
 * Per-vertex colours of the mesh, rgb 0–1: the photographic colour (grey
 * where it has none) blended toward its class colour by `blend`, unlabelled
 * vertices keeping theirs, then every vertex but the isolated class dimmed.
 * `count` is the vertex count when neither array says.
 */
export function meshColours(
  colors: Uint8Array | null,
  labels: ArrayLike<number> | null,
  schema: readonly SchemaClass[],
  blend: number,
  isolate: number | null,
  count = colors ? Math.floor(colors.length / 3) : (labels?.length ?? 0),
): Float32Array {
  const out = new Float32Array(count * 3);
  const own = (i: number, k: number) =>
    (colors ? (colors[i * 3 + k] ?? MESH_GREY) : MESH_GREY) / 255;
  const lut = new Map<number, [number, number, number]>();
  for (const c of schema) {
    const [r, g, b] = parseHex(c.color);
    lut.set(c.id, [r / 255, g / 255, b / 255]);
  }
  const usable = labels !== null && labels.length === count;
  const t = Math.min(1, Math.max(0, blend));
  for (let i = 0; i < count; i++) {
    let rgb: Rgb = [own(i, 0), own(i, 1), own(i, 2)];
    const label = usable ? (labels[i] ?? -1) : -1;
    const target = label >= 0 ? lut.get(label) : undefined;
    if (target && t > 0) rgb = blendColour(rgb, target, t);
    if (isolate !== null && usable && label !== isolate)
      rgb = [rgb[0] * DIMMED, rgb[1] * DIMMED, rgb[2] * DIMMED];
    out[i * 3] = rgb[0];
    out[i * 3 + 1] = rgb[1];
    out[i * 3 + 2] = rgb[2];
  }
  return out;
}

export interface HomePose {
  position: Vec3;
  target: Vec3;
  up: Vec3;
  /** The box's diagonal, for the clipping planes. */
  diagonal: number;
}

const cross = (a: Vec3, b: Vec3): Vec3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const unit = (v: Vec3): Vec3 => {
  const n = Math.hypot(...v) || 1;
  return [v[0] / n, v[1] / n, v[2] / n];
};

/**
 * The home view (ui/viewer3d.py `_reset_view`): from the south-east and
 * elevated, looking at the box's centre, `up` vertical. `ref` is the world x
 * axis, or z when `up` is nearly along x.
 */
export function homePose(bbox: Bounds, up: Vec3): HomePose {
  const u = unit(up);
  const centre = centreOf(bbox);
  const diagonal = diagonalOf(bbox) || 1;
  let ref: Vec3 = [1, 0, 0];
  if (Math.abs(ref[0] * u[0] + ref[1] * u[1] + ref[2] * u[2]) > 0.9) ref = [0, 0, 1];
  const east = unit(cross(ref, u));
  const south = cross(u, east);
  const k = Math.SQRT1_2;
  const se: Vec3 = [(east[0] + south[0]) * k, (east[1] + south[1]) * k, (east[2] + south[2]) * k];
  return {
    position: [
      centre[0] + diagonal * (0.8 * se[0] + 0.6 * u[0]),
      centre[1] + diagonal * (0.8 * se[1] + 0.6 * u[1]),
      centre[2] + diagonal * (0.8 * se[2] + 0.6 * u[2]),
    ],
    target: centre,
    up: u,
    diagonal,
  };
}

/** A count for a label: 500k, 2M, 6.5M. */
export function pointsLabel(n: number): string {
  if (n >= 1_000_000) return `${Number((n / 1_000_000).toFixed(1))}M`;
  if (n >= 1000) return `${Math.round(n / 1000)}k`;
  return String(n);
}
