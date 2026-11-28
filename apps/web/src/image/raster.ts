// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

// The Image pane's pixel work, pure: filling polygons and painting strokes
// into a mask (row-major class ids, −1 unlabelled), turning an external mask
// file's pixels into a selection (as `core/raster.py`'s `mask_from_file`
// does), colouring a mask for display, and the pan and zoom maths.

/** A pixel rectangle, `x1` and `y1` excluded. */
export interface Rect {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export type Point = [number, number];

/** Where a pixel's label came from (ADR-0006), as the Engine codes it. */
export const SOURCE = { none: 0, hand: 2, cloud: 3, predicted: 4 } as const;

/** The class to paint with to accept what a pixel shows as hand-drawn. */
export const KEEP = -2;

/**
 * What a stroke leaves besides the mask: `touched` is 1 where it went (the
 * Engine keeps those pixels as the user's), `sources` is the source shown.
 */
export interface Marks {
  touched: Uint8Array;
  sources: Uint8Array;
}

/** Paint one pixel: a class (−1 erases), or {@link KEEP} to accept a label shown. */
function paintPixel(mask: Int32Array, i: number, classId: number, marks?: Marks): boolean {
  if (classId === KEEP) {
    if ((mask[i] as number) < 0) return false;
  } else {
    mask[i] = classId;
  }
  if (marks) {
    marks.touched[i] = 1;
    marks.sources[i] = classId === -1 ? SOURCE.none : SOURCE.hand;
  }
  return true;
}

export function unionRect(a: Rect | null, b: Rect | null): Rect | null {
  if (!a) return b;
  if (!b) return a;
  return {
    x0: Math.min(a.x0, b.x0),
    y0: Math.min(a.y0, b.y0),
    x1: Math.max(a.x1, b.x1),
    y1: Math.max(a.y1, b.y1),
  };
}

/**
 * Paint the inside of a polygon (vertices in image pixels) with `classId`:
 * even-odd rule, a pixel counting as inside when its centre is, clipped to the
 * image. Returns the rectangle touched, or null when nothing was.
 */
export function fillPolygon(
  mask: Int32Array,
  w: number,
  h: number,
  polygon: readonly Point[],
  classId: number,
  marks?: Marks,
): Rect | null {
  const n = polygon.length;
  if (n < 3 || w <= 0 || h <= 0) return null;
  let top = Infinity;
  let bottom = -Infinity;
  for (const [, y] of polygon) {
    top = Math.min(top, y);
    bottom = Math.max(bottom, y);
  }
  const yFirst = Math.max(0, Math.ceil(top - 0.5));
  const yLast = Math.min(h - 1, Math.ceil(bottom - 0.5) - 1);
  let rect: Rect | null = null;
  const crossings: number[] = [];
  for (let y = yFirst; y <= yLast; y++) {
    const yc = y + 0.5;
    crossings.length = 0;
    for (let i = 0; i < n; i++) {
      const a = polygon[i] as Point;
      const b = polygon[(i + 1) % n] as Point;
      // Half-open in y, so a vertex on the scanline counts once.
      if (a[1] <= yc !== b[1] <= yc) {
        crossings.push(a[0] + ((yc - a[1]) / (b[1] - a[1])) * (b[0] - a[0]));
      }
    }
    crossings.sort((p, q) => p - q);
    for (let k = 0; k + 1 < crossings.length; k += 2) {
      const xa = Math.max(0, Math.ceil((crossings[k] as number) - 0.5));
      const xb = Math.min(w, Math.ceil((crossings[k + 1] as number) - 0.5));
      if (xb <= xa) continue;
      if (classId !== KEEP && !marks) mask.fill(classId, y * w + xa, y * w + xb);
      else for (let i = y * w + xa; i < y * w + xb; i++) paintPixel(mask, i, classId, marks);
      rect = unionRect(rect, { x0: xa, y0: y, x1: xb, y1: y + 1 });
    }
  }
  return rect;
}

/** Paint a disc (centre in image pixels) of the given radius; the pixel under the centre always is. */
export function stampDisc(
  mask: Int32Array,
  w: number,
  h: number,
  cx: number,
  cy: number,
  radius: number,
  classId: number,
  marks?: Marks,
): Rect | null {
  const r = Math.max(radius, 0);
  const x0 = Math.max(0, Math.floor(cx - r));
  const x1 = Math.min(w, Math.ceil(cx + r) + 1);
  const y0 = Math.max(0, Math.floor(cy - r));
  const y1 = Math.min(h, Math.ceil(cy + r) + 1);
  const px = Math.floor(cx);
  const py = Math.floor(cy);
  const r2 = r * r;
  let touched = false;
  for (let y = y0; y < y1; y++) {
    const dy = y + 0.5 - cy;
    for (let x = x0; x < x1; x++) {
      const dx = x + 0.5 - cx;
      if (dx * dx + dy * dy <= r2 || (x === px && y === py)) {
        if (paintPixel(mask, y * w + x, classId, marks)) touched = true;
      }
    }
  }
  return touched ? { x0, y0, x1, y1 } : null;
}

/** Stamp discs along a segment, about every radius/2, ends included. */
export function strokeLine(
  mask: Int32Array,
  w: number,
  h: number,
  from: Point,
  to: Point,
  radius: number,
  classId: number,
  marks?: Marks,
): Rect | null {
  const length = Math.hypot(to[0] - from[0], to[1] - from[1]);
  const spacing = Math.max(radius / 2, 0.5);
  const steps = Math.max(1, Math.ceil(length / spacing));
  let rect: Rect | null = null;
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    const x = from[0] + (to[0] - from[0]) * t;
    const y = from[1] + (to[1] - from[1]) * t;
    rect = unionRect(rect, stampDisc(mask, w, h, x, y, radius, classId, marks));
  }
  return rect;
}

/**
 * The selection an external mask file makes, painted with `classId` over a
 * copy of `base`, like `mask_from_file`: a .jpg selects its highest-intensity
 * pixels (grayscale as PIL's "L": (299 R + 587 G + 114 B) / 1000), a .png its
 * highest-alpha pixels. As there, an all-equal file selects everything.
 */
export function maskFromPixels(
  rgba: Uint8ClampedArray,
  w: number,
  h: number,
  kind: "jpg" | "png",
  classId: number,
  base: Int32Array,
): Int32Array {
  const n = w * h;
  const value = new Uint8Array(n);
  let max = 0;
  for (let i = 0; i < n; i++) {
    const o = i * 4;
    const v =
      kind === "png"
        ? (rgba[o + 3] as number)
        : Math.floor(
            ((rgba[o] as number) * 299 +
              (rgba[o + 1] as number) * 587 +
              (rgba[o + 2] as number) * 114) /
              1000,
          );
    value[i] = v;
    if (v > max) max = v;
  }
  const out = base.slice();
  for (let i = 0; i < n; i++) if (value[i] === max) out[i] = classId;
  return out;
}

/** The colours of the sources, when the overlay shows them instead of the classes. */
export const SOURCE_COLOURS: Record<number, [number, number, number]> = {
  [SOURCE.hand]: [63, 185, 80],
  [SOURCE.cloud]: [88, 166, 255],
  [SOURCE.predicted]: [255, 166, 87],
};

/**
 * How opaque a pixel of a source is drawn: hand-drawn solid, from the cloud
 * lighter, predicted hatched (diagonal stripes), so the source shows at a glance.
 */
export function sourceAlpha(source: number, x: number, y: number): number {
  if (source === SOURCE.cloud) return 170;
  if (source === SOURCE.predicted) return ((x + y) >> 2) & 1 ? 150 : 55;
  return 255;
}

/**
 * Colour a mask into `out` (RGBA; the display opacity is the canvas's).
 * Unlabelled (−1) and ids without a colour are transparent. `rect` limits the
 * work to the pixels that changed. With `sources`, each pixel's opacity says
 * where its label came from ({@link sourceAlpha}), and with `bySource` its
 * colour does too ({@link SOURCE_COLOURS}) instead of its class.
 */
export function maskToRgba(
  mask: Int32Array,
  w: number,
  h: number,
  colours: Map<number, [number, number, number]>,
  out: Uint8ClampedArray,
  rect?: Rect,
  sources?: Uint8Array | null,
  bySource = false,
): void {
  const x0 = rect ? Math.max(0, rect.x0) : 0;
  const x1 = rect ? Math.min(w, rect.x1) : w;
  const y0 = rect ? Math.max(0, rect.y0) : 0;
  const y1 = rect ? Math.min(h, rect.y1) : h;
  let lastId = -2;
  let last: [number, number, number] | undefined;
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      const i = y * w + x;
      const id = mask[i] as number;
      if (id !== lastId) {
        lastId = id;
        last = id < 0 ? undefined : colours.get(id);
      }
      const o = i * 4;
      const source = sources ? (sources[i] as number) : SOURCE.hand;
      const colour = last && bySource ? (SOURCE_COLOURS[source] ?? last) : last;
      if (colour) {
        out[o] = colour[0];
        out[o + 1] = colour[1];
        out[o + 2] = colour[2];
        out[o + 3] = sources ? sourceAlpha(source, x, y) : 255;
      } else {
        out[o] = out[o + 1] = out[o + 2] = out[o + 3] = 0;
      }
    }
  }
}

/** "#rrggbb" → [r, g, b]; white when malformed. */
export function parseColour(hex: string): [number, number, number] {
  const m = /^#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(hex.trim());
  if (!m) return [255, 255, 255];
  return [
    Number.parseInt(m[1] as string, 16),
    Number.parseInt(m[2] as string, 16),
    Number.parseInt(m[3] as string, 16),
  ];
}

// -- pan and zoom --------------------------------------------------------------

/** Screen = image × scale + (x, y), in CSS pixels of the viewport. */
export interface View {
  scale: number;
  x: number;
  y: number;
}

export const MIN_SCALE = 0.02;
export const MAX_SCALE = 64;
export const ZOOM_STEP = 1.25;

/** The view that shows the whole image, centred. */
export function fitView(imageW: number, imageH: number, viewW: number, viewH: number): View {
  if (imageW <= 0 || imageH <= 0 || viewW <= 0 || viewH <= 0) return { scale: 1, x: 0, y: 0 };
  const scale = Math.min(viewW / imageW, viewH / imageH);
  return { scale, x: (viewW - imageW * scale) / 2, y: (viewH - imageH * scale) / 2 };
}

/** Zoom by `factor` keeping the image point under (cx, cy) fixed on screen. */
export function zoomAbout(view: View, factor: number, cx: number, cy: number): View {
  const scale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, view.scale * factor));
  const k = scale / view.scale;
  return { scale, x: cx - (cx - view.x) * k, y: cy - (cy - view.y) * k };
}

export function panBy(view: View, dx: number, dy: number): View {
  return { ...view, x: view.x + dx, y: view.y + dy };
}

export function screenToImage(view: View, sx: number, sy: number): Point {
  return [(sx - view.x) / view.scale, (sy - view.y) / view.scale];
}

export function imageToScreen(view: View, ix: number, iy: number): Point {
  return [ix * view.scale + view.x, iy * view.scale + view.y];
}

/** The wheel's zoom factor: ×1.25 for a scroll up, ×0.8 for down. */
export function wheelFactor(deltaY: number): number {
  return deltaY < 0 ? ZOOM_STEP : 1 / ZOOM_STEP;
}
