// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

import { describe, expect, it } from "vitest";
import {
  fillPolygon,
  fitView,
  imageToScreen,
  maskFromPixels,
  maskToRgba,
  parseColour,
  screenToImage,
  stampDisc,
  strokeLine,
  wheelFactor,
  zoomAbout,
} from "./raster";

const blank = (w: number, h: number) => new Int32Array(w * h).fill(-1);
const count = (m: Int32Array, id: number) => m.filter((v) => v === id).length;
const rows = (m: Int32Array, w: number) =>
  Array.from({ length: m.length / w }, (_, y) =>
    Array.from(m.slice(y * w, (y + 1) * w), (v) => (v < 0 ? "." : String(v))).join(""),
  );

describe("fillPolygon", () => {
  it("fills pixels whose centres are inside", () => {
    const m = blank(6, 6);
    fillPolygon(
      m,
      6,
      6,
      [
        [1, 1],
        [4, 1],
        [4, 3],
        [1, 3],
      ],
      2,
    );
    expect(rows(m, 6)).toEqual(["......", ".222..", ".222..", "......", "......", "......"]);
  });

  it("samples at centres: a triangle's diagonal", () => {
    const m = blank(4, 4);
    fillPolygon(
      m,
      4,
      4,
      [
        [0, 0],
        [4, 0],
        [0, 4],
      ],
      1,
    );
    expect(rows(m, 4)).toEqual(["111.", "11..", "1...", "...."]);
  });

  it("uses the even-odd rule: a self-overlap leaves a hole", () => {
    const m = blank(8, 8);
    // Outer square and an inner one drawn the same way round: nonzero would fill both.
    fillPolygon(
      m,
      8,
      8,
      [
        [0, 0],
        [8, 0],
        [8, 8],
        [0, 8],
        [0, 0],
        [2, 2],
        [6, 2],
        [6, 6],
        [2, 6],
        [2, 2],
      ],
      0,
    );
    expect(m[4 * 8 + 4]).toBe(-1);
    expect(m[1 * 8 + 1]).toBe(0);
  });

  it("clips to the image", () => {
    const m = blank(4, 4);
    const r = fillPolygon(
      m,
      4,
      4,
      [
        [-10, -10],
        [10, -10],
        [10, 10],
        [-10, 10],
      ],
      3,
    );
    expect(count(m, 3)).toBe(16);
    expect(r).toEqual({ x0: 0, y0: 0, x1: 4, y1: 4 });
  });

  it("does nothing for fewer than three vertices or outside the image", () => {
    const m = blank(4, 4);
    expect(
      fillPolygon(
        m,
        4,
        4,
        [
          [0, 0],
          [3, 3],
        ],
        1,
      ),
    ).toBeNull();
    expect(
      fillPolygon(
        m,
        4,
        4,
        [
          [10, 10],
          [12, 10],
          [12, 12],
        ],
        1,
      ),
    ).toBeNull();
    expect(count(m, 1)).toBe(0);
  });

  it("can erase with -1", () => {
    const m = new Int32Array(16).fill(5);
    fillPolygon(
      m,
      4,
      4,
      [
        [0, 0],
        [2, 0],
        [2, 2],
        [0, 2],
      ],
      -1,
    );
    expect(count(m, -1)).toBe(4);
  });
});

describe("stampDisc and strokeLine", () => {
  it("stamps the pixels whose centres are within the radius", () => {
    const m = blank(7, 7);
    stampDisc(m, 7, 7, 3.5, 3.5, 2, 1);
    expect(rows(m, 7)).toEqual([
      ".......",
      "...1...",
      "..111..",
      ".11111.",
      "..111..",
      "...1...",
      ".......",
    ]);
  });

  it("always paints the pixel under the centre, even for a tiny radius", () => {
    const m = blank(4, 4);
    stampDisc(m, 4, 4, 1.9, 2.1, 0.1, 4);
    expect(count(m, 4)).toBe(1);
    expect(m[2 * 4 + 1]).toBe(4);
  });

  it("clips at the borders and ignores a disc wholly outside", () => {
    const m = blank(4, 4);
    expect(stampDisc(m, 4, 4, -20, -20, 2, 1)).toBeNull();
    stampDisc(m, 4, 4, 0, 0, 2, 1);
    expect(m[0]).toBe(1);
  });

  it("strokes without gaps along a line", () => {
    const m = blank(40, 9);
    strokeLine(m, 40, 9, [2, 4.5], [37, 4.5], 2, 1);
    for (let x = 2; x <= 37; x++) expect(m[4 * 40 + x]).toBe(1);
    expect(m[4 * 40 + 39]).toBe(-1);
  });

  it("a zero-length stroke is one stamp, and the erase class works", () => {
    const m = new Int32Array(25).fill(2);
    strokeLine(m, 5, 5, [2.5, 2.5], [2.5, 2.5], 1, -1);
    expect(m[12]).toBe(-1);
  });
});

describe("maskFromPixels", () => {
  const px = (...p: [number, number, number, number][]) => new Uint8ClampedArray(p.flat());

  it("a jpg selects the highest-intensity pixels", () => {
    const rgba = px(
      [0, 0, 0, 255],
      [255, 255, 255, 255],
      [200, 200, 200, 255],
      [255, 255, 255, 255],
    );
    const out = maskFromPixels(rgba, 2, 2, "jpg", 3, blank(2, 2));
    expect(Array.from(out)).toEqual([-1, 3, -1, 3]);
  });

  it("weights the channels as PIL's L does", () => {
    const rgba = px([100, 0, 0, 255], [0, 100, 0, 255], [0, 0, 100, 255], [0, 0, 0, 255]);
    expect(Array.from(maskFromPixels(rgba, 2, 2, "jpg", 1, blank(2, 2)))).toEqual([-1, 1, -1, -1]);
  });

  it("a png selects the highest-alpha pixels, whatever the colour", () => {
    const rgba = px([255, 255, 255, 10], [0, 0, 0, 200], [9, 9, 9, 200], [0, 0, 0, 0]);
    expect(Array.from(maskFromPixels(rgba, 2, 2, "png", 0, blank(2, 2)))).toEqual([-1, 0, 0, -1]);
  });

  it("paints over a copy of the base and leaves the rest of it alone", () => {
    const base = Int32Array.from([5, 5, 6, -1]);
    const rgba = px([0, 0, 0, 255], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]);
    const out = maskFromPixels(rgba, 2, 2, "png", 2, base);
    expect(Array.from(out)).toEqual([2, 5, 6, -1]);
    expect(Array.from(base)).toEqual([5, 5, 6, -1]);
  });

  it("an all-equal file selects everything, as mask_from_file does", () => {
    const rgba = px([0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255]);
    expect(count(maskFromPixels(rgba, 2, 2, "jpg", 1, blank(2, 2)), 1)).toBe(4);
  });
});

describe("maskToRgba", () => {
  const colours = new Map<number, [number, number, number]>([
    [0, [255, 0, 0]],
    [1, [0, 255, 0]],
  ]);

  it("colours classes and leaves unlabelled and unknown ids transparent", () => {
    const out = new Uint8ClampedArray(16).fill(9);
    maskToRgba(Int32Array.from([0, 1, -1, 7]), 2, 2, colours, out);
    expect(Array.from(out)).toEqual([255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0]);
  });

  it("only touches the given rectangle", () => {
    const out = new Uint8ClampedArray(16).fill(9);
    maskToRgba(Int32Array.from([0, 0, 0, 0]), 2, 2, colours, out, { x0: 1, y0: 1, x1: 2, y1: 2 });
    expect(Array.from(out.slice(0, 4))).toEqual([9, 9, 9, 9]);
    expect(Array.from(out.slice(12))).toEqual([255, 0, 0, 255]);
  });

  it("parses schema colours", () => {
    expect(parseColour("#10ff80")).toEqual([16, 255, 128]);
    expect(parseColour("nope")).toEqual([255, 255, 255]);
  });
});

describe("view maths", () => {
  it("fits a wide image to the width and centres it", () => {
    const v = fitView(2000, 1000, 800, 600);
    expect(v.scale).toBeCloseTo(0.4);
    expect(v.x).toBeCloseTo(0);
    expect(v.y).toBeCloseTo(100);
  });

  it("fits a tall image to the height", () => {
    const v = fitView(500, 1000, 800, 600);
    expect(v.scale).toBeCloseTo(0.6);
    expect(v.x).toBeCloseTo(250);
  });

  it("survives empty sizes", () => {
    expect(fitView(0, 0, 100, 100)).toEqual({ scale: 1, x: 0, y: 0 });
    expect(fitView(10, 10, 0, 0)).toEqual({ scale: 1, x: 0, y: 0 });
  });

  it("zooms about the cursor: the image point under it stays put", () => {
    const v = { scale: 0.5, x: 30, y: -20 };
    const before = screenToImage(v, 200, 150);
    const z = zoomAbout(v, 1.25, 200, 150);
    expect(z.scale).toBeCloseTo(0.625);
    const after = imageToScreen(z, before[0], before[1]);
    expect(after[0]).toBeCloseTo(200);
    expect(after[1]).toBeCloseTo(150);
  });

  it("wheel up is ×1.25 and down ×0.8, and zoom is bounded", () => {
    expect(wheelFactor(-100)).toBe(1.25);
    expect(wheelFactor(100)).toBeCloseTo(0.8);
    expect(zoomAbout({ scale: 64, x: 0, y: 0 }, 1.25, 0, 0).scale).toBe(64);
  });

  it("converts back and forth", () => {
    const v = { scale: 2, x: 10, y: 20 };
    expect(imageToScreen(v, 5, 5)).toEqual([20, 30]);
    expect(screenToImage(v, 20, 30)).toEqual([5, 5]);
  });
});
