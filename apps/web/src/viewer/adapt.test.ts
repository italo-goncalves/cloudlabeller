// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

import { describe, expect, it } from "vitest";
import type { SchemaClass } from "../engine/api";
import {
  boundsOf,
  classColours,
  classificationFor,
  dimFor,
  frustumOf,
  frustumSize,
  homePose,
  meshColours,
  NODE_LIMIT,
  nodeLimit,
  pointsLabel,
  shownIndices,
  singleNodeSource,
  statusColour,
} from "./adapt";

const schema: SchemaClass[] = [
  { id: 0, name: "Quartz", color: "#ff0000", hotkey: null },
  { id: 1, name: "Mica", color: "#0000ff", hotkey: null },
];

describe("singleNodeSource", () => {
  const xyz = Float32Array.from([1000, 2000, 30, 1010, 2004, 32, 1002, 2002, 31]);
  const origin: [number, number, number] = [1005, 2002, 31];

  it("holds one node whose cube encloses the recentred points", async () => {
    const src = singleNodeSource(xyz, null, new Float32Array(3), origin);
    expect(src.nodes).toEqual([{ key: "0-0-0-0", depth: 0, x: 0, y: 0, z: 0, pointCount: 3 }]);
    expect(src.zOrigin).toBe(31);
    const node = await src.load("0-0-0-0");
    expect(Array.from(node.positions.slice(0, 3))).toEqual([-5, -2, -1]);
    const [x0, y0, z0, x1, y1, z1] = src.cube;
    for (let i = 0; i < 3; i++) {
      expect(node.positions[i * 3]).toBeGreaterThanOrEqual(x0);
      expect(node.positions[i * 3]).toBeLessThanOrEqual(x1);
      expect(node.positions[i * 3 + 1]).toBeGreaterThanOrEqual(y0);
      expect(node.positions[i * 3 + 1]).toBeLessThanOrEqual(y1);
      expect(node.positions[i * 3 + 2]).toBeGreaterThanOrEqual(z0);
      expect(node.positions[i * 3 + 2]).toBeLessThanOrEqual(z1);
    }
    expect(x1 - x0).toBeCloseTo(y1 - y0);
    expect(y1 - y0).toBeCloseTo(z1 - z0);
  });

  it("hands the classes and colours it was given, and survives a single point", async () => {
    const rgb = Uint8Array.from([1, 2, 3]);
    const cls = Float32Array.from([2]);
    const src = singleNodeSource(Float32Array.from([5, 5, 5]), rgb, cls, [5, 5, 5]);
    const node = await src.load("0-0-0-0");
    expect(node.classification).toBe(cls);
    expect(node.colour).toBe(rgb);
    expect(src.cube[3]).toBeGreaterThan(src.cube[0]);
  });
});

describe("classes", () => {
  it("is the label plus one, gathered through the index", () => {
    const labels = Int32Array.from([-1, 0, 1, 1, -1]);
    expect(Array.from(classificationFor(labels, Int32Array.from([4, 2, 1])))).toEqual([0, 2, 1]);
    expect(Array.from(classificationFor(labels, null))).toEqual([0, 1, 2, 2, 0]);
  });

  it("treats an index beyond the labels as unlabelled", () => {
    expect(Array.from(classificationFor(Int32Array.from([0]), Int32Array.from([0, 7])))).toEqual([
      1, 0,
    ]);
  });

  it("colours unlabelled grey and class k at k + 1", () => {
    const c = classColours(schema);
    expect(c).toEqual([
      [200, 200, 200],
      [255, 0, 0],
      [0, 0, 255],
    ]);
  });

  it("leaves gaps in a sparse schema grey", () => {
    const c = classColours([{ id: 3, name: "x", color: "#00ff00", hotkey: null }]);
    expect(c.length).toBe(5);
    expect(c[4]).toEqual([0, 255, 0]);
    expect(c[2]).toEqual([200, 200, 200]);
  });

  it("dims everything but the isolated class, and unlabelled", () => {
    expect(dimFor(schema, null)).toEqual([]);
    expect(dimFor(schema, 1)).toEqual([true, true, false]);
  });
});

describe("shownIndices", () => {
  it("maps full indices to shown positions and drops what is not shown", () => {
    const index = Int32Array.from([3, 5, 9, 12]);
    expect(Array.from(shownIndices([5, 12, 4, 100, 3], index))).toEqual([1, 3, 0]);
  });

  it("is the identity when every point is shown, within the count", () => {
    expect(Array.from(shownIndices([0, 4, 9], null, 5))).toEqual([0, 4]);
    expect(Array.from(shownIndices([0, 4, 9], null))).toEqual([0, 4, 9]);
  });
});

describe("cameras", () => {
  const K = [1000, 0, 500, 0, 1000, 400, 0, 0, 1];
  // a quarter turn about z: x' = y, y' = -x
  const R = [0, 1, 0, -1, 0, 0, 0, 0, 1];

  it("transposes R, finds the field of view and aspect", () => {
    // a flat list too, as an older Engine sent
    const flat = (m: number[]) => m as unknown as number[][];
    const f = frustumOf({
      width: 1000,
      height: 800,
      K: flat(K),
      R: flat(R),
      position: [10, 20, 30],
    });
    expect(f.rotation).toEqual([0, -1, 0, 1, 0, 0, 0, 0, 1]);
    expect(f.fovY).toBeCloseTo(2 * Math.atan(0.4));
    expect(f.aspect).toBeCloseTo(1.25);
    expect(f.position).toEqual([10, 20, 30]);
  });

  it("recentres the position and takes rows as well as a flat list", () => {
    const rows = {
      width: 2,
      height: 2,
      K: [
        [1, 0, 1],
        [0, 1, 1],
        [0, 0, 1],
      ],
      R: [
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 1],
      ],
      position: [10, 20, 30],
    };
    // biome-ignore lint/suspicious/noExplicitAny: rows stand in for a flat list
    const f = frustumOf(rows as any, [10, 10, 10]);
    expect(f.position).toEqual([0, 10, 20]);
    expect(f.rotation).toEqual([0, -1, 0, 1, 0, 0, 0, 0, 1]);
    expect(f.fovY).toBeCloseTo(2 * Math.atan(1));
  });

  it("sizes frusta at 5% of the diagonal times the scale", () => {
    expect(frustumSize(200, 0.5)).toBeCloseTo(5);
  });

  it("colours by status as Qt's status.py", () => {
    expect(statusColour("user")).toEqual([51, 204, 51]);
    expect(statusColour("auto")).toEqual([255, 212, 0]);
    expect(statusColour("ml")).toEqual([255, 212, 0]);
    expect(statusColour("none")).toEqual([255, 51, 51]);
  });
});

describe("meshColours", () => {
  const colors = Uint8Array.from([255, 255, 255, 0, 0, 0, 100, 100, 100]);
  const labels = Int32Array.from([-1, 0, 1]);

  it("keeps unlabelled vertices as they are and blends labelled ones", () => {
    const c = meshColours(colors, labels, schema, 0.5, null);
    expect(Array.from(c.slice(0, 3))).toEqual([1, 1, 1]);
    expect(Array.from(c.slice(3, 6))).toEqual([0.5, 0, 0]);
    expect(c[8]).toBeCloseTo((100 / 255) * 0.5 + 0.5);
    expect(c[6]).toBeCloseTo((100 / 255) * 0.5);
  });

  it("does not blend at 0 and reaches the class colour at 1", () => {
    expect(Array.from(meshColours(colors, labels, schema, 0, null).slice(3, 6))).toEqual([0, 0, 0]);
    expect(Array.from(meshColours(colors, labels, schema, 1, null).slice(3, 6))).toEqual([1, 0, 0]);
  });

  it("dims every vertex outside the isolated class to 15%", () => {
    const c = meshColours(colors, labels, schema, 1, 0);
    expect(c[0]).toBeCloseTo(0.15);
    expect(Array.from(c.slice(3, 6))).toEqual([1, 0, 0]);
    expect(c[8]).toBeCloseTo(0.15);
  });

  it("is grey without photographic colours and ignores labels of the wrong length", () => {
    const c = meshColours(null, Int32Array.from([0, 0]), schema, 1, null, 3);
    expect(c.length).toBe(9);
    expect(c[0]).toBeCloseTo(188 / 255);
  });
});

describe("home", () => {
  const bbox = {
    min: [0, 0, 0] as [number, number, number],
    max: [10, 10, 10] as [number, number, number],
  };

  it("looks at the centre from the south-east, elevated, along up", () => {
    const h = homePose(bbox, [0, 0, 1]);
    const d = Math.hypot(10, 10, 10);
    expect(h.target).toEqual([5, 5, 5]);
    expect(h.diagonal).toBeCloseTo(d);
    // east = ref x up = (1,0,0)x(0,0,1) = (0,-1,0); south = up x east = (1,0,0)
    const k = Math.SQRT1_2;
    expect(h.position[0]).toBeCloseTo(5 + d * 0.8 * k);
    expect(h.position[1]).toBeCloseTo(5 - d * 0.8 * k);
    expect(h.position[2]).toBeCloseTo(5 + d * 0.6);
  });

  it("picks another reference when up runs along x", () => {
    const h = homePose(bbox, [1, 0, 0]);
    const offset = h.position.map((p, i) => p - (h.target[i] ?? 0));
    expect(offset[0]).toBeCloseTo(h.diagonal * 0.6);
    expect(Number.isFinite(offset[1])).toBe(true);
  });
});

describe("limits and bounds", () => {
  it("caps a view at what one node can hold", () => {
    expect(nodeLimit(0)).toBe(NODE_LIMIT);
    expect(nodeLimit(500_000)).toBe(500_000);
    expect(nodeLimit(99_000_000)).toBe(NODE_LIMIT);
  });

  it("finds a box, or none", () => {
    expect(boundsOf([])).toBeNull();
    expect(boundsOf([1, 2, 3, -1, 5, 0])).toEqual({ min: [-1, 2, 0], max: [1, 5, 3] });
  });
});

describe("pointsLabel", () => {
  it("shortens counts", () => {
    expect(pointsLabel(500_000)).toBe("500k");
    expect(pointsLabel(2_000_000)).toBe("2M");
    expect(pointsLabel(6_500_000)).toBe("6.5M");
    expect(pointsLabel(120)).toBe("120");
  });
});
