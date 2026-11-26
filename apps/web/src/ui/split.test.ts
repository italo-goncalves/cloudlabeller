// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

import { describe, expect, it } from "vitest";
import { dragSize, readSizes } from "./split";

describe("dragSize", () => {
  it("grows with the pointer, or against it when inverted", () => {
    expect(dragSize(300, 40, 200, 600)).toBe(340);
    expect(dragSize(300, 40, 200, 600, true)).toBe(260);
  });

  it("stays within its bounds, the minimum winning when they cross", () => {
    expect(dragSize(300, -500, 200, 600)).toBe(200);
    expect(dragSize(300, 500, 200, 600)).toBe(600);
    expect(dragSize(300, 0, 200, 100)).toBe(200);
  });
});

describe("readSizes", () => {
  const defaults = { left: 320, log: 176 };

  it("takes the saved sizes it knows and falls back for the rest", () => {
    expect(readSizes({ left: 400, log: "x", other: 9 }, defaults)).toEqual({ left: 400, log: 176 });
  });

  it("ignores nothing saved, and sizes that aren't positive", () => {
    expect(readSizes(null, defaults)).toEqual(defaults);
    expect(readSizes({ left: -5, log: Number.NaN }, defaults)).toEqual(defaults);
  });
});
