// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

import { describe, expect, it } from "vitest";
import { countClasses, share } from "./counts";

describe("class counts", () => {
  it("counts each class and the unlabelled", () => {
    const { counts, total } = countClasses(Int32Array.of(-1, 0, 0, 2, -1, -1));
    expect(total).toBe(6);
    expect([...counts.entries()].sort((a, b) => a[0] - b[0])).toEqual([
      [-1, 3],
      [0, 2],
      [2, 1],
    ]);
  });

  it("words a share of the total", () => {
    expect(share(1234, 10000)).toBe("1,234 (12.3%)");
    expect(share(0, 0)).toBe("");
  });
});
