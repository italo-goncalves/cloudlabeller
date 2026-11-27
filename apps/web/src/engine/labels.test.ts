// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

import { describe, expect, it } from "vitest";
import { narrow, widen } from "./labels";

describe("label ids on the wire", () => {
  it("narrows a mask to int8 and widens it back", () => {
    const mask = Int32Array.from([-1, 0, 3, 127, -1]);
    const n = narrow(mask);
    expect(n.dtype).toBe("int8");
    expect(n.data.byteLength).toBe(5);
    expect(Array.from(widen(n.data))).toEqual(Array.from(mask));
  });

  it("keeps ids past 127 whole", () => {
    expect(narrow(Int32Array.from([-1, 128])).dtype).toBe("int16");
    expect(narrow(Int32Array.from([-1, 40000])).dtype).toBe("int32");
    expect(Array.from(narrow(Int32Array.from([-1, 300])).data)).toEqual([-1, 300]);
  });

  it("widens without copying what is already int32", () => {
    const ids = Int32Array.from([1, 2]);
    expect(widen(ids)).toBe(ids);
  });
});
