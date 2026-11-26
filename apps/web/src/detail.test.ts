// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

import { describe, expect, it } from "vitest";
import { defaultLevel, detailChoices, MVS_LEVELS, SFM_LEVELS } from "./detail";

describe("reconstruction detail", () => {
  it("caps each level at the long side over its divisor, labelled by area", () => {
    const c = detailChoices(MVS_LEVELS, [5472, 3648], 2000);
    expect(c.map((x) => x.pixels)).toEqual([0, 2736, 1368, 684]);
    expect(c[0]?.label).toBe("Ultra — full resolution (5472 px)");
    expect(c[2]?.label).toBe("Medium — 1/16 resolution (1368 px)");
  });

  it("scales the custom default when the photos' size is unknown", () => {
    const c = detailChoices(SFM_LEVELS, null, 3200);
    expect(c.map((x) => x.pixels)).toEqual([0, 3200, 1600]);
    expect(c[0]?.label).toBe("High — full resolution");
  });

  it("finds the default level by name, else the second", () => {
    expect(defaultLevel(SFM_LEVELS, "medium")).toBe(1);
    expect(defaultLevel(MVS_LEVELS, "LOW")).toBe(3);
    expect(defaultLevel(SFM_LEVELS, "nope")).toBe(1);
  });
});
