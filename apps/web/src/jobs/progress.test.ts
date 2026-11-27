// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

import { describe, expect, it } from "vitest";
import { progressLog } from "./progress";

describe("progressLog", () => {
  it("logs each new message with its percentage", () => {
    const p = progressLog();
    expect(p.line(0.1, "Cutting photos into Tiles: 1 of 376")).toBe(
      "[ 10%] Cutting photos into Tiles: 1 of 376",
    );
    expect(p.line(0.104, "Cutting photos into Tiles: 2 of 376")).toBe(
      "[ 10%] Cutting photos into Tiles: 2 of 376",
    );
    expect(p.line(1, "Training complete")).toBe("[100%] Training complete");
  });

  it("skips a repeated or empty message", () => {
    const p = progressLog();
    expect(p.line(0.2, "Fusing")).not.toBeNull();
    expect(p.line(0.3, "Fusing")).toBeNull();
    expect(p.line(0.4, "  ")).toBeNull();
    expect(p.line(0.5, "Meshing")).not.toBeNull();
  });

  it("logs a message without a fraction as is", () => {
    expect(progressLog().line(null, "Started")).toBe("Started");
  });
});
