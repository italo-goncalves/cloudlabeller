// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

import { describe, expect, it } from "vitest";
import type { EnvironmentSaid } from "../desktop/bridge";
import { explain, fold, initialSetup, MAX_LINES } from "./setup";

const run = (...said: EnvironmentSaid[]) => said.reduce(fold, initialSetup);

describe("the Environment screen", () => {
  it("follows an install to ready", () => {
    const s = run(
      {
        kind: "chose",
        choice: {
          own: { machine: "wsl", distro: "Ubuntu" },
          why: "gpu",
          gpu: "GPU 0: RTX",
          distros: ["Ubuntu"],
        },
      },
      { kind: "doing", what: "install" },
      { kind: "step", i: 2, n: 5, what: "venv" },
      { kind: "out", line: "Creating venv" },
      { kind: "err", line: "warning" },
      {
        kind: "ready",
        kept: {
          environment: { machine: "wsl", distro: "Ubuntu", python: { kind: "default" } },
          own: true,
        },
      },
    );
    expect(s.phase).toBe("ready");
    expect(s.doing).toBe("install");
    expect(s.step).toEqual({ i: 2, n: 5, what: "venv" });
    expect(s.lines).toEqual([
      { n: 1, text: "Creating venv", err: false },
      { n: 2, text: "warning", err: true },
    ]);
    expect(s.choice && explain(s.choice)).toContain("Ubuntu");
  });

  it("keeps the last lines only", () => {
    const lines = Array.from({ length: MAX_LINES + 10 }, (_, i) => ({
      kind: "out" as const,
      line: String(i),
    }));
    const s = run(...lines);
    expect(s.lines).toHaveLength(MAX_LINES);
    expect(s.lines[0]?.text).toBe("10");
  });

  it("shows a failure's Problem", () => {
    const s = run({
      kind: "failed",
      problem: { code: "install.cancelled", english: "Cancelled." },
    });
    expect(s.phase).toBe("failed");
    expect(s.problem?.code).toBe("install.cancelled");
  });
});
