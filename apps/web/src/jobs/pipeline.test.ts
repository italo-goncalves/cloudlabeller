// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

import { describe, expect, it } from "vitest";
import type { JobOutcome } from "../state/app";
import { runPipeline, type Stage, stages } from "./pipeline";

const all = stages({ sfm: { gpu: true }, georef: true, mvs: {}, mesh: { method: "poisson" } });
const ok: JobOutcome = { ok: true };
const failed = (code = "job.failed"): JobOutcome => ({ ok: false, error: { code, english: code } });

function scripted(outcomes: Partial<Record<Stage["kind"], JobOutcome>>) {
  const ran: string[] = [];
  const lines: string[] = [];
  let t = 0;
  const run = async (s: Stage) => {
    ran.push(s.kind);
    t += 60_000;
    return outcomes[s.kind] ?? ok;
  };
  return { ran, lines, run, say: (l: string) => lines.push(l), now: () => t };
}

describe("the pipeline", () => {
  it("chains SfM, georeferencing, MVS and the mesh, and times it", async () => {
    expect(all.map((s) => s.kind)).toEqual(["sfm", "georef", "mvs", "mesh"]);
    expect(stages({ sfm: {}, georef: false, mvs: {}, mesh: {} }).map((s) => s.kind)).toEqual([
      "sfm",
      "mvs",
      "mesh",
    ]);
    const p = scripted({});
    const end = await runPipeline(all, p.run, p.say, p.now);
    expect(end).toEqual({
      done: ["sfm", "georef", "mvs", "mesh"],
      stopped: null,
      local: false,
      minutes: 4,
    });
    expect(p.lines[0]).toBe("=== Pipeline 1/4: Structure from Motion ===");
    expect(p.lines.at(-1)).toBe("=== Pipeline done in 4.0 min ===");
  });

  it("carries on in the local frame when georeferencing fails", async () => {
    const p = scripted({ georef: failed("job.georef.fit") });
    const end = await runPipeline(all, p.run, p.say, p.now);
    expect(p.ran).toEqual(["sfm", "georef", "mvs", "mesh"]);
    expect(end.local).toBe(true);
    expect(end.done).toEqual(["sfm", "mvs", "mesh"]);
  });

  it("stops the chain when a stage is cancelled, georeferencing included", async () => {
    const p = scripted({ georef: failed("job.cancelled") });
    const end = await runPipeline(all, p.run, p.say, p.now);
    expect(p.ran).toEqual(["sfm", "georef"]);
    expect(end.stopped).toEqual({ kind: "georef", why: "cancelled", english: "job.cancelled" });
  });

  it("stops at the first other failure, or when a Job can't even start", async () => {
    const p = scripted({ mvs: failed("job.mvs.no_colmap") });
    expect((await runPipeline(all, p.run, p.say, p.now)).stopped?.kind).toBe("mvs");
    expect(p.ran).toEqual(["sfm", "georef", "mvs"]);
    const thrown = await runPipeline(
      all,
      async () => {
        throw { code: "job.busy", english: "busy" };
      },
      () => {},
    );
    expect(thrown.stopped).toEqual({ kind: "sfm", why: "failed", english: "busy" });
  });
});
