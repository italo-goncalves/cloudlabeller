// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

// The full pipeline, which the window chains (the Engine runs one Job at a
// time): SfM, then georeferencing if asked, then MVS, then the mesh. Each
// stage is a Job; the chain stops at the first that fails or is stopped,
// except georeferencing, whose failure leaves the Project in its local frame
// and lets the chain go on (as the Qt app did).

import type { JobOutcome } from "../state/app";

export interface Stage {
  kind: "sfm" | "georef" | "mvs" | "mesh";
  label: string;
  options: Record<string, unknown>;
}

export interface PipelineOptions {
  sfm: Record<string, unknown>;
  georef: boolean;
  mvs: Record<string, unknown>;
  mesh: Record<string, unknown>;
}

export interface PipelineEnd {
  /** The stages that ended well, in order. */
  done: Stage["kind"][];
  /** Why the chain stopped early, or null when every stage ran. */
  stopped: { kind: Stage["kind"]; why: "cancelled" | "failed"; english: string } | null;
  /** Whether georeferencing failed and the Project kept its local frame. */
  local: boolean;
  minutes: number;
}

export function stages(o: PipelineOptions): Stage[] {
  const out: Stage[] = [{ kind: "sfm", label: "Structure from Motion", options: o.sfm }];
  if (o.georef) out.push({ kind: "georef", label: "Georeferencing to EXIF GPS", options: {} });
  out.push({ kind: "mvs", label: "Dense reconstruction (MVS)", options: o.mvs });
  out.push({ kind: "mesh", label: "Mesh", options: o.mesh });
  return out;
}

const cancelled = (o: JobOutcome) => o.error?.code === "job.cancelled";

/**
 * Run the stages in order through `run`, saying each start and end through
 * `say`. `now` is the clock, in ms.
 */
export async function runPipeline(
  list: Stage[],
  run: (stage: Stage) => Promise<JobOutcome>,
  say: (line: string) => void,
  now: () => number = () => Date.now(),
): Promise<PipelineEnd> {
  const t0 = now();
  const done: Stage["kind"][] = [];
  let local = false;
  const end = (stopped: PipelineEnd["stopped"]): PipelineEnd => {
    const minutes = (now() - t0) / 60_000;
    say(
      stopped
        ? `=== Pipeline stopped at ${stopped.kind} after ${minutes.toFixed(1)} min ===`
        : `=== Pipeline done in ${minutes.toFixed(1)} min ===`,
    );
    return { done, stopped, local, minutes };
  };
  for (const [i, stage] of list.entries()) {
    say(`=== Pipeline ${i + 1}/${list.length}: ${stage.label} ===`);
    let o: JobOutcome;
    try {
      o = await run(stage);
    } catch (e) {
      const english = typeof e === "object" && e && "english" in e ? String(e.english) : String(e);
      return end({ kind: stage.kind, why: "failed", english });
    }
    if (o.ok) {
      done.push(stage.kind);
      continue;
    }
    if (cancelled(o))
      return end({ kind: stage.kind, why: "cancelled", english: o.error?.english ?? "" });
    if (stage.kind === "georef") {
      local = true;
      say(
        `Georeferencing failed (${o.error?.english ?? "no reason given"}); carrying on in the local frame.`,
      );
      continue;
    }
    return end({ kind: stage.kind, why: "failed", english: o.error?.english ?? "" });
  }
  return end(null);
}
