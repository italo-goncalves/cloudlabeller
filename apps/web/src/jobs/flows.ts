// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

// What a menu item sets going: one Job or a short chain of them, with the
// follow-ups the Qt app made (georeferencing after SfM, a mesh after MVS)
// and the questions around them (photos already in the Project, COLMAP
// missing).

import { enginePath, isProblem, keepAwake, pickFolder } from "../desktop/bridge";
import * as api from "../engine/api";
import { createSurvey } from "../shell/surveys";
import type { App, JobOutcome } from "../state/app";
import { type PipelineOptions, runPipeline, stages } from "./pipeline";

/** The default mesh rebuilt after MVS: Poisson at the cloud's own density. */
export const AFTER_MVS_MESH = { method: "poisson", trim: 10, point_weight: 1 } as const;

/** Say a Job's failure, unless it was the user's own Stop. */
function failed(app: App, o: JobOutcome, title: string) {
  if (o.ok || o.error?.code === "job.cancelled") return;
  if (o.error?.code === "job.mvs.no_colmap") {
    app.dispatch({ type: "dialog", dialog: { name: "colmap" } });
    return;
  }
  app.tell(title, o.error?.english ?? "The Job failed; the log says why.");
}

/** Run a Job, telling the user if it can't start or fails. */
export async function job(
  app: App,
  kind: string,
  options: Record<string, unknown>,
  label: string,
): Promise<JobOutcome> {
  try {
    const o = await app.runJob(kind, options, label);
    failed(app, o, `${label} failed`);
    return o;
  } catch (e) {
    if (isProblem(e) && e.code === "job.mvs.no_colmap")
      app.dispatch({ type: "dialog", dialog: { name: "colmap" } });
    else app.fail(e, `${label} couldn't start`);
    return { ok: false, error: isProblem(e) ? e : { code: "job.call", english: String(e) } };
  }
}

/** Add photos from a folder the user picks. */
export async function importPhotos(app: App): Promise<void> {
  const folder = await pickFolder("Choose a folder of photos");
  if (!folder) return;
  const project = app.state.project;
  const current = project?.surveys.find((s) => s.id === project.survey);
  if (current && current.images.length > 0) {
    const answer = await app.choose(
      "Add photos",
      `Add these photos to ${current.name}, or start a new Survey with them (another flight or day)?`,
      ["New Survey", `Add to ${current.name}`],
    );
    if (!answer) return;
    if (answer === "New Survey") {
      const name = folder.split(/[/]/).filter(Boolean).pop() ?? "";
      if ((await createSurvey(app, { name })) === null) return;
    }
  }
  try {
    const where = await enginePath(folder);
    const { conflicts } = await api.ingestConflicts(where);
    let onConflict = "skip";
    if (conflicts.length > 0) {
      const shown =
        conflicts.slice(0, 8).join("\n") +
        (conflicts.length > 8 ? `\n… and ${conflicts.length - 8} more` : "");
      const answer = await app.choose(
        "Photos already in the Project",
        `${conflicts.length} of these photos have the same name as photos already in the Project:\n\n${shown}\n\nOverwrite them, or keep the ones already there?`,
        ["Keep existing", "Overwrite"],
      );
      if (!answer) return;
      onConflict = answer === "Overwrite" ? "overwrite" : "skip";
    }
    await job(app, "ingest", { folder: where, on_conflict: onConflict }, "Adding photos");
  } catch (e) {
    app.fail(e, "Couldn't add the photos");
  }
}

/** Georeference to EXIF GPS, reporting the fit; `quiet` leaves a failure to the caller. */
export async function georeference(app: App, quiet = false): Promise<JobOutcome> {
  const o = quiet
    ? await app
        .runJob("georef", {}, "Georeferencing")
        .catch((e) => ({ ok: false, error: e }) as JobOutcome)
    : await job(app, "georef", {}, "Georeferencing");
  if (o.ok && !quiet) app.tell("Georeferenced", fitReport(o.result));
  return o;
}

/** The fit's report, as the Qt app gave it. */
export function fitReport(result: unknown): string {
  const r = (result ?? {}) as {
    scale_m_per_unit?: number;
    origin_lla?: [number, number, number];
    n_gps?: number;
    n_inliers?: number;
    fit_rms_m?: number;
  };
  const lines = ["The model is now in a local East-North-Up frame, in metres."];
  if (r.scale_m_per_unit !== undefined)
    lines.push(`Scale: ${r.scale_m_per_unit.toFixed(4)} m per model unit`);
  if (r.origin_lla)
    lines.push(
      `Origin: ${r.origin_lla[0].toFixed(6)}°, ${r.origin_lla[1].toFixed(6)}°, ${r.origin_lla[2].toFixed(1)} m`,
    );
  if (r.n_inliers !== undefined && r.n_gps !== undefined)
    lines.push(`GPS photos used: ${r.n_inliers} of ${r.n_gps}`);
  if (r.fit_rms_m !== undefined) lines.push(`Fit RMS: ${r.fit_rms_m.toFixed(2)} m`);
  return lines.join("\n");
}

export async function sfm(app: App, options: Record<string, unknown>, thenGeoref: boolean) {
  const o = await job(app, "sfm", options, "Structure from Motion");
  if (o.ok && thenGeoref) {
    const g = await georeference(app, true);
    if (g.ok) app.tell("SfM done, georeferenced", fitReport(g.result));
    else if (g.error?.code !== "job.cancelled")
      app.tell(
        "SfM done; georeferencing failed",
        `${g.error?.english ?? ""}\n\nThe model stays in its local frame.`,
      );
  }
}

export async function mvs(app: App, options: Record<string, unknown>) {
  const o = await job(app, "mvs", options, "Dense reconstruction (MVS)");
  if (o.ok) await job(app, "mesh", { ...AFTER_MVS_MESH }, "Mesh");
}

export async function mesh(app: App, options: Record<string, unknown>) {
  await job(app, "mesh", options, "Mesh");
}

/** SfM to mesh unattended, the computer kept awake. */
export async function pipeline(app: App, options: PipelineOptions) {
  await keepAwake(true).catch(() => undefined);
  try {
    const end = await runPipeline(
      stages(options),
      (s) => app.runJob(s.kind, s.options, s.label),
      (line) => app.dispatch({ type: "log", text: line }),
    );
    if (end.stopped?.why === "failed") {
      if (end.stopped.english.includes("COLMAP") || end.stopped.english.includes("colmap"))
        app.dispatch({ type: "dialog", dialog: { name: "colmap" } });
      else app.tell("The pipeline stopped", `${end.stopped.kind}: ${end.stopped.english}`);
    } else if (!end.stopped)
      app.tell(
        "The pipeline is done",
        `Finished in ${end.minutes.toFixed(1)} min.${end.local ? "\nGeoreferencing failed, so the model is in its local frame." : ""}`,
      );
  } finally {
    await keepAwake(false).catch(() => undefined);
  }
}

/** Images → Cloud, asking whether automatic labels count too. */
export async function imagesToCloud(app: App) {
  const images = app.state.project?.images ?? [];
  const user = images.filter((i) => i.status === "user").length;
  const auto = images.filter((i) => i.status === "auto" || i.status === "ml").length;
  if (user + auto === 0) {
    app.tell(
      "Nothing to transfer",
      "No photo has labels yet. Paint some, or project the cloud's labels onto the photos first.",
    );
    return;
  }
  let includeAuto = false;
  if (auto > 0) {
    const answer = await app.choose(
      "Images → Cloud",
      `${user} photos are labelled by you and ${auto} automatically (from the cloud or the model). Your labels always outrank automatic ones. Include the automatic ones?`,
      ["Mine only", "Include automatic"],
    );
    if (!answer) return;
    includeAuto = answer === "Include automatic";
  }
  try {
    const r = await app.long("Images → Cloud", (t) => api.imagesToCloud(includeAuto, t));
    app.dispatch({
      type: "log",
      text: `Images → Cloud: ${r.images} photos voted; ${r.sparse} sparse${r.dense !== undefined ? ` and ${r.dense} dense` : ""} points updated`,
    });
  } catch (e) {
    app.fail(e, "Images → Cloud failed");
  }
}

export async function cloudToImages(app: App) {
  try {
    const r = await app.long("Cloud → Images", (t) => api.cloudToImages(t));
    app.dispatch({
      type: "log",
      text: `Cloud → Images: ${r.auto_labelled} photos labelled from the ${r.cloud} cloud`,
    });
  } catch (e) {
    app.fail(e, "Cloud → Images failed");
  }
}

export async function undo(app: App, back: boolean) {
  try {
    const r = back ? await api.undo() : await api.redo();
    if (!r.changed) return;
    app.dispatch(
      r.changed.modality === "cloud"
        ? { type: "changed", parts: ["labels"] }
        : { type: "changed", parts: ["images"], images: r.changed.image ? [r.changed.image] : [] },
    );
    await app.refresh();
  } catch (e) {
    app.fail(e, back ? "Couldn't undo" : "Couldn't redo");
  }
}
