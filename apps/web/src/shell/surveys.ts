// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

// Moving between a Project's Surveys, and the questions around making and
// deleting one. The window shows one Survey at a time (`api.setSurvey`).

import * as api from "../engine/api";
import type { App } from "../state/app";

/** Show Survey `id`: its photos, its active Reconstruction, its labels. */
export async function switchSurvey(app: App, id: string): Promise<void> {
  if (app.state.busy) {
    app.tell("Busy", "Wait for the running work to finish before changing Surveys.");
    return;
  }
  try {
    api.setSurvey(id);
    app.dispatch({ type: "switched", project: await api.state() });
  } catch (e) {
    app.fail(e, "Couldn't open the Survey");
  }
}

/** Make a Survey and show it; its name and the rest come from `meta`. */
export async function createSurvey(app: App, meta: api.SurveyMeta): Promise<string | null> {
  try {
    const made = await api.surveyCreate(meta);
    await switchSurvey(app, made.id);
    app.dispatch({ type: "log", text: `New Survey: ${made.name}` });
    return made.id;
  } catch (e) {
    app.fail(e, "Couldn't make the Survey");
    return null;
  }
}

/** Delete the current Survey, after asking. */
export async function deleteSurvey(app: App): Promise<void> {
  const project = app.state.project;
  if (!project) return;
  const survey = project.surveys.find((s) => s.id === project.survey);
  if (!survey) return;
  if (project.surveys.length < 2) {
    app.tell("Delete Survey", "A Project keeps at least one Survey.");
    return;
  }
  const ok = await app.confirm(
    `Delete ${survey.name}?`,
    `This deletes the Survey's ${survey.images.length} photos, its masks, predictions and ` +
      `all ${survey.reconstructions} of its Reconstructions from the Project folder. ` +
      "The classes, GCPs and trained models stay. It can't be undone.",
    "Delete",
  );
  if (!ok) return;
  try {
    const out = await api.surveyDelete(survey.id);
    const next = out.surveys[0];
    if (next) await switchSurvey(app, next);
    app.dispatch({
      type: "log",
      text: `Deleted ${survey.name}: ${api.sizeText(out.freed_bytes)} freed`,
    });
  } catch (e) {
    app.fail(e, "Couldn't delete the Survey");
  }
}

/** "2026-11-20T10:15:00" as the window shows it. */
export function whenText(timestamp: string | null): string {
  if (!timestamp) return "";
  return timestamp.replace("T", " ").slice(0, 16);
}
