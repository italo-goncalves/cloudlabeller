// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

// Opening, making and closing a Project. A Project is a folder holding
// project.json; the user picks the folder in the core's own dialog.

import {
  addRecentProject,
  closeProject,
  isProblem,
  openProject,
  pickFolder,
} from "../desktop/bridge";
import { clearSurvey, type RawProjectState, view } from "../engine/api";
import type { App } from "../state/app";

async function adopt(app: App, root: string, create: boolean): Promise<boolean> {
  try {
    clearSurvey();
    const project = view(
      (await openProject(root, create, app.onNote)) as unknown as RawProjectState,
    );
    app.dispatch({ type: "opened", project });
    app.dispatch({ type: "log", text: `Opened ${project.root}` });
    await addRecentProject(root).catch(() => undefined);
    return true;
  } catch (e) {
    if (isProblem(e) && e.code === "project.exists" && create) {
      if (
        await app.confirm(
          "Folder already has a project",
          "This folder already holds a CloudLabeller Project. Open it instead?",
          "Open it",
        )
      )
        return adopt(app, root, false);
      return false;
    }
    if (isProblem(e) && e.code === "project.not_found") {
      app.tell(
        "Not a project",
        `No CloudLabeller Project in:\n${root}\n\n(A Project folder holds a project.json file.)`,
      );
      return false;
    }
    app.fail(e, create ? "Couldn't make the Project" : "Couldn't open the Project");
    return false;
  }
}

/** Open a Project the user picks, or the one at `root` (a recent one). */
export async function openFlow(app: App, root?: string): Promise<boolean> {
  const path = root ?? (await pickFolder("Open a Project folder"));
  return path ? adopt(app, path, false) : false;
}

/** Make a Project in a folder the user picks (an empty one, ideally). */
export async function newFlow(app: App): Promise<boolean> {
  const path = await pickFolder("Choose a folder for the new Project");
  return path ? adopt(app, path, true) : false;
}

export async function closeFlow(app: App): Promise<void> {
  try {
    await closeProject();
    app.dispatch({ type: "closed" });
  } catch (e) {
    app.fail(e, "Couldn't close the Project");
  }
}
