// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

// The main window with a Project open: the menus, the panes (Labels, View
// and Dataset on the left, the 3D view in the middle, the photo on the
// right, the log below) and the status bar. Keys that work everywhere —
// undo, redo, the classes by number — are handled here.

import { getCurrentWindow } from "@tauri-apps/api/window";
import { useEffect, useMemo, useRef, useState } from "react";
import { engineCall } from "../desktop/bridge";
import { DialogHost } from "../dialogs/DialogHost";
import * as api from "../engine/api";
import { ImageView } from "../image/ImageView";
import * as flows from "../jobs/flows";
import { DatasetPane } from "../panes/DatasetPane";
import { LabelsPane } from "../panes/LabelsPane";
import { LogPane } from "../panes/LogPane";
import { useApp } from "../state/app";
import type { DialogName, Panes } from "../state/store";
import { Button, ProgressBar } from "../ui/controls";
import { Viewer3D } from "../viewer/Viewer3D";
import { ViewPane } from "../viewer/ViewPane";
import { type Menu, Menubar } from "./Menubar";
import { closeFlow, newFlow, openFlow } from "./projects";
import { Welcome } from "./Welcome";

const PANES_KEY = "cloudlabeller.panes";

function loadPanes(): Partial<Panes> {
  try {
    return JSON.parse(localStorage.getItem(PANES_KEY) ?? "{}") as Partial<Panes>;
  } catch {
    return {};
  }
}

/** Whether a key press belongs to a text field rather than to the window. */
function typing(e: KeyboardEvent): boolean {
  const t = e.target as HTMLElement | null;
  return (
    !!t &&
    (t.tagName === "INPUT" ||
      t.tagName === "TEXTAREA" ||
      t.tagName === "SELECT" ||
      t.isContentEditable)
  );
}

function useMenus(): Menu[] {
  const app = useApp();
  const { project, busy, panes } = app.state;
  const open = (name: DialogName) => () => app.dispatch({ type: "dialog", dialog: { name } });
  const has = !!project;
  const idle = !busy;
  const cloud = (project?.sparse ?? 0) > 0;
  const dense = (project?.dense ?? 0) > 0;
  // biome-ignore lint/correctness/useExhaustiveDependencies: rebuilt as the state they read changes
  return useMemo<Menu[]>(
    () => [
      {
        title: "File",
        items: [
          { label: "New Project…", run: () => void newFlow(app), disabled: !idle },
          { label: "Open Project…", run: () => void openFlow(app), disabled: !idle },
          { label: "Close Project", run: () => void closeFlow(app), disabled: !has || !idle },
          "separator",
          {
            label: "Add Images…",
            run: () => void flows.importPhotos(app),
            disabled: !has || !idle,
          },
          { label: "Project Info…", run: open("project_info"), disabled: !has },
          "separator",
          { label: "Export Point Cloud…", run: open("export_cloud"), disabled: !cloud && !dense },
          { label: "Export Mesh…", run: open("export_mesh"), disabled: !project?.mesh },
          "separator",
          { label: "Quit", run: () => void getCurrentWindow().close() },
        ],
      },
      {
        title: "Photogrammetry",
        items: [
          { label: "Run Full Pipeline…", run: open("pipeline"), disabled: !has || !idle },
          "separator",
          { label: "Run SfM…", run: open("sfm"), disabled: !has || !idle },
          { label: "Run Dense (MVS)…", run: open("mvs"), disabled: !cloud || !idle },
          { label: "Create Mesh…", run: open("mesh"), disabled: !dense || !idle },
          "separator",
          { label: "Clean Sparse Cloud…", run: open("clean"), disabled: !cloud || !idle },
          { label: "Download COLMAP…", run: open("colmap"), disabled: !idle },
        ],
      },
      {
        title: "Georeferencing",
        items: [
          {
            label: "Align to EXIF GPS…",
            run: async () => {
              if (project?.georeferenced) {
                app.tell(
                  "Already georeferenced",
                  "This Project is already georeferenced. To move it into a projected CRS, use Reproject to CRS.",
                );
                return;
              }
              if (
                await app.confirm(
                  "Align to EXIF GPS",
                  "This fits the model to the photos' GPS positions and puts it in a local East-North-Up frame, in metres, at the first photo. Every product (sparse, dense, mesh) moves with it.\n\nIf a dense cloud was made from only some of the photos, it moves too but may not match the new SfM exactly.",
                  "Align",
                )
              )
                await flows.georeference(app);
            },
            disabled: !cloud || !idle,
          },
          {
            label: "Reproject to CRS…",
            run: open("reproject"),
            disabled: !project?.georeferenced || !idle,
          },
        ],
      },
      {
        title: "Transfer",
        items: [
          {
            label: "Images → Cloud",
            run: () => void flows.imagesToCloud(app),
            disabled: !cloud || !idle,
          },
          {
            label: "Cloud → Images",
            run: () => void flows.cloudToImages(app),
            disabled: !cloud || !idle,
          },
        ],
      },
      {
        title: "Model",
        items: [
          { label: "Model Settings…", run: open("model_spec"), disabled: !has },
          { label: "Train U-Net…", run: open("train"), disabled: !has || !idle },
          { label: "Predict All Images…", run: open("predict"), disabled: !has || !idle },
        ],
      },
      {
        title: "Edit",
        items: [
          {
            label: "Undo",
            shortcut: "Ctrl+Z",
            run: () => void flows.undo(app, true),
            disabled: !project?.can_undo,
          },
          {
            label: "Redo",
            shortcut: "Ctrl+Y",
            run: () => void flows.undo(app, false),
            disabled: !project?.can_redo,
          },
        ],
      },
      {
        title: "Panes",
        items: (["labels", "view", "dataset", "image", "log"] as const).map((p) => ({
          label: { labels: "Labels", view: "View", dataset: "Dataset", image: "Image", log: "Log" }[
            p
          ],
          checked: panes[p],
          run: () => app.dispatch({ type: "pane", pane: p }),
        })),
      },
      { title: "Help", items: [{ label: "About CloudLabeller…", run: open("about") }] },
    ],
    [app, project, busy, panes],
  );
}

function StatusBar() {
  const app = useApp();
  const { project, busy } = app.state;
  const [frame, setFrame] = useState<{ short: string; full: string } | null>(null);
  // biome-ignore lint/correctness/useExhaustiveDependencies: the frame changes with georeferencing
  useEffect(() => {
    if (!project) return;
    api
      .crsFrame()
      .then(setFrame)
      .catch(() => setFrame(null));
  }, [project?.georeferenced, project?.root, app.state.geometryVersion]);
  if (!project) return null;
  const solved = project.images.filter((i) => i.solved).length;
  const points = project.dense ?? project.sparse;
  const words = {
    none: "not reconstructed",
    current: "reconstruction current",
    outdated: "reconstruction outdated",
  }[project.reconstruction];
  return (
    <div className="flex h-7 items-center gap-3 border-t border-border bg-surface-1 px-3 text-table text-fg-muted">
      {frame && <span title={frame.full}>{frame.short}</span>}
      <button
        type="button"
        className="truncate hover:text-fg"
        title="Project Info"
        onClick={() => app.dispatch({ type: "dialog", dialog: { name: "project_info" } })}
      >
        {project.root} · {project.images.length} photos · {solved} cameras
        {points ? ` · ${points.toLocaleString("en")} points` : ""} · {words}
      </button>
      <span className="flex-1" />
      {busy && (
        <>
          <span className="truncate text-fg">
            {busy.label}
            {busy.waiting !== null
              ? ` — waiting for the GPU (place ${busy.waiting})`
              : busy.message
                ? ` — ${busy.message}`
                : ""}
          </span>
          <ProgressBar fraction={busy.fraction} />
          <Button kind="quiet" className="h-6" onClick={app.stop}>
            Stop
          </Button>
        </>
      )}
    </div>
  );
}

export function Window() {
  const app = useApp();
  const { project, panes, dialog } = app.state;
  const menus = useMenus();
  const appRef = useRef(app);
  appRef.current = app;

  // the panes the user had shown last time
  // biome-ignore lint/correctness/useExhaustiveDependencies: once, on start
  useEffect(() => {
    const saved = loadPanes();
    for (const [k, v] of Object.entries(saved))
      app.dispatch({ type: "pane", pane: k as keyof Panes, shown: Boolean(v) });
  }, []);
  useEffect(() => {
    try {
      localStorage.setItem(PANES_KEY, JSON.stringify(panes));
    } catch {
      // a window without storage keeps its panes for the session only
    }
  }, [panes]);

  // keys for the whole window
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      const a = appRef.current;
      if (typing(e) || a.state.dialog || !a.state.project) return;
      const k = e.key.toLowerCase();
      if (e.ctrlKey && !e.altKey && (k === "z" || k === "y")) {
        e.preventDefault();
        void flows.undo(a, k === "z" && !e.shiftKey);
        return;
      }
      if (e.ctrlKey || e.altKey || e.metaKey) return;
      if (k === "0") a.dispatch({ type: "active_class", id: -1 });
      const byKey = a.state.project.schema.find((c) => c.hotkey === e.key);
      if (byKey) a.dispatch({ type: "active_class", id: byKey.id });
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, []);

  // closing the window: ask while something runs, then close the Project
  useEffect(() => {
    let off: (() => void) | undefined;
    getCurrentWindow()
      .onCloseRequested(async (e) => {
        const a = appRef.current;
        let running = !!a.state.busy;
        if (!running && a.state.project) {
          const jobs = await engineCall<{ jobs: unknown[] }>("jobs.list").catch(() => ({
            jobs: [],
          }));
          running = jobs.jobs.length > 0;
        }
        if (
          running &&
          !(await a.confirm(
            "Something is still running",
            "A Job is still running. Quit anyway? It will be stopped.",
            "Quit",
          ))
        ) {
          e.preventDefault();
        }
      })
      .then((f) => {
        off = f;
      })
      .catch(() => undefined);
    return () => off?.();
  }, []);

  return (
    <div className="flex h-full flex-col">
      <Menubar menus={menus} />
      {project ? (
        <div className="flex min-h-0 flex-1 flex-col">
          <div className="flex min-h-0 flex-1">
            {(panes.labels || panes.view || panes.dataset) && (
              <aside className="flex w-dock-wide shrink-0 flex-col border-r border-border bg-surface-1">
                {panes.labels && (
                  <section className="flex max-h-[34%] min-h-0 flex-col border-b border-border">
                    <LabelsPane />
                  </section>
                )}
                {panes.view && (
                  <section className="min-h-0 shrink-0 overflow-auto border-b border-border">
                    <ViewPane />
                  </section>
                )}
                {panes.dataset && (
                  <section className="flex min-h-0 flex-1 flex-col">
                    <DatasetPane />
                  </section>
                )}
              </aside>
            )}
            <main className="relative min-w-0 flex-1 bg-viewport">
              <Viewer3D />
            </main>
            {panes.image && (
              <aside className="flex w-[38%] min-w-80 shrink-0 flex-col border-l border-border bg-surface-1">
                <ImageView />
              </aside>
            )}
          </div>
          {panes.log && (
            <section className="h-44 shrink-0 border-t border-border bg-surface-1">
              <LogPane />
            </section>
          )}
        </div>
      ) : (
        <div className="min-h-0 flex-1">
          <Welcome />
        </div>
      )}
      <StatusBar />
      {dialog && <DialogHost />}
    </div>
  );
}
