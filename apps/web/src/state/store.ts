// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

// What the window knows, in one reducer: the open Project, what the user is
// working on (class, photo, selection), how the viewer draws, the running
// Job or long request, the log, and which dialog is open. Pure, so tested.

import type { Kept } from "../desktop/bridge";
import type { CloudName, ProjectState } from "../engine/api";

export type Representation = "sparse" | "dense" | "mesh";
/** How thinned the cloud is on screen (see `detail.ts`'s `VIEW_DETAIL`). */
export type ViewDetail = "low" | "medium" | "high" | "full";
export type ImageTool = "pan" | "polygon" | "brush" | "eraser";

export interface ViewSettings {
  representation: Representation;
  /** RGB → class colour, 0–1. */
  blend: number;
  pointSize: number;
  detail: ViewDetail;
  showCameras: boolean;
  /** Frustum size as a fraction of the base (5% of the cloud's diagonal). */
  frustumScale: number;
  /** The class shown on its own (others dimmed), or null. */
  isolate: number | null;
  lasso: boolean;
}

export interface Panes {
  labels: boolean;
  view: boolean;
  dataset: boolean;
  image: boolean;
  log: boolean;
}

export interface Busy {
  /** What the status bar says. */
  label: string;
  /** A Job's plan number, or an Engine request's ticket. */
  plan?: number;
  ticket?: string;
  fraction: number | null;
  message: string;
  /** Place in the GPU queue while waiting for the Lease. */
  waiting: number | null;
}

export interface LogLine {
  n: number;
  text: string;
  err: boolean;
}

export type DialogName =
  | "welcome"
  | "new_project"
  | "import"
  | "sfm"
  | "mvs"
  | "mesh"
  | "pipeline"
  | "clean"
  | "colmap"
  | "georef"
  | "reproject"
  | "export_cloud"
  | "export_mesh"
  | "transfer_to_cloud"
  | "model_spec"
  | "train"
  | "predict"
  | "project_info"
  | "about"
  | "confirm"
  | "message";

export interface Dialog {
  name: DialogName;
  /** What the dialog needs beyond the state: a message, a question, a callback. */
  data?: Record<string, unknown>;
}

export interface AppState {
  kept: Kept | null;
  project: ProjectState | null;
  activeClass: number;
  /** The photo shown in the Image pane (and highlighted among the frusta). */
  selected: string | null;
  view: ViewSettings;
  tool: ImageTool;
  brushSize: number;
  overlayOpacity: number;
  /** Points the lasso took, by full-cloud index. */
  selection: { cloud: CloudName; indices: Uint32Array } | null;
  /** Sparse points the cleaning would remove, previewed. */
  cleanPreview: Uint32Array | null;
  /** Bumped when the classes of points or vertices change: the viewer re-reads them. */
  labelsVersion: number;
  /** Bumped when a cloud or the mesh changes shape: the viewer reloads it. */
  geometryVersion: number;
  /** Bumped when photos' masks change, with the photos named (empty: any). */
  masks: { version: number; images: string[] };
  busy: Busy | null;
  log: LogLine[];
  dialog: Dialog | null;
  panes: Panes;
}

export const MAX_LOG = 5000;

export const initialView: ViewSettings = {
  representation: "sparse",
  blend: 0.6,
  pointSize: 2,
  detail: "high",
  showCameras: true,
  frustumScale: 0.2,
  isolate: null,
  lasso: false,
};

export const initialState: AppState = {
  kept: null,
  project: null,
  activeClass: 0,
  selected: null,
  view: initialView,
  tool: "pan",
  brushSize: 24,
  overlayOpacity: 0.5,
  selection: null,
  cleanPreview: null,
  labelsVersion: 0,
  geometryVersion: 0,
  masks: { version: 0, images: [] },
  busy: null,
  log: [],
  dialog: null,
  panes: { labels: true, view: true, dataset: true, image: true, log: true },
};

export type Action =
  | { type: "kept"; kept: Kept }
  | { type: "opened"; project: ProjectState }
  | { type: "closed" }
  | { type: "project"; project: ProjectState }
  /** The Engine's `project.changed`. */
  | { type: "changed"; parts: string[]; images?: string[] }
  | { type: "active_class"; id: number }
  | { type: "select_image"; name: string | null }
  | { type: "step_image"; by: 1 | -1 }
  | { type: "view"; patch: Partial<ViewSettings> }
  | { type: "tool"; tool: ImageTool }
  | { type: "brush_size"; size: number }
  | { type: "overlay"; opacity: number }
  | { type: "selection"; selection: AppState["selection"] }
  | { type: "clean_preview"; removed: Uint32Array | null }
  | { type: "busy"; busy: Busy | null }
  | { type: "progress"; fraction: number | null; message: string }
  | { type: "waiting"; place: number | null }
  | { type: "log"; text: string; err?: boolean }
  | { type: "clear_log" }
  | { type: "dialog"; dialog: Dialog | null }
  | { type: "pane"; pane: keyof Panes; shown?: boolean };

/** The representation to show: the one asked for if the Project has it, else the best it has. */
export function available(project: ProjectState | null, want: Representation): Representation {
  const has = {
    sparse: (project?.sparse ?? 0) > 0,
    dense: (project?.dense ?? 0) > 0,
    mesh: project?.mesh != null,
  };
  if (has[want]) return want;
  return has.sparse ? "sparse" : has.dense ? "dense" : has.mesh ? "mesh" : "sparse";
}

/** Whether a class id names a class of the Project (or is −1, unlabelled). */
const inSchema = (project: ProjectState | null, id: number) =>
  id === -1 || (project?.schema.some((c) => c.id === id) ?? false);

let logN = 0;

export function reduce(s: AppState, a: Action): AppState {
  switch (a.type) {
    case "kept":
      return { ...s, kept: a.kept };
    case "opened":
      return {
        ...initialState,
        kept: s.kept,
        panes: s.panes,
        log: s.log,
        view: {
          ...s.view,
          representation: available(a.project, "dense"),
          isolate: null,
          lasso: false,
        },
        project: a.project,
        activeClass: a.project.schema[0]?.id ?? -1,
        selected: a.project.images[0]?.name ?? null,
        labelsVersion: s.labelsVersion + 1,
        geometryVersion: s.geometryVersion + 1,
        masks: { version: s.masks.version + 1, images: [] },
      };
    case "closed":
      return { ...initialState, kept: s.kept, panes: s.panes, log: s.log };
    case "project": {
      const project = a.project;
      const selected =
        s.selected && project.images.some((i) => i.name === s.selected)
          ? s.selected
          : (project.images[0]?.name ?? null);
      return {
        ...s,
        project,
        selected,
        activeClass: inSchema(project, s.activeClass)
          ? s.activeClass
          : (project.schema[0]?.id ?? -1),
        view: {
          ...s.view,
          representation: available(project, s.view.representation),
          isolate:
            s.view.isolate !== null && inSchema(project, s.view.isolate) ? s.view.isolate : null,
        },
      };
    }
    case "changed": {
      const geometry = a.parts.some((p) => p === "sparse" || p === "dense" || p === "mesh");
      const labels = geometry || a.parts.includes("labels") || a.parts.includes("schema");
      // a photo's own labels changed (an undo of a mask edit names it)
      const namedLabels = a.parts.includes("labels") && (a.images?.length ?? 0) > 0;
      const masks =
        namedLabels ||
        a.parts.includes("images") ||
        a.parts.includes("status") ||
        a.parts.includes("schema");
      return {
        ...s,
        geometryVersion: geometry ? s.geometryVersion + 1 : s.geometryVersion,
        labelsVersion: labels ? s.labelsVersion + 1 : s.labelsVersion,
        masks: masks ? { version: s.masks.version + 1, images: a.images ?? [] } : s.masks,
        // a cloud that changed shape: its old indices mean nothing
        selection: geometry ? null : s.selection,
        cleanPreview: geometry ? null : s.cleanPreview,
      };
    }
    case "active_class":
      return inSchema(s.project, a.id) ? { ...s, activeClass: a.id } : s;
    case "select_image":
      return { ...s, selected: a.name };
    case "step_image": {
      const images = s.project?.images ?? [];
      if (images.length === 0) return s;
      const i = images.findIndex((x) => x.name === s.selected);
      const j = Math.min(images.length - 1, Math.max(0, Math.max(i, 0) + a.by));
      return { ...s, selected: images[j]?.name ?? null };
    }
    case "view": {
      const view = { ...s.view, ...a.patch };
      if (a.patch.representation)
        view.representation = available(s.project, a.patch.representation);
      const representationChanged = view.representation !== s.view.representation;
      return {
        ...s,
        view: { ...view, lasso: view.representation === "mesh" ? false : view.lasso },
        selection: representationChanged ? null : s.selection,
      };
    }
    case "tool":
      return { ...s, tool: a.tool };
    case "brush_size":
      return { ...s, brushSize: Math.min(400, Math.max(1, Math.round(a.size))) };
    case "overlay":
      return { ...s, overlayOpacity: Math.min(1, Math.max(0, a.opacity)) };
    case "selection":
      return {
        ...s,
        selection: a.selection && a.selection.indices.length > 0 ? a.selection : null,
      };
    case "clean_preview":
      return { ...s, cleanPreview: a.removed };
    case "busy":
      return { ...s, busy: a.busy };
    case "progress":
      return s.busy
        ? { ...s, busy: { ...s.busy, fraction: a.fraction, message: a.message, waiting: null } }
        : s;
    case "waiting":
      return s.busy ? { ...s, busy: { ...s.busy, waiting: a.place } } : s;
    case "log": {
      const lines = a.text.split(/\r?\n/).map((text) => ({ n: logN++, text, err: a.err ?? false }));
      const log = s.log.concat(lines);
      return { ...s, log: log.length > MAX_LOG ? log.slice(log.length - MAX_LOG) : log };
    }
    case "clear_log":
      return { ...s, log: [] };
    case "dialog":
      return { ...s, dialog: a.dialog };
    case "pane":
      return { ...s, panes: { ...s.panes, [a.pane]: a.shown ?? !s.panes[a.pane] } };
  }
}
