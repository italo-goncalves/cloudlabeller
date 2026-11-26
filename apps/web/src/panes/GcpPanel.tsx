// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

// The GCPs: a table of them with where each is marked in the Survey shown
// and how far off the last fit left it. Choosing one makes it the GCP a
// click places, with the GCP tool of the Image view or of the View pane.

import { cn } from "@geokit/ui";
import { useEffect, useState } from "react";
import * as api from "../engine/api";
import { useApp } from "../state/app";
import { Button } from "../ui/controls";

function metres(d: [number, number, number] | undefined): string {
  if (!d) return "";
  const m = Math.hypot(d[0], d[1], d[2]);
  return m < 1 ? `${(m * 100).toFixed(1)} cm` : `${m.toFixed(2)} m`;
}

/** The GCPs, reloaded when they or their Marks change. */
export function useGcps(): api.Gcps | null {
  const { state, fail } = useApp();
  const [list, setList] = useState<api.Gcps | null>(null);
  const { gcpVersion, project } = state;
  // biome-ignore lint/correctness/useExhaustiveDependencies: reloaded on its version and Survey
  useEffect(() => {
    if (!project) return;
    let live = true;
    api
      .gcps()
      .then((g) => live && setList(g))
      .catch((e) => live && fail(e, "Couldn't read the GCPs"));
    return () => {
      live = false;
    };
  }, [gcpVersion, project?.survey, project?.root, fail]);
  return list;
}

export function GcpPanel() {
  const app = useApp();
  const { project, activeGcp, busy } = app.state;
  const list = useGcps();
  if (!project) return null;
  const sid = project.survey;
  const gcps = list?.gcps ?? [];
  const fit = list?.fit?.surveys[sid];
  const open = (name: "gcp" | "gcp_import", data?: Record<string, unknown>) =>
    app.dispatch({ type: "dialog", dialog: { name, ...(data ? { data } : {}) } });

  const add = async () => {
    try {
      const out = await api.gcpCreate();
      app.dispatch({ type: "active_gcp", id: out.id });
    } catch (e) {
      app.fail(e, "Couldn't add the GCP");
    }
  };
  const remove = async (g: api.Gcp) => {
    if (!(await app.confirm(`Delete ${g.name}?`, "Its Marks in every Survey go too.", "Delete")))
      return;
    try {
      await api.gcpDelete(g.id);
      if (activeGcp === g.id) app.dispatch({ type: "active_gcp", id: null });
    } catch (e) {
      app.fail(e, "Couldn't delete the GCP");
    }
  };
  const align = async () => {
    try {
      const out = await app.long("Aligning to the GCPs", (t) => api.alignFit(t));
      const lines = project.surveys.map((s) => {
        const r = out.surveys[s.id];
        if (!r) return `${s.name}: no Reconstruction`;
        if (r.how === "gcp")
          return `${s.name}: aligned to ${r.n_gcps} GCPs, ${r.rms === null ? "" : `${metres([r.rms, 0, 0])} RMS`}`;
        if (r.how === "reference") return `${s.name}: the reference, kept as it was`;
        return `${s.name}: ${r.n_gcps} GCP${r.n_gcps === 1 ? "" : "s"} usable, 3 needed; ${r.aligned ? "kept its alignment" : "not aligned"}`;
      });
      app.tell(
        "Aligned to the GCPs",
        (out.frame_created ? "The Project frame now comes from the GCPs' CRS.\n\n" : "") +
          lines.join("\n"),
      );
    } catch (e) {
      app.fail(e, "Couldn't align to the GCPs");
    }
  };

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex flex-wrap gap-1 border-b border-border px-2 py-1.5">
        <Button onClick={() => void add()}>Add</Button>
        <Button onClick={() => open("gcp_import")}>Import…</Button>
        <Button
          kind="primary"
          disabled={gcps.length === 0 || busy !== null}
          onClick={() => void align()}
        >
          Align
        </Button>
      </div>
      {fit && (
        <p className="border-b border-border px-3 py-1 text-fg-faint">
          {fit.how === "gcp"
            ? `This Survey is aligned to ${fit.n_gcps} GCPs${fit.rms === null ? "" : `, ${metres([fit.rms, 0, 0])} RMS`}.`
            : fit.how === "reference"
              ? "This Survey is the reference the others align to."
              : `This Survey sees ${fit.n_gcps} usable GCPs; 3 are needed.`}
        </p>
      )}
      {gcps.length === 0 ? (
        <p className="px-3 py-2 text-fg-faint">
          No GCPs yet. Import their coordinates, or add them and mark them on the photos and the
          cloud: Surveys that share three GCPs line up.
        </p>
      ) : (
        <ul className="min-h-0 flex-1 overflow-auto py-1" aria-label="GCPs">
          {gcps.map((g) => {
            const here = g.marks[sid];
            const res = list?.fit?.residuals[g.id]?.[sid];
            const marked = here
              ? [
                  here.cloud ? "cloud" : "",
                  here.photos.length ? `${here.photos.length} photos` : "",
                ]
                  .filter(Boolean)
                  .join(", ")
              : "not marked";
            return (
              <li key={g.id}>
                <div
                  className={cn(
                    "flex items-center gap-2 px-3 py-1 hover:bg-surface-2",
                    g.id === activeGcp && "bg-selection",
                  )}
                >
                  <button
                    type="button"
                    className="flex min-w-0 flex-1 items-center gap-2 text-left"
                    title="Choose it: the GCP tools place its Marks"
                    onClick={() =>
                      app.dispatch({ type: "active_gcp", id: g.id === activeGcp ? null : g.id })
                    }
                  >
                    <span
                      className={cn(
                        "size-2.5 shrink-0 rotate-45",
                        g.use === "check" ? "border border-fg-muted" : "bg-accent",
                      )}
                    />
                    <span className="min-w-0 flex-1 truncate">
                      {g.name}
                      {!g.xyz && <span className="text-fg-faint"> (no coordinates)</span>}
                    </span>
                    <span className="shrink-0 text-table text-fg-faint">{marked}</span>
                    {res && (
                      <span className="w-16 shrink-0 text-right font-mono text-table text-fg-muted">
                        {metres(res)}
                      </span>
                    )}
                  </button>
                  <button
                    type="button"
                    className="rounded px-1 text-fg-faint hover:bg-surface-2 hover:text-fg"
                    title="Edit"
                    onClick={() => open("gcp", { id: g.id })}
                  >
                    Edit
                  </button>
                  <button
                    type="button"
                    className="rounded px-1 text-fg-faint hover:bg-surface-2 hover:text-danger"
                    title="Delete"
                    onClick={() => void remove(g)}
                  >
                    ✕
                  </button>
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
