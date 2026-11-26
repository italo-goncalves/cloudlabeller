// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

// The photos: a status dot each (green: drawn by the user; yellow: from the
// cloud or the model; red: none), and where a solved photo's camera stood.
// Choosing one shows it in the Image pane and highlights its frustum.

import { cn } from "@geokit/ui";
import { useEffect, useRef, useState } from "react";
import * as api from "../engine/api";
import { deleteSurvey, switchSurvey, whenText } from "../shell/surveys";
import { useApp } from "../state/app";
import { PaneTitle } from "../ui/controls";

export const STATUS_COLOUR: Record<api.Status, string> = {
  user: "#3fb950",
  auto: "#d29922",
  ml: "#d29922",
  none: "#f85149",
};

const STATUS_WORDS: Record<api.Status, string> = {
  user: "labelled by you",
  auto: "labelled from the cloud",
  ml: "labelled by the model",
  none: "not labelled",
};

function fmt(p: [number, number, number]) {
  return p.map((v) => v.toFixed(2)).join(", ");
}

export function DatasetPane() {
  const app = useApp();
  const { project, selected, geometryVersion } = app.state;
  const [where, setWhere] = useState<Map<string, [number, number, number]>>(new Map());
  const list = useRef<HTMLUListElement>(null);

  // biome-ignore lint/correctness/useExhaustiveDependencies: cameras move only when the reconstruction does
  useEffect(() => {
    if (!project?.images.some((i) => i.solved)) {
      setWhere(new Map());
      return;
    }
    let live = true;
    api
      .cameras()
      .then(
        (r) =>
          live && setWhere(new Map(r.cameras.map((c) => [c.name, c.position_crs ?? c.position]))),
      )
      .catch(() => live && setWhere(new Map()));
    return () => {
      live = false;
    };
  }, [geometryVersion, project?.reconstruction]);

  // biome-ignore lint/correctness/useExhaustiveDependencies: scroll to the photo chosen
  useEffect(() => {
    list.current?.querySelector('[data-selected="true"]')?.scrollIntoView({ block: "nearest" });
  }, [selected]);

  if (!project) return null;
  const solved = project.images.filter((i) => i.solved).length;
  return (
    <div className="flex min-h-0 flex-col">
      <PaneTitle
        right={
          <span className="font-normal text-fg-faint">{`${project.images.length} photos, ${solved} solved`}</span>
        }
      >
        Dataset
      </PaneTitle>
      <SurveyLine />
      <ul ref={list} className="min-h-0 flex-1 overflow-auto py-1" aria-label="Photos">
        {project.images.map((im) => {
          const p = where.get(im.name);
          return (
            <li key={im.name} data-selected={im.name === selected}>
              <button
                type="button"
                className={cn(
                  "flex w-full items-center gap-2 px-3 py-1 text-left hover:bg-surface-2",
                  im.name === selected && "bg-selection",
                )}
                title={`${im.name}: ${STATUS_WORDS[im.status]}${im.solved ? "" : ", not solved"}`}
                onClick={() => app.dispatch({ type: "select_image", name: im.name })}
              >
                <span
                  className="size-2.5 shrink-0 rounded-full"
                  style={{ background: STATUS_COLOUR[im.status] }}
                />
                <span className={cn("min-w-0 flex-1 truncate", !im.solved && "text-fg-faint")}>
                  {im.name}
                </span>
                {p && <span className="shrink-0 font-mono text-table text-fg-faint">{fmt(p)}</span>}
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

const LINK = "rounded px-1.5 text-fg-muted hover:bg-surface-2 hover:text-fg disabled:opacity-40";

/** Which Survey is shown, and its Reconstruction: one compact line while
 * the Project has one Survey, a chooser once it has more. */
function SurveyLine() {
  const app = useApp();
  const { project, busy } = app.state;
  if (!project) return null;
  const current = project.surveys.find((s) => s.id === project.survey);
  const many = project.surveys.length > 1;
  const dialog = (name: "survey" | "reconstructions", data?: Record<string, unknown>) =>
    app.dispatch({ type: "dialog", dialog: { name, ...(data ? { data } : {}) } });
  return (
    <div className="flex flex-col gap-1 border-b border-border px-3 py-1.5">
      <div className="flex items-center gap-1">
        {many ? (
          <select
            aria-label="Survey"
            className="h-control min-w-0 flex-1 rounded-md border border-border bg-surface-0 px-1"
            value={project.survey}
            disabled={busy !== null}
            onChange={(e) => void switchSurvey(app, e.target.value)}
          >
            {project.surveys.map((s) => (
              <option key={s.id} value={s.id}>
                {s.name}
                {s.timestamp ? ` · ${whenText(s.timestamp)}` : ""}
              </option>
            ))}
          </select>
        ) : (
          <span className="min-w-0 flex-1 truncate" title={whenText(current?.timestamp ?? null)}>
            {current?.name}
          </span>
        )}
        <button
          type="button"
          className={LINK}
          title="Survey details"
          onClick={() => dialog("survey")}
        >
          Details
        </button>
        <button
          type="button"
          className={LINK}
          title="New Survey"
          disabled={busy !== null}
          onClick={() => dialog("survey", { create: true })}
        >
          New
        </button>
        {many && (
          <button
            type="button"
            className={LINK}
            title="Delete this Survey"
            disabled={busy !== null}
            onClick={() => void deleteSurvey(app)}
          >
            Delete
          </button>
        )}
      </div>
      {current && current.reconstructions > 0 && (
        <button
          type="button"
          className="truncate text-left text-fg-faint hover:text-fg"
          onClick={() => dialog("reconstructions")}
        >
          {`Reconstruction ${current.active?.slice(1) ?? "—"} of ${current.reconstructions}${current.aligned ? ", aligned" : ""}`}
        </button>
      )}
    </div>
  );
}
