// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

// The photos: a status dot each (green: drawn by the user; yellow: from the
// cloud or the model; red: none), and where a solved photo's camera stood.
// Choosing one shows it in the Image pane and highlights its frustum.

import { cn } from "@geokit/ui";
import { useEffect, useRef, useState } from "react";
import * as api from "../engine/api";
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
