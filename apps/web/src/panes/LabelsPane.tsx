// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

// The classes: the unlabelled row, then each class with its colour, id,
// name and share of the cloud. Clicking a row makes it the class painted;
// double-clicking a name renames it.

import { cn } from "@geokit/ui";
import { useEffect, useState } from "react";
import * as api from "../engine/api";
import { useApp } from "../state/app";
import { Button, PaneTitle } from "../ui/controls";
import { countClasses, share } from "./counts";

export function LabelsPane() {
  const app = useApp();
  const { project, activeClass, labelsVersion, view } = app.state;
  const [counts, setCounts] = useState<{ counts: Map<number, number>; total: number } | null>(null);
  const [renaming, setRenaming] = useState<{ id: number; name: string } | null>(null);
  const cloud: api.CloudName | null = project?.dense ? "dense" : project?.sparse ? "sparse" : null;

  // biome-ignore lint/correctness/useExhaustiveDependencies: re-read when the labels change
  useEffect(() => {
    if (!cloud) {
      setCounts(null);
      return;
    }
    let live = true;
    api
      .cloudLabels(cloud)
      .then((l) => live && setCounts(countClasses(l)))
      .catch(() => live && setCounts(null));
    return () => {
      live = false;
    };
  }, [cloud, labelsVersion]);

  if (!project) return null;
  const edit = async (f: () => Promise<unknown>, title: string) => {
    try {
      await f();
      await app.refresh();
    } catch (e) {
      app.fail(e, title);
    }
  };
  const remove = async () => {
    const c = project.schema.find((x) => x.id === activeClass);
    if (!c) return;
    if (
      await app.confirm(
        "Remove the class",
        `Remove "${c.name}"? Its points and pixels become unlabelled, the classes after it move down by one, and the undo history is cleared.`,
        "Remove",
      )
    )
      await edit(() => api.schemaDelete(c.id), "Couldn't remove the class");
  };
  const total = counts?.total ?? 0;
  const row = "flex h-8 cursor-pointer items-center gap-2 px-3 hover:bg-surface-2";

  return (
    <div className="flex min-h-0 flex-col">
      <PaneTitle right={cloud && <span className="font-normal text-fg-faint">{cloud} cloud</span>}>
        Labels
      </PaneTitle>
      <div className="min-h-0 flex-1 overflow-auto py-1">
        {/* biome-ignore lint/a11y/useSemanticElements: a row holds its own inputs, which a <button> can't */}
        <div
          className={cn(row, activeClass === -1 && "bg-selection")}
          onClick={() => app.dispatch({ type: "active_class", id: -1 })}
          onKeyDown={(e) => e.key === "Enter" && app.dispatch({ type: "active_class", id: -1 })}
          role="button"
          tabIndex={0}
        >
          <span className="size-4 rounded-sm border border-border-strong" />
          <span className="w-5 text-fg-faint">0</span>
          <span className="flex-1 text-fg-muted">unlabelled (erase)</span>
          <span className="text-table text-fg-faint">
            {share(counts?.counts.get(-1) ?? 0, total)}
          </span>
        </div>
        {project.schema.map((c) => (
          // biome-ignore lint/a11y/useSemanticElements: a row holds its own inputs, which a <button> can't
          <div
            key={c.id}
            className={cn(row, activeClass === c.id && "bg-selection")}
            onClick={() => app.dispatch({ type: "active_class", id: c.id })}
            onKeyDown={(e) => e.key === "Enter" && app.dispatch({ type: "active_class", id: c.id })}
            role="button"
            tabIndex={0}
          >
            <input
              type="color"
              aria-label={`${c.name}'s colour`}
              className="size-4 cursor-pointer rounded-sm border-0 bg-transparent p-0"
              value={c.color}
              onClick={(e) => e.stopPropagation()}
              onChange={(e) =>
                edit(() => api.schemaSetColor(c.id, e.target.value), "Couldn't change the colour")
              }
            />
            <span className="w-5 text-fg-faint">{c.hotkey ?? ""}</span>
            {renaming?.id === c.id ? (
              <input
                // biome-ignore lint/a11y/noAutofocus: the rename field takes the focus it was opened for
                autoFocus
                className="h-6 flex-1 rounded border border-accent bg-surface-0 px-1"
                value={renaming.name}
                onChange={(e) => setRenaming({ id: c.id, name: e.target.value })}
                onClick={(e) => e.stopPropagation()}
                onKeyDown={(e) => {
                  e.stopPropagation();
                  if (e.key === "Enter" && renaming.name.trim()) {
                    const name = renaming.name.trim();
                    setRenaming(null);
                    void edit(() => api.schemaRename(c.id, name), "Couldn't rename the class");
                  } else if (e.key === "Escape") setRenaming(null);
                }}
                onBlur={() => setRenaming(null)}
              />
            ) : (
              <button
                type="button"
                className="flex-1 truncate text-left"
                onDoubleClick={() => setRenaming({ id: c.id, name: c.name })}
                title="Double-click to rename"
              >
                {c.name}
              </button>
            )}
            <span className="text-table text-fg-faint">
              {share(counts?.counts.get(c.id) ?? 0, total)}
            </span>
          </div>
        ))}
      </div>
      <div className="flex flex-wrap gap-1 border-t border-border p-2">
        <Button onClick={() => edit(() => api.schemaAdd(), "Couldn't add a class")}>Add</Button>
        <Button disabled={activeClass < 0} onClick={remove}>
          Remove
        </Button>
        <Button
          className={cn(view.isolate !== null && "bg-selection")}
          disabled={activeClass < 0 && view.isolate === null}
          onClick={() =>
            app.dispatch({
              type: "view",
              patch: { isolate: view.isolate === null ? activeClass : null },
            })
          }
          title="Show the active class on its own; the others are dimmed"
        >
          {view.isolate === null ? "Isolate" : "Show all"}
        </Button>
      </div>
    </div>
  );
}
