// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

// What the window shows with no Project open: New, Open and the recent ones.

import { useEffect, useState } from "react";
import { recentProjects } from "../desktop/bridge";
import { useApp } from "../state/app";
import { Button } from "../ui/controls";
import { newFlow, openFlow } from "./projects";

export function Welcome() {
  const app = useApp();
  const [recent, setRecent] = useState<string[]>([]);
  const [opening, setOpening] = useState(false);

  useEffect(() => {
    recentProjects()
      .then(setRecent)
      .catch(() => setRecent([]));
  }, []);

  const go = async (f: () => Promise<boolean>) => {
    setOpening(true);
    try {
      await f();
    } finally {
      setOpening(false);
    }
  };

  return (
    <section className="mx-auto flex h-full w-full max-w-xl flex-col justify-center gap-5 p-6">
      <div>
        <h1 className="text-xl font-semibold">CloudLabeller</h1>
        <p className="text-fg-muted">Create a new Project or open an existing one to begin.</p>
      </div>
      <div className="flex gap-2">
        <Button kind="primary" disabled={opening} onClick={() => go(() => newFlow(app))}>
          New Project…
        </Button>
        <Button disabled={opening} onClick={() => go(() => openFlow(app))}>
          Open Project…
        </Button>
      </div>
      {recent.length > 0 && (
        <div>
          <h2 className="mb-2 text-fg-muted">Recent Projects</h2>
          <ul className="overflow-hidden rounded-md border border-border">
            {recent.map((r) => (
              <li key={r}>
                <button
                  type="button"
                  disabled={opening}
                  className="block w-full truncate border-b border-border px-3 py-2 text-left last:border-b-0 hover:bg-surface-2 disabled:opacity-50"
                  title={r}
                  onClick={() => go(() => openFlow(app, r))}
                >
                  {r}
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
      {opening && <p className="text-fg-muted">Opening… (the Engine starts first)</p>}
    </section>
  );
}
