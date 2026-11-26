// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

import { useCallback, useState } from "react";
import { inDesktop, type Kept } from "./desktop/bridge";
import { DeveloperPage } from "./developer/DeveloperPage";
import { EnvironmentSetup } from "./environment/EnvironmentSetup";
import { Updates } from "./Updates";

/** The window: the Environment first, then the Project. */
export function App() {
  const [kept, setKept] = useState<Kept | null>(null);
  const ready = useCallback((k: Kept) => setKept(k), []);

  if (!inDesktop()) {
    return (
      <main className="flex h-full flex-col gap-4 bg-surface-0 p-6 text-fg">
        <h1 className="text-lg font-semibold">CloudLabeller</h1>
        <p className="text-fg-muted">
          This page runs inside the CloudLabeller desktop app (<code>pnpm desktop</code>).
        </p>
      </main>
    );
  }
  return (
    <main className="flex h-full flex-col bg-surface-0 text-fg">
      <Updates />
      <div className="min-h-0 flex-1">
        {kept ? <DeveloperPage kept={kept} /> : <EnvironmentSetup onReady={ready} />}
      </div>
    </main>
  );
}
