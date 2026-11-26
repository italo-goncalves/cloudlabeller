// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

import { useEffect, useState } from "react";
import {
  checkUpdate,
  type Downloading,
  installUpdate,
  isProblem,
  type Offered,
} from "./desktop/bridge";

/** Offers a newer version once, at start, with Later. */
export function Updates() {
  const [offered, setOffered] = useState<Offered | null>(null);
  const [state, setState] = useState<Downloading | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    checkUpdate()
      .then(setOffered)
      .catch(() => setOffered(null)); // offline, or no Release yet: say nothing
  }, []);

  if (!offered) return null;
  const pct =
    state?.kind === "progress" && state.total ? Math.round((100 * state.done) / state.total) : null;
  return (
    <div className="flex items-center gap-3 border-b border-border bg-surface-1 px-4 py-2">
      <span className="mr-auto">
        CloudLabeller {offered.version} is out (you have {offered.current}).
        {error && <span className="ml-2 text-danger">{error}</span>}
      </span>
      {state ? (
        <span className="text-fg-muted">
          {state.kind === "installing"
            ? "Installing…"
            : `Downloading${pct !== null ? ` ${pct}%` : "…"}`}
        </span>
      ) : (
        <>
          <button
            type="button"
            className="h-control rounded-md bg-accent px-3 text-accent-fg hover:bg-accent-hover"
            onClick={() =>
              installUpdate(setState).catch((e: unknown) => {
                setState(null);
                setError(isProblem(e) ? e.english : String(e));
              })
            }
          >
            Update and restart
          </button>
          <button
            type="button"
            className="h-control rounded-md bg-surface-2 px-3 hover:bg-surface-3"
            onClick={() => setOffered(null)}
          >
            Later
          </button>
        </>
      )}
    </div>
  );
}
