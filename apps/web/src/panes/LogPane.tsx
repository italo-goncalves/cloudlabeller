// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

// What the Jobs and the window have said, newest last, kept to MAX_LOG lines.

import { useEffect, useRef } from "react";
import { useApp } from "../state/app";
import { Button, PaneTitle } from "../ui/controls";

export function LogPane() {
  const app = useApp();
  const box = useRef<HTMLPreElement>(null);
  const stick = useRef(true);
  const { log } = app.state;

  // biome-ignore lint/correctness/useExhaustiveDependencies: follow new lines
  useEffect(() => {
    const b = box.current;
    if (b && stick.current) b.scrollTop = b.scrollHeight;
  }, [log]);

  return (
    <div className="flex h-full min-h-0 flex-col">
      <PaneTitle
        right={
          <Button kind="quiet" onClick={() => app.dispatch({ type: "clear_log" })}>
            Clear
          </Button>
        }
      >
        Log
      </PaneTitle>
      <pre
        ref={box}
        className="min-h-0 flex-1 overflow-auto px-3 py-1 font-mono text-table select-text"
        onScroll={(e) => {
          const b = e.currentTarget;
          stick.current = b.scrollTop + b.clientHeight >= b.scrollHeight - 8;
        }}
      >
        {log.map((l) => (
          <div key={l.n} className={l.err ? "text-fg-muted" : undefined}>
            {l.text}
          </div>
        ))}
      </pre>
    </div>
  );
}
