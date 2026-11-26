// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

// A menu bar: click a title to open its menu, move along to open the next,
// click outside or press Escape to close.

import { cn } from "@geokit/ui";
import { useEffect, useRef, useState } from "react";

export type MenuItem =
  | { label: string; run: () => void; disabled?: boolean; shortcut?: string; checked?: boolean }
  | "separator";

export interface Menu {
  title: string;
  items: MenuItem[];
}

export function Menubar({ menus }: { menus: Menu[] }) {
  const [open, setOpen] = useState<number | null>(null);
  const bar = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (open === null) return;
    const away = (e: MouseEvent) => {
      if (!bar.current?.contains(e.target as Node)) setOpen(null);
    };
    const key = (e: KeyboardEvent) => e.key === "Escape" && setOpen(null);
    window.addEventListener("mousedown", away);
    window.addEventListener("keydown", key);
    return () => {
      window.removeEventListener("mousedown", away);
      window.removeEventListener("keydown", key);
    };
  }, [open]);
  return (
    <div
      ref={bar}
      className="flex h-8 items-center gap-0.5 border-b border-border bg-surface-1 px-1"
      role="menubar"
    >
      {menus.map((m, i) => (
        <div key={m.title} className="relative">
          <button
            type="button"
            role="menuitem"
            aria-haspopup="menu"
            aria-expanded={open === i}
            className={cn("h-7 rounded px-2.5 hover:bg-surface-2", open === i && "bg-surface-2")}
            onClick={() => setOpen(open === i ? null : i)}
            onMouseEnter={() => open !== null && setOpen(i)}
          >
            {m.title}
          </button>
          {open === i && (
            <div
              role="menu"
              className="absolute top-full left-0 z-40 mt-0.5 min-w-60 rounded-md border border-border bg-surface-1 py-1 shadow-xl"
            >
              {m.items.map((it, k) =>
                it === "separator" ? (
                  // biome-ignore lint/suspicious/noArrayIndexKey: separators have nothing else to key on
                  <div key={`s${k}`} className="my-1 border-t border-border" />
                ) : (
                  <button
                    key={it.label}
                    type="button"
                    role="menuitem"
                    disabled={it.disabled}
                    className="flex w-full items-center gap-4 px-3 py-1 text-left hover:bg-surface-2 disabled:cursor-default disabled:opacity-40 disabled:hover:bg-transparent"
                    onClick={() => {
                      setOpen(null);
                      it.run();
                    }}
                  >
                    <span className="w-3">{it.checked ? "✓" : ""}</span>
                    <span className="flex-1">{it.label}</span>
                    {it.shortcut && <span className="text-table text-fg-faint">{it.shortcut}</span>}
                  </button>
                ),
              )}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
