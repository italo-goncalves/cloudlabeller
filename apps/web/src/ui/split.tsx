// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

// Splitters: thin handles between panes that drag a pane's size, and the
// sizes they keep from one run to the next.

import { cn } from "@geokit/ui";
import { useCallback, useEffect, useRef, useState } from "react";

/** A size after dragging `delta` pixels from `start`, kept within [min, max].
 *  `invert` is for a pane on the far side of its handle (the right dock, the
 *  log), which grows as the pointer moves left or up. */
export function dragSize(
  start: number,
  delta: number,
  min: number,
  max: number,
  invert = false,
): number {
  const size = start + (invert ? -delta : delta);
  return Math.round(Math.min(Math.max(size, min), Math.max(min, max)));
}

/** Sizes read back from storage: only finite numbers for the names known. */
export function readSizes<T extends Record<string, number>>(saved: unknown, defaults: T): T {
  const out = { ...defaults };
  if (saved && typeof saved === "object")
    for (const k of Object.keys(defaults) as (keyof T)[]) {
      const v = (saved as Record<keyof T, unknown>)[k];
      if (typeof v === "number" && Number.isFinite(v) && v > 0) out[k] = v as T[keyof T];
    }
  return out;
}

/** Pane sizes in pixels, kept in localStorage under `key`. */
export function useSizes<T extends Record<string, number>>(key: string, defaults: T) {
  const [sizes, setSizes] = useState<T>(() => {
    try {
      return readSizes(JSON.parse(localStorage.getItem(key) ?? "null"), defaults);
    } catch {
      return defaults;
    }
  });
  useEffect(() => {
    try {
      localStorage.setItem(key, JSON.stringify(sizes));
    } catch {
      // a window without storage keeps its sizes for the session only
    }
  }, [key, sizes]);
  const set = useCallback(
    (name: keyof T, value: number) => setSizes((s) => ({ ...s, [name]: value })),
    [],
  );
  return [sizes, set] as const;
}

/** A handle between two panes. `axis` "x" drags a width, "y" a height. */
export function Splitter({
  axis,
  value,
  min,
  max,
  invert,
  onChange,
  label,
}: {
  axis: "x" | "y";
  value: number;
  min: number;
  /** The largest size, worked out when the drag starts (the window may have changed). */
  max: () => number;
  invert?: boolean;
  onChange: (value: number) => void;
  label: string;
}) {
  const drag = useRef<{ from: number; start: number; max: number } | null>(null);
  const at = (e: React.PointerEvent) => (axis === "x" ? e.clientX : e.clientY);
  const step = (by: number) => onChange(dragSize(value, by, min, max(), invert));
  return (
    // biome-ignore lint/a11y/useSemanticElements: a draggable separator has no native element
    <div
      role="separator"
      aria-label={label}
      aria-orientation={axis === "x" ? "vertical" : "horizontal"}
      aria-valuenow={value}
      aria-valuemin={min}
      tabIndex={0}
      className={cn(
        "relative z-10 shrink-0 bg-border transition-colors hover:bg-accent focus:bg-accent focus:outline-none",
        axis === "x" ? "w-px cursor-col-resize" : "h-px cursor-row-resize",
        // a wider invisible grip than the 1px line
        axis === "x"
          ? "before:absolute before:inset-y-0 before:-left-1 before:w-2"
          : "before:absolute before:inset-x-0 before:-top-1 before:h-2",
      )}
      onPointerDown={(e) => {
        e.preventDefault();
        e.currentTarget.setPointerCapture(e.pointerId);
        drag.current = { from: at(e), start: value, max: max() };
      }}
      onPointerMove={(e) => {
        const d = drag.current;
        if (d) onChange(dragSize(d.start, at(e) - d.from, min, d.max, invert));
      }}
      onPointerUp={() => {
        drag.current = null;
      }}
      onPointerCancel={() => {
        drag.current = null;
      }}
      onKeyDown={(e) => {
        const back = axis === "x" ? "ArrowLeft" : "ArrowUp";
        const fwd = axis === "x" ? "ArrowRight" : "ArrowDown";
        if (e.key === back || e.key === fwd) {
          e.preventDefault();
          step((e.key === fwd ? 1 : -1) * (e.shiftKey ? 64 : 16));
        }
      }}
    />
  );
}
