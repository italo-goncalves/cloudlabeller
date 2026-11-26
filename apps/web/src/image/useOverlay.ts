// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

// The mask drawn on a canvas in the schema's colours: all of it when it was
// loaded or replaced, only the touched rectangle while a stroke goes on.

import { useCallback, useEffect, useMemo, useRef } from "react";
import type { SchemaClass } from "../engine/api";
import { maskToRgba, parseColour, type Rect } from "./raster";
import type { MaskEditor } from "./useMask";

export function useOverlay(editor: MaskEditor, schema: readonly SchemaClass[]) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const pixels = useRef<ImageData | null>(null);
  const colours = useMemo(
    () => new Map(schema.map((c) => [c.id, parseColour(c.color)] as const)),
    [schema],
  );
  const { size, mask, rev } = editor;

  // biome-ignore lint/correctness/useExhaustiveDependencies: `rev` is what says the mask changed
  useEffect(() => {
    const el = canvas.current;
    const ctx = el?.getContext("2d");
    if (!el || !ctx) return;
    if (!size) {
      pixels.current = null;
      el.width = el.height = 0;
      return;
    }
    if (el.width !== size.w || el.height !== size.h || !pixels.current) {
      el.width = size.w;
      el.height = size.h;
      pixels.current = ctx.createImageData(size.w, size.h);
    }
    const m = mask.current;
    if (!m) {
      ctx.clearRect(0, 0, size.w, size.h);
      return;
    }
    maskToRgba(m, size.w, size.h, colours, pixels.current.data);
    ctx.putImageData(pixels.current, 0, 0);
  }, [rev, size, colours, mask]);

  /** Redraw the pixels of `rect` from the working mask. */
  const paint = useCallback(
    (rect: Rect | null) => {
      const ctx = canvas.current?.getContext("2d");
      const img = pixels.current;
      const m = mask.current;
      if (!rect || !ctx || !img || !m || !size) return;
      maskToRgba(m, size.w, size.h, colours, img.data, rect);
      ctx.putImageData(img, 0, 0, rect.x0, rect.y0, rect.x1 - rect.x0, rect.y1 - rect.y0);
    },
    [colours, mask, size],
  );

  return { canvas, paint };
}
