// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

// The mask drawn on a canvas in the schema's colours: all of it when it was
// loaded or replaced, only the touched rectangle while a stroke goes on. Each
// pixel's opacity says where its label came from (solid hand-drawn, lighter
// from the cloud, hatched predicted); `bySource` colours the sources instead.

import { useCallback, useEffect, useMemo, useRef } from "react";
import type { SchemaClass } from "../engine/api";
import { maskToRgba, parseColour, type Rect } from "./raster";
import type { MaskEditor } from "./useMask";

export function useOverlay(editor: MaskEditor, schema: readonly SchemaClass[], bySource = false) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const pixels = useRef<ImageData | null>(null);
  const colours = useMemo(
    () => new Map(schema.map((c) => [c.id, parseColour(c.color)] as const)),
    [schema],
  );
  const { size, mask, marks, rev } = editor;

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
    maskToRgba(
      m,
      size.w,
      size.h,
      colours,
      pixels.current.data,
      undefined,
      marks.current?.sources,
      bySource,
    );
    ctx.putImageData(pixels.current, 0, 0);
  }, [rev, size, colours, mask, marks, bySource]);

  /** Redraw the pixels of `rect` from the working mask. */
  const paint = useCallback(
    (rect: Rect | null) => {
      const ctx = canvas.current?.getContext("2d");
      const img = pixels.current;
      const m = mask.current;
      if (!rect || !ctx || !img || !m || !size) return;
      maskToRgba(m, size.w, size.h, colours, img.data, rect, marks.current?.sources, bySource);
      ctx.putImageData(img, 0, 0, rect.x0, rect.y0, rect.x1 - rect.x0, rect.y1 - rect.y0);
    },
    [colours, mask, marks, size, bySource],
  );

  return { canvas, paint };
}
