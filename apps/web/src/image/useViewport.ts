// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

// Pan and zoom of the photo in its pane: it fits when it loads (and while the
// user has not moved it, when the pane resizes); the wheel zooms about the cursor.

import { useCallback, useEffect, useRef, useState } from "react";
import {
  fitView,
  type Point,
  panBy,
  screenToImage,
  type View,
  wheelFactor,
  zoomAbout,
} from "./raster";

export function useViewport(size: { w: number; h: number } | null, key: string | null) {
  const ref = useRef<HTMLDivElement>(null);
  const [box, setBox] = useState({ w: 0, h: 0 });
  const [view, setView] = useState<View>({ scale: 1, x: 0, y: 0 });
  const fitted = useRef(true);
  const viewRef = useRef(view);
  viewRef.current = view;

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const watch = new ResizeObserver(() => setBox({ w: el.clientWidth, h: el.clientHeight }));
    watch.observe(el);
    setBox({ w: el.clientWidth, h: el.clientHeight });
    return () => watch.disconnect();
  }, []);

  // A new photo (or a new size of it) starts fitted; a resize refits only
  // while the user has not moved the view.
  const shown = useRef<string | null>(null);
  useEffect(() => {
    const id = size ? `${key}:${size.w}x${size.h}` : null;
    if (id !== shown.current) {
      shown.current = id;
      fitted.current = true;
    }
    if (fitted.current && size) setView(fitView(size.w, size.h, box.w, box.h));
  }, [key, size, box.w, box.h]);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const wheel = (e: WheelEvent) => {
      e.preventDefault();
      const r = el.getBoundingClientRect();
      fitted.current = false;
      setView((v) => zoomAbout(v, wheelFactor(e.deltaY), e.clientX - r.left, e.clientY - r.top));
    };
    el.addEventListener("wheel", wheel, { passive: false });
    return () => el.removeEventListener("wheel", wheel);
  }, []);

  const pan = useCallback((dx: number, dy: number) => {
    fitted.current = false;
    setView((v) => panBy(v, dx, dy));
  }, []);

  /** A pointer event's position in the viewport, CSS pixels. */
  const local = useCallback((clientX: number, clientY: number): Point => {
    const r = ref.current?.getBoundingClientRect();
    return [clientX - (r?.left ?? 0), clientY - (r?.top ?? 0)];
  }, []);

  /** A pointer event's position in image pixels. */
  const toImage = useCallback(
    (clientX: number, clientY: number): Point => {
      const [x, y] = local(clientX, clientY);
      return screenToImage(viewRef.current, x, y);
    },
    [local],
  );

  return { ref, view, pan, local, toImage };
}
