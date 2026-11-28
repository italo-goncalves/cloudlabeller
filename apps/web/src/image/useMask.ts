// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

// The photo's mask as the pane edits it: loaded from the Engine (the user's
// labels over the cloud's and the model's), changed in place while a stroke
// is under way, and saved whole, one undoable edit at a time. The Engine
// keeps only what differs from what it showed: the user's strokes.

import { type RefObject, useCallback, useEffect, useRef, useState } from "react";
import { currentSurvey, imageInfo, imageMask, saveMask } from "../engine/api";
import { useApp } from "../state/app";

export interface MaskEditor {
  /** The photo's size, once known. */
  size: { w: number; h: number } | null;
  /** The working mask; null while there is none (nothing yet painted or loaded). */
  mask: RefObject<Int32Array | null>;
  /** Bumped whenever the whole mask changed under the overlay: repaint it all. */
  rev: number;
  /** True from the first move of a stroke until it is handed to `save`. */
  stroking: RefObject<boolean>;
  /** The working mask, made all unlabelled at the photo's size if there is none. */
  ensure: () => Int32Array | null;
  /** Save `mask` as one edit, after any earlier one; resolves when the Engine has it. */
  save: (mask: Int32Array) => Promise<void>;
  /** Replace the working mask with another (an import) and save it. */
  replace: (mask: Int32Array) => Promise<void>;
  /** Resolves when every save so far has finished. */
  settled: () => Promise<void>;
}

export function useMask(name: string | null): MaskEditor {
  const { state, fail } = useApp();
  const [size, setSize] = useState<{ w: number; h: number } | null>(null);
  const [rev, setRev] = useState(0);
  const mask = useRef<Int32Array | null>(null);
  const stroking = useRef(false);
  const pending = useRef(0);
  const queue = useRef<Promise<void>>(Promise.resolve());
  const loaded = useRef<string | null>(null);
  const seq = useRef(0);
  const sizeRef = useRef(size);
  sizeRef.current = size;

  const { version, images } = state.masks;
  useEffect(() => {
    void version; // a change of it is what asks for the reload
    if (!name) {
      seq.current++;
      loaded.current = null;
      mask.current = null;
      setSize(null);
      setRev((r) => r + 1);
      return;
    }
    const switched = loaded.current !== name;
    if (!switched && !(images.length === 0 || images.includes(name))) return;
    if (!switched && stroking.current) return;
    loaded.current = name;
    if (switched) {
      mask.current = null;
      setSize(null);
      setRev((r) => r + 1);
    }
    // Not tied to the effect's cleanup: a reload that was not asked for again
    // must not cancel the one in flight. Only a newer request supersedes it.
    const mine = ++seq.current;
    (async () => {
      try {
        const [info, m] = await Promise.all([imageInfo(name), imageMask(name)]);
        // Stale, or edited since the request: what is here is newer.
        if (mine !== seq.current || stroking.current || pending.current > 0) return;
        mask.current = m.mask;
        setSize(m.mask ? { w: m.width, h: m.height } : { w: info.width, h: info.height });
        setRev((r) => r + 1);
      } catch (e) {
        if (mine === seq.current) fail(e);
      }
    })();
  }, [name, version, images, fail]);

  const ensure = useCallback(() => {
    if (mask.current) return mask.current;
    const s = sizeRef.current;
    if (!s) return null;
    mask.current = new Int32Array(s.w * s.h).fill(-1);
    return mask.current;
  }, []);

  const save = useCallback(
    (m: Int32Array) => {
      const s = sizeRef.current;
      if (!name || !s) return Promise.resolve();
      // A copy: the next stroke edits the working mask while this one is written.
      const copy = m.slice();
      const survey = currentSurvey();
      pending.current++;
      const run = queue.current.then(async () => {
        try {
          await saveMask(name, copy, s.w, s.h, survey);
        } catch (e) {
          fail(e);
        } finally {
          pending.current--;
        }
      });
      queue.current = run;
      return run;
    },
    [name, fail],
  );

  const replace = useCallback(
    (m: Int32Array) => {
      mask.current = m;
      setRev((r) => r + 1);
      return save(m);
    },
    [save],
  );

  const settled = useCallback(() => queue.current, []);

  return { size, mask, rev, stroking, ensure, save, replace, settled };
}
