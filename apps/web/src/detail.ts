// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

// Two kinds of detail level.
//
// Reconstruction: named levels as COLMAP's `max_image_size`, relative to the
// photos' own resolution (ported from the Qt app's ui/detail_level.py). A
// level is a linear divisor of the long side; its label gives the area
// fraction, so divisor 2 reads "1/4 resolution". 0 px means no cap.
//
// Viewing: how many points of a cloud reach the screen (`VIEW_DETAIL`).

import type { ViewDetail } from "./state/store";

export type Level = [name: string, divisor: number];

/** Feature extraction is cheap enough to keep detail. */
export const SFM_LEVELS: Level[] = [
  ["High", 1],
  ["Medium", 2],
  ["Low", 4],
];
/** Patch-match stereo is far hungrier: it starts one level lower and reaches one deeper. */
export const MVS_LEVELS: Level[] = [
  ["Ultra", 1],
  ["High", 2],
  ["Medium", 4],
  ["Low", 8],
];

export interface DetailChoice {
  label: string;
  /** `max_image_size`; 0 is full resolution. */
  pixels: number;
}

/**
 * Each level's label and pixel cap. With no known resolution the presets
 * scale `customDefault` so that divisor 2 lands on it.
 */
export function detailChoices(
  levels: Level[],
  source: [number, number] | null,
  customDefault: number,
): DetailChoice[] {
  const long = source ? Math.max(source[0], source[1]) : null;
  return levels.map(([name, div]) => {
    if (div === 1)
      return {
        label: long ? `${name} — full resolution (${long} px)` : `${name} — full resolution`,
        pixels: 0,
      };
    const px = long ? Math.floor(long / div) : Math.floor((customDefault * 2) / div);
    return { label: `${name} — 1/${div * div} resolution (${px} px)`, pixels: px };
  });
}

/** The default level's index for a name, or the second level. */
export function defaultLevel(levels: Level[], name: string): number {
  const i = levels.findIndex(([n]) => n.toLowerCase() === name.toLowerCase());
  return i >= 0 ? i : Math.min(1, levels.length - 1);
}

/** Points sent to the viewer for each viewing level; 0 is every point. */
export const VIEW_DETAIL: Record<ViewDetail, number> = {
  low: 500_000,
  medium: 2_000_000,
  high: 6_000_000,
  full: 0,
};
