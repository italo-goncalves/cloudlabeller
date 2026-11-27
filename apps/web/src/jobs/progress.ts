// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

// A Job's progress, written into the Log as well as the status bar.

/**
 * Remembers a Job's last progress message: `line` returns the Log line for a
 * new message ("[ 42%] Cutting photos into Tiles: 95 of 376"), or null when
 * the message is empty or the same as the last one (only its fraction moved).
 */
export function progressLog() {
  let last = "";
  return {
    line(fraction: number | null, message: string): string | null {
      const text = message.trim();
      if (!text || text === last) return null;
      last = text;
      if (fraction == null || !Number.isFinite(fraction)) return text;
      const pct = Math.round(Math.min(1, Math.max(0, fraction)) * 100);
      return `[${String(pct).padStart(3)}%] ${text}`;
    },
  };
}
