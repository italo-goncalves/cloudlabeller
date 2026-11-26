// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

// The View pane's Home button reaches the 3D viewer through here: the viewer
// listens, anything may ask. A module-level emitter, so the two panes need
// share nothing but this file.

const listeners = new Set<() => void>();

/** Ask every viewer to go to its home view. */
export function goHome(): void {
  for (const l of [...listeners]) l();
}

/** Listen for `goHome`; the function it returns stops listening. */
export function onHome(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}
