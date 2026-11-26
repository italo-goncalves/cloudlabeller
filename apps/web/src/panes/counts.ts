// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

/** How many points each class has (−1 is unlabelled), and the total. */
export function countClasses(labels: ArrayLike<number>): {
  counts: Map<number, number>;
  total: number;
} {
  const counts = new Map<number, number>();
  for (let i = 0; i < labels.length; i++) {
    const c = labels[i] ?? -1;
    counts.set(c, (counts.get(c) ?? 0) + 1);
  }
  return { counts, total: labels.length };
}

/** "12,345 (6.7%)", or "" when there are no points. */
export function share(n: number, total: number): string {
  if (total <= 0) return "";
  return `${n.toLocaleString("en")} (${((100 * n) / total).toFixed(1)}%)`;
}
