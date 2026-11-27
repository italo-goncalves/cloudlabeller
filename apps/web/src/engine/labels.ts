// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

// Class ids cross to and from the Engine in the smallest integer type that
// holds them: a photo's Mask is 20 MB as int8 and 80 MB as int32, and in the
// installed app the Engine reads them through WSL's slow `/mnt/c`. The pane
// itself works on Int32Array.

type Ids = ArrayLike<number> & ArrayBufferView;

/** `ids` as an Int32Array (the same array when it already is one). */
export function widen(ids: Ids): Int32Array {
  return ids instanceof Int32Array ? ids : new Int32Array(ids as ArrayLike<number>);
}

/** `ids` in the narrowest of int8, int16 and int32 that holds every value. */
export function narrow(ids: Int32Array): {
  data: Int8Array | Int16Array | Int32Array;
  dtype: "int8" | "int16" | "int32";
} {
  let lo = 0;
  let hi = 0;
  for (let i = 0; i < ids.length; i++) {
    const v = ids[i] as number;
    if (v < lo) lo = v;
    else if (v > hi) hi = v;
  }
  if (lo >= -128 && hi <= 127) return { data: Int8Array.from(ids), dtype: "int8" };
  if (lo >= -32768 && hi <= 32767) return { data: Int16Array.from(ids), dtype: "int16" };
  return { data: ids, dtype: "int32" };
}
