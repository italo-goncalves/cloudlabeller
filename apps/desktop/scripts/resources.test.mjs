// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

import assert from "node:assert/strict";
import { test } from "node:test";
import { checkDigest, sha256, UV } from "./resources.mjs";

test("sha256 is the hex digest", () => {
  assert.equal(
    sha256(Buffer.from("abc")),
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
  );
});

test("a wrong digest is refused", () => {
  assert.throws(() => checkDigest("x", Buffer.from("abc"), "0".repeat(64)), /SHA-256/);
});

test("uv is pinned for a native and a WSL Environment", () => {
  assert.deepEqual(Object.keys(UV).sort(), ["linux", "windows"]);
  for (const spec of Object.values(UV)) {
    assert.match(spec.sha256, /^[0-9a-f]{64}$/);
  }
});
