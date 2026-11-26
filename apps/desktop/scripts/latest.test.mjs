// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

import assert from "node:assert/strict";
import { test } from "node:test";
import { manifest } from "./latest.mjs";

test("latest.json points at the release's installer", () => {
  const m = manifest({
    version: "0.3.0",
    notes: "n",
    date: "2026-11-25T00:00:00Z",
    file: "CloudLabeller_0.3.0_x64-setup.exe",
    signature: "sig",
  });
  assert.equal(m.version, "0.3.0");
  assert.equal(
    m.platforms["windows-x86_64"].url,
    "https://github.com/italo-goncalves/cloudlabeller/releases/download/v0.3.0/CloudLabeller_0.3.0_x64-setup.exe",
  );
  assert.equal(m.platforms["windows-x86_64"].signature, "sig");
});
