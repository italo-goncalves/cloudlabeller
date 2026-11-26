// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

// After `tauri build`: copies the NSIS installer (and its .sig, when the
// build was signed) into apps/desktop/publish/, and writes latest.json, the
// file the updater reads from the GitHub Release:
//
//   https://github.com/italo-goncalves/cloudlabeller/releases/latest/download/latest.json
//
// Usage: node scripts/latest.mjs [notes]

import {
  copyFileSync,
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  writeFileSync,
} from "node:fs";
import { basename, dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

export const REPO = "italo-goncalves/cloudlabeller";

/**
 * The updater's manifest for one Windows installer.
 * @param {{version: string, notes: string, date: string, file: string, signature: string}} a
 */
export function manifest({ version, notes, date, file, signature }) {
  return {
    version,
    notes,
    pub_date: date,
    platforms: {
      "windows-x86_64": {
        signature,
        url: `https://github.com/${REPO}/releases/download/v${version}/${encodeURIComponent(file)}`,
      },
    },
  };
}

function main() {
  const here = dirname(fileURLToPath(import.meta.url));
  const desktop = resolve(here, "..");
  const version = JSON.parse(readFileSync(join(desktop, "package.json"), "utf8")).version;
  const nsis = join(desktop, "src-tauri", "target", "release", "bundle", "nsis");
  const installer = readdirSync(nsis).find((f) => f.endsWith("-setup.exe") && f.includes(version));
  if (!installer) throw new Error(`no ${version} installer in ${nsis}`);
  const out = join(desktop, "publish");
  mkdirSync(out, { recursive: true });
  copyFileSync(join(nsis, installer), join(out, installer));
  const sig = join(nsis, `${installer}.sig`);
  if (!existsSync(sig)) {
    console.log(`latest: ${installer} is unsigned — no latest.json (the updater won't offer it)`);
    return;
  }
  copyFileSync(sig, join(out, basename(sig)));
  const json = manifest({
    version,
    notes: process.argv[2] ?? `CloudLabeller ${version}`,
    date: new Date().toISOString(),
    file: installer,
    signature: readFileSync(sig, "utf8").trim(),
  });
  writeFileSync(join(out, "latest.json"), `${JSON.stringify(json, null, 2)}\n`);
  console.log(`latest: publish/${installer}, its .sig and latest.json`);
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main();
}
