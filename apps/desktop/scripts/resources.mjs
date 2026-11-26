// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

// Fills apps/desktop/resources/ (git-ignored), which the installer ships:
//
//   uv/windows/uv.exe, uv/linux/uv   uv for a native and a WSL Environment
//                                    (the WSL install then needs no root)
//   engine/cloudlabeller-*.whl       the app's own wheel, installed last
//   engine/requirements.lock         the Environment's pinned packages
//
// Run by `pnpm desktop` and `pnpm bundle`. uv is fetched once per version
// and checked against its pinned SHA-256.

import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import {
  copyFileSync,
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

export const UV_VERSION = "0.13.9";

/** The uv archives shipped, with their published SHA-256. */
export const UV = {
  windows: {
    archive: "uv-x86_64-pc-windows-msvc.zip",
    sha256: "edd61f7f5728a56cc62455386661860bdfc7bab65872f39b0c25bdcc1d8e1beb",
    binary: "uv.exe",
  },
  linux: {
    archive: "uv-x86_64-unknown-linux-musl.tar.gz",
    sha256: "58ccf5f1792b8a5a88d01bc910ea131a8b3db4a319a17ad5fad75e0845a3e271",
    binary: "uv",
  },
};

const here = dirname(fileURLToPath(import.meta.url));
const desktop = resolve(here, "..");
const repo = resolve(desktop, "..", "..");
const out = join(desktop, "resources");

/** Hex SHA-256 of a buffer. @param {Buffer} data */
export function sha256(data) {
  return createHash("sha256").update(data).digest("hex");
}

/**
 * Throws unless `data` has the expected digest.
 * @param {string} name @param {Buffer} data @param {string} expected
 */
export function checkDigest(name, data, expected) {
  const got = sha256(data);
  if (got !== expected) {
    throw new Error(`${name}: SHA-256 ${got}, expected ${expected}`);
  }
}

/** @param {string} cmd @param {string[]} args @param {import('node:child_process').SpawnSyncOptions} [opts] */
function run(cmd, args, opts = {}) {
  const r = spawnSync(cmd, args, { stdio: "inherit", ...opts });
  if (r.status !== 0) {
    throw new Error(`${cmd} ${args.join(" ")} failed (${r.status ?? r.error})`);
  }
}

/** Windows' own bsdtar unpacks both .zip and .tar.gz. */
function tarCommand() {
  if (process.platform === "win32") {
    return join(process.env.SystemRoot ?? "C:\\Windows", "System32", "tar.exe");
  }
  return "tar";
}

async function fetchUv() {
  const dir = join(out, "uv");
  const stamp = join(dir, "version.txt");
  if (existsSync(stamp) && readFileSync(stamp, "utf8").trim() === UV_VERSION) {
    return;
  }
  rmSync(dir, { recursive: true, force: true });
  for (const [platform, spec] of Object.entries(UV)) {
    const url = `https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/${spec.archive}`;
    console.log(`resources: fetching ${url}`);
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
    const data = Buffer.from(await res.arrayBuffer());
    checkDigest(spec.archive, data, spec.sha256);
    const target = join(dir, platform);
    const unpack = join(dir, `${platform}.unpack`);
    mkdirSync(unpack, { recursive: true });
    const archive = join(unpack, spec.archive);
    writeFileSync(archive, data);
    run(tarCommand(), ["-xf", archive, "-C", unpack]);
    const found = findFile(unpack, spec.binary);
    if (!found) throw new Error(`${spec.archive}: no ${spec.binary} inside`);
    mkdirSync(target, { recursive: true });
    copyFileSync(found, join(target, spec.binary));
    rmSync(unpack, { recursive: true, force: true });
  }
  writeFileSync(stamp, `${UV_VERSION}\n`);
}

/** @param {string} dir @param {string} name @returns {string | null} */
function findFile(dir, name) {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (entry.isFile() && entry.name === name) return path;
    if (entry.isDirectory()) {
      /** @type {string | null} */
      const inner = findFile(path, name);
      if (inner) return inner;
    }
  }
  return null;
}

function buildWheel() {
  const dir = join(out, "engine");
  mkdirSync(dir, { recursive: true });
  for (const f of readdirSync(dir)) {
    if (f.endsWith(".whl")) rmSync(join(dir, f));
  }
  console.log("resources: building the cloudlabeller wheel");
  run("uv", ["build", "--wheel", "--out-dir", dir, repo]);
}

function copyLock() {
  const lock = join(desktop, "environment", "requirements.lock");
  const target = join(out, "engine", "requirements.lock");
  if (existsSync(lock)) {
    copyFileSync(lock, target);
  } else {
    rmSync(target, { force: true });
    console.warn(
      "resources: no environment/requirements.lock yet — the Environment " +
        "installs the wheel's own dependencies unpinned",
    );
  }
}

async function main() {
  mkdirSync(out, { recursive: true });
  await fetchUv();
  buildWheel();
  copyLock();
  console.log(`resources: ready in ${out}`);
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((err) => {
    console.error(err instanceof Error ? err.message : err);
    process.exit(1);
  });
}
