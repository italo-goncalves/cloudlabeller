// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

// The page's only way to the desktop core: Tauri commands and Channels.
// Each command here has its Rust twin in apps/desktop/src-tauri/src, and
// each type mirrors the Rust one it is named after.

import { Channel, invoke } from "@tauri-apps/api/core";

/** Whether the page runs inside the desktop shell (not a plain browser). */
export function inDesktop(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

/** An error to tell the user, by code (geokit's and the Engine's shape). */
export interface Problem {
  code: string;
  english: string;
  params?: Record<string, unknown>;
}

export function isProblem(x: unknown): x is Problem {
  return typeof x === "object" && x !== null && "code" in x && "english" in x;
}

// -- the Environment ----------------------------------------------------------

export type Own =
  | { machine: "windows" }
  | { machine: "wsl"; distro: string }
  | { machine: "posix" };

export type Why = "gpu" | "no_gpu_in_wsl" | "no_wsl" | "not_windows";

export interface Choice {
  own: Own;
  why: Why;
  gpu: string | null;
  distros: string[];
}

export interface Environment {
  machine: "windows" | "wsl" | "posix";
  distro: string;
  python: { kind: "path"; path: string } | { kind: "conda"; name: string } | { kind: "default" };
}

export interface Kept {
  environment: Environment;
  own: boolean;
  installed?: string;
}

export type EnvironmentSaid =
  | { kind: "chose"; choice: Choice }
  | { kind: "doing"; what: "check" | "install" | "upgrade" | "repair" }
  | { kind: "step"; i: number; n: number; what: string }
  | { kind: "out"; line: string }
  | { kind: "err"; line: string }
  | { kind: "ready"; kept: Kept }
  | { kind: "failed"; problem: Problem };

/** Make the Environment ready; every message arrives on `on`, the last is
 * `ready` or `failed`. */
export async function ensureEnvironment(
  reinstall: boolean,
  on: (said: EnvironmentSaid) => void,
): Promise<void> {
  const channel = new Channel<EnvironmentSaid>();
  channel.onmessage = on;
  await invoke("environment_ensure", { reinstall, on: channel });
}

export function cancelEnvironment(): Promise<void> {
  return invoke("environment_cancel");
}

export function currentEnvironment(): Promise<Kept | null> {
  return invoke("environment_current");
}

// -- the Project and its Engine ---------------------------------------------

/** A notification from the Engine (docs/engine.md), or `engine.exited`. */
export interface EngineNote {
  method: string;
  params: Record<string, unknown>;
}

/** Ask the user for a folder; only folders picked here can be opened. */
export function pickFolder(title: string): Promise<string | null> {
  return invoke("folder_pick", { title });
}

/** Open (or, with `create`, make) a Project in a new Engine; its
 * notifications arrive on `on`. Answers the Project's state. */
export function openProject(
  root: string,
  create: boolean,
  on: (note: EngineNote) => void,
  classes?: { name?: string; color?: string }[],
): Promise<Record<string, unknown>> {
  const channel = new Channel<EngineNote>();
  channel.onmessage = on;
  return invoke("project_open", { root, create, classes: classes ?? null, on: channel });
}

export function closeProject(): Promise<void> {
  return invoke("project_close");
}

/** Any Engine method but opening, closing and Jobs. With a `ticket`, its
 * `request.progress` notifications carry it and `cancelEngineCall` stops it. */
export function engineCall<T = unknown>(
  method: string,
  params: Record<string, unknown> = {},
  ticket?: string,
): Promise<T> {
  return invoke("engine_call", { method, params, ticket: ticket ?? null });
}

export function cancelEngineCall(ticket: string): Promise<boolean> {
  return invoke("engine_cancel", { ticket });
}

/** An Array file's descriptor (docs/engine.md). */
export interface ArrayRef {
  path: string;
  dtype: "uint8" | "int8" | "uint16" | "int16" | "uint32" | "int32" | "float32" | "float64";
  shape: number[];
}

export async function readArray(ref: ArrayRef): Promise<ArrayBuffer> {
  return invoke<ArrayBuffer>("array_read", { path: ref.path });
}

/** Write an inbound Array file under `.arrays/in/<name>.bin`. */
export async function writeArray(
  name: string,
  data: ArrayBufferView,
  dtype: ArrayRef["dtype"],
  shape: number[],
): Promise<ArrayRef> {
  const path = `.arrays/in/${name}.bin`;
  const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  await invoke("array_write", bytes, { headers: { "x-path": path } });
  return { path, dtype, shape };
}

// -- Jobs ---------------------------------------------------------------------

export type JobSaid =
  | { kind: "planned"; plan: { plan: number; kind: string; gpu: boolean; log: string } }
  | { kind: "waiting"; place: number }
  | { kind: "staging"; direction: "in" | "out" }
  | { kind: "started" }
  | { kind: "progress"; fraction: number; message: string }
  | { kind: "line"; line: string; err: boolean }
  | {
      kind: "finished";
      outcome: { ok: boolean; result?: unknown; error?: Problem; stopped?: boolean };
    }
  | { kind: "failed"; problem: Problem };

/** Run a Job; everything it says arrives on `on`, the last message
 * `finished` or `failed`. */
export async function runJob(
  kind: string,
  options: Record<string, unknown>,
  on: (said: JobSaid) => void,
): Promise<void> {
  const channel = new Channel<JobSaid>();
  channel.onmessage = on;
  await invoke("job_run", { kind, options, on: channel });
}

export function cancelJob(plan: number): Promise<boolean> {
  return invoke("job_cancel", { plan });
}

export function gpuLease(): Promise<{ holder: string | null; waiting: string[] }> {
  return invoke("gpu_lease");
}

// -- updates --------------------------------------------------------------------

export interface Offered {
  version: string;
  current: string;
  notes: string | null;
  date: string | null;
}

export type Downloading =
  | { kind: "progress"; done: number; total: number | null }
  | { kind: "installing" };

/** A newer version on GitHub Releases, if any. */
export function checkUpdate(): Promise<Offered | null> {
  return invoke("update_check");
}

/** Download and install the offered update; the app then restarts. */
export async function installUpdate(on: (d: Downloading) => void): Promise<void> {
  const channel = new Channel<Downloading>();
  channel.onmessage = on;
  await invoke("update_install", { on: channel });
}
