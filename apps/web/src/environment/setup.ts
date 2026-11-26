// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

// What the Environment screen shows, folded from the core's messages.

import type { Choice, EnvironmentSaid, Kept, Problem } from "../desktop/bridge";

export const MAX_LINES = 400;

export interface SetupState {
  phase: "starting" | "working" | "ready" | "failed";
  doing: "check" | "install" | "upgrade" | "repair" | null;
  choice: Choice | null;
  step: { i: number; n: number; what: string } | null;
  /** The last lines, each with its number since the start. */
  lines: { n: number; text: string; err: boolean }[];
  seen: number;
  kept: Kept | null;
  problem: Problem | null;
}

export const initialSetup: SetupState = {
  phase: "starting",
  doing: null,
  choice: null,
  step: null,
  lines: [],
  seen: 0,
  kept: null,
  problem: null,
};

export function fold(state: SetupState, said: EnvironmentSaid): SetupState {
  switch (said.kind) {
    case "chose":
      return { ...state, choice: said.choice };
    case "doing":
      return { ...state, phase: "working", doing: said.what };
    case "step":
      return { ...state, phase: "working", step: { i: said.i, n: said.n, what: said.what } };
    case "out":
    case "err":
      return {
        ...state,
        seen: state.seen + 1,
        lines: [
          ...state.lines,
          { n: state.seen + 1, text: said.line, err: said.kind === "err" },
        ].slice(-MAX_LINES),
      };
    case "ready":
      return { ...state, phase: "ready", kept: said.kept, problem: null };
    case "failed":
      return { ...state, phase: "failed", problem: said.problem };
  }
}

/** Why the Environment goes where it goes, in words. */
export function explain(choice: Choice): string {
  switch (choice.why) {
    case "gpu":
      return `WSL (${choice.own.machine === "wsl" ? choice.own.distro : "?"}) reaches your GPU (${
        choice.gpu ?? "NVIDIA"
      }): reconstruction and the U-Net run there, on the GPU.`;
    case "no_gpu_in_wsl":
      return "WSL is installed, but no distribution reaches a GPU (nvidia-smi found none), so CloudLabeller runs natively on Windows, on the CPU.";
    case "no_wsl":
      return "WSL isn't installed, so CloudLabeller runs natively on Windows, on the CPU. TensorFlow has no GPU support on native Windows: to train and predict on an NVIDIA GPU, install WSL 2 and run Reinstall.";
    case "not_windows":
      return "CloudLabeller runs natively on this system.";
  }
}

/** The step codes geokit's install tells, in words. */
export const STEP_WORDS: Record<string, string> = {
  remove: "Removing the old Environment",
  python: "Getting Python",
  venv: "Making the Environment",
  lockfile: "Installing the pinned packages",
  packages: "Installing packages",
  wheels: "Installing CloudLabeller",
  check: "Checking it works",
  wsl: "Installing in WSL",
};

export const WSL_GUIDE = "https://learn.microsoft.com/windows/wsl/install";
