// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

import { useCallback, useEffect, useReducer, useRef } from "react";
import { cancelEnvironment, ensureEnvironment, type Kept } from "../desktop/bridge";
import { explain, fold, initialSetup, STEP_WORDS, WSL_GUIDE } from "./setup";

type Action = Parameters<typeof fold>[1] | { kind: "reset" };

function reduce(state: typeof initialSetup, action: Action) {
  return action.kind === "reset" ? initialSetup : fold(state, action);
}

/** The Environment screen: it sets the Environment up unattended on start
 * and hands over to the app once it is ready. */
export function EnvironmentSetup({ onReady }: { onReady: (kept: Kept) => void }) {
  const [state, dispatch] = useReducer(reduce, initialSetup);
  const started = useRef(false);

  const start = useCallback((reinstall: boolean) => {
    dispatch({ kind: "reset" });
    ensureEnvironment(reinstall, dispatch).catch((e: unknown) =>
      dispatch({
        kind: "failed",
        problem:
          typeof e === "object" && e !== null && "code" in e
            ? (e as { code: string; english: string })
            : { code: "environment.call", english: String(e) },
      }),
    );
  }, []);

  useEffect(() => {
    if (!started.current) {
      started.current = true;
      start(false);
    }
  }, [start]);

  useEffect(() => {
    if (state.phase === "ready" && state.kept) onReady(state.kept);
  }, [state.phase, state.kept, onReady]);

  const working = state.phase === "starting" || state.phase === "working";
  const title =
    state.doing === "check"
      ? "Checking the Environment"
      : state.doing === "repair"
        ? "Repairing the Environment"
        : state.doing === "upgrade"
          ? "Updating the Environment"
          : "Setting up the Environment";

  return (
    <section className="mx-auto flex h-full w-full max-w-3xl flex-col gap-4 py-8">
      <h2 className="text-lg font-semibold">{title}</h2>
      {state.choice && <p className="text-fg-muted">{explain(state.choice)}</p>}
      {state.choice?.why === "no_wsl" && (
        <p>
          <a className="text-accent underline" href={WSL_GUIDE} target="_blank" rel="noreferrer">
            How to install WSL (Microsoft)
          </a>
        </p>
      )}
      {state.step && (
        <p>
          Step {state.step.i} of {state.step.n}: {STEP_WORDS[state.step.what] ?? state.step.what}
        </p>
      )}
      {state.problem && (
        <p className="rounded-md border border-danger p-3 text-danger">
          {state.problem.english} <span className="text-fg-faint">({state.problem.code})</span>
        </p>
      )}
      <pre className="min-h-0 flex-1 overflow-auto rounded-md bg-surface-1 p-3 font-mono text-table">
        {state.lines.map((l) => (
          <div key={l.n} className={l.err ? "text-fg-muted" : undefined}>
            {l.text}
          </div>
        ))}
      </pre>
      <div className="flex gap-2">
        {working ? (
          <button
            type="button"
            className="h-control rounded-md bg-surface-2 px-4 hover:bg-surface-3"
            onClick={() => cancelEnvironment()}
          >
            Cancel
          </button>
        ) : (
          <>
            <button
              type="button"
              className="h-control rounded-md bg-accent px-4 text-accent-fg hover:bg-accent-hover"
              onClick={() => start(false)}
            >
              Try again
            </button>
            <button
              type="button"
              className="h-control rounded-md bg-surface-2 px-4 hover:bg-surface-3"
              onClick={() => start(true)}
            >
              Reinstall from nothing
            </button>
          </>
        )}
      </div>
    </section>
  );
}
