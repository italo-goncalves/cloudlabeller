// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

// The window's state, handed to every screen, and what screens do with the
// core: run a Job or a long Engine request with the status bar following it,
// say a Problem, ask a question.

import {
  createContext,
  type Dispatch,
  type ReactNode,
  useCallback,
  useContext,
  useMemo,
  useReducer,
  useRef,
} from "react";
import {
  cancelEngineCall,
  cancelJob,
  runJob as coreRunJob,
  type EngineNote,
  isProblem,
  type JobSaid,
  type Problem,
} from "../desktop/bridge";
import * as api from "../engine/api";
import { progressLog } from "../jobs/progress";
import { type Action, type AppState, initialState, reduce } from "./store";

export interface JobOutcome {
  ok: boolean;
  result?: unknown;
  error?: Problem;
  stopped?: boolean;
}

export interface App {
  state: AppState;
  dispatch: Dispatch<Action>;
  /** Re-read the Project's state. */
  refresh: () => Promise<void>;
  /** What the Engine says between answers. */
  onNote: (note: EngineNote) => void;
  /** Run a Job to its end, the status bar and log following it. */
  runJob: (kind: string, options: Record<string, unknown>, label: string) => Promise<JobOutcome>;
  /** A long Engine request that can be cancelled, the status bar following it. */
  long: <T>(label: string, call: (ticket: string) => Promise<T>) => Promise<T>;
  /** Stop what the status bar shows. */
  stop: () => void;
  /** Tell the user about a failure. */
  fail: (e: unknown, title?: string) => void;
  /** Tell the user something. */
  tell: (title: string, text: string) => void;
  /** Ask a yes/no question; true for yes. */
  confirm: (title: string, text: string, yes?: string) => Promise<boolean>;
  /** Ask a question with several answers; the one chosen, or null. */
  choose: (title: string, text: string, answers: string[]) => Promise<string | null>;
}

const AppContext = createContext<App | null>(null);

export function useApp(): App {
  const app = useContext(AppContext);
  if (!app) throw new Error("useApp outside AppProvider");
  return app;
}

/** A Problem's words for the user, or any other error's. */
export function words(e: unknown): string {
  if (isProblem(e)) return e.english;
  if (e instanceof Error) return e.message;
  return String(e);
}

let tickets = 0;

export function AppProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reduce, initialState);
  const busyRef = useRef<{ plan?: number; ticket?: string } | null>(null);

  const refresh = useCallback(async () => {
    try {
      dispatch({ type: "project", project: await api.state() });
    } catch (e) {
      dispatch({ type: "log", text: `Couldn't read the Project: ${words(e)}`, err: true });
    }
  }, []);

  const onNote = useCallback(
    (note: EngineNote) => {
      const p = note.params;
      switch (note.method) {
        case "project.changed":
          // A change in another Survey than the one shown only refreshes.
          if (typeof p.survey === "string" && p.survey !== api.currentSurveyOrNull()) {
            void refresh();
            break;
          }
          dispatch({
            type: "changed",
            parts: (p.parts as string[]) ?? [],
            ...(Array.isArray(p.images) ? { images: p.images as string[] } : {}),
          });
          void refresh();
          break;
        case "request.progress":
          dispatch({
            type: "progress",
            fraction: typeof p.fraction === "number" ? p.fraction : null,
            message: String(p.message ?? ""),
          });
          break;
        case "engine.log":
          dispatch({ type: "log", text: `Engine: ${String(p.message ?? "")}`, err: true });
          break;
        case "engine.exited":
          dispatch({ type: "closed" });
          dispatch({
            type: "dialog",
            dialog: {
              name: "message",
              data: {
                title: "The Engine stopped",
                text: "The Engine ended unexpectedly, so the Project was closed. Open it again to carry on; its engine.log says why.",
              },
            },
          });
          break;
      }
    },
    [refresh],
  );

  const tell = useCallback((title: string, text: string) => {
    dispatch({ type: "dialog", dialog: { name: "message", data: { title, text } } });
  }, []);

  const fail = useCallback(
    (e: unknown, title = "That didn't work") => {
      if (isProblem(e) && e.code === "request.cancelled") return;
      dispatch({ type: "log", text: `${title}: ${words(e)}`, err: true });
      tell(title, words(e));
    },
    [tell],
  );

  const choose = useCallback(
    (title: string, text: string, answers: string[]) =>
      new Promise<string | null>((resolve) => {
        dispatch({
          type: "dialog",
          dialog: { name: "confirm", data: { title, text, answers, resolve } },
        });
      }),
    [],
  );

  const confirm = useCallback(
    async (title: string, text: string, yes = "OK") => (await choose(title, text, [yes])) === yes,
    [choose],
  );

  const runJob = useCallback(
    (kind: string, options: Record<string, unknown>, label: string) =>
      new Promise<JobOutcome>((resolve, reject) => {
        dispatch({ type: "busy", busy: { label, fraction: null, message: "", waiting: null } });
        dispatch({ type: "log", text: `=== ${label} ===` });
        const progressLines = progressLog();
        let settled = false;
        const end = (outcome: JobOutcome | null, problem?: Problem) => {
          if (settled) return;
          settled = true;
          busyRef.current = null;
          dispatch({ type: "busy", busy: null });
          if (outcome) resolve(outcome);
          else reject(problem);
        };
        const on = (said: JobSaid) => {
          switch (said.kind) {
            case "planned":
              busyRef.current = { plan: said.plan.plan };
              dispatch({
                type: "busy",
                busy: { label, plan: said.plan.plan, fraction: null, message: "", waiting: null },
              });
              break;
            case "waiting":
              dispatch({ type: "waiting", place: said.place });
              dispatch({ type: "log", text: `Waiting for the GPU (place ${said.place} in line)` });
              break;
            case "staging":
              dispatch({
                type: "progress",
                fraction: null,
                message: said.direction === "in" ? "Copying to WSL" : "Copying back",
              });
              break;
            case "started":
              dispatch({ type: "progress", fraction: null, message: "Started" });
              break;
            case "progress": {
              dispatch({ type: "progress", fraction: said.fraction, message: said.message });
              const text = progressLines.line(said.fraction, said.message);
              if (text) dispatch({ type: "log", text });
              break;
            }
            case "line":
              dispatch({ type: "log", text: said.line, err: said.err });
              break;
            case "finished": {
              const o = said.outcome;
              if (o.ok)
                dispatch({ type: "log", text: o.stopped ? `${label}: stopped` : `${label}: done` });
              else if (o.error)
                dispatch({ type: "log", text: `${label}: ${o.error.english}`, err: true });
              void refresh();
              end(o);
              break;
            }
            case "failed":
              dispatch({ type: "log", text: `${label}: ${said.problem.english}`, err: true });
              end(null, said.problem);
              break;
          }
        };
        // Survey-scoped Jobs read `survey`; the others ignore it.
        coreRunJob(kind, { survey: api.currentSurveyOrNull(), ...options }, on).catch(
          (e: unknown) => end(null, isProblem(e) ? e : { code: "job.call", english: words(e) }),
        );
      }),
    [refresh],
  );

  const long = useCallback(async <T,>(label: string, call: (ticket: string) => Promise<T>) => {
    const ticket = `t${++tickets}`;
    busyRef.current = { ticket };
    dispatch({ type: "busy", busy: { label, ticket, fraction: null, message: "", waiting: null } });
    try {
      return await call(ticket);
    } finally {
      busyRef.current = null;
      dispatch({ type: "busy", busy: null });
    }
  }, []);

  const stop = useCallback(() => {
    const b = busyRef.current;
    if (b?.plan !== undefined) void cancelJob(b.plan);
    else if (b?.ticket) void cancelEngineCall(b.ticket);
  }, []);

  const app = useMemo<App>(
    () => ({ state, dispatch, refresh, onNote, runJob, long, stop, fail, tell, confirm, choose }),
    [state, refresh, onNote, runJob, long, stop, fail, tell, confirm, choose],
  );
  return <AppContext.Provider value={app}>{children}</AppContext.Provider>;
}
