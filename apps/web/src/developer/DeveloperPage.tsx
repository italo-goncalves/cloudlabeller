// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

// A temporary page for step 3: open a Project, call the Engine, run Jobs.
// Step 4's screens replace it.

import { useEffect, useState } from "react";
import {
  cancelJob,
  closeProject,
  type EngineNote,
  engineCall,
  isProblem,
  type JobSaid,
  type Kept,
  openProject,
  pickFolder,
  runJob,
} from "../desktop/bridge";

const MAX = 400;

function show(e: unknown): string {
  return isProblem(e) ? `${e.english} (${e.code})` : String(e);
}

function describe(kept: Kept): string {
  const env = kept.environment;
  const where = env.machine === "wsl" ? `WSL ${env.distro}` : env.machine;
  return `${where}${kept.installed ? `, ${kept.installed}` : ""}${kept.own ? "" : " (developer's)"}`;
}

export function DeveloperPage({ kept }: { kept: Kept }) {
  const [state, setState] = useState<Record<string, unknown> | null>(null);
  const [notes, setNotes] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [kinds, setKinds] = useState<{ kind: string; description: string }[]>([]);
  const [kind, setKind] = useState("sfm");
  const [options, setOptions] = useState("{}");
  const [jobLines, setJobLines] = useState<string[]>([]);
  const [progress, setProgress] = useState<{ fraction: number; message: string } | null>(null);
  const [plan, setPlan] = useState<number | null>(null);
  const [method, setMethod] = useState("project.summary");
  const [params, setParams] = useState("{}");
  const [answer, setAnswer] = useState("");

  const note = (n: EngineNote) =>
    setNotes((all) => [...all, `${n.method} ${JSON.stringify(n.params)}`].slice(-MAX));

  useEffect(() => {
    if (!state) return;
    engineCall<{ kinds: { kind: string; description: string }[] }>("jobs.kinds")
      .then((r) => setKinds(r.kinds))
      .catch((e) => setError(show(e)));
  }, [state]);

  async function open(create: boolean) {
    setError(null);
    const folder = await pickFolder(create ? "Where the new Project goes" : "Open a Project");
    if (!folder) return;
    const root = create
      ? `${folder}\\${prompt("Project name", "project") ?? "project"}.clproj`
      : folder;
    try {
      setState(await openProject(root, create, note));
    } catch (e) {
      setError(show(e));
    }
  }

  async function refresh() {
    try {
      setState(await engineCall("project.state"));
    } catch (e) {
      setError(show(e));
    }
  }

  async function call() {
    setError(null);
    try {
      setAnswer(JSON.stringify(await engineCall(method, JSON.parse(params), "dev"), null, 2));
    } catch (e) {
      setAnswer(show(e));
    }
  }

  async function run() {
    setError(null);
    setJobLines([]);
    setProgress(null);
    const line = (s: string) => setJobLines((all) => [...all, s].slice(-MAX));
    const on = (said: JobSaid) => {
      switch (said.kind) {
        case "planned":
          setPlan(said.plan.plan);
          line(`planned #${said.plan.plan} (gpu: ${said.plan.gpu}, log: ${said.plan.log})`);
          break;
        case "waiting":
          line(`waiting for GPU (place ${said.place})`);
          break;
        case "staging":
          line(`staging the Workspace ${said.direction}`);
          break;
        case "started":
          line("started");
          break;
        case "progress":
          setProgress({ fraction: said.fraction, message: said.message });
          break;
        case "line":
          line(said.err ? `! ${said.line}` : said.line);
          break;
        case "finished":
          setPlan(null);
          line(`finished: ${JSON.stringify(said.outcome)}`);
          refresh();
          break;
        case "failed":
          setPlan(null);
          line(`failed: ${show(said.problem)}`);
          break;
      }
    };
    try {
      await runJob(kind, JSON.parse(options), on);
    } catch (e) {
      setError(show(e));
    }
  }

  const button = "h-control rounded-md bg-surface-2 px-3 hover:bg-surface-3 disabled:opacity-50";
  const area = "w-full rounded-md bg-surface-1 p-2 font-mono text-table";

  return (
    <section className="flex h-full flex-col gap-3 overflow-auto p-4">
      <header className="flex items-center gap-2">
        <h2 className="mr-auto text-lg font-semibold">Developer page</h2>
        <span className="text-fg-muted">Environment: {describe(kept)}</span>
        <button type="button" className={button} onClick={() => open(false)}>
          Open Project…
        </button>
        <button type="button" className={button} onClick={() => open(true)}>
          New Project…
        </button>
        <button
          type="button"
          className={button}
          disabled={!state}
          onClick={() => closeProject().then(() => setState(null))}
        >
          Close
        </button>
      </header>
      {error && <p className="text-danger">{error}</p>}
      {state && (
        <div className="grid grid-cols-2 gap-3">
          <div className="flex flex-col gap-2">
            <h3 className="font-semibold">Project</h3>
            <pre className={`${area} max-h-64 overflow-auto`}>{JSON.stringify(state, null, 2)}</pre>
            <h3 className="font-semibold">Engine call</h3>
            <input className={area} value={method} onChange={(e) => setMethod(e.target.value)} />
            <textarea
              className={area}
              rows={3}
              value={params}
              onChange={(e) => setParams(e.target.value)}
            />
            <button type="button" className={button} onClick={call}>
              Call
            </button>
            <pre className={`${area} max-h-48 overflow-auto`}>{answer}</pre>
          </div>
          <div className="flex flex-col gap-2">
            <h3 className="font-semibold">Job</h3>
            <select className={area} value={kind} onChange={(e) => setKind(e.target.value)}>
              {kinds.map((k) => (
                <option key={k.kind} value={k.kind}>
                  {k.kind}: {k.description}
                </option>
              ))}
            </select>
            <textarea
              className={area}
              rows={3}
              value={options}
              onChange={(e) => setOptions(e.target.value)}
            />
            <div className="flex gap-2">
              <button type="button" className={button} disabled={plan !== null} onClick={run}>
                Run
              </button>
              <button
                type="button"
                className={button}
                disabled={plan === null}
                onClick={() => plan !== null && cancelJob(plan)}
              >
                Cancel
              </button>
            </div>
            {progress && (
              <p>
                {Math.round(progress.fraction * 100)}% {progress.message}
              </p>
            )}
            <pre className={`${area} h-64 overflow-auto`}>{jobLines.join("\n")}</pre>
            <h3 className="font-semibold">Engine notifications</h3>
            <pre className={`${area} h-40 overflow-auto`}>{notes.join("\n")}</pre>
          </div>
        </div>
      )}
    </section>
  );
}
