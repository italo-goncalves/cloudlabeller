// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

// A Survey's details (making one, or editing the current one), and the
// current Survey's Reconstructions: which one is shown, renaming,
// deleting, and copying labels from one into the shown one.

import { cn } from "@geokit/ui";
import { useCallback, useEffect, useState } from "react";
import * as api from "../engine/api";
import { createSurvey } from "../shell/surveys";
import { useApp } from "../state/app";
import { Button, Dialog, Field, TextInput } from "../ui/controls";
import { useClose } from "./common";

const INPUT =
  "h-control w-full rounded-md border border-border bg-surface-0 px-2 focus:border-accent focus:outline-none";

/** Making a Survey (`create`) or editing the current one. */
export function SurveyDialog({ create }: { create: boolean }) {
  const app = useApp();
  const close = useClose();
  const project = app.state.project;
  const current = project?.surveys.find((s) => s.id === project.survey);
  const start = create ? null : current;
  const [name, setName] = useState(start?.name ?? `Survey ${(project?.surveys.length ?? 0) + 1}`);
  const [when, setWhen] = useState(start?.timestamp?.slice(0, 16) ?? "");
  const [operator, setOperator] = useState(start?.operator ?? "");
  const [camera, setCamera] = useState(start?.camera ?? "");
  const [notes, setNotes] = useState(start?.notes ?? "");
  const meta = (): api.SurveyMeta => ({
    name: name.trim(),
    timestamp: when ? (when.length === 16 ? `${when}:00` : when) : "",
    operator,
    camera,
    notes,
  });
  const save = async () => {
    if (!name.trim()) return;
    if (create) {
      const m = meta();
      if (!m.timestamp) delete m.timestamp;
      if ((await createSurvey(app, m)) !== null) close();
      return;
    }
    if (!current) return;
    try {
      await api.surveyUpdate(current.id, meta());
      await app.refresh();
      close();
    } catch (e) {
      app.fail(e, "Couldn't save the Survey");
    }
  };
  return (
    <Dialog
      title={create ? "New Survey" : "Survey"}
      onClose={close}
      footer={
        <>
          <Button onClick={close}>Cancel</Button>
          <Button kind="primary" disabled={!name.trim()} onClick={() => void save()}>
            {create ? "Make it" : "Save"}
          </Button>
        </>
      }
    >
      <Field label="Name">{(id) => <TextInput id={id} value={name} onChange={setName} />}</Field>
      <Field
        label="When"
        hint={
          create
            ? "Left empty, it is set from the photos' EXIF once they are added."
            : "When the photos were taken."
        }
      >
        {(id) => (
          <input
            id={id}
            type="datetime-local"
            className={INPUT}
            value={when}
            onChange={(e) => setWhen(e.target.value)}
          />
        )}
      </Field>
      <Field label="Operator">
        {(id) => <TextInput id={id} value={operator} onChange={setOperator} />}
      </Field>
      <Field label="Drone or camera">
        {(id) => (
          <TextInput id={id} value={camera} onChange={setCamera} placeholder="e.g. DJI Mavic 3E" />
        )}
      </Field>
      <Field label="Notes">
        {(id) => (
          <textarea
            id={id}
            className={cn(INPUT, "h-20 py-1")}
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
          />
        )}
      </Field>
    </Dialog>
  );
}

const STEPS: [keyof api.ReconstructionRow["products"], string][] = [
  ["sfm", "SfM"],
  ["mvs", "dense"],
  ["mesh", "mesh"],
];

/** The current Survey's Reconstructions. */
export function ReconstructionsDialog() {
  const app = useApp();
  const close = useClose();
  const [list, setList] = useState<api.Reconstructions | null>(null);
  const [renaming, setRenaming] = useState<{ id: string; name: string } | null>(null);
  const busy = app.state.busy !== null;
  const load = useCallback(() => {
    api
      .reconstructions()
      .then(setList)
      .catch((e) => app.fail(e, "Couldn't list the Reconstructions"));
  }, [app.fail]);
  useEffect(load, [load]);

  const act = async (what: string, call: () => Promise<api.Reconstructions>) => {
    try {
      setList(await call());
      await app.refresh();
    } catch (e) {
      app.fail(e, what);
    }
  };
  const remove = async (r: api.ReconstructionRow) => {
    const ok = await app.confirm(
      `Delete ${r.name}?`,
      `${r.name} and its labels go, and so do the products no other Reconstruction uses ` +
        `(${api.sizeText(r.freed_bytes)}). It can't be undone.`,
      "Delete",
    );
    if (!ok) {
      app.dispatch({ type: "dialog", dialog: { name: "reconstructions" } });
      return;
    }
    await act("Couldn't delete the Reconstruction", async () => {
      const out = await api.reconstructionDelete(r.id);
      app.dispatch({
        type: "log",
        text: `Deleted ${r.name}: ${api.sizeText(out.freed_bytes)} freed`,
      });
      return out;
    });
    app.dispatch({ type: "dialog", dialog: { name: "reconstructions" } });
  };
  const copy = async (r: api.ReconstructionRow) => {
    try {
      const out = await app.long(`Copying labels from ${r.name}`, (t) =>
        api.transferReconstruction(r.id, t),
      );
      app.tell(
        "Labels copied",
        `${out.sparse.toLocaleString()} sparse points` +
          (out.dense !== undefined ? ` and ${out.dense.toLocaleString()} dense points` : "") +
          ` took their labels from ${r.name}.`,
      );
    } catch (e) {
      app.fail(e, "Couldn't copy the labels");
    }
  };

  const rows = list?.reconstructions ?? [];
  const names = new Map(rows.map((r) => [r.id, r.name]));
  return (
    <Dialog
      wide
      title="Reconstructions"
      onClose={close}
      footer={
        <Button kind="primary" onClick={close}>
          Close
        </Button>
      }
    >
      <p className="mb-3 text-fg-muted">
        Running a step again makes a new Reconstruction beside the old one; it shares the steps
        before it. The one marked shown is what the views, labelling and export use.
      </p>
      {rows.length === 0 ? (
        <p className="text-fg-faint">No Reconstruction yet: run SfM.</p>
      ) : (
        <table className="w-full text-left">
          <thead className="text-fg-faint">
            <tr>
              <th className="py-1 font-normal">Name</th>
              <th className="py-1 font-normal">Steps</th>
              <th className="py-1 font-normal">From</th>
              <th className="py-1 font-normal">Aligned</th>
              <th className="py-1 font-normal">Frees</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.id} className={cn("border-t border-border", r.active && "bg-selection")}>
                <td className="py-1 pr-2">
                  {renaming?.id === r.id ? (
                    <form
                      className="flex gap-1"
                      onSubmit={(e) => {
                        e.preventDefault();
                        const name = renaming.name.trim();
                        setRenaming(null);
                        if (name && name !== r.name)
                          void act("Couldn't rename", () => api.reconstructionRename(r.id, name));
                      }}
                    >
                      <TextInput
                        value={renaming.name}
                        onChange={(name) => setRenaming({ id: r.id, name })}
                      />
                      <Button type="submit">OK</Button>
                    </form>
                  ) : (
                    <button
                      type="button"
                      className="text-left hover:underline"
                      title="Rename"
                      onClick={() => setRenaming({ id: r.id, name: r.name })}
                    >
                      {r.name}
                      {r.active && <span className="ml-2 text-fg-faint">(shown)</span>}
                    </button>
                  )}
                </td>
                <td className="py-1 pr-2 text-fg-muted">
                  {STEPS.filter(([k]) => r.products[k])
                    .map(([, w]) => w)
                    .join(", ")}
                </td>
                <td className="py-1 pr-2 text-fg-muted">
                  {r.parent ? (names.get(r.parent) ?? r.parent) : "—"}
                </td>
                <td className="py-1 pr-2 text-fg-muted">{r.aligned ? "yes" : "no"}</td>
                <td className="py-1 pr-2 font-mono text-table text-fg-faint">
                  {api.sizeText(r.freed_bytes)}
                </td>
                <td className="py-1 text-right whitespace-nowrap">
                  {!r.active && (
                    <>
                      <Button
                        disabled={busy}
                        onClick={() =>
                          void act("Couldn't show it", () => api.reconstructionActivate(r.id))
                        }
                      >
                        Show
                      </Button>{" "}
                      <Button
                        disabled={busy}
                        title="Copy its labels into the one shown"
                        onClick={() => void copy(r)}
                      >
                        Copy labels
                      </Button>{" "}
                    </>
                  )}
                  <Button kind="quiet" disabled={busy} onClick={() => void remove(r)}>
                    Delete
                  </Button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </Dialog>
  );
}
