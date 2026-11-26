// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

// The U-Net's dialogs: its settings, training, and predicting with a saved model.

import { cn } from "@geokit/ui";
import { useCallback, useEffect, useState } from "react";
import * as api from "../engine/api";
import { job } from "../jobs/flows";
import { useApp } from "../state/app";
import { Button, Check, Dialog, Field, NumberInput, Radios, TextInput } from "../ui/controls";
import { useClose } from "./common";
import { useFacts } from "./reconstruction";

export function ModelSpecDialog() {
  const app = useApp();
  const close = useClose();
  const [spec, setSpec] = useState<api.ModelSpec | null>(null);
  const [preview, setPreview] = useState<{ size: [number, number] | null; problem: string | null }>(
    { size: null, problem: null },
  );
  useEffect(() => {
    api
      .modelSpec()
      .then((r) => {
        setSpec(r.spec);
        setPreview({ size: r.training_size, problem: r.problem });
      })
      .catch((e) => app.fail(e, "Couldn't read the model's settings"));
  }, [app.fail]);
  const set = (p: Partial<api.ModelSpec>) => spec && setSpec({ ...spec, ...p });
  const save = async () => {
    if (!spec) return;
    try {
      const r = await api.setModelSpec(spec);
      if (r.problem) setPreview({ size: r.training_size, problem: r.problem });
      else close();
    } catch (e) {
      app.fail(e, "Couldn't save the model's settings");
    }
  };
  return (
    <Dialog
      title="Model settings"
      onClose={close}
      footer={
        <>
          <Button onClick={close}>Cancel</Button>
          <Button kind="primary" disabled={!spec} onClick={save}>
            Save
          </Button>
        </>
      }
    >
      {spec && (
        <>
          <Field
            label="Resolution divisor"
            inline
            hint="Photos are trained at their size over this (snapped to the U-Net's multiple)."
          >
            {(id) => (
              <NumberInput
                id={id}
                min={1}
                max={100}
                value={spec.resolution_divisor}
                onChange={(v) => set({ resolution_divisor: v })}
              />
            )}
          </Field>
          <Field label="Channels in the first layer" inline>
            {(id) => (
              <NumberInput
                id={id}
                min={1}
                max={256}
                value={spec.channels}
                onChange={(v) => set({ channels: v })}
              />
            )}
          </Field>
          <Field label="Blocks (depth)" inline>
            {(id) => (
              <NumberInput
                id={id}
                min={1}
                max={8}
                value={spec.blocks}
                onChange={(v) => set({ blocks: v })}
              />
            )}
          </Field>
          <Field label="Dropout" inline>
            {(id) => (
              <NumberInput
                id={id}
                min={0}
                max={0.9}
                step={0.05}
                value={spec.dropout}
                onChange={(v) => set({ dropout: v })}
              />
            )}
          </Field>
          <Field label="Filter size (odd)" inline>
            {(id) => (
              <NumberInput
                id={id}
                min={1}
                max={11}
                step={2}
                value={spec.filter_size}
                onChange={(v) => set({ filter_size: v })}
              />
            )}
          </Field>
          {preview.problem ? (
            <p className="text-danger">{preview.problem}</p>
          ) : (
            preview.size && (
              <p className="text-fg-muted">
                Trains at {preview.size[0]} × {preview.size[1]} px (when saved).
              </p>
            )
          )}
        </>
      )}
    </Dialog>
  );
}

const fact = (m: api.ModelManifest | undefined, k: string) =>
  m && m[k] !== undefined ? String(m[k]) : "—";

export function TrainDialog() {
  const app = useApp();
  const close = useClose();
  const facts = useFacts("train");
  const [manifests, setManifests] = useState<api.ModelManifest[]>([]);
  const [name, setName] = useState("unet");
  const [resume, setResume] = useState(false);
  const [epochs, setEpochs] = useState(100);
  const [batch, setBatch] = useState(2);
  const [val, setVal] = useState(0.2);
  const [rolls, setRolls] = useState(5);
  const [includeAuto, setIncludeAuto] = useState(false);
  useEffect(() => {
    api
      .models()
      .then((r) => setManifests(r.models))
      .catch(() => setManifests([]));
  }, []);
  const saved = manifests.find((m) => m.name === name.trim());
  const user = facts?.user_images ?? 0;
  const auto = facts?.auto_images ?? 0;
  const n = user + (includeAuto ? auto : 0);
  const problem = !name.trim()
    ? "Name the model."
    : n === 0
      ? "No labelled photos to train on: paint some first."
      : saved && resume && Number(saved.n_classes) !== (app.state.project?.schema.length ?? 0)
        ? "The saved model has a different number of classes; start fresh instead."
        : null;
  const run = () => {
    close();
    void job(
      app,
      "train",
      {
        name: name.trim(),
        epochs,
        batch_size: batch,
        val_fraction: val,
        augment_rolls: rolls,
        include_auto: includeAuto,
        resume: Boolean(saved) && resume,
      },
      `Training "${name.trim()}"`,
    ).then((o) => {
      if (!o.ok) return;
      const r = (o.result ?? {}) as Record<string, unknown>;
      const metrics = Object.entries(r)
        .filter(([, v]) => typeof v === "number")
        .map(
          ([k, v]) =>
            `${k}: ${Number(v)
              .toFixed(4)
              .replace(/\.?0+$/, "")}`,
        )
        .join("\n");
      app.tell(
        o.stopped ? "Training stopped; the model is saved" : "Training done",
        metrics || "The model is saved.",
      );
    });
  };
  return (
    <Dialog
      title="Train the U-Net"
      onClose={close}
      footer={
        <>
          <Button onClick={close}>Cancel</Button>
          <Button kind="primary" disabled={problem !== null || !facts} onClick={run}>
            Train
          </Button>
        </>
      }
    >
      <p className="mb-3 text-fg-muted">
        {user} photos labelled by you and {auto} from the cloud.
      </p>
      <Field label="Model name">
        {(id) => (
          <>
            <TextInput id={id} list={`${id}-models`} value={name} onChange={setName} />
            <datalist id={`${id}-models`}>
              {manifests.map((m) => (
                <option key={m.name} value={m.name} />
              ))}
            </datalist>
          </>
        )}
      </Field>
      {saved && (
        <div className="mb-3 rounded-md bg-surface-2 p-2">
          <p className="mb-2 text-fg-muted">
            A model of that name exists: {fact(saved, "n_classes")} classes, made{" "}
            {fact(saved, "created")}.
          </p>
          <Radios
            name="resume"
            value={resume ? "resume" : "fresh"}
            onChange={(v) => setResume(v === "resume")}
            options={[
              { value: "fresh", label: "Start fresh (overwrite it)" },
              { value: "resume", label: "Continue training it" },
            ]}
          />
        </div>
      )}
      <Field label={saved && resume ? "Epochs (more)" : "Epochs"} inline>
        {(id) => <NumberInput id={id} min={1} max={10000} value={epochs} onChange={setEpochs} />}
      </Field>
      <Field label="Batch size" inline>
        {(id) => <NumberInput id={id} min={1} max={64} value={batch} onChange={setBatch} />}
      </Field>
      <Field label="Validation fraction" inline>
        {(id) => (
          <NumberInput id={id} min={0} max={0.5} step={0.05} value={val} onChange={setVal} />
        )}
      </Field>
      <Field label="Augmentation rolls" inline>
        {(id) => <NumberInput id={id} min={0} max={20} value={rolls} onChange={setRolls} />}
      </Field>
      <Check checked={includeAuto} onChange={setIncludeAuto} disabled={auto === 0}>
        Include {auto} photos labelled from the cloud, at half weight
      </Check>
      {problem && <p className="text-danger">{problem}</p>}
      <p className="text-fg-faint">
        Stop ends training after the current epoch and still saves the model.
      </p>
    </Dialog>
  );
}

const COLUMNS: [string, string][] = [
  ["name", "Name"],
  ["created", "Created"],
  ["input_size", "Input"],
  ["n_classes", "Classes"],
  ["channels", "Channels"],
  ["blocks", "Blocks"],
  ["dropout", "Dropout"],
  ["filter_size", "Filter"],
];

export function PredictDialog() {
  const app = useApp();
  const close = useClose();
  const [manifests, setManifests] = useState<api.ModelManifest[] | null>(null);
  const [chosen, setChosen] = useState<string | null>(null);
  const load = useCallback(() => {
    api
      .models()
      .then((r) => {
        setManifests(r.models);
        setChosen((c) => c ?? r.models[0]?.name ?? null);
      })
      .catch((e) => app.fail(e, "Couldn't list the models"));
  }, [app.fail]);
  useEffect(load, [load]);
  const remove = async () => {
    if (!chosen) return;
    if (
      !(await app.confirm(
        "Delete the model",
        `Delete the saved model "${chosen}"? This can't be undone.`,
        "Delete",
      ))
    ) {
      app.dispatch({ type: "dialog", dialog: { name: "predict" } });
      return;
    }
    try {
      await api.deleteModel(chosen);
      setChosen(null);
    } catch (e) {
      app.fail(e, "Couldn't delete the model");
    }
    app.dispatch({ type: "dialog", dialog: { name: "predict" } });
  };
  const run = (name: string) => {
    close();
    void job(app, "predict", { model: name }, `Predicting with "${name}"`).then((o) => {
      if (o.ok) {
        const predicted = (o.result as { predicted?: string[] })?.predicted ?? [];
        app.dispatch({ type: "log", text: `Predicted ${predicted.length} photos` });
      }
    });
  };
  const show = (v: unknown) =>
    Array.isArray(v) ? v.join(" × ") : v === undefined || v === null ? "—" : String(v);
  const classes = app.state.project?.schema.length ?? 0;
  const pick = manifests?.find((m) => m.name === chosen);
  return (
    <Dialog
      title="Predict every photo"
      wide
      onClose={close}
      footer={
        <>
          <Button kind="danger" className="mr-auto" disabled={!chosen} onClick={remove}>
            Delete…
          </Button>
          <Button onClick={close}>Cancel</Button>
          <Button
            kind="primary"
            disabled={!pick || Number(pick.n_classes) !== classes}
            onClick={() => chosen && run(chosen)}
          >
            Predict
          </Button>
        </>
      }
    >
      <p className="mb-3 text-fg-muted">
        The model labels every photo you haven't labelled yourself.
      </p>
      {manifests === null ? (
        <p className="text-fg-muted">Reading the models…</p>
      ) : manifests.length === 0 ? (
        <p>No saved models: train one first.</p>
      ) : (
        <table className="w-full text-table">
          <thead>
            <tr className="text-left text-fg-muted">
              {COLUMNS.map(([, h]) => (
                <th key={h} className="px-2 py-1 font-normal">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {manifests.map((m) => (
              <tr
                key={m.name}
                className={cn(
                  "cursor-pointer border-t border-border hover:bg-surface-2",
                  m.name === chosen && "bg-selection",
                )}
                onClick={() => setChosen(m.name)}
                onDoubleClick={() => Number(m.n_classes) === classes && run(m.name)}
              >
                {COLUMNS.map(([k]) => (
                  <td key={k} className="px-2 py-1">
                    {show(m[k])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {pick && Number(pick.n_classes) !== classes && (
        <p className="mt-2 text-danger">
          This model has {String(pick.n_classes)} classes and the Project {classes}.
        </p>
      )}
    </Dialog>
  );
}
