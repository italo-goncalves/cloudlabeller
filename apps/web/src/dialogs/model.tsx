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
  const [defaultTile, setDefaultTile] = useState<number | null>(null);
  useEffect(() => {
    api
      .modelSpec()
      .then((r) => {
        setSpec(r.spec);
        setPreview({ size: r.training_size, problem: r.problem });
        setDefaultTile(r.default_tile_size);
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
          <p className="mb-3 text-fg-muted">
            The network learns and predicts on Tiles: square pieces of the photos, cut on an
            overlapping grid and shrunk before it sees them. Photos of any size give Tiles of the
            same size, so one model covers Surveys taken with different cameras.
          </p>
          <Field
            label="Tile size (photo pixels)"
            inline
            hint={
              defaultTile
                ? `For these photos: ${defaultTile} px, the first power of two above a quarter of the longest side.`
                : undefined
            }
          >
            {(id) => (
              <NumberInput
                id={id}
                min={32}
                max={8192}
                step={32}
                value={spec.tile_size}
                onChange={(v) => set({ tile_size: v })}
              />
            )}
          </Field>
          <Field label="Shrink by" inline>
            {(id) => (
              <select
                id={id}
                className="h-control rounded-md border border-border bg-surface-0 px-2"
                value={spec.shrink}
                onChange={(e) => set({ shrink: Number(e.target.value) })}
              >
                {[1, 2, 4, 8].map((k) => (
                  <option key={k} value={k}>
                    {k === 1 ? "1 (full detail)" : k}
                  </option>
                ))}
              </select>
            )}
          </Field>
          <Field label="Overlap" inline hint="Of a Tile, shared with its neighbours (0 to 0.45).">
            {(id) => (
              <NumberInput
                id={id}
                min={0}
                max={0.45}
                step={0.05}
                value={spec.overlap}
                onChange={(v) => set({ overlap: v })}
              />
            )}
          </Field>
          <Field
            label="Least labelled"
            inline
            hint="Training skips Tiles with less than this fraction labelled."
          >
            {(id) => (
              <NumberInput
                id={id}
                min={0}
                max={1}
                step={0.01}
                value={spec.min_labelled}
                onChange={(v) => set({ min_labelled: v })}
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
          <Check checked={Boolean(spec.normals)} onChange={(normals) => set({ normals })}>
            Also learn from the cloud's normals{" "}
            <span className="rounded bg-surface-2 px-1 text-fg-muted">experimental</span>
          </Check>
          {spec.normals && (
            <p className="mb-2 text-fg-muted">
              Needs a dense cloud: the model trains on and predicts only photos with a solved camera
              in a Survey that has a dense Reconstruction, and Train and Predict refuse Surveys
              without one. On one site it halved the errors.
            </p>
          )}
          {preview.problem ? (
            <p className="text-danger">{preview.problem}</p>
          ) : (
            preview.size && (
              <p className="text-fg-muted">
                The network sees {preview.size[0]} × {preview.size[1]} px per Tile (when saved).
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

/** Which Surveys a Job covers: all of them by default. */
function SurveyPicker({ value, onChange }: { value: string[]; onChange: (ids: string[]) => void }) {
  const surveys = useApp().state.project?.surveys ?? [];
  if (surveys.length < 2) return null;
  return (
    <Field label="Surveys">
      {() => (
        <div className="flex flex-col">
          {surveys.map((s) => (
            <Check
              key={s.id}
              checked={value.includes(s.id)}
              onChange={(on) =>
                onChange(
                  on
                    ? surveys.map((x) => x.id).filter((id) => id === s.id || value.includes(id))
                    : value.filter((id) => id !== s.id),
                )
              }
            >
              {s.name}
            </Check>
          ))}
        </div>
      )}
    </Field>
  );
}

export function TrainDialog() {
  const app = useApp();
  const close = useClose();
  const facts = useFacts("train");
  const [manifests, setManifests] = useState<api.ModelManifest[]>([]);
  const [name, setName] = useState("unet");
  const [resume, setResume] = useState(false);
  const [steps, setSteps] = useState(50000);
  const [patience, setPatience] = useState(10);
  const [batch, setBatch] = useState(2);
  const [val, setVal] = useState(0.2);
  const [rolls, setRolls] = useState(5);
  const [includeAuto, setIncludeAuto] = useState(false);
  const [byUncertainty, setByUncertainty] = useState(false);
  const all = app.state.project?.surveys.map((s) => s.id) ?? [];
  const [surveys, setSurveys] = useState<string[]>(all);
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
    : surveys.length === 0
      ? "Choose at least one Survey."
      : saved && resume && !saved.tiled
        ? "That model was trained on whole photos, before Tiles: start fresh instead."
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
        steps,
        patience,
        batch_size: batch,
        val_fraction: val,
        augment_rolls: rolls,
        include_auto: includeAuto,
        weight_by_uncertainty: includeAuto && byUncertainty,
        resume: Boolean(saved) && resume,
        ...(surveys.length < all.length ? { surveys } : {}),
      },
      `Training "${name.trim()}"`,
    ).then((o) => {
      if (!o.ok) return;
      const r = (o.result ?? {}) as Record<string, unknown>;
      const kept =
        typeof r.best_epoch === "number"
          ? `Kept epoch ${r.best_epoch} of ${r.epochs_run}${r.stopped_early ? " (stopped early)" : ""}.
`
          : "";
      const metrics = Object.entries((r.metrics ?? {}) as Record<string, unknown>)
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
        kept + (metrics || "The model is saved."),
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
        {user} photos labelled by you and {auto} from the cloud, in every Survey.
      </p>
      <SurveyPicker value={surveys} onChange={setSurveys} />
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
      <Field
        label={saved && resume ? "Steps (more)" : "Steps"}
        inline
        hint="Batches to train for. The epochs (every Tile's windows once) are as many as that takes, rounded up, worked out once the photos are cut."
      >
        {(id) => (
          <NumberInput
            id={id}
            min={1}
            max={10000000}
            step={1000}
            value={steps}
            onChange={setSteps}
          />
        )}
      </Field>
      <Field
        label="Stop early after"
        inline
        hint="Epochs without a better validation loss (0: never stop early)."
      >
        {(id) => <NumberInput id={id} min={0} max={1000} value={patience} onChange={setPatience} />}
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
      <Check checked={byUncertainty} onChange={setByUncertainty} disabled={!includeAuto}>
        Weigh their labels by how sure the cloud is of them, too
      </Check>
      <p className="text-fg-faint">
        The model never learns from predictions, its own or through the cloud, and it is validated
        on photos you painted only.
      </p>
      {problem && <p className="text-danger">{problem}</p>}
      <p className="text-fg-faint">
        The model saved is the epoch with the lowest validation loss (training loss when nothing is
        held out). Stop ends training after the current epoch and still saves it.
      </p>
    </Dialog>
  );
}

/** The models table: what a manifest holds, its spec's values from `spec`. */
const COLUMNS: [string, string, (m: api.ModelManifest) => unknown][] = [
  ["name", "Name", (m) => m.name],
  ["created", "Created", (m) => m.created],
  ["tile", "Tile", (m) => (m.tiled ? `${m.spec?.tile_size} px ÷${m.spec?.shrink}` : "whole photo")],
  ["inputs", "Inputs", (m) => (m.spec?.normals ? "RGB + normals" : "RGB")],
  ["n_classes", "Classes", (m) => m.n_classes],
  ["channels", "Channels", (m) => m.spec?.channels],
  ["blocks", "Blocks", (m) => m.spec?.blocks],
  ["dropout", "Dropout", (m) => m.spec?.dropout],
  ["filter_size", "Filter", (m) => m.spec?.filter_size],
];

export function PredictDialog() {
  const app = useApp();
  const close = useClose();
  const [manifests, setManifests] = useState<api.ModelManifest[] | null>(null);
  const [chosen, setChosen] = useState<string | null>(null);
  const all = app.state.project?.surveys.map((s) => s.id) ?? [];
  const [surveys, setSurveys] = useState<string[]>(all);
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
    const options = { model: name, ...(surveys.length < all.length ? { surveys } : {}) };
    void job(app, "predict", options, `Predicting with "${name}"`).then((o) => {
      if (o.ok) {
        const predicted = (o.result as { predicted?: unknown[] })?.predicted ?? [];
        app.dispatch({ type: "log", text: `Predicted ${predicted.length} photos` });
      }
    });
  };
  const show = (v: unknown) =>
    Array.isArray(v) ? v.join(" × ") : v === undefined || v === null ? "—" : String(v);
  const classes = app.state.project?.schema.length ?? 0;
  const pick = manifests?.find((m) => m.name === chosen);
  const usable = (m: api.ModelManifest) => m.tiled && Number(m.n_classes) === classes;
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
            disabled={!pick || !usable(pick) || surveys.length === 0}
            onClick={() => chosen && run(chosen)}
          >
            Predict
          </Button>
        </>
      }
    >
      <p className="mb-3 text-fg-muted">
        The model predicts every photo. Its labels fill only the gaps your labels and the cloud's
        leave, and nothing trains on them.
      </p>
      <SurveyPicker value={surveys} onChange={setSurveys} />
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
                onDoubleClick={() => usable(m) && run(m.name)}
              >
                {COLUMNS.map(([k, , get]) => (
                  <td key={k} className={cn("px-2 py-1", !m.tiled && "text-fg-faint")}>
                    {show(get(m))}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {pick && !pick.tiled && (
        <p className="mt-2 text-danger">
          This model was trained on whole photos, before Tiles: train a new one.
        </p>
      )}
      {pick?.tiled && Number(pick.n_classes) !== classes && (
        <p className="mt-2 text-danger">
          This model has {String(pick.n_classes)} classes and the Project {classes}.
        </p>
      )}
    </Dialog>
  );
}
