// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

// The reconstruction dialogs: SfM, MVS, the mesh, the whole pipeline,
// downloading COLMAP and cleaning the sparse cloud. Options and defaults as
// in the Qt app; what the machine can do comes from `jobs.facts`.

import { useEffect, useMemo, useState } from "react";
import { type DetailChoice, defaultLevel, detailChoices, MVS_LEVELS, SFM_LEVELS } from "../detail";
import * as api from "../engine/api";
import * as flows from "../jobs/flows";
import { useApp } from "../state/app";
import { Button, Check, Dialog, Field, NumberInput, Radios, Select, Slider } from "../ui/controls";
import { useClose } from "./common";

const FALLBACK_MODELS = [
  "SIMPLE_RADIAL",
  "RADIAL",
  "OPENCV",
  "FULL_OPENCV",
  "PINHOLE",
  "SIMPLE_PINHOLE",
];
const MATCHER_WORDS: Record<string, string> = {
  spatial: "Spatial — uses the photos' GPS (drone surveys)",
  sequential: "Sequential — photos taken in order (video, walk-around)",
  exhaustive: "Exhaustive — every pair (small sets, slow)",
};

export function useFacts(kind: string): api.Facts | null {
  const app = useApp();
  const [facts, setFacts] = useState<api.Facts | null>(null);
  useEffect(() => {
    let live = true;
    api
      .facts(kind)
      .then((f) => live && setFacts(f))
      .catch((e) => {
        if (live) setFacts({});
        app.dispatch({
          type: "log",
          text: `Couldn't ask what ${kind} can do: ${String((e as { english?: string }).english ?? e)}`,
          err: true,
        });
      });
    return () => {
      live = false;
    };
  }, [kind, app.dispatch]);
  return facts;
}

/** A detail level chooser, with a pixel box for Custom. */
function Detail({
  choices,
  initial,
  custom,
  onChange,
}: {
  choices: DetailChoice[];
  initial: number;
  custom: number;
  onChange: (px: number) => void;
}) {
  const [i, setI] = useState(initial);
  const [px, setPx] = useState(custom);
  const isCustom = i >= choices.length;
  // biome-ignore lint/correctness/useExhaustiveDependencies: report the pixel cap whenever it changes
  useEffect(() => onChange(isCustom ? px : (choices[i]?.pixels ?? 0)), [i, px, choices]);
  return (
    <div className="flex gap-2">
      <Select
        value={String(i)}
        onChange={(v) => setI(Number(v))}
        options={[
          ...choices.map((c, k) => ({ value: String(k), label: c.label })),
          { value: String(choices.length), label: "Custom…" },
        ]}
      />
      {isCustom && <NumberInput value={px} min={0} max={20000} step={200} onChange={setPx} />}
    </div>
  );
}

interface SfmOptions {
  matcher: string;
  single_camera: boolean;
  gpu: boolean;
  camera_model: string;
  max_image_size: number;
}

function SfmFields({
  facts,
  value,
  onChange,
}: {
  facts: api.Facts;
  value: SfmOptions;
  onChange: (v: SfmOptions) => void;
}) {
  const app = useApp();
  const choices = useMemo(
    () => detailChoices(SFM_LEVELS, app.state.project?.source_resolution ?? null, 3200),
    [app.state.project],
  );
  const set = (p: Partial<SfmOptions>) => onChange({ ...value, ...p });
  return (
    <>
      <Field label="Matching">
        {() => (
          <Radios
            name="matcher"
            value={value.matcher}
            onChange={(m) => set({ matcher: m })}
            options={(facts.matchers ?? Object.keys(MATCHER_WORDS)).map((m) => ({
              value: m,
              label: MATCHER_WORDS[m] ?? m,
            }))}
          />
        )}
      </Field>
      <Check checked={value.single_camera} onChange={(v) => set({ single_camera: v })}>
        One camera for every photo (one drone, one lens)
      </Check>
      <Check checked={value.gpu} onChange={(v) => set({ gpu: v })} disabled={facts.gpu === false}>
        Use the GPU{facts.gpu === false ? " (none that COLMAP can use here)" : ""}
      </Check>
      <Field label="Camera model">
        {(id) => (
          <Select
            id={id}
            value={value.camera_model}
            onChange={(m) => set({ camera_model: m })}
            options={(facts.camera_models ?? FALLBACK_MODELS).map((m) => ({ value: m, label: m }))}
          />
        )}
      </Field>
      <Field
        label="Detail"
        hint="Photos are scaled down to this before matching. More detail is slower and needs more memory."
      >
        {() => (
          <Detail
            choices={choices}
            initial={defaultLevel(SFM_LEVELS, "medium")}
            custom={3200}
            onChange={(px) => set({ max_image_size: px })}
          />
        )}
      </Field>
    </>
  );
}

const sfmDefaults = (facts: api.Facts): SfmOptions => ({
  matcher: "spatial",
  single_camera: true,
  gpu: facts.gpu !== false,
  camera_model: (facts.camera_models ?? FALLBACK_MODELS)[0] ?? "SIMPLE_RADIAL",
  max_image_size: 0,
});

function GeorefCheck({
  facts,
  value,
  onChange,
}: {
  facts: api.Facts;
  value: boolean;
  onChange: (v: boolean) => void;
}) {
  const gps = facts.gps_images ?? 0;
  return (
    <Check checked={value && gps >= 3} onChange={onChange} disabled={gps < 3}>
      Georeference to EXIF GPS afterwards ({gps} photos have GPS{gps < 3 ? "; it needs 3" : ""})
    </Check>
  );
}

export function SfmDialog() {
  const app = useApp();
  const close = useClose();
  const facts = useFacts("sfm");
  const [opts, setOpts] = useState<SfmOptions | null>(null);
  const [georef, setGeoref] = useState(true);
  useEffect(() => {
    if (facts && !opts) setOpts(sfmDefaults(facts));
  }, [facts, opts]);
  const images = app.state.project?.images.length ?? 0;
  const run = () => {
    if (!opts) return;
    close();
    void flows.sfm(
      app,
      { ...opts },
      georef && (facts?.gps_images ?? 0) >= 3 && !app.state.project?.georeferenced,
    );
  };
  return (
    <Dialog
      title="Run Structure from Motion"
      onClose={close}
      footer={
        <>
          <Button onClick={close}>Cancel</Button>
          <Button kind="primary" disabled={!opts || images === 0} onClick={run}>
            Run
          </Button>
        </>
      }
    >
      {!facts || !opts ? (
        <p className="text-fg-muted">Looking at the photos…</p>
      ) : (
        <>
          {images === 0 && <p className="mb-3 text-danger">Add photos first.</p>}
          <SfmFields facts={facts} value={opts} onChange={setOpts} />
          {!app.state.project?.georeferenced && (
            <GeorefCheck facts={facts} value={georef} onChange={setGeoref} />
          )}
        </>
      )}
    </Dialog>
  );
}

const QUALITY = [
  { value: "standard", label: "Standard — two-pass patch-match (recommended)" },
  { value: "draft", label: "Draft — ~3-4× faster stereo, noisier cloud" },
] as const;

export function MvsDialog() {
  const app = useApp();
  const close = useClose();
  const facts = useFacts("mvs");
  const choices = useMemo(
    () => detailChoices(MVS_LEVELS, app.state.project?.source_resolution ?? null, 2000),
    [app.state.project],
  );
  const [px, setPx] = useState(0);
  const [quality, setQuality] = useState<"standard" | "draft">("standard");
  const run = () => {
    close();
    void flows.mvs(app, { max_image_size: px, quality });
  };
  return (
    <Dialog
      title="Run dense reconstruction (MVS)"
      onClose={close}
      footer={
        <>
          <Button onClick={close}>Cancel</Button>
          <Button kind="primary" disabled={!facts} onClick={run}>
            Run
          </Button>
        </>
      }
    >
      <Field
        label="Detail"
        hint="Patch-match stereo is memory-hungry: start at Medium on a 12 GB GPU."
      >
        {() => (
          <Detail
            choices={choices}
            initial={defaultLevel(MVS_LEVELS, "medium")}
            custom={2000}
            onChange={setPx}
          />
        )}
      </Field>
      <Field label="Quality">
        {(id) => <Select id={id} value={quality} onChange={setQuality} options={[...QUALITY]} />}
      </Field>
      {facts?.colmap === false && (
        <p className="mb-2 text-danger">
          MVS needs COLMAP with CUDA, which this Environment doesn't have.{" "}
          <button
            type="button"
            className="text-accent underline"
            onClick={() => app.dispatch({ type: "dialog", dialog: { name: "colmap" } })}
          >
            Download COLMAP…
          </button>
        </p>
      )}
      <p className="text-fg-muted">
        The mesh is rebuilt afterwards (Poisson, at the dense cloud's density).
      </p>
    </Dialog>
  );
}

const MESH_DEPTHS = [13, 12, 11, 10];

function MeshDetail({
  facts,
  value,
  onChange,
}: {
  facts: api.Facts;
  value: number | null;
  onChange: (d: number | null) => void;
}) {
  const d = facts.suggested_depth;
  return (
    <Select
      value={value === null ? "auto" : String(value)}
      onChange={(v) => onChange(v === "auto" ? null : Number(v))}
      options={[
        {
          value: "auto",
          label: d
            ? `Match the cloud's density — depth ${d} (recommended)`
            : "Match the cloud's density (recommended)",
        },
        ...MESH_DEPTHS.map((x) => ({
          value: String(x),
          label: `Depth ${x}${x === 13 ? " — finest" : x === 10 ? " — coarse" : ""}`,
        })),
      ]}
    />
  );
}

export function MeshDialog() {
  const app = useApp();
  const close = useClose();
  const facts = useFacts("mesh");
  const [method, setMethod] = useState<"poisson" | "delaunay">("poisson");
  const [depth, setDepth] = useState<number | null>(null);
  const [trim, setTrim] = useState(10);
  const [weight, setWeight] = useState(1);
  const [cost, setCost] = useState<{ minutes: number; ram_gb: number; vertices: number } | null>(
    null,
  );
  const effective = depth ?? facts?.suggested_depth ?? null;

  useEffect(() => {
    if (method !== "poisson" || effective === null) {
      setCost(null);
      return;
    }
    let live = true;
    api
      .meshEstimate(method, effective)
      .then((c) => live && setCost(c))
      .catch(() => live && setCost(null));
    return () => {
      live = false;
    };
  }, [method, effective]);

  const run = () => {
    close();
    void flows.mesh(app, {
      method,
      ...(depth !== null ? { depth } : {}),
      trim,
      point_weight: weight,
    });
  };
  const ram = facts?.ram_gb;
  return (
    <Dialog
      title="Create the mesh"
      onClose={close}
      footer={
        <>
          <Button onClick={close}>Cancel</Button>
          <Button kind="primary" disabled={!facts} onClick={run}>
            Create
          </Button>
        </>
      }
    >
      {!facts ? (
        <p className="text-fg-muted">Looking at the dense cloud…</p>
      ) : (
        <>
          <Field label="Method">
            {() => (
              <Radios
                name="method"
                value={method}
                onChange={setMethod}
                options={[
                  { value: "poisson", label: "Poisson — watertight, smooths noise" },
                  {
                    value: "delaunay",
                    label: "Delaunay — keeps the finest detail",
                    disabled: !facts.delaunay,
                  },
                ]}
              />
            )}
          </Field>
          {!facts.delaunay && (
            <p className="-mt-2 mb-3 text-fg-faint">
              Delaunay needs COLMAP's dense workspace and colmap.exe.
            </p>
          )}
          {method === "poisson" && (
            <>
              <Field
                label="Detail"
                hint="Each step of depth doubles the mesh's resolution and roughly doubles time and memory."
              >
                {() => <MeshDetail facts={facts} value={depth} onChange={setDepth} />}
              </Field>
              <Field
                label={`Trim: ${trim}`}
                hint="Cuts away surface the cloud doesn't support; higher trims more."
              >
                {(id) => (
                  <Slider id={id} min={0} max={20} step={0.5} value={trim} onChange={setTrim} />
                )}
              </Field>
              <Field label="Point weight" inline>
                {(id) => (
                  <NumberInput
                    id={id}
                    min={0.5}
                    max={20}
                    step={0.5}
                    value={weight}
                    onChange={setWeight}
                  />
                )}
              </Field>
              {cost && (
                <p className={ram && cost.ram_gb > ram * 0.8 ? "text-danger" : "text-fg-muted"}>
                  Roughly {Math.max(1, Math.round(cost.minutes))} min, {cost.ram_gb.toFixed(1)} GB
                  of memory, {(cost.vertices / 1e6).toFixed(1)} M vertices
                  {ram ? ` (this computer has ${ram.toFixed(0)} GB)` : ""}.
                </p>
              )}
            </>
          )}
        </>
      )}
    </Dialog>
  );
}

export function PipelineDialog() {
  const app = useApp();
  const close = useClose();
  const sfmFacts = useFacts("sfm");
  const [sfm, setSfm] = useState<SfmOptions | null>(null);
  const [georef, setGeoref] = useState(true);
  const mvsChoices = useMemo(
    () => detailChoices(MVS_LEVELS, app.state.project?.source_resolution ?? null, 2000),
    [app.state.project],
  );
  const [mvsPx, setMvsPx] = useState(0);
  const [quality, setQuality] = useState<"standard" | "draft">("standard");
  const [method, setMethod] = useState<"poisson" | "delaunay">("poisson");
  const [depth, setDepth] = useState<number | null>(null);
  useEffect(() => {
    if (sfmFacts && !sfm) setSfm(sfmDefaults(sfmFacts));
  }, [sfmFacts, sfm]);
  const images = app.state.project?.images.length ?? 0;
  const run = () => {
    if (!sfm) return;
    close();
    void flows.pipeline(app, {
      sfm: { ...sfm },
      georef: georef && (sfmFacts?.gps_images ?? 0) >= 3 && !app.state.project?.georeferenced,
      mvs: { max_image_size: mvsPx, quality },
      mesh: { method, ...(depth !== null ? { depth } : {}), trim: 10, point_weight: 1 },
    });
  };
  return (
    <Dialog
      title="Run the full pipeline"
      wide
      onClose={close}
      footer={
        <>
          <Button onClick={close}>Cancel</Button>
          <Button kind="primary" disabled={!sfm || images === 0} onClick={run}>
            Run all
          </Button>
        </>
      }
    >
      {!sfmFacts || !sfm ? (
        <p className="text-fg-muted">Looking at the photos…</p>
      ) : (
        <div className="grid grid-cols-2 gap-6">
          <div>
            <h3 className="mb-2 font-semibold">1. Structure from Motion</h3>
            <SfmFields facts={sfmFacts} value={sfm} onChange={setSfm} />
            {!app.state.project?.georeferenced && (
              <GeorefCheck facts={sfmFacts} value={georef} onChange={setGeoref} />
            )}
          </div>
          <div>
            <h3 className="mb-2 font-semibold">2. Dense reconstruction</h3>
            <Field label="Detail">
              {() => (
                <Detail
                  choices={mvsChoices}
                  initial={defaultLevel(MVS_LEVELS, "medium")}
                  custom={2000}
                  onChange={setMvsPx}
                />
              )}
            </Field>
            <Field label="Quality">
              {(id) => (
                <Select id={id} value={quality} onChange={setQuality} options={[...QUALITY]} />
              )}
            </Field>
            <h3 className="mt-4 mb-2 font-semibold">3. Mesh</h3>
            <Field label="Method">
              {(id) => (
                <Select
                  id={id}
                  value={method}
                  onChange={setMethod}
                  options={[
                    { value: "poisson", label: "Poisson — watertight, smooths noise" },
                    { value: "delaunay", label: "Delaunay — maximum detail" },
                  ]}
                />
              )}
            </Field>
            {method === "poisson" && (
              <Field label="Detail">
                {() => <MeshDetail facts={{}} value={depth} onChange={setDepth} />}
              </Field>
            )}
            <p className="text-fg-muted">
              The computer is kept awake until it is done. Stop ends the current stage and the
              chain.
            </p>
          </div>
        </div>
      )}
    </Dialog>
  );
}

export function ColmapDialog() {
  const app = useApp();
  const close = useClose();
  const machine = app.state.kept?.environment.machine;
  const [variant, setVariant] = useState<"cuda" | "cpu">("cuda");
  const run = () => {
    close();
    void flows.job(app, "colmap", { variant }, "Downloading COLMAP").then((o) => {
      if (o.ok)
        app.tell(
          "COLMAP is ready",
          `COLMAP was installed at:\n${String((o.result as { path?: string })?.path ?? "")}`,
        );
    });
  };
  return (
    <Dialog
      title="COLMAP is needed"
      onClose={close}
      footer={
        <>
          <Button onClick={close}>Cancel</Button>
          <Button kind="primary" disabled={machine === "wsl"} onClick={run}>
            Download
          </Button>
        </>
      }
    >
      {machine === "wsl" ? (
        <p>
          This Environment runs COLMAP through pycolmap in WSL; the GPU it needs wasn't found there.
          Check that `nvidia-smi` works in WSL.
        </p>
      ) : (
        <>
          <p className="mb-3">
            Dense reconstruction needs COLMAP with CUDA. CloudLabeller can download the official
            Windows release.
          </p>
          <Radios
            name="variant"
            value={variant}
            onChange={setVariant}
            options={[
              { value: "cuda", label: "With CUDA (an NVIDIA GPU; needed for MVS)" },
              { value: "cpu", label: "Without CUDA (SfM only)" },
            ]}
          />
        </>
      )}
    </Dialog>
  );
}

export function CleanDialog() {
  const app = useApp();
  const [minImages, setMinImages] = useState(3);
  const [maxError, setMaxError] = useState(2);
  const [preview, setPreview] = useState<{ keep: number; total: number } | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const close = () => {
    app.dispatch({ type: "clean_preview", removed: null });
    app.dispatch({ type: "dialog", dialog: null });
  };
  // biome-ignore lint/correctness/useExhaustiveDependencies: the preview follows the thresholds
  useEffect(() => {
    app.dispatch({ type: "view", patch: { representation: "sparse" } });
    let live = true;
    const t = setTimeout(() => {
      api
        .cleanPreview(minImages, maxError)
        .then((p) => {
          if (!live) return;
          setProblem(null);
          setPreview({ keep: p.keep, total: p.total });
          app.dispatch({ type: "clean_preview", removed: p.removed });
        })
        .catch((e) => live && setProblem((e as { english?: string }).english ?? String(e)));
    }, 150);
    return () => {
      live = false;
      clearTimeout(t);
    };
  }, [minImages, maxError]);
  const apply = async () => {
    try {
      const r = await api.clean(minImages, maxError);
      close();
      app.dispatch({
        type: "log",
        text: `Cleaned the sparse cloud: ${r.removed} points removed, ${r.n_points} kept`,
      });
      await app.refresh();
    } catch (e) {
      app.fail(e, "Couldn't clean the cloud");
    }
  };
  const ok = preview && preview.keep > 0 && preview.keep < preview.total;
  return (
    // not modal in spirit: it sits over the viewer, which shows what goes
    <div className="fixed top-16 right-6 z-40 w-80 rounded-lg border border-border bg-surface-1 p-4 shadow-xl">
      <h2 className="mb-3 font-semibold">Clean the sparse cloud</h2>
      <Field label={`Seen in at least ${minImages} photos`}>
        {(id) => <Slider id={id} min={2} max={50} value={minImages} onChange={setMinImages} />}
      </Field>
      <Field label={`Reprojection error at most ${maxError.toFixed(1)} px`}>
        {(id) => (
          <Slider id={id} min={0.1} max={20} step={0.1} value={maxError} onChange={setMaxError} />
        )}
      </Field>
      {problem ? (
        <p className="mb-3 text-danger">{problem}</p>
      ) : (
        preview && (
          <p className="mb-3 text-fg-muted">
            Keeps {preview.keep.toLocaleString("en")} of {preview.total.toLocaleString("en")}{" "}
            points; the <span style={{ color: "#ff00ff" }}>magenta</span> ones go. Their labels go
            with them, and undo is cleared.
          </p>
        )
      )}
      <div className="flex justify-end gap-2">
        <Button onClick={close}>Cancel</Button>
        <Button kind="primary" disabled={!ok} onClick={apply}>
          Remove {preview ? (preview.total - preview.keep).toLocaleString("en") : ""} points
        </Button>
      </div>
    </div>
  );
}
