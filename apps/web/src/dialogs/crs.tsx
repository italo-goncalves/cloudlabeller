// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

// Georeferencing's dialogs: reprojecting into a projected CRS, and the
// export of a cloud or the mesh with the CRS it is written in. The CRS
// picker searches the Engine's catalogue of projected CRSs as you type.

import { useEffect, useState } from "react";
import { saveFile } from "../desktop/bridge";
import * as api from "../engine/api";
import { job } from "../jobs/flows";
import { useApp } from "../state/app";
import { Button, Check, Dialog, Field, Radios, TextInput } from "../ui/controls";
import { useClose } from "./common";

/** A projected CRS chosen by searching "EPSG:code — name". */
export function CrsPicker({
  value,
  onChange,
}: {
  value: api.Crs | null;
  onChange: (c: api.Crs | null) => void;
}) {
  const [query, setQuery] = useState(value ? `EPSG:${value.epsg} — ${value.name}` : "");
  const [results, setResults] = useState<api.Crs[]>([]);
  const [open, setOpen] = useState(false);
  useEffect(() => {
    if (value) setQuery(`EPSG:${value.epsg} — ${value.name}`);
  }, [value]);
  useEffect(() => {
    if (!open) return;
    let live = true;
    const t = setTimeout(() => {
      const q = query.replace(/^EPSG:\s*/i, "").split(" — ")[0] ?? "";
      api
        .crsSearch(q.trim(), 40)
        .then((r) => live && setResults(r.results))
        .catch(() => live && setResults([]));
    }, 120);
    return () => {
      live = false;
      clearTimeout(t);
    };
  }, [query, open]);
  return (
    <div className="relative">
      <TextInput
        value={query}
        placeholder="Search by EPSG code or name, e.g. 31982 or UTM zone 22S"
        onChange={(q) => {
          setQuery(q);
          setOpen(true);
          onChange(null);
        }}
      />
      {open && results.length > 0 && (
        <ul className="absolute z-10 mt-1 max-h-60 w-full overflow-auto rounded-md border border-border bg-surface-1 shadow-lg">
          {results.map((c) => (
            <li key={c.epsg}>
              <button
                type="button"
                className="block w-full px-2 py-1 text-left hover:bg-surface-2"
                onClick={() => {
                  onChange(c);
                  setOpen(false);
                }}
              >
                <span className="font-mono text-table">EPSG:{c.epsg}</span> — {c.name}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function useSuggested(): api.Crs | null {
  const [crs, setCrs] = useState<api.Crs | null>(null);
  useEffect(() => {
    api
      .crsSuggest()
      .then((s) => s.epsg && setCrs({ epsg: s.epsg, name: s.name ?? "" }))
      .catch(() => setCrs(null));
  }, []);
  return crs;
}

function useFrame(): { short: string; full: string } | null {
  const [frame, setFrame] = useState<{ short: string; full: string } | null>(null);
  useEffect(() => {
    api
      .crsFrame()
      .then(setFrame)
      .catch(() => setFrame(null));
  }, []);
  return frame;
}

export function ReprojectDialog() {
  const app = useApp();
  const close = useClose();
  const suggested = useSuggested();
  const frame = useFrame();
  const [crs, setCrs] = useState<api.Crs | null>(null);
  const [orthometric, setOrthometric] = useState(true);
  useEffect(() => {
    if (suggested && !crs) setCrs(suggested);
  }, [suggested, crs]);
  const run = () => {
    if (!crs) return;
    close();
    void job(app, "reproject", { epsg: crs.epsg, orthometric }, `Reprojecting to EPSG:${crs.epsg}`);
  };
  return (
    <Dialog
      title="Reproject to a CRS"
      onClose={close}
      footer={
        <>
          <Button onClick={close}>Cancel</Button>
          <Button kind="primary" disabled={!crs} onClick={run}>
            Reproject
          </Button>
        </>
      }
    >
      {frame && <p className="mb-3 text-fg-muted">Now: {frame.full}</p>}
      <Field
        label="Projected CRS"
        hint={suggested ? `Suggested from the origin: EPSG:${suggested.epsg}` : undefined}
      >
        {() => <CrsPicker value={crs} onChange={setCrs} />}
      </Field>
      <Check checked={orthometric} onChange={setOrthometric}>
        Heights above sea level (EGM96), not the ellipsoid
      </Check>
      <p className="text-fg-faint">
        Reprojecting moves every product. Reprojecting again later works, but each step rounds a
        little: choose the CRS you'll deliver in.
      </p>
    </Dialog>
  );
}

const CLOUD_FORMATS = [
  { value: "las", label: "LAS", ext: "las" },
  { value: "csv", label: "CSV", ext: "csv" },
  { value: "ply", label: "PLY", ext: "ply" },
] as const;
const MESH_FORMATS = [
  { value: "ply", label: "PLY", ext: "ply" },
  { value: "obj", label: "OBJ", ext: "obj" },
  { value: "stl", label: "STL", ext: "stl" },
] as const;

export function ExportDialog({ what }: { what: "cloud" | "mesh" }) {
  const app = useApp();
  const close = useClose();
  const project = app.state.project;
  const enu = project?.georeferenced ?? false;
  const suggested = useSuggested();
  const formats = what === "cloud" ? CLOUD_FORMATS : MESH_FORMATS;
  const [format, setFormat] = useState<string>(formats[0].value);
  const [mode, setMode] = useState<"local" | "projected">("local");
  const [crs, setCrs] = useState<api.Crs | null>(null);
  const [orthometric, setOrthometric] = useState(true);
  const [settings, setSettings] = useState<api.ExportSettings | null>(null);

  useEffect(() => {
    if (!enu) return;
    api
      .exportSettings()
      .then((s) => {
        setSettings(s);
        setMode(s.export_mode ?? "local");
        setOrthometric(s.export_orthometric);
      })
      .catch(() => setSettings(null));
  }, [enu]);
  useEffect(() => {
    if (crs) return;
    if (settings?.export_epsg) setCrs({ epsg: settings.export_epsg, name: "" });
    else if (suggested) setCrs(suggested);
  }, [settings, suggested, crs]);

  const cloud = project?.dense ? "dense" : "sparse";
  const go = async () => {
    const f = formats.find((x) => x.value === format) ?? formats[0];
    const base = project?.root.split(/[\\/]/).pop() ?? "export";
    const path = await saveFile(
      `Export the ${what === "cloud" ? `${cloud} cloud` : "mesh"}`,
      `${base}_${what === "cloud" ? cloud : "mesh"}.${f.ext}`,
      [{ name: f.label, extensions: [f.ext] }],
    );
    if (!path) return;
    const target: api.ExportCrs =
      enu && mode === "projected" && crs
        ? { mode: "projected", epsg: crs.epsg, orthometric }
        : { mode: "local" };
    close();
    try {
      if (enu)
        await api
          .setExportSettings({
            export_mode: target.mode,
            export_epsg: target.epsg ?? null,
            export_orthometric: orthometric,
          })
          .catch(() => undefined);
      const where = await import("../desktop/bridge").then((b) => b.enginePath(path));
      const send = () =>
        what === "cloud"
          ? app.long("Exporting the cloud", (t) =>
              api.exportCloud(where, f.value as "las", target, t),
            )
          : app.long("Exporting the mesh", (t) =>
              api.exportMesh(where, f.value as "ply", target, t),
            );
      try {
        await send();
      } catch (e) {
        if ((e as { code?: string }).code !== "export.needs_geoid" || !target.epsg) throw e;
        const o = await job(app, "geoid", { epsg: target.epsg }, "Downloading the EGM96 geoid");
        if (!o.ok) return;
        await send();
      }
      app.dispatch({ type: "log", text: `Exported to ${path}` });
    } catch (e) {
      app.fail(e, "The export failed");
    }
  };
  const nothing = what === "cloud" ? !project?.sparse && !project?.dense : !project?.mesh;
  return (
    <Dialog
      title={what === "cloud" ? "Export the point cloud" : "Export the mesh"}
      onClose={close}
      footer={
        <>
          <Button onClick={close}>Cancel</Button>
          <Button
            kind="primary"
            disabled={nothing || (enu && mode === "projected" && !crs)}
            onClick={go}
          >
            Export…
          </Button>
        </>
      }
    >
      {nothing ? (
        <p className="text-danger">
          {what === "cloud" ? "There is no cloud yet: run SfM first." : "There is no mesh yet."}
        </p>
      ) : (
        <>
          {what === "cloud" && (
            <p className="mb-3 text-fg-muted">The {cloud} cloud, with its labels.</p>
          )}
          <Field label="Format">
            {() => (
              <Radios
                name="format"
                value={format}
                onChange={setFormat}
                options={formats.map((f) => ({ value: f.value, label: f.label }))}
              />
            )}
          </Field>
          {enu && (
            <>
              <Field label="Coordinates">
                {() => (
                  <Radios
                    name="mode"
                    value={mode}
                    onChange={setMode}
                    options={[
                      { value: "local", label: "Local East-North-Up (metres)" },
                      { value: "projected", label: "A projected CRS" },
                    ]}
                  />
                )}
              </Field>
              {mode === "projected" && (
                <>
                  <Field label="CRS">{() => <CrsPicker value={crs} onChange={setCrs} />}</Field>
                  <Check checked={orthometric} onChange={setOrthometric}>
                    Heights above sea level (EGM96); the geoid grid (~3 MB) is downloaded the first
                    time
                  </Check>
                </>
              )}
            </>
          )}
          {!enu && (
            <p className="text-fg-faint">
              Coordinates are written in the Project's own frame
              {project?.georeferenced ? "" : " (not georeferenced)"}.
            </p>
          )}
        </>
      )}
    </Dialog>
  );
}
