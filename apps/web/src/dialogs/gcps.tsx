// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

// A GCP's details (name, use, surveyed coordinates and their CRS), and
// importing GCPs from a coordinates file.

import { useEffect, useState } from "react";
import { pickFile } from "../desktop/bridge";
import * as api from "../engine/api";
import { useGcps } from "../panes/GcpPanel";
import { useApp } from "../state/app";
import { Button, Check, Dialog, Field, NumberInput, Radios, TextInput } from "../ui/controls";
import { useClose } from "./common";
import { CrsPicker } from "./crs";

const WGS84: api.Crs = { epsg: 4326, name: "WGS 84 longitude, latitude" };

/** A CRS for coordinates: projected (searched) or WGS 84 longitude/latitude. */
function CoordinateCrs({
  value,
  onChange,
}: {
  value: api.Crs | null;
  onChange: (c: api.Crs | null) => void;
}) {
  const geographic = value?.epsg === 4326 || value?.epsg === 4979;
  return (
    <>
      <Check checked={geographic} onChange={(on) => onChange(on ? WGS84 : null)}>
        WGS 84 longitude and latitude (x = longitude, y = latitude)
      </Check>
      {!geographic && (
        <div className="mt-2">
          <CrsPicker value={value} onChange={onChange} />
        </div>
      )}
    </>
  );
}

const HEIGHTS: { value: "ellipsoidal" | "orthometric"; label: string }[] = [
  { value: "ellipsoidal", label: "Ellipsoidal (GNSS)" },
  { value: "orthometric", label: "Above sea level (EGM96)" },
];
const USES: { value: api.GcpUse; label: string }[] = [
  { value: "control", label: "Control: aligns the Surveys" },
  { value: "check", label: "Check: only measured" },
];

export function GcpDialog({ id }: { id: string }) {
  const app = useApp();
  const close = useClose();
  const gcp = useGcps()?.gcps.find((g) => g.id === id) ?? null;
  const [name, setName] = useState("");
  const [use, setUse] = useState<api.GcpUse>("control");
  const [surveyed, setSurveyed] = useState(false);
  const [xyz, setXyz] = useState<[number, number, number]>([0, 0, 0]);
  const [crs, setCrs] = useState<api.Crs | null>(null);
  const [ortho, setOrtho] = useState(false);
  const [accuracy, setAccuracy] = useState<number | null>(null);
  const [ready, setReady] = useState(false);
  useEffect(() => {
    if (!gcp || ready) return;
    setName(gcp.name);
    setUse(gcp.use);
    setSurveyed(gcp.xyz !== null);
    if (gcp.xyz) setXyz(gcp.xyz);
    if (gcp.epsg !== null)
      setCrs(gcp.epsg === 4326 || gcp.epsg === 4979 ? WGS84 : { epsg: gcp.epsg, name: "" });
    setOrtho(gcp.orthometric);
    setAccuracy(gcp.accuracy);
    setReady(true);
  }, [gcp, ready]);
  const ok = name.trim() !== "" && (!surveyed || crs !== null);
  const save = async () => {
    try {
      await api.gcpUpdate(id, {
        name: name.trim(),
        use,
        xyz: surveyed ? xyz : null,
        ...(surveyed && crs ? { epsg: crs.epsg, orthometric: ortho } : {}),
        accuracy,
      });
      close();
    } catch (e) {
      app.fail(e, "Couldn't save the GCP");
    }
  };
  const axis = (i: 0 | 1 | 2, label: string) => (
    <Field label={label} inline>
      {(fid) => (
        <NumberInput
          id={fid}
          className="w-40"
          step={0.001}
          value={xyz[i]}
          onChange={(v) => setXyz((p) => p.map((q, k) => (k === i ? v : q)) as typeof p)}
        />
      )}
    </Field>
  );
  return (
    <Dialog
      title="GCP"
      onClose={close}
      footer={
        <>
          <Button onClick={close}>Cancel</Button>
          <Button kind="primary" disabled={!ok || !ready} onClick={() => void save()}>
            Save
          </Button>
        </>
      }
    >
      <Field label="Name">{(fid) => <TextInput id={fid} value={name} onChange={setName} />}</Field>
      <Field label="Use">
        {() => <Radios name="gcp-use" value={use} options={USES} onChange={setUse} />}
      </Field>
      <Check checked={surveyed} onChange={setSurveyed}>
        Surveyed coordinates
      </Check>
      {surveyed && (
        <div className="mt-3 ml-6">
          <Field label="Coordinate system">
            {() => <CoordinateCrs value={crs} onChange={setCrs} />}
          </Field>
          {axis(0, "X / easting / longitude")}
          {axis(1, "Y / northing / latitude")}
          {axis(2, "Height")}
          <Field label="Heights">
            {() => (
              <Radios
                name="gcp-heights"
                value={ortho ? "orthometric" : "ellipsoidal"}
                options={HEIGHTS}
                onChange={(v) => setOrtho(v === "orthometric")}
              />
            )}
          </Field>
          <Field label="Accuracy (m)" inline>
            {(fid) => (
              <NumberInput
                id={fid}
                step={0.01}
                min={0}
                value={accuracy ?? Number.NaN}
                onChange={(v) => setAccuracy(v)}
              />
            )}
          </Field>
        </div>
      )}
    </Dialog>
  );
}

export function GcpImportDialog() {
  const app = useApp();
  const close = useClose();
  const [path, setPath] = useState<string | null>(null);
  const [crs, setCrs] = useState<api.Crs | null>(null);
  const [ortho, setOrtho] = useState(false);
  const [use, setUse] = useState<api.GcpUse>("control");
  const choose = async () => {
    const p = await pickFile("Choose a coordinates file", [
      { name: "Coordinates", extensions: ["csv", "txt"] },
    ]);
    if (p) setPath(p);
  };
  const run = async () => {
    if (!path || !crs) return;
    try {
      const out = await api.gcpImportCsv(path, crs.epsg, ortho, use);
      close();
      app.dispatch({
        type: "log",
        text: `GCPs: ${out.added} added, ${out.updated} updated from ${path}`,
      });
    } catch (e) {
      app.fail(e, "Couldn't import the GCPs");
    }
  };
  return (
    <Dialog
      title="Import GCPs"
      onClose={close}
      footer={
        <>
          <Button onClick={close}>Cancel</Button>
          <Button kind="primary" disabled={!path || !crs} onClick={() => void run()}>
            Import
          </Button>
        </>
      }
    >
      <p className="mb-3 text-fg-muted">
        One GCP per line: name, x, y, z, and optionally its accuracy in metres. Commas, semicolons
        or tabs between them; a header line is fine. A name already here takes the new coordinates.
      </p>
      <Field label="File">
        {() => (
          <div className="flex items-center gap-2">
            <Button onClick={() => void choose()}>Choose…</Button>
            <span className="min-w-0 flex-1 truncate text-fg-muted">{path ?? "none chosen"}</span>
          </div>
        )}
      </Field>
      <Field label="Coordinate system">
        {() => <CoordinateCrs value={crs} onChange={setCrs} />}
      </Field>
      <Field label="Heights">
        {() => (
          <Radios
            name="gcp-heights"
            value={ortho ? "orthometric" : "ellipsoidal"}
            options={HEIGHTS}
            onChange={(v) => setOrtho(v === "orthometric")}
          />
        )}
      </Field>
      <Field label="Use">
        {() => <Radios name="gcp-use" value={use} options={USES} onChange={setUse} />}
      </Field>
    </Dialog>
  );
}
