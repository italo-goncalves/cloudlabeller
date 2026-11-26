// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

// The centre pane: the Project's sparse cloud, dense cloud or mesh in 3D, with
// the photos' cameras, a lasso, and the marks of a selection or a cleaning
// preview. What is drawn follows `state.view`; `rig.ts` does the drawing, on
// demand. Replaces the Qt viewer (ui/viewer3d.py).

import { simplifyPath, type Vec2 } from "@geokit/viewer";
import { useEffect, useMemo, useRef, useState } from "react";
import { VIEW_DETAIL } from "../detail";
import * as api from "../engine/api";
import { useApp } from "../state/app";
import {
  classColours,
  classMask,
  diagonalOf,
  dimFor,
  frustumOf,
  frustumSize,
  meshColours,
  nodeLimit,
  pointsLabel,
  type Rgb,
  shownIndices,
  statusColour,
  type Vec3,
} from "./adapt";
import { onHome } from "./home";
import { Rig } from "./rig";

const ORANGE: Rgb = [255, 140, 0];
const MAGENTA: Rgb = [255, 0, 255];
const DEFAULT_UP: Vec3 = [0, 0, 1];

interface Cams {
  /** The `geometryVersion` they were read at. */
  version: number;
  list: api.Camera[];
  up: Vec3;
}

/** A click that moved less than this (CSS px) is a click, not a drag. */
const CLICK_SLOP = 4;

export function Viewer3D() {
  const app = useApp();
  const { state } = app;
  const { project, view, selection, cleanPreview, selected, labelsVersion, geometryVersion } =
    state;

  const box = useRef<HTMLDivElement>(null);
  const rigRef = useRef<Rig | null>(null);
  const live = useRef(app);
  live.current = app;

  const [dataRev, setDataRev] = useState(0);
  const [frustaRev, setFrustaRev] = useState(0);
  const [loading, setLoading] = useState(false);
  const [cams, setCams] = useState<Cams | null>(null);
  const [broken, setBroken] = useState<string | null>(null);
  const camsRef = useRef<Cams | null>(null);
  camsRef.current = cams;

  const rep = view.representation;
  const present = rep === "mesh" ? project?.mesh != null : (project?.[rep] ?? 0) > 0;
  const root = project?.root ?? null;

  const schema = project?.schema;
  const schemaKey = (schema ?? []).map((c) => `${c.id}${c.color}`).join(",");
  // biome-ignore lint/correctness/useExhaustiveDependencies: the key stands for the schema's ids and colours
  const colours = useMemo(() => {
    const s = schema ?? [];
    return { colours: classColours(s), mask: classMask(s) };
  }, [schemaKey]);

  // -- the rig: renderer, camera, controls, pointer ----------------------------

  useEffect(() => {
    const parent = box.current;
    if (!parent) return;
    let rig: Rig;
    try {
      rig = new Rig(parent);
    } catch (e) {
      setBroken(`WebGL isn't available here (${e instanceof Error ? e.message : String(e)}).`);
      return;
    }
    rigRef.current = rig;
    rig.onError = (e) => live.current.fail(e, "Couldn't draw the cloud");

    const size = () => {
      const r = parent.getBoundingClientRect();
      rig.resize(r.width, r.height);
    };
    size();
    const watch = new ResizeObserver(size);
    watch.observe(parent);
    const unhome = onHome(() => rig.home(camsRef.current?.up ?? DEFAULT_UP));

    const at = (e: PointerEvent | MouseEvent): Vec2 => {
      const r = rig.canvas.getBoundingClientRect();
      return [e.clientX - r.left, e.clientY - r.top];
    };

    // lasso
    let path: Vec2[] | null = null;
    // click
    let down: Vec2 | null = null;

    const select = async (polygon: Vec2[]) => {
      const g = rig.geometry;
      if (!g?.cloud) return;
      const cloud = g.cloud;
      const poly = simplifyPath(polygon, 40);
      if (poly.length < 3) return;
      try {
        const indices = await api.cloudSelect(
          cloud,
          rig.worldViewProjection(),
          poly,
          rig.width,
          rig.height,
        );
        live.current.dispatch({ type: "selection", selection: { cloud, indices } });
      } catch (e) {
        live.current.fail(e, "Couldn't select the points");
      }
    };

    // a GCP's cloud Mark, on the point the click snaps to
    const placeGcp = async (cloud: api.CloudName, x: number, y: number) => {
      const app = live.current;
      const gcp = app.state.activeGcp;
      if (!gcp) {
        app.tell("GCP tool", "Choose a GCP in the GCPs tab of the Dataset pane first.");
        return;
      }
      try {
        const hit = await api.cloudPick(
          cloud,
          rig.worldViewProjection(),
          x,
          y,
          rig.width,
          rig.height,
        );
        if (hit.index === null) {
          app.dispatch({ type: "log", text: "No cloud point under the click: try closer." });
          return;
        }
        await api.markCloud(gcp, cloud, hit.index);
      } catch (e) {
        app.fail(e, "Couldn't place the Mark");
      }
    };

    const onDown = (e: PointerEvent) => {
      if (e.button !== 0) return;
      if (live.current.state.view.lasso && rig.geometry?.cloud) {
        path = [at(e)];
        rig.canvas.setPointerCapture(e.pointerId);
        rig.lasso.set(path, rig.width, rig.height);
        rig.invalidate();
      } else down = at(e);
    };
    const onMove = (e: PointerEvent) => {
      if (!path) return;
      path.push(at(e));
      rig.lasso.set(path, rig.width, rig.height);
      rig.invalidate();
    };
    const onUp = (e: PointerEvent) => {
      if (path) {
        const done = path;
        path = null;
        rig.lasso.set([], rig.width, rig.height);
        rig.invalidate();
        if (rig.canvas.hasPointerCapture(e.pointerId))
          rig.canvas.releasePointerCapture(e.pointerId);
        void select(done);
        return;
      }
      const start = down;
      down = null;
      if (!start || e.button !== 0) return;
      const [x, y] = at(e);
      if (Math.hypot(x - start[0], y - start[1]) > CLICK_SLOP) return;
      if (live.current.state.view.gcpPick && rig.geometry?.cloud) {
        void placeGcp(rig.geometry.cloud, x, y);
        return;
      }
      const i = rig.pickFrustum(x, y);
      const name = i === null ? undefined : camsRef.current?.list[i]?.name;
      if (name !== undefined) live.current.dispatch({ type: "select_image", name });
    };
    const onCancel = () => {
      path = null;
      down = null;
      rig.lasso.set([], rig.width, rig.height);
      rig.invalidate();
    };
    const onDouble = (e: MouseEvent) => {
      const [x, y] = at(e);
      const i = rig.pickFrustum(x, y);
      if (i !== null) rig.flyToFrustum(i);
    };
    rig.canvas.addEventListener("pointerdown", onDown);
    rig.canvas.addEventListener("pointermove", onMove);
    rig.canvas.addEventListener("pointerup", onUp);
    rig.canvas.addEventListener("pointercancel", onCancel);
    rig.canvas.addEventListener("dblclick", onDouble);

    return () => {
      unhome();
      watch.disconnect();
      rig.canvas.removeEventListener("pointerdown", onDown);
      rig.canvas.removeEventListener("pointermove", onMove);
      rig.canvas.removeEventListener("pointerup", onUp);
      rig.canvas.removeEventListener("pointercancel", onCancel);
      rig.canvas.removeEventListener("dblclick", onDouble);
      rig.dispose();
      rigRef.current = null;
    };
  }, []);

  // where the GCPs are, in this Survey's active Reconstruction
  const { gcpVersion, activeGcp } = app.state;
  // biome-ignore lint/correctness/useExhaustiveDependencies: reloaded when the GCPs or the geometry change
  useEffect(() => {
    const rig = rigRef.current;
    if (!rig || !project) return;
    let alive = true;
    api
      .marksCloud()
      .then(
        (r) =>
          alive && rig.setGcps(r.marks.map((m) => ({ xyz: m.xyz, active: m.gcp === activeGcp }))),
      )
      .catch(() => alive && rig.setGcps([]));
    return () => {
      alive = false;
    };
  }, [gcpVersion, activeGcp, project?.survey]);

  // the lasso takes the left button while it is on
  useEffect(() => {
    rigRef.current?.setLasso(view.lasso);
  }, [view.lasso]);

  // -- geometry ---------------------------------------------------------------

  const detailKey = rep === "mesh" ? "-" : view.detail;
  // biome-ignore lint/correctness/useExhaustiveDependencies: a new Project, cloud, detail or geometry reloads; the rest is read as it is
  useEffect(() => {
    const rig = rigRef.current;
    if (!rig) return;
    if (root === null || !present) {
      rig.clearGeometry();
      setDataRev((n) => n + 1);
      return;
    }
    let dead = false;
    setLoading(true);
    const lv = live.current.state.labelsVersion;
    void (async () => {
      try {
        if (rep === "mesh") {
          const [geo, labels] = await Promise.all([api.meshGeometry(), api.meshLabels()]);
          if (!dead) rig.setMesh(geo, labels, geometryVersion, lv);
        } else {
          const [cloud, labels] = await Promise.all([
            api.cloudView(rep, nodeLimit(VIEW_DETAIL[view.detail])),
            api.cloudLabels(rep),
          ]);
          if (!dead) rig.setCloud(rep, cloud, labels, geometryVersion, lv);
        }
        if (!dead) setDataRev((n) => n + 1);
      } catch (e) {
        if (!dead) live.current.fail(e, "Couldn't load the view");
      } finally {
        if (!dead) setLoading(false);
      }
    })();
    return () => {
      dead = true;
    };
  }, [root, present, rep, detailKey, geometryVersion]);

  // new labels: re-read them and recolour in place
  // biome-ignore lint/correctness/useExhaustiveDependencies: `dataRev` says what is loaded now
  useEffect(() => {
    const rig = rigRef.current;
    const g = rig?.geometry;
    if (!rig || !g || g.labelsVersion === labelsVersion || g.geomVersion !== geometryVersion)
      return;
    if ((g.kind === "mesh") !== (rep === "mesh") || (g.cloud !== null && g.cloud !== rep)) return;
    let dead = false;
    void (async () => {
      try {
        const labels =
          g.kind === "mesh" ? await api.meshLabels() : await api.cloudLabels(rep as api.CloudName);
        if (!dead && rig.geometry === g) {
          rig.setLabels(labels, labelsVersion);
          setDataRev((n) => n + 1);
        }
      } catch (e) {
        if (!dead) live.current.fail(e, "Couldn't read the labels");
      }
    })();
    return () => {
      dead = true;
    };
  }, [labelsVersion, geometryVersion, rep, dataRev]);

  // -- look -------------------------------------------------------------------

  // biome-ignore lint/correctness/useExhaustiveDependencies: `colours` follows the schema's key
  useEffect(() => {
    const rig = rigRef.current;
    const g = rig?.geometry;
    if (!rig || !g) return;
    if (g.kind === "mesh") {
      rig.setMeshColours(
        meshColours(g.colors, g.labels, schema ?? [], view.blend, view.isolate, g.shown),
      );
    } else {
      rig.setCloudLook({
        colourBy: "classification",
        blend: view.blend,
        size: view.pointSize,
        classColours: colours.colours,
        classMask: colours.mask,
        dim: dimFor(schema ?? [], view.isolate),
        ownColourClass: 0,
      });
    }
  }, [dataRev, view.blend, view.pointSize, view.isolate, colours, schemaKey]);

  // marks: the cleaning preview wins over the selection
  // biome-ignore lint/correctness/useExhaustiveDependencies: `dataRev` says what is loaded now, to be marked again
  useEffect(() => {
    const rig = rigRef.current;
    const g = rig?.geometry;
    if (!rig || !g?.cloud) return;
    const preview = cleanPreview && g.cloud === "sparse" ? cleanPreview : null;
    const picked = !preview && selection && selection.cloud === g.cloud ? selection.indices : null;
    const full = preview ?? picked;
    rig.setMarks(full ? shownIndices(full, g.index, g.shown) : null, preview ? MAGENTA : ORANGE);
  }, [dataRev, selection, cleanPreview]);

  // -- cameras ----------------------------------------------------------------

  const solved = project?.images.filter((i) => i.solved).length ?? 0;
  useEffect(() => {
    if (root === null) {
      setCams(null);
      return;
    }
    const none: Cams = { version: geometryVersion, list: [], up: DEFAULT_UP };
    if (solved === 0) {
      setCams(none);
      return;
    }
    let dead = false;
    api
      .cameras()
      .then((r) => {
        if (!dead) setCams({ version: geometryVersion, list: r.cameras, up: r.up ?? DEFAULT_UP });
      })
      .catch((e: unknown) => {
        if (dead) return;
        live.current.fail(e, "Couldn't read the cameras");
        setCams(none);
      });
    return () => {
      dead = true;
    };
  }, [root, geometryVersion, solved]);

  // the first sight of a Project: the home view, once the cameras have said which way is up
  // biome-ignore lint/correctness/useExhaustiveDependencies: `dataRev` says what is loaded now
  useEffect(() => {
    const rig = rigRef.current;
    if (!rig?.geometry || !cams || root === null || rig.homedFor === root) return;
    if (cams.version !== geometryVersion) return;
    rig.homedFor = root;
    rig.home(cams.up);
  }, [dataRev, cams, root, geometryVersion]);

  const showCameras = view.showCameras;
  // biome-ignore lint/correctness/useExhaustiveDependencies: `dataRev` says what is loaded now
  useEffect(() => {
    const rig = rigRef.current;
    if (!rig) return;
    if (!rig.geometry || !cams || !showCameras || cams.list.length === 0) {
      rig.setFrusta(null, null);
      setFrustaRev((n) => n + 1);
      return;
    }
    rig.setFrusta(
      cams.list.map((c) => frustumOf(c, rig.origin)),
      { size: 1, colours: [255, 255, 255], highlight: null },
    );
    setFrustaRev((n) => n + 1);
  }, [dataRev, cams, showCameras]);

  const statuses = project?.images ?? [];
  const statusKey = statuses.map((i) => i.status[0]).join("");
  // biome-ignore lint/correctness/useExhaustiveDependencies: `statusKey` stands for the photos' statuses
  useEffect(() => {
    const rig = rigRef.current;
    const g = rig?.geometry;
    if (!rig || !g || !cams || !rig.hasFrusta) return;
    const byName = new Map(statuses.map((i) => [i.name, i.status]));
    const at = cams.list.findIndex((c) => c.name === selected);
    rig.applyFrusta({
      size: frustumSize(diagonalOf(g.bounds), view.frustumScale),
      colours: cams.list.map((c) => statusColour(byName.get(c.name) ?? "none")),
      highlight: at >= 0 ? at : null,
    });
  }, [frustaRev, view.frustumScale, selected, statusKey, cams]);

  // -- what the pane says when it draws nothing ---------------------------------

  const g = rigRef.current?.geometry ?? null;
  const hint = broken
    ? broken
    : root === null
      ? "Open or create a Project to see it here."
      : !present
        ? "This Project has no point cloud or mesh yet. Run the reconstruction to make one."
        : null;

  return (
    <div className="relative h-full min-h-0 w-full overflow-hidden bg-surface-0">
      <div ref={box} className="absolute inset-0" />
      {hint && (
        <div className="absolute inset-0 flex items-center justify-center p-6 text-center text-fg-muted">
          {hint}
        </div>
      )}
      {!hint && loading && (
        <div className="pointer-events-none absolute top-2 left-3 text-fg-muted">Loading…</div>
      )}
      {!hint && !loading && g && (
        <div className="pointer-events-none absolute bottom-2 left-3 text-fg-faint">
          {g.shown === g.total
            ? `${pointsLabel(g.total)} ${g.kind === "mesh" ? "vertices" : "points"}`
            : `${pointsLabel(g.shown)} of ${pointsLabel(g.total)} points`}
        </div>
      )}
    </div>
  );
}
