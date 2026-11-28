// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

// The Image pane: a photo with its mask over it, and the tools to label it
// (polygon, brush, eraser), as the 0.2.x Qt `ui/image_view.py` did, the
// Accept brush, which keeps what a pixel shows as the user's own label
// (ADR-0006), and the GCP tool, which places the chosen GCP's Mark on the
// photo. The pixel work is in `raster.ts`, the mask's life in `useMask`, the
// view in `useViewport`.

import { cn } from "@geokit/ui";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  imageNormals,
  imageUncertainty,
  markDelete,
  markPhoto,
  marksPhoto,
  type PhotoMarks,
  propagate,
} from "../engine/api";
import { useGcps } from "../panes/GcpPanel";
import { useApp } from "../state/app";
import type { ImageTool } from "../state/store";
import { Button, PaneTitle, Slider } from "../ui/controls";
import { pickMaskPixels } from "./importMask";
import {
  fillPolygon,
  imageToScreen,
  KEEP,
  maskFromPixels,
  type Point,
  type Rect,
  stampDisc,
  strokeLine,
  unionRect,
} from "./raster";
import { useMask } from "./useMask";
import { useOverlay } from "./useOverlay";
import { usePhoto } from "./usePhoto";
import { useViewport } from "./useViewport";

const TOOLS: { tool: ImageTool; label: string }[] = [
  { tool: "pan", label: "Pan" },
  { tool: "polygon", label: "Polygon (D)" },
  { tool: "brush", label: "Brush (B)" },
  { tool: "eraser", label: "Eraser (E)" },
  { tool: "accept", label: "Accept (A)" },
  { tool: "gcp", label: "GCP (G)" },
];

type Drag =
  | { kind: "pan"; last: Point }
  | { kind: "stroke"; last: Point; mask: Int32Array; classId: number; dirty: Rect | null };

function typing(target: EventTarget | null): boolean {
  return target instanceof HTMLElement && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName);
}

export function ImageView() {
  const { state, dispatch, long, fail, tell } = useApp();
  const { selected: name, tool, brushSize, overlayOpacity, activeClass, project } = state;
  const schema = project?.schema ?? [];
  const row = project?.images.find((i) => i.name === name);

  const photo = usePhoto(name, project?.survey ?? null);
  const { activeGcp, gcpVersion } = state;
  const gcpNames = new Map((useGcps()?.gcps ?? []).map((g) => [g.id, g.name]));
  const [gcpMarks, setGcpMarks] = useState<PhotoMarks>({ marks: [], hints: [] });
  // how unsure each pixel's label is, drawn on its own canvas: the cloud's
  // points' Uncertainty, the model's, 0 where hand-drawn
  const [showUnsure, setShowUnsure] = useState(false);
  const unsureCanvas = useRef<HTMLCanvasElement>(null);
  const predicted = Boolean(row && row.status !== "none");
  // biome-ignore lint/correctness/useExhaustiveDependencies: redrawn per photo and prediction
  useEffect(() => {
    const canvas = unsureCanvas.current;
    if (!canvas || !showUnsure || !predicted || !name) return;
    let live = true;
    imageUncertainty(name)
      .then((u) => {
        if (!live || !u) return;
        canvas.width = u.width;
        canvas.height = u.height;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        const img = ctx.createImageData(u.width, u.height);
        for (let i = 0; i < u.values.length; i++) {
          const v = u.values[i] ?? Number.NaN;
          if (!Number.isFinite(v)) continue; // unlabelled or unknown: transparent
          img.data[i * 4] = 255;
          img.data[i * 4 + 1] = Math.round(64 * (1 - v));
          img.data[i * 4 + 2] = 255;
          img.data[i * 4 + 3] = Math.round(230 * v);
        }
        ctx.putImageData(img, 0, 0);
      })
      .catch(() => undefined);
    return () => {
      live = false;
    };
  }, [name, showUnsure, predicted, state.masks.version, project?.survey]);
  // the photo coloured by the dense cloud's normals, over it (the photo shows where none)
  const [showNormals, setShowNormals] = useState(false);
  const normalsCanvas = useRef<HTMLCanvasElement>(null);
  const canNormals = Boolean(project?.normals) && Boolean(row?.solved);
  // biome-ignore lint/correctness/useExhaustiveDependencies: redrawn per photo and geometry
  useEffect(() => {
    const canvas = normalsCanvas.current;
    if (!canvas || !showNormals || !canNormals || !name) return;
    let live = true;
    imageNormals(name)
      .then((n) => {
        if (!live || !n) return;
        canvas.width = n.width;
        canvas.height = n.height;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        const img = ctx.createImageData(n.width, n.height);
        for (let i = 0, k = 0; i < n.width * n.height; i++, k += 3) {
          const r = n.rgb[k] ?? 128;
          const g = n.rgb[k + 1] ?? 128;
          const b = n.rgb[k + 2] ?? 128;
          img.data[i * 4] = r;
          img.data[i * 4 + 1] = g;
          img.data[i * 4 + 2] = b;
          img.data[i * 4 + 3] = r === 128 && g === 128 && b === 128 ? 0 : 255;
        }
        ctx.putImageData(img, 0, 0);
      })
      .catch(() => undefined);
    return () => {
      live = false;
    };
  }, [name, showNormals, canNormals, state.geometryVersion, project?.survey]);
  // biome-ignore lint/correctness/useExhaustiveDependencies: the Marks change with their version
  useEffect(() => {
    if (!name) {
      setGcpMarks({ marks: [], hints: [] });
      return;
    }
    let live = true;
    marksPhoto(name)
      .then((m) => live && setGcpMarks(m))
      .catch(() => live && setGcpMarks({ marks: [], hints: [] }));
    return () => {
      live = false;
    };
  }, [name, gcpVersion, project?.survey]);
  const editor = useMask(name);
  const { size } = editor;
  // the overlay coloured by where its labels came from, not by class
  const [bySource, setBySource] = useState(false);
  const overlay = useOverlay(editor, schema, bySource);
  const viewport = useViewport(size, name);
  const { view } = viewport;

  const [polygon, setPolygon] = useState<Point[]>([]);
  const [hover, setHover] = useState<Point | null>(null);
  const drag = useRef<Drag | null>(null);

  const radius = brushSize / 2;
  const classColour = schema.find((c) => c.id === activeClass)?.color ?? "#ffffff";

  // A new photo or tool leaves no half-drawn polygon behind.
  // biome-ignore lint/correctness/useExhaustiveDependencies: these are the triggers
  useEffect(() => setPolygon([]), [name, tool]);

  const brushing = tool === "brush" || tool === "eraser" || tool === "accept";
  const paintClass = tool === "eraser" ? -1 : tool === "accept" ? KEEP : activeClass;

  const finishPolygon = useCallback(() => {
    if (!size || polygon.length < 3) return;
    const mask = editor.ensure();
    if (!mask) return;
    const rect = fillPolygon(
      mask,
      size.w,
      size.h,
      polygon,
      activeClass,
      editor.marks.current ?? undefined,
    );
    setPolygon([]);
    if (!rect) return;
    overlay.paint(rect);
    void editor.save(mask);
  }, [size, polygon, editor, overlay, activeClass]);

  const clamp = useCallback(
    (p: Point): Point => [
      Math.min(Math.max(p[0], 0), (size?.w ?? 1) - 1),
      Math.min(Math.max(p[1], 0), (size?.h ?? 1) - 1),
    ],
    [size],
  );

  // -- the pointer ------------------------------------------------------------

  const onPointerDown = (e: React.PointerEvent) => {
    const panning = e.button === 1 || (e.button === 0 && tool === "pan");
    const here = viewport.toImage(e.clientX, e.clientY);
    if (panning) {
      drag.current = { kind: "pan", last: viewport.local(e.clientX, e.clientY) };
      e.currentTarget.setPointerCapture(e.pointerId);
      return;
    }
    if (!size) return;
    if (tool === "gcp" && name) {
      if (!activeGcp) {
        tell("GCP tool", "Choose a GCP in the GCPs tab of the Dataset pane first.");
        return;
      }
      const at: [number, number] = [
        Math.min(Math.max(here[0], 0), size.w),
        Math.min(Math.max(here[1], 0), size.h),
      ];
      if (e.button === 0) markPhoto(activeGcp, name, at).catch((err) => fail(err));
      else if (e.button === 2) markDelete(activeGcp, name).catch((err) => fail(err));
      return;
    }
    if (tool === "polygon") {
      if (e.button === 0) setPolygon((p) => [...p, clamp(here)]);
      else if (e.button === 2) setPolygon((p) => p.slice(0, -1));
    } else if (brushing && e.button === 0) {
      const mask = editor.ensure();
      if (!mask) return;
      editor.stroking.current = true;
      const marks = editor.marks.current ?? undefined;
      const dirty = stampDisc(mask, size.w, size.h, here[0], here[1], radius, paintClass, marks);
      overlay.paint(dirty);
      drag.current = { kind: "stroke", last: here, mask, classId: paintClass, dirty };
      e.currentTarget.setPointerCapture(e.pointerId);
    }
  };

  const onPointerMove = (e: React.PointerEvent) => {
    const at = viewport.local(e.clientX, e.clientY);
    setHover(at);
    const d = drag.current;
    if (!d) return;
    if (d.kind === "pan") {
      viewport.pan(at[0] - d.last[0], at[1] - d.last[1]);
      d.last = at;
    } else if (size) {
      const here = viewport.toImage(e.clientX, e.clientY);
      const marks = editor.marks.current ?? undefined;
      const rect = strokeLine(d.mask, size.w, size.h, d.last, here, radius, d.classId, marks);
      overlay.paint(rect);
      d.dirty = unionRect(d.dirty, rect);
      d.last = here;
    }
  };

  const endDrag = () => {
    const d = drag.current;
    drag.current = null;
    if (d?.kind !== "stroke") return;
    editor.stroking.current = false;
    // One stroke, one save, one undo step.
    if (d.dirty) void editor.save(d.mask);
  };

  // -- the keyboard -----------------------------------------------------------

  const keys = useRef<(e: KeyboardEvent) => void>(() => {});
  keys.current = (e) => {
    if (typing(e.target) || state.dialog || e.ctrlKey || e.metaKey || e.altKey) return;
    const toggle = (t: ImageTool) => dispatch({ type: "tool", tool: tool === t ? "pan" : t });
    switch (e.key) {
      case "d":
      case "D":
        toggle("polygon");
        break;
      case "b":
      case "B":
        toggle("brush");
        break;
      case "e":
      case "E":
        toggle("eraser");
        break;
      case "a":
      case "A":
        toggle("accept");
        break;
      case "g":
      case "G":
        toggle("gcp");
        break;
      case "Enter":
        if (tool !== "polygon" || polygon.length < 3) return;
        finishPolygon();
        break;
      case "Escape":
        if (polygon.length === 0) return;
        setPolygon([]);
        break;
      case "ArrowLeft":
        dispatch({ type: "step_image", by: -1 });
        break;
      case "ArrowRight":
        dispatch({ type: "step_image", by: 1 });
        break;
      default:
        return;
    }
    e.preventDefault();
  };
  useEffect(() => {
    const on = (e: KeyboardEvent) => keys.current(e);
    window.addEventListener("keydown", on);
    return () => window.removeEventListener("keydown", on);
  }, []);

  // -- the buttons --------------------------------------------------------------

  const importMask = async () => {
    if (!name || !size) return;
    if (activeClass < 0) {
      tell(
        "No active class",
        "Select a label class first: the mask is applied with the active class.",
      );
      return;
    }
    try {
      const file = await pickMaskPixels();
      if (!file) return;
      if (file.width !== size.w || file.height !== size.h) {
        tell(
          "Import mask failed",
          `Mask size is ${file.width} × ${file.height} px but the photo is ${size.w} × ${size.h} px: the mask must match the photo exactly.`,
        );
        return;
      }
      const base = editor.ensure();
      if (!base) return;
      await editor.replace(maskFromPixels(file.rgba, size.w, size.h, file.kind, activeClass, base));
    } catch (e) {
      fail(e, "Import mask failed");
    }
  };

  const spread = async () => {
    if (!name) return;
    try {
      await editor.settled();
      await long("Propagating labels", (t) => propagate(name, t));
    } catch (e) {
      fail(e);
    }
  };

  const ready = !!name && !!size;
  const cursor = drag.current?.kind === "pan" ? "grabbing" : tool === "pan" ? "grab" : "crosshair";
  const polyScreen = polygon.map((p) => imageToScreen(view, p[0], p[1]));
  const path = polyScreen.map((p) => p.join(",")).join(" ");

  return (
    <div className="flex h-full min-h-0 flex-col bg-surface-1">
      <PaneTitle right={row && <span className="font-normal text-fg-muted">{row.status}</span>}>
        <span className="truncate">{name ?? "Image"}</span>
      </PaneTitle>

      <div className="flex flex-wrap items-center gap-x-2 gap-y-1 border-b border-border px-2 py-1">
        {TOOLS.map((t) => (
          <Button
            key={t.tool}
            kind={tool === t.tool ? "primary" : "plain"}
            aria-pressed={tool === t.tool}
            disabled={!ready}
            onClick={() => dispatch({ type: "tool", tool: t.tool })}
          >
            {t.label}
          </Button>
        ))}
        <span className="flex items-center gap-1 text-fg-muted">
          Size
          <Slider
            label="Brush size"
            className="w-24"
            min={1}
            max={200}
            value={brushSize}
            onChange={(v) => dispatch({ type: "brush_size", size: v })}
          />
          <span className="w-8 text-right tabular-nums">{brushSize}</span>
        </span>
        <span className="flex items-center gap-1 text-fg-muted">
          Opacity
          <Slider
            label="Overlay opacity"
            className="w-24"
            min={0}
            max={1}
            step={0.05}
            value={overlayOpacity}
            onChange={(v) => dispatch({ type: "overlay", opacity: v })}
          />
        </span>
        <span className="ml-auto flex items-center gap-1">
          <Button
            kind={bySource ? "primary" : "plain"}
            aria-pressed={bySource}
            disabled={!ready}
            title="Colour the labels by where they came from: green hand-drawn, blue from the cloud, orange predicted. Only hand-drawn and cloud labels train the model."
            onClick={() => setBySource((v) => !v)}
          >
            Sources
          </Button>
          <Button
            kind={showNormals && canNormals ? "primary" : "plain"}
            aria-pressed={showNormals && canNormals}
            disabled={!canNormals}
            title={
              canNormals
                ? "Colour the photo by the dense cloud's normals: red east, green north, blue up"
                : "Normals need a dense cloud and a solved photo"
            }
            onClick={() => setShowNormals((v) => !v)}
          >
            Normals
          </Button>
          <Button
            kind={showUnsure && predicted ? "primary" : "plain"}
            aria-pressed={showUnsure && predicted}
            disabled={!predicted}
            title="Show how unsure each label is: the brighter, the less sure. From the cloud, it comes from the photos' votes; predicted, from the model; hand-drawn labels are sure."
            onClick={() => setShowUnsure((v) => !v)}
          >
            Uncertainty
          </Button>
          <Button disabled={!ready} onClick={importMask}>
            Import Mask…
          </Button>
          <Button disabled={!ready} onClick={spread}>
            Propagate →
          </Button>
          <Button
            kind="quiet"
            aria-label="Previous photo"
            onClick={() => dispatch({ type: "step_image", by: -1 })}
          >
            ◀
          </Button>
          <Button
            kind="quiet"
            aria-label="Next photo"
            onClick={() => dispatch({ type: "step_image", by: 1 })}
          >
            ▶
          </Button>
        </span>
      </div>

      <div
        ref={viewport.ref}
        role="application"
        aria-label="Photo"
        className="relative min-h-0 flex-1 touch-none overflow-hidden bg-surface-0 select-none"
        style={{ cursor }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
        onPointerLeave={() => setHover(null)}
        onMouseDown={(e) => e.button === 1 && e.preventDefault()}
        onContextMenu={(e) => e.preventDefault()}
      >
        {size && (
          <div
            className="absolute top-0 left-0 origin-top-left"
            style={{
              width: size.w,
              height: size.h,
              transform: `translate(${view.x}px, ${view.y}px) scale(${view.scale})`,
            }}
          >
            {photo.url && (
              <img
                src={photo.url}
                alt=""
                draggable={false}
                className="block h-full w-full"
                style={{ imageOrientation: "none" }}
              />
            )}
            {showNormals && canNormals && (
              <canvas
                ref={normalsCanvas}
                className="pointer-events-none absolute inset-0 h-full w-full"
                style={{ imageRendering: "auto" }}
              />
            )}
            {showUnsure && predicted && (
              <canvas
                ref={unsureCanvas}
                className="pointer-events-none absolute inset-0 h-full w-full"
                style={{ imageRendering: "auto" }}
              />
            )}
            <canvas
              ref={overlay.canvas}
              className="absolute inset-0 h-full w-full"
              style={{
                opacity: overlayOpacity,
                imageRendering: view.scale > 4 ? "pixelated" : "auto",
              }}
            />
          </div>
        )}

        <svg className="pointer-events-none absolute inset-0 h-full w-full" aria-hidden="true">
          {polyScreen.length > 1 && (
            <polygon
              points={path}
              fill={classColour}
              fillOpacity={0.15}
              stroke={classColour}
              strokeWidth={2}
            />
          )}
          {polyScreen.map((p, i) => (
            <circle
              // biome-ignore lint/suspicious/noArrayIndexKey: vertices only ever append or pop
              key={i}
              cx={p[0]}
              cy={p[1]}
              r={4}
              fill={classColour}
              stroke="white"
            />
          ))}
          {gcpMarks.hints.map((h) => {
            const [x, y] = imageToScreen(view, h.uv[0], h.uv[1]);
            return (
              <g key={`h-${h.gcp}`} opacity={0.85}>
                <circle
                  cx={x}
                  cy={y}
                  r={14}
                  fill="none"
                  stroke="#58a6ff"
                  strokeWidth={2}
                  strokeDasharray="4 3"
                />
                <text
                  x={x + 17}
                  y={y + 4}
                  fill="#58a6ff"
                  fontSize={12}
                  stroke="black"
                  strokeWidth={3}
                  paintOrder="stroke"
                >
                  {gcpNames.get(h.gcp) ?? h.gcp}?
                </text>
              </g>
            );
          })}
          {gcpMarks.marks.map((m) => {
            const [x, y] = imageToScreen(view, m.uv[0], m.uv[1]);
            const colour = m.gcp === activeGcp ? "#ffd33d" : "#ff7b72";
            return (
              <g key={`m-${m.gcp}`}>
                <path
                  d={`M${x - 10} ${y}H${x + 10}M${x} ${y - 10}V${y + 10}`}
                  stroke="black"
                  strokeWidth={4}
                />
                <path
                  d={`M${x - 10} ${y}H${x + 10}M${x} ${y - 10}V${y + 10}`}
                  stroke={colour}
                  strokeWidth={2}
                />
                <text
                  x={x + 12}
                  y={y - 8}
                  fill={colour}
                  fontSize={12}
                  stroke="black"
                  strokeWidth={3}
                  paintOrder="stroke"
                >
                  {gcpNames.get(m.gcp) ?? m.gcp}
                </text>
              </g>
            );
          })}
          {hover && brushing && (
            <circle
              cx={hover[0]}
              cy={hover[1]}
              r={Math.max(radius * view.scale, 1)}
              fill="none"
              stroke={tool === "eraser" ? "white" : tool === "accept" ? "#3fb950" : classColour}
              strokeWidth={2}
              strokeDasharray={tool === "brush" ? undefined : "4 3"}
            />
          )}
        </svg>

        {(photo.loading || (name && !size)) && !photo.problem && (
          <div className="absolute inset-0 flex items-center justify-center text-fg-muted">
            Loading…
          </div>
        )}
        {!name && (
          <div className="absolute inset-0 flex items-center justify-center text-fg-muted">
            Select a photo.
          </div>
        )}
        {photo.problem && (
          <div
            className={cn(
              "absolute top-2 left-2 max-w-md rounded-md bg-surface-2 px-3 py-2 text-fg-muted",
            )}
          >
            {photo.problem}
          </div>
        )}
      </div>
    </div>
  );
}
