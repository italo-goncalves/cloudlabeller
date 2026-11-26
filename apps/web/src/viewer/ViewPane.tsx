// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

// The left "View" pane: a toolbar (the home view, the lasso and what it does
// to its selection), then what the 3D viewer shows and how, and the cameras. Replaces the
// Qt app's ui/view_panel.py.

import { type ReactNode, useId } from "react";
import { VIEW_DETAIL } from "../detail";
import * as api from "../engine/api";
import { useApp } from "../state/app";
import type { Representation, ViewDetail, ViewSettings } from "../state/store";
import { Check, IconButton, PaneTitle, Radios, Select, Slider } from "../ui/controls";
import { ClearIcon, GcpIcon, HomeIcon, LabelIcon, LassoIcon, TrashIcon } from "../ui/icons";
import { pointsLabel } from "./adapt";
import { goHome } from "./home";

const DETAIL_NAMES: Record<ViewDetail, string> = {
  low: "Low",
  medium: "Medium",
  high: "High",
  full: "Full",
};

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="mb-4">
      <h3 className="mb-1.5 font-semibold text-fg-muted">{title}</h3>
      {children}
    </section>
  );
}

export function ViewPane() {
  const { state, dispatch, fail, confirm } = useApp();
  const { project, view, selection, activeClass } = state;
  const detailId = useId();
  const set = (patch: Partial<ViewSettings>) => dispatch({ type: "view", patch });

  const has = {
    sparse: (project?.sparse ?? 0) > 0,
    dense: (project?.dense ?? 0) > 0,
    mesh: project?.mesh != null,
  };
  const count = selection?.indices.length ?? 0;
  const className = project?.schema.find((c) => c.id === activeClass)?.name;
  const canPaint = selection !== null && (project?.schema.length ?? 0) > 0;
  const canDelete = selection !== null && selection.cloud === "sparse";

  const apply = async () => {
    if (!selection) return;
    try {
      await api.cloudPaint(selection.cloud, selection.indices, activeClass);
      dispatch({ type: "selection", selection: null });
    } catch (e) {
      fail(e, "Couldn't apply the label");
    }
  };

  const remove = async () => {
    if (selection?.cloud !== "sparse") return;
    const n = selection.indices.length;
    const yes = await confirm(
      "Delete points",
      `Delete ${n.toLocaleString()} ${n === 1 ? "point" : "points"} from the sparse cloud?`,
      "Delete",
    );
    if (!yes) return;
    try {
      await api.cloudDelete("sparse", selection.indices);
      dispatch({ type: "selection", selection: null });
    } catch (e) {
      fail(e, "Couldn't delete the points");
    }
  };

  return (
    <div className="flex h-full min-h-0 flex-col">
      <PaneTitle>View</PaneTitle>
      <div className="flex items-center gap-0.5 border-b border-border px-2 py-1">
        <IconButton title="Home view" onClick={goHome}>
          <HomeIcon />
        </IconButton>
        <span className="mx-1 h-5 w-px bg-border" />
        <IconButton
          title={
            view.representation === "mesh"
              ? "Lasso select (on a cloud only)"
              : "Lasso select: left-drag draws a loop on the cloud; middle and right still move the view"
          }
          pressed={view.lasso}
          disabled={view.representation === "mesh"}
          onClick={() => set({ lasso: !view.lasso })}
        >
          <LassoIcon />
        </IconButton>
        <IconButton
          title={
            view.representation === "mesh"
              ? "Place a GCP (on a cloud only)"
              : "Place the chosen GCP: a click snaps its Mark to the nearest cloud point"
          }
          pressed={view.gcpPick}
          disabled={view.representation === "mesh"}
          onClick={() => set({ gcpPick: !view.gcpPick })}
        >
          <GcpIcon />
        </IconButton>
        <IconButton
          title={
            className ? `Label the selected points “${className}”` : "Label the selected points"
          }
          disabled={!canPaint}
          onClick={() => void apply()}
        >
          <LabelIcon />
        </IconButton>
        <IconButton
          title="Delete the selected points (sparse cloud only)"
          className="hover:text-danger"
          disabled={!canDelete}
          onClick={() => void remove()}
        >
          <TrashIcon />
        </IconButton>
        <IconButton
          title="Clear the selection"
          disabled={selection === null}
          onClick={() => dispatch({ type: "selection", selection: null })}
        >
          <ClearIcon />
        </IconButton>
        <span className="ml-auto truncate pl-1 text-table text-fg-muted">
          {count === 0 ? "" : `${count.toLocaleString()} selected`}
        </span>
      </div>
      <div className="min-h-0 flex-1 overflow-auto p-3">
        <Section title="Show">
          <Radios<Representation>
            name="representation"
            value={view.representation}
            onChange={(representation) => set({ representation })}
            options={[
              { value: "sparse", label: "Sparse", disabled: !has.sparse },
              { value: "dense", label: "Dense", disabled: !has.dense },
              { value: "mesh", label: "Mesh", disabled: !has.mesh },
            ]}
          />
        </Section>

        <Section title="Colour">
          <Slider
            label="Blend from photo colours to labels"
            value={view.blend}
            min={0}
            max={1}
            step={0.01}
            onChange={(blend) => set({ blend })}
          />
          <div className="flex justify-between text-fg-faint">
            <span>RGB</span>
            <span>Labels</span>
          </div>
        </Section>

        <Section title="Points">
          <div className="mb-1 flex items-center gap-2">
            <span className="w-16 shrink-0 text-fg-muted">Size</span>
            <Slider
              label="Point size"
              value={view.pointSize}
              min={1}
              max={10}
              step={0.5}
              onChange={(pointSize) => set({ pointSize })}
            />
            <span className="w-6 text-right tabular-nums">{view.pointSize}</span>
          </div>
          <div className="flex items-center gap-2">
            <label htmlFor={detailId} className="w-16 shrink-0 text-fg-muted">
              Detail
            </label>
            <Select<ViewDetail>
              id={detailId}
              value={view.detail}
              onChange={(detail) => set({ detail })}
              options={(Object.keys(VIEW_DETAIL) as ViewDetail[]).map((d) => ({
                value: d,
                label: `${DETAIL_NAMES[d]} (${VIEW_DETAIL[d] === 0 ? "all points" : `${pointsLabel(VIEW_DETAIL[d])} points`})`,
              }))}
            />
          </div>
        </Section>

        <Section title="Cameras">
          <Check checked={view.showCameras} onChange={(showCameras) => set({ showCameras })}>
            Show cameras
          </Check>
          <div className="flex items-center gap-2">
            <span className="w-16 shrink-0 text-fg-muted">Size</span>
            <Slider
              label="Camera size"
              value={view.frustumScale}
              min={0.02}
              max={0.8}
              step={0.01}
              onChange={(frustumScale) => set({ frustumScale })}
            />
          </div>
        </Section>
      </div>
    </div>
  );
}
