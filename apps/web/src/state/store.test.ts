// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

import { describe, expect, it } from "vitest";
import type { ProjectState } from "../engine/api";
import { type AppState, available, initialState, MAX_LOG, reduce } from "./store";

const project = (over: Partial<ProjectState> = {}): ProjectState => ({
  root: "C:/p",
  surveys: [],
  survey: "s1",
  schema: [
    { id: 0, name: "rock", color: "#ff0000", hotkey: "1" },
    { id: 1, name: "soil", color: "#00ff00", hotkey: "2" },
  ],
  images: [
    { name: "a.jpg", status: "none", solved: true, user_mask: false },
    { name: "b.jpg", status: "user", solved: true, user_mask: true },
  ],
  reconstruction: "current",
  sparse: 100,
  dense: null,
  mesh: null,
  georeferenced: false,
  source_resolution: [4000, 3000],
  can_undo: false,
  can_redo: false,
  dirty: false,
  ...over,
});

const opened = (p = project()): AppState => reduce(initialState, { type: "opened", project: p });

describe("the store", () => {
  it("opens on the best representation the Project has, its first class and photo", () => {
    const s = opened();
    expect(s.view.representation).toBe("sparse");
    expect(s.activeClass).toBe(0);
    expect(s.selected).toBe("a.jpg");
    expect(opened(project({ dense: 5 })).view.representation).toBe("dense");
  });

  it("falls back from a representation the Project lacks", () => {
    expect(available(project(), "mesh")).toBe("sparse");
    expect(available(project({ sparse: null, mesh: { vertices: 3, faces: 1 } }), "dense")).toBe(
      "mesh",
    );
    const s = reduce(opened(), { type: "view", patch: { representation: "mesh", lasso: true } });
    expect(s.view.representation).toBe("sparse");
  });

  it("turns the lasso off on the mesh and drops a selection when the representation changes", () => {
    let s = opened(project({ dense: 5, mesh: { vertices: 3, faces: 1 } }));
    s = reduce(s, { type: "selection", selection: { cloud: "dense", indices: Uint32Array.of(1) } });
    s = reduce(s, { type: "view", patch: { lasso: true } });
    expect(s.selection).not.toBeNull();
    s = reduce(s, { type: "view", patch: { representation: "mesh" } });
    expect(s.view.lasso).toBe(false);
    expect(s.selection).toBeNull();
  });

  it("keeps the active class and isolation only while the class exists", () => {
    let s = reduce(opened(), { type: "active_class", id: 1 });
    s = reduce(s, { type: "view", patch: { isolate: 1 } });
    expect(s.activeClass).toBe(1);
    expect(reduce(s, { type: "active_class", id: 9 }).activeClass).toBe(1);
    s = reduce(s, {
      type: "project",
      project: project({ schema: [{ id: 0, name: "rock", color: "#f00", hotkey: "1" }] }),
    });
    expect(s.activeClass).toBe(0);
    expect(s.view.isolate).toBeNull();
    expect(reduce(s, { type: "active_class", id: -1 }).activeClass).toBe(-1);
  });

  it("steps through the photos without running off either end", () => {
    let s = reduce(opened(), { type: "step_image", by: 1 });
    expect(s.selected).toBe("b.jpg");
    s = reduce(s, { type: "step_image", by: 1 });
    expect(s.selected).toBe("b.jpg");
    s = reduce(s, { type: "step_image", by: -1 });
    expect(s.selected).toBe("a.jpg");
  });

  it("bumps what must be re-read when the Project changes", () => {
    let s = opened();
    s = reduce(s, {
      type: "selection",
      selection: { cloud: "sparse", indices: Uint32Array.of(1, 2) },
    });
    const labels = reduce(s, { type: "changed", parts: ["labels"] });
    expect(labels.labelsVersion).toBe(s.labelsVersion + 1);
    expect(labels.geometryVersion).toBe(s.geometryVersion);
    expect(labels.selection).not.toBeNull();
    const shape = reduce(s, { type: "changed", parts: ["sparse"] });
    expect(shape.geometryVersion).toBe(s.geometryVersion + 1);
    expect(shape.selection).toBeNull();
    const masks = reduce(s, { type: "changed", parts: ["status"], images: ["a.jpg"] });
    expect(masks.masks).toEqual({ version: s.masks.version + 1, images: ["a.jpg"] });
    // an undone mask edit comes as labels, with the photo named
    const undone = reduce(s, { type: "changed", parts: ["labels"], images: ["b.jpg"] });
    expect(undone.masks).toEqual({ version: s.masks.version + 1, images: ["b.jpg"] });
    expect(labels.masks).toEqual(s.masks);
  });

  it("keeps the last lines of the log", () => {
    let s = initialState;
    for (let i = 0; i < MAX_LOG + 10; i++) s = reduce(s, { type: "log", text: `line ${i}` });
    expect(s.log).toHaveLength(MAX_LOG);
    expect(s.log.at(-1)?.text).toBe(`line ${MAX_LOG + 9}`);
    expect(
      reduce(s, { type: "log", text: "a\nb" })
        .log.slice(-2)
        .map((l) => l.text),
    ).toEqual(["a", "b"]);
  });

  it("shows another Survey's first photo, with nothing chosen and every view reloaded", () => {
    let s = opened();
    s = reduce(s, { type: "select_image", name: "b.jpg" });
    s = reduce(s, {
      type: "selection",
      selection: { cloud: "sparse", indices: new Uint32Array([1, 2]) },
    });
    s = reduce(s, { type: "active_class", id: 1 });
    const other = project({
      survey: "s2",
      images: [{ name: "x.jpg", status: "none", solved: false, user_mask: false }],
      sparse: null,
    });
    const t = reduce(s, { type: "switched", project: other });
    expect(t.selected).toBe("x.jpg");
    expect(t.selection).toBeNull();
    expect(t.activeClass).toBe(1);
    expect(t.view.representation).toBe("sparse");
    expect(t.geometryVersion).toBe(s.geometryVersion + 1);
    expect(t.masks.version).toBe(s.masks.version + 1);
  });

  it("clamps the brush and the overlay", () => {
    expect(reduce(initialState, { type: "brush_size", size: 0 }).brushSize).toBe(1);
    expect(reduce(initialState, { type: "overlay", opacity: 2 }).overlayOpacity).toBe(1);
  });
});
