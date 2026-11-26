// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

import { afterEach, describe, expect, it } from "vitest";
import {
  clearSurvey,
  currentSurvey,
  currentSurveyOrNull,
  type RawProjectState,
  type SurveyState,
  view,
} from "./api";

const survey = (id: string, over: Partial<SurveyState> = {}): SurveyState => ({
  id,
  name: `Survey ${id}`,
  timestamp: null,
  operator: "",
  camera: "",
  notes: "",
  images: [{ name: `${id}.jpg`, status: "none", solved: true, user_mask: false }],
  reconstruction: "current",
  active: "r1",
  aligned: false,
  sparse: 10,
  dense: null,
  mesh: null,
  source_resolution: [40, 30],
  can_undo: false,
  can_redo: false,
  ...over,
});

const raw = (over: Partial<RawProjectState> = {}): RawProjectState => ({
  root: "C:/p",
  schema: [],
  surveys: [survey("s1"), survey("s2", { aligned: true, sparse: 99 })],
  georeferenced: { frame: "ENU" },
  dirty: false,
  ...over,
});

afterEach(() => clearSurvey());

describe("view", () => {
  it("shows the first Survey of a Project just opened, and makes it current", () => {
    const p = view(raw());
    expect(p.survey).toBe("s1");
    expect(p.images[0]?.name).toBe("s1.jpg");
    expect(p.surveys.map((s) => s.id)).toEqual(["s1", "s2"]);
    expect(currentSurvey()).toBe("s1");
  });

  it("stays on the current Survey, or the one asked for", () => {
    view(raw(), "s2");
    const p = view(raw());
    expect(p.survey).toBe("s2");
    expect(p.sparse).toBe(99);
  });

  it("falls back to the first Survey when the current one is gone", () => {
    view(raw(), "s2");
    expect(view(raw({ surveys: [survey("s1")] })).survey).toBe("s1");
  });

  it("is georeferenced only when the Project has a frame and the Survey is aligned", () => {
    expect(view(raw(), "s1").georeferenced).toBe(false);
    expect(view(raw(), "s2").georeferenced).toBe(true);
    expect(view(raw({ georeferenced: null }), "s2").georeferenced).toBe(false);
  });

  it("has no current Survey once cleared", () => {
    view(raw());
    clearSurvey();
    expect(currentSurveyOrNull()).toBeNull();
    expect(() => currentSurvey()).toThrow();
  });
});
