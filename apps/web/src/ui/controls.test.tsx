// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

import { act, useState } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Dialog } from "./controls";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement | null = null;
afterEach(() => {
  host?.remove();
  host = null;
});

function render(node: React.ReactNode) {
  host = document.createElement("div");
  document.body.append(host);
  const root = createRoot(host);
  act(() => root.render(node));
  return root;
}

/** Like the Train dialog: a parent that re-renders, and hands a new onClose, on each change. */
function TwoFields({ onClose }: { onClose: () => void }) {
  const [name, setName] = useState("unet");
  const [epochs, setEpochs] = useState("100");
  return (
    <Dialog title="Two fields" onClose={() => onClose()}>
      <input aria-label="name" value={name} onChange={(e) => setName(e.target.value)} />
      <input aria-label="epochs" value={epochs} onChange={(e) => setEpochs(e.target.value)} />
    </Dialog>
  );
}

function type(input: HTMLInputElement, value: string) {
  const set = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")?.set;
  set?.call(input, value);
  input.dispatchEvent(new Event("input", { bubbles: true }));
}

describe("Dialog", () => {
  it("focuses the first field when it opens", () => {
    render(<TwoFields onClose={() => undefined} />);
    expect(document.activeElement?.getAttribute("aria-label")).toBe("name");
  });

  it("keeps the focus where the user typed when the parent re-renders", () => {
    render(<TwoFields onClose={() => undefined} />);
    const epochs = document.querySelector<HTMLInputElement>('[aria-label="epochs"]');
    if (!epochs) throw new Error("no field");
    act(() => epochs.focus());
    act(() => type(epochs, "1000"));
    expect(epochs.value).toBe("1000");
    expect(document.activeElement).toBe(epochs);
  });

  it("closes on Escape with the latest onClose", () => {
    const onClose = vi.fn();
    render(<TwoFields onClose={onClose} />);
    act(() => {
      window.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    });
    expect(onClose).toHaveBeenCalledOnce();
  });
});
