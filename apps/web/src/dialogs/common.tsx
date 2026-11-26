// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

// The plain dialogs: a message, a question, Project Info and About.

import { useCallback, useEffect, useState } from "react";
import * as api from "../engine/api";
import { useApp } from "../state/app";
import { Button, Dialog } from "../ui/controls";

export function useClose() {
  const { dispatch } = useApp();
  return useCallback(() => dispatch({ type: "dialog", dialog: null }), [dispatch]);
}

export function MessageDialog({ title, text }: { title: string; text: string }) {
  const close = useClose();
  return (
    <Dialog
      title={title}
      onClose={close}
      footer={
        <Button kind="primary" onClick={close}>
          OK
        </Button>
      }
    >
      <p className="whitespace-pre-wrap">{text}</p>
    </Dialog>
  );
}

export function ConfirmDialog({
  title,
  text,
  answers,
  resolve,
}: {
  title: string;
  text: string;
  answers: string[];
  resolve: (answer: string | null) => void;
}) {
  const app = useApp();
  const answer = (a: string | null) => {
    app.dispatch({ type: "dialog", dialog: null });
    resolve(a);
  };
  return (
    <Dialog
      title={title}
      onClose={() => answer(null)}
      footer={
        <>
          <Button onClick={() => answer(null)}>Cancel</Button>
          {answers.map((a, i) => (
            <Button
              key={a}
              kind={i === answers.length - 1 ? "primary" : "plain"}
              onClick={() => answer(a)}
            >
              {a}
            </Button>
          ))}
        </>
      }
    >
      <p className="whitespace-pre-wrap">{text}</p>
    </Dialog>
  );
}

export function ProjectInfoDialog() {
  const app = useApp();
  const close = useClose();
  const [rows, setRows] = useState<[string, string][] | null>(null);
  useEffect(() => {
    api
      .summary()
      .then((r) => setRows(r.rows))
      .catch((e) => app.fail(e, "Couldn't read the Project's summary"));
  }, [app.fail]);
  const text = (rows ?? []).map(([k, v]) => `${k}: ${v}`).join("\n");
  return (
    <Dialog
      title="Project Info"
      wide
      onClose={close}
      footer={
        <>
          <Button onClick={() => void navigator.clipboard?.writeText(text)}>Copy</Button>
          <Button kind="primary" onClick={close}>
            Close
          </Button>
        </>
      }
    >
      {rows ? (
        <table className="w-full text-table select-text">
          <tbody>
            {rows.map(([k, v]) => (
              <tr key={k} className="border-b border-border last:border-0">
                <td className="py-1 pr-4 align-top text-fg-muted">{k}</td>
                <td className="py-1 break-all">{v}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p className="text-fg-muted">Reading…</p>
      )}
    </Dialog>
  );
}

export function AboutDialog() {
  const close = useClose();
  const [version, setVersion] = useState<string | null>(null);
  useEffect(() => {
    api
      .hello()
      .then((h) => setVersion(h.version))
      .catch(() => setVersion(null));
  }, []);
  return (
    <Dialog
      title="About CloudLabeller"
      onClose={close}
      footer={
        <Button kind="primary" onClick={close}>
          Close
        </Button>
      }
    >
      <div className="flex flex-col gap-3">
        <p>
          <span className="text-lg font-semibold">CloudLabeller</span>{" "}
          {version && <span className="text-fg-muted">{version}</span>}
        </p>
        <p>Photogrammetric reconstruction and 2D ↔ 3D labelling with U-Net label propagation.</p>
        <p>
          Created by <b>Ítalo Gomes Gonçalves</b>
        </p>
        <p className="rounded-md border border-border bg-surface-2 p-3">
          CloudLabeller is an <b>AI-assisted project</b>: most of its application code was written,
          tested and debugged by an AI coding assistant (Anthropic's Claude) working under the
          direction of the author, who defined the requirements, the scientific approach and the
          machine-learning methodology. As with any software — AI-assisted or otherwise — results
          should be independently validated before use in scientific or operational work.
        </p>
        <p className="text-table text-fg-muted">
          Licensed under the GNU GPL v3 or later; commercial licenses are available from the author
          (see README.md). Built on COLMAP / pycolmap, TensorFlow, laspy and trimesh; the window on
          Tauri, React, three.js and RockAnalytica's geokit (Apache-2.0).
        </p>
      </div>
    </Dialog>
  );
}
