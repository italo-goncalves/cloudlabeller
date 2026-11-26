// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

// The one dialog open at a time, by its name in the state.

import { useApp } from "../state/app";
import { AboutDialog, ConfirmDialog, MessageDialog, ProjectInfoDialog } from "./common";
import { ExportDialog, ReprojectDialog } from "./crs";
import { ModelSpecDialog, PredictDialog, TrainDialog } from "./model";
import {
  CleanDialog,
  ColmapDialog,
  MeshDialog,
  MvsDialog,
  PipelineDialog,
  SfmDialog,
} from "./reconstruction";
import { ReconstructionsDialog, SurveyDialog } from "./surveys";

export function DialogHost() {
  const { dialog } = useApp().state;
  if (!dialog) return null;
  const d = dialog.data ?? {};
  switch (dialog.name) {
    case "message":
      return <MessageDialog title={String(d.title ?? "")} text={String(d.text ?? "")} />;
    case "confirm":
      return (
        <ConfirmDialog
          title={String(d.title ?? "")}
          text={String(d.text ?? "")}
          answers={(d.answers as string[]) ?? ["OK"]}
          resolve={(d.resolve as (a: string | null) => void) ?? (() => undefined)}
        />
      );
    case "project_info":
      return <ProjectInfoDialog />;
    case "about":
      return <AboutDialog />;
    case "survey":
      return <SurveyDialog create={d.create === true} />;
    case "reconstructions":
      return <ReconstructionsDialog />;
    case "sfm":
      return <SfmDialog />;
    case "mvs":
      return <MvsDialog />;
    case "mesh":
      return <MeshDialog />;
    case "pipeline":
      return <PipelineDialog />;
    case "colmap":
      return <ColmapDialog />;
    case "clean":
      return <CleanDialog />;
    case "reproject":
      return <ReprojectDialog />;
    case "export_cloud":
      return <ExportDialog what="cloud" />;
    case "export_mesh":
      return <ExportDialog what="mesh" />;
    case "model_spec":
      return <ModelSpecDialog />;
    case "train":
      return <TrainDialog />;
    case "predict":
      return <PredictDialog />;
    default:
      return null;
  }
}
