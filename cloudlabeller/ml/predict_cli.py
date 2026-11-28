# CloudLabeller — photogrammetric reconstruction and bidirectional 2D <-> 3D
# point-cloud labelling with U-Net label propagation.
# Copyright (C) 2026 Ítalo Gomes Gonçalves
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program. If not, see <https://www.gnu.org/licenses/>.
#
# Commercial licensing: this program is also available under a separate
# commercial license from the author — see README.md.

"""Subprocess entry point for U-Net prediction.

TensorFlow runs in its own process (see ``train_cli`` for why: GIL deadlock
against Qt when run on an in-process thread).

Usage::

    python -m cloudlabeller.ml.predict_cli PROJECT_ROOT --model NAME

Predicts every image whose status is neither 'user' nor 'auto', writing masks
at model resolution into ``project/ml/predictions/``. Statuses are NOT written
here — the parent owns the manifest and marks images 'ml' on success. Emits
``PROGRESS <frac> <msg>`` lines; prints ``RESULT <json>`` with the predicted
names on success.
"""

from __future__ import annotations

import argparse
import logging
import sys

from cloudlabeller.logging_setup import setup_logging
from cloudlabeller.workers.result import print_result

log = logging.getLogger("cloudlabeller.ml")

#: Statuses Predict leaves alone: labelled by the user, or from the cloud.
PROTECTED = ("user", "auto")


def _parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="predict_cli")
    p.add_argument("project_root")
    p.add_argument("--model", required=True, help="saved model name (ml/models/<name>)")
    p.add_argument("--surveys", default=None,
                   help="comma-separated Survey ids to predict (default: every one)")
    return p.parse_args(argv[1:])


def main(argv: list[str]) -> int:
    args = _parse(argv)
    setup_logging(logfile=None, console=True)     # stderr; parent captures it
    from cloudlabeller.workers.resources import limit_subprocess_resources
    limit_subprocess_resources()

    def progress(frac: float, msg: str = "") -> None:
        print(f"PROGRESS {frac:.4f} {msg}", flush=True)

    from cloudlabeller.core.project import Project
    from cloudlabeller.ml import inference, model_store
    from cloudlabeller.ml.inputs import load_input

    progress(0.0, "Opening project…")
    proj = Project.open(args.project_root)

    manifests = {m["name"]: m for m in model_store.list_models(proj.ml_dir)}
    manifest = manifests.get(model_store.sanitize_name(args.model))
    if manifest is None:
        log.error("model %r not found under %s", args.model, proj.ml_dir / "models")
        return 2
    from cloudlabeller.ml.model_spec import ModelSpec, is_tiled

    if not is_tiled(manifest):
        log.error("model %r was trained on whole photos (before Tiles): retrain it",
                  manifest["name"])
        return 2
    spec = ModelSpec.from_dict(manifest.get("spec"))
    n_classes = len(proj.schema.classes)
    wanted = [s for s in (args.surveys or "").split(",") if s] or list(proj.surveys)
    if manifest["n_classes"] != len(proj.schema.classes):
        log.error("model has %d classes but the project has %d — retrain first",
                  manifest["n_classes"], len(proj.schema.classes))
        return 2

    jobs = []                                      # (survey, [names]) per Survey
    for sid in wanted:
        if sid not in proj.surveys:
            log.error("no Survey %s in the project", sid)
            return 2
        survey = proj.survey(sid)
        # Neither the user's photos nor the ones labelled from the cloud: a
        # prediction must not hide labels that came from a better source
        # (ADR-0006).
        names = [r.name for r in survey.dataset.images
                 if survey.labels.status_of(r.name) not in PROTECTED]
        if spec.normals:
            # Experimental: a model trained on normals predicts only the photos
            # the dense cloud's normals reach.
            from cloudlabeller.ml.inputs import can_have_normals

            reached = [n for n in names if can_have_normals(survey, n)]
            if len(reached) < len(names):
                log.warning("Survey %s: %d photos have no normals (no camera or no dense "
                            "cloud) and are not predicted", sid, len(names) - len(reached))
            names = reached
        if names:
            jobs.append((survey, names))
    total = sum(len(n) for _, n in jobs)
    if not total:
        print_result({"predicted": []})
        return 0

    progress(0.01, f"Loading model '{manifest['name']}'…")
    model, _ = model_store.load_model(proj.ml_dir, manifest["name"])

    predicted, done = [], 0
    for survey, names in jobs:
        records = {r.name: r.path for r in survey.dataset.images}
        start, share = done, len(names) / total
        written = inference.predict_all(
            model, names,
            lambda n, rec=records, sv=survey: load_input(sv, n, rec[n], spec.normals),
            survey.predictions_dir, spec, n_classes, manifest["name"],
            progress=lambda f, m="", s0=start, sh=share:
                progress(0.05 + 0.95 * (s0 / total + sh * f), m))
        predicted += [{"survey": survey.id, "name": n, "uncertainty": written[n]}
                      for n in sorted(written)]
        done += len(names)
    print_result({"predicted": predicted})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
