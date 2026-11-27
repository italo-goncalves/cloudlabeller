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

"""Experiment: do the dense cloud's normals help the U-Net?

Each photo gets a **normal map**: the normals of the dense-cloud points
visible in it, rotated into the camera's frame (x right, y down, z forward)
and splatted into their pixels as the labels are for an auto mask. They
become three more input channels beside RGB, stored as uint8 like the
photo: ``round(127.5 * (n + 1))``, so 128 means "no normal here".

Training runs twice on the same Tiles and the same steps: on RGB alone
(the first three channels) and on RGB + normals. It trains on the photos
labelled from the cloud and validates on the photos the user painted, so
the score is against ground truth the cloud didn't make (a cloud-made
mask and its normal map share their edges, which would flatter the
normals).

The Project is only read. Visibility comes from the Survey's cache
(``cache/pmap``) through :func:`read_only_visibility`, which never writes.

Usage (in the WSL Environment, which has TensorFlow with the GPU)::

    PYTHONPATH=<repo> python tools/experiments/normals_as_features.py
        PROJECT OUT [--survey s1] [--steps 50000] [--patience 10] [--seeds 2]

``OUT`` gets ``tiles.npz`` (cut once, reused) and ``results.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from cloudlabeller.transfer.normal_map import NO_NORMAL, normal_map

# The normal maps (``transfer.normal_map``) and the augmentation that turns
# the normals' x around in mirrored windows (``ml.tile_dataset.augment``)
# are the app's own since the experiment (ModelSpec.normals).


def read_only_visibility(cache_dir: Path):
    """A ``VisibilityIndex`` that reads the cache and never writes to it."""
    from cloudlabeller.transfer.visibility import VisibilityIndex

    class ReadOnlyVisibility(VisibilityIndex):
        def _store(self, *args, **kwargs) -> None:
            pass

    return ReadOnlyVisibility(cache_dir)


# -- cutting ------------------------------------------------------------------------
def cut_tiles(project_root: Path, survey_id: str, out: Path, spec) -> dict:
    """Cut 6-channel Tiles of the cloud-labelled photos (training) and the
    user-painted ones (validation), once; ``out/tiles.npz`` keeps them."""
    cache = out / "tiles.npz"
    if cache.exists():
        with np.load(cache) as d:
            return {k: d[k] for k in d.files}

    from PIL import Image

    from cloudlabeller.core.project import Project
    from cloudlabeller.ml.tiles import cut
    from cloudlabeller.transfer.project_to_image import cloud_to_image

    t0 = time.perf_counter()
    proj = Project.open(project_root)
    survey = proj.survey(survey_id)
    lb = survey.labels
    cloud, labels = survey.auto_mask_source()
    assert cloud is survey.dataset.dense_cloud, "the experiment needs the dense cloud labelled"
    normals = cloud.normals
    vis = read_only_visibility(survey.cache_dir / "pmap")
    print(f"loaded in {time.perf_counter() - t0:.0f} s; {cloud.n_points} points", flush=True)

    groups: dict[str, list] = {"train": [], "val": []}
    for r in survey.dataset.images:
        if r.camera is None:
            continue
        status = lb.status_of(r.name)
        if status == "auto":
            groups["train"].append(r)
        elif status == "user" and r.name in lb.image_masks:
            groups["val"].append(r)

    arrays: dict = {}
    for part, records in groups.items():
        xs, ys, names, seen = [], [], [], []
        for k, r in enumerate(records):
            idx, pix = vis.get(r.name, cloud, r.camera)
            nmap = normal_map(normals[idx], pix, r.camera.width, r.camera.height, r.camera.R)
            seen.append(float((nmap != NO_NORMAL).any(axis=2).mean()))
            if part == "train":
                mask = cloud_to_image(cloud, labels, r.camera, name=r.name, visibility=vis)
            else:
                mask = lb.image_masks[r.name]
            with Image.open(r.path) as im:
                rgb = np.asarray(im.convert("RGB"))
            if rgb.shape[:2] != nmap.shape[:2]:
                print(f"skip {r.name}: photo {rgb.shape[:2]} vs camera {nmap.shape[:2]}")
                continue
            image = np.concatenate([rgb, nmap], axis=2)
            for (i, j), img, m in cut(image, mask, spec.tile_size, spec.overlap, spec.shrink,
                                      spec.min_labelled):
                xs.append(img)
                ys.append(m)
                names.append(f"{r.name}#{i},{j}")
            if (k + 1) % 25 == 0 or k + 1 == len(records):
                print(f"{part}: {k + 1}/{len(records)} photos, {len(xs)} Tiles, "
                      f"{time.perf_counter() - t0:.0f} s", flush=True)
        arrays[f"{part}_x"] = np.stack(xs)
        arrays[f"{part}_y"] = np.stack(ys)
        arrays[f"{part}_names"] = np.array(names)
        arrays[f"coverage_{part}"] = np.array(seen)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **arrays)
    return arrays


# -- training -----------------------------------------------------------------------
def train_variant(tiles: dict, channels: int, spec, n_classes: int, *, steps: int,
                  patience: int, batch_size: int, rolls: int, seed: int) -> dict:
    """One training run on the first ``channels`` channels of the Tiles."""
    import tensorflow as tf

    from cloudlabeller.ml import trainer
    from cloudlabeller.ml.dataset_builder import Sample, TrainingSet
    from cloudlabeller.ml.unet import build_model_from_spec

    tf.keras.utils.set_random_seed(seed)

    def tset(part):
        return TrainingSet([Sample(str(n), np.ascontiguousarray(x[..., :channels]), y, 1.0)
                            for n, x, y in zip(tiles[f"{part}_names"], tiles[f"{part}_x"],
                                               tiles[f"{part}_y"])])
    train_set, val_set = tset("train"), tset("val")
    model = build_model_from_spec(spec, n_classes, in_channels=channels)
    t0 = time.perf_counter()
    model, history, outcome = trainer.train(
        model, train_set, val_set, steps=steps, batch_size=batch_size, augment_rolls=rolls,
        patience=patience,
        progress=lambda f, m="": print(f"  [{channels}ch s{seed}] {m}", flush=True))
    best = outcome["best_epoch"]
    return {"channels": channels, "seed": seed, "minutes": (time.perf_counter() - t0) / 60,
            "outcome": outcome, "best": {k: v[best] for k, v in history.items()},
            "history": history}


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="normals_as_features")
    p.add_argument("project")
    p.add_argument("out")
    p.add_argument("--survey", default="s1")
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--rolls", type=int, default=5)
    p.add_argument("--seeds", type=int, default=2)
    a = p.parse_args(argv[1:])

    from cloudlabeller.core.project import Project

    proj = Project.open(a.project)
    spec = proj.model_spec()
    n_classes = len(proj.schema.classes)
    out = Path(a.out)
    print(f"spec: {spec.to_dict()}, {n_classes} classes", flush=True)
    tiles = cut_tiles(Path(a.project), a.survey, out, spec)
    print(f"Tiles: {len(tiles['train_x'])} training, {len(tiles['val_x'])} validation; "
          f"normals cover {tiles['coverage_train'].mean():.0%} of the training photos, "
          f"{tiles['coverage_val'].mean():.0%} of the validation ones", flush=True)

    results_path = out / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else []
    done = {(r["channels"], r["seed"]) for r in results}
    for seed in range(a.seeds):
        for channels in (3, 6):
            if (channels, seed) in done:
                continue
            r = train_variant(tiles, channels, spec, n_classes, steps=a.steps,
                              patience=a.patience, batch_size=a.batch_size, rolls=a.rolls,
                              seed=seed)
            results.append(r)
            results_path.write_text(json.dumps(results, indent=1))
            print(f"RESULT {channels}ch seed {seed}: best epoch "
                  f"{r['outcome']['best_epoch'] + 1} {json.dumps(r['best'])}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
