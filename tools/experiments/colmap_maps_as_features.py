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

"""Experiment: COLMAP's own depth and normal maps as U-Net inputs.

COLMAP's dense step (PatchMatch stereo) leaves a depth map and a normal map
per photo in ``dense/stereo/``, made on the *undistorted* photos at the MVS
resolution. This script moves the geometric (consistency-checked) ones into
the original photos' pixels and compares four inputs on the same Tiles:

``rgb``
    the photo alone;
``cloud``
    RGB + the dense cloud's normals splatted into the photo
    (``transfer.normal_map``, what ``ModelSpec.normals`` uses now);
``colmap``
    RGB + COLMAP's geometric normal map, camera frame;
``invariant``
    RGB + four channels that don't depend on where the camera was: a
    validity flag, the slope (angle of the world-frame normal to the
    vertical) and the relief at two scales (the depth's local smoothed value
    minus the depth, in metres: what sticks out towards the camera is
    positive, whatever the distance).

Everything runs at 1/``--reduce`` of the photos' size (the network's own
resolution with shrink 8), so the Tiles are ``--tile / --reduce`` px and
shrink 1. Training and validation both use the user-painted photos, split by
whole photos; each split seed trains every variant (a paired comparison).

It reads a **0.3.0-layout** Project (``images/``, ``labels/images/``,
``reconstruction/``) and writes nothing in it: everything goes to ``OUT``.

    PYTHONPATH=<repo> python tools/experiments/colmap_maps_as_features.py PROJECT OUT
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from cloudlabeller.transfer.normal_map import NO_NORMAL, encode_normals, unit

VARIANTS = {"rgb": [0, 1, 2], "cloud": [0, 1, 2, 3, 4, 5], "colmap": [0, 1, 2, 6, 7, 8],
            "invariant": [0, 1, 2, 9, 10, 11, 12]}


# -- COLMAP maps --------------------------------------------------------------------
def read_colmap_array(path: Path) -> np.ndarray:
    """A COLMAP ``.bin`` depth or normal map: (H, W) or (H, W, 3) float32."""
    with open(path, "rb") as f:
        raw = f.read()
    seps, i = [], 0
    while len(seps) < 3:
        i = raw.index(b"&", i)
        seps.append(i)
        i += 1
    w, h, c = (int(x) for x in raw[:seps[2]].split(b"&"))
    data = np.frombuffer(raw[seps[2] + 1:], np.float32, count=w * h * c)
    return np.transpose(data.reshape((w, h, c), order="F"), (1, 0, 2)).squeeze()


def colmap_into_photo(depth, normals, K_undist, camera, scale: float, reduce: int):
    """COLMAP's maps moved into the original photo, at 1/``reduce`` of its
    size: (depth in metres, camera-frame normals, valid), z-buffered and
    with one-pixel holes filled from a neighbour.

    Each map pixel is lifted to its 3D point (in metres: ``scale`` turns the
    SfM units the maps were made in into the Project frame's), then
    projected by the original camera, lens distortion included."""
    from scipy import ndimage

    from cloudlabeller.transfer.projection import project_points

    h8, w8 = int(camera.height) // reduce, int(camera.width) // reduce
    rows, cols = np.nonzero(depth > 0)
    d = depth[rows, cols].astype(np.float64) * scale
    fx, fy, cx, cy = K_undist
    cam_pts = np.column_stack([(cols + 0.5 - cx) / fx * d, (rows + 0.5 - cy) / fy * d, d])
    R, t = np.asarray(camera.R, float), np.asarray(camera.t, float)
    world = (cam_pts - t) @ R                                   # R^T (x - t)
    pp, vis = project_points(camera, world)
    u, v = pp[:, 0] / reduce, pp[:, 1] / reduce
    ok = vis & (u >= 0) & (u < w8) & (v >= 0) & (v < h8)
    u, v, d, n = u[ok].astype(int), v[ok].astype(int), d[ok], normals[rows[ok], cols[ok]]
    order = np.argsort(-d)                                       # nearest written last
    out_d = np.zeros((h8, w8), np.float32)
    out_n = np.zeros((h8, w8, 3), np.float32)
    out_d[v[order], u[order]] = d[order]
    out_n[v[order], u[order]] = n[order]
    valid = out_d > 0
    dist, (iy, ix) = ndimage.distance_transform_edt(~valid, return_indices=True)
    fill = (~valid) & (dist <= 1.5)
    out_d[fill] = out_d[iy[fill], ix[fill]]
    out_n[fill] = out_n[iy[fill], ix[fill]]
    return out_d, out_n, valid | fill


def invariant_channels(depth_m, normals_cam, valid, R, focal: float, scales):
    """(H, W, 4) uint8: valid, slope, relief at each of the two ``scales``
    (metres). Slope 0..90 degrees maps to 0..255; relief r to
    ``128 + 127 tanh(2 r / scale)`` (128: flat or unknown)."""
    from scipy import ndimage

    n_world = unit(normals_cam) @ np.asarray(R, float)          # R^T n
    slope = np.degrees(np.arccos(np.clip(np.abs(n_world[..., 2]), 0, 1)))
    out = np.zeros(depth_m.shape + (4,), np.uint8)
    out[..., 0] = np.where(valid, 255, 0)
    out[..., 1] = np.where(valid, np.rint(slope / 90 * 255), 0).astype(np.uint8)
    gsd = float(np.median(depth_m[valid])) / focal if valid.any() else 1.0
    w = valid.astype(np.float32)
    for k, s in enumerate(scales):
        sigma = max(0.5, s / gsd)
        num = ndimage.gaussian_filter(depth_m * w, sigma)
        den = ndimage.gaussian_filter(w, sigma)
        relief = np.where(den > 1e-3, num / np.maximum(den, 1e-3), depth_m) - depth_m
        enc = 128 + 127 * np.tanh(2 * relief / s)
        out[..., 2 + k] = np.where(valid, np.rint(enc), 128).astype(np.uint8)
    return out, gsd


def cloud_normals(cloud, camera, reduce: int):
    """The dense cloud's normals splatted into the photo at 1/``reduce``."""
    from cloudlabeller.transfer.normal_map import normal_map
    from cloudlabeller.transfer.projection import visible_point_pixels

    idx, pix = visible_point_pixels(cloud, camera)
    h8, w8 = int(camera.height) // reduce, int(camera.width) // reduce
    pix8 = np.minimum(pix // reduce, [w8 - 1, h8 - 1])
    return normal_map(cloud.normals[idx], pix8, w8, h8, camera.R, splat=1)


# -- the photos ---------------------------------------------------------------------
def prepare(project: Path, out: Path, reduce: int, scales) -> dict:
    """Every user-painted photo at 1/``reduce``: 13 uint8 channels (RGB,
    cloud normals, COLMAP normals, invariants) and its mask. Cached in
    ``out/photos.npz``."""
    cache = out / "photos.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as d:
            return {k: d[k] for k in d.files}
    import pycolmap
    from PIL import Image

    from cloudlabeller.io.geometry import load_cloud_npz
    from cloudlabeller.photogrammetry.pipeline import load_cameras

    t0 = time.perf_counter()
    manifest = json.loads((project / "project.json").read_text(encoding="utf-8"))
    settings = manifest["settings"]
    geo = settings.get("georeferenced") or {}
    if isinstance(geo, str):
        import ast
        geo = ast.literal_eval(geo)
    scale = float(geo.get("scale_m_per_unit", 1.0))
    status = manifest.get("image_status", {})
    rec = project / "reconstruction"
    cameras = load_cameras(rec)
    undist = pycolmap.Reconstruction(str(rec / "dense" / "sparse"))
    k_of = {im.name: tuple(undist.cameras[im.camera_id].params) for im in undist.images.values()}
    cloud = load_cloud_npz(rec / "dense.npz")
    names = sorted(n for n, s in status.items() if s == "user"
                   and (project / "labels" / "images" / f"{n}.npy").exists() and n in cameras)
    print(f"{len(names)} painted photos, {cloud.n_points} dense points, "
          f"scale {scale:.3f} m per SfM unit ({time.perf_counter() - t0:.0f} s)", flush=True)

    stacks, masks, stats = [], [], []
    stereo = rec / "dense" / "stereo"
    for k, name in enumerate(names):
        cam = cameras[name]
        h8, w8 = int(cam.height) // reduce, int(cam.width) // reduce
        with Image.open(project / "images" / name) as im:
            rgb = np.asarray(im.convert("RGB").resize((w8, h8), Image.BOX))
        mask = np.load(project / "labels" / "images" / f"{name}.npy")
        mask = mask[reduce // 2::reduce, reduce // 2::reduce][:h8, :w8].astype(np.int16)
        a = cloud_normals(cloud, cam, reduce)
        depth = read_colmap_array(stereo / "depth_maps" / f"{name}.geometric.bin")
        nmap = read_colmap_array(stereo / "normal_maps" / f"{name}.geometric.bin")
        d_m, n_cam, valid = colmap_into_photo(depth, nmap, k_of[name], cam, scale, reduce)
        b = np.where(valid[..., None], encode_normals(unit(n_cam)), NO_NORMAL).astype(np.uint8)
        inv, gsd = invariant_channels(d_m, n_cam, valid, cam.R, float(cam.K[0, 0]) / reduce,
                                      scales)
        # do the two normal maps agree where both have one?
        both = (a != NO_NORMAL).any(axis=2) & valid
        na = a[both].astype(np.float32) / 127.5 - 1
        cos = float(np.median(np.sum(unit(na) * unit(n_cam[both]), axis=1))) if both.any() else None
        stats.append({"name": name, "cloud_cover": float((a != NO_NORMAL).any(axis=2).mean()),
                      "colmap_cover": float(valid.mean()), "gsd_m": gsd,
                      "median_depth_m": float(np.median(d_m[valid])) if valid.any() else None,
                      "median_cos": cos})
        stacks.append(np.concatenate([rgb, a, b, inv], axis=2))
        masks.append(mask)
        print(f"{k + 1}/{len(names)} {name}: cover cloud {stats[-1]['cloud_cover']:.0%}, "
              f"COLMAP {stats[-1]['colmap_cover']:.0%}; GSD {gsd * 100:.1f} cm/px; "
              f"normals agree cos {cos if cos is None else round(cos, 3)} "
              f"({time.perf_counter() - t0:.0f} s)", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    arrays = {"names": np.array(names), "x": np.stack(stacks), "y": np.stack(masks),
              "stats": np.array(json.dumps(stats)),
              "n_classes": np.array(len(manifest["schema"]["classes"]))}
    np.savez(cache, **arrays)
    return arrays


# -- training -----------------------------------------------------------------------
def augment_without_flip(image, mask, window, rolls: int, log_gamma):
    """The app's mirror/slide/gamma augmentation for channels that don't
    change under a mirror (slope, relief): gamma on RGB only, nothing flipped."""
    import tensorflow as tf

    width = tf.shape(image)[1]
    shift = (tf.cast(window, tf.int32) + 1) * (width // rolls)
    strip_x = tf.concat([image, tf.reverse(image, axis=[1])], axis=1)
    strip_y = tf.concat([mask, tf.reverse(mask, axis=[1])], axis=1)
    x = tf.roll(strip_x, shift, axis=1)[:, :width]
    y = tf.roll(strip_y, shift, axis=1)[:, :width]
    gamma = tf.exp(tf.cast(log_gamma, x.dtype))
    return tf.concat([tf.pow(x[..., :3], gamma), x[..., 3:]], axis=-1), y


def tiles_of(photos: dict, keep: list[int], channels: list[int], tile: int, overlap: float,
             min_labelled: float):
    from cloudlabeller.ml.dataset_builder import Sample, TrainingSet
    from cloudlabeller.ml.tiles import cut

    samples = []
    for i in keep:
        image = np.ascontiguousarray(photos["x"][i][..., channels])
        for (r, c), img, m in cut(image, photos["y"][i], tile, overlap, 1, min_labelled):
            samples.append(Sample(f"{photos['names'][i]}#{r},{c}", img, m, 1.0))
    return TrainingSet(samples)


def run(photos: dict, variant: str, split: int, a) -> dict:
    import tensorflow as tf

    from cloudlabeller.ml import tile_dataset, trainer
    from cloudlabeller.ml.model_spec import ModelSpec
    from cloudlabeller.ml.unet import build_model_from_spec

    n = len(photos["names"])
    order = np.random.default_rng(split).permutation(n)
    val = sorted(order[:max(1, round(n * a.val_fraction))].tolist())
    train = sorted(order[len(val):].tolist())
    channels = VARIANTS[variant]
    original = tile_dataset.augment
    if variant == "invariant":
        tile_dataset.augment = augment_without_flip
    try:
        tile = a.tile // a.reduce
        train_set = tiles_of(photos, train, channels, tile, a.overlap, a.min_labelled)
        val_set = tiles_of(photos, val, channels, tile, a.overlap, a.min_labelled)
        tf.keras.utils.set_random_seed(split)
        spec = ModelSpec(tile_size=tile, shrink=1, channels=a.channels, blocks=a.blocks,
                         dropout=a.dropout, filter_size=a.filter_size)
        model = build_model_from_spec(spec, int(photos["n_classes"]), in_channels=len(channels))
        t0 = time.perf_counter()
        _, history, outcome = trainer.train(
            model, train_set, val_set, steps=a.steps, batch_size=a.batch_size,
            augment_rolls=a.rolls, patience=a.patience,
            progress=lambda f, m="": print(f"  [{variant} split {split}] {m}", flush=True))
    finally:
        tile_dataset.augment = original
    best = outcome["best_epoch"]
    return {"variant": variant, "split": split, "val_photos": [str(photos["names"][i]) for i in val],
            "n_train": len(train_set.samples), "n_val": len(val_set.samples),
            "minutes": (time.perf_counter() - t0) / 60, "outcome": outcome,
            "best": {k: v[best] for k, v in history.items()}, "history": history}


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="colmap_maps_as_features")
    p.add_argument("project")
    p.add_argument("out")
    p.add_argument("--reduce", type=int, default=8)
    p.add_argument("--scales", type=float, nargs=2, default=[1.0, 5.0])
    p.add_argument("--tile", type=int, default=2048)
    p.add_argument("--overlap", type=float, default=0.35)
    p.add_argument("--min-labelled", type=float, default=0.05)
    p.add_argument("--val-fraction", type=float, default=0.25)
    p.add_argument("--splits", type=int, default=2)
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--rolls", type=int, default=5)
    p.add_argument("--channels", type=int, default=16)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--filter-size", type=int, default=5)
    p.add_argument("--variants", default=",".join(VARIANTS))
    a = p.parse_args(argv[1:])
    out = Path(a.out)
    photos = prepare(Path(a.project), out, a.reduce, a.scales)
    stats = json.loads(str(photos["stats"]))
    cos = [s["median_cos"] for s in stats if s["median_cos"] is not None]
    print(f"{len(stats)} photos; cover: cloud {np.mean([s['cloud_cover'] for s in stats]):.0%}, "
          f"COLMAP {np.mean([s['colmap_cover'] for s in stats]):.0%}; normals agree "
          f"(median cos) {np.median(cos):.3f}", flush=True)

    results_path = out / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else []
    done = {(r["variant"], r["split"]) for r in results}
    for split in range(a.splits):
        for variant in a.variants.split(","):
            if (variant, split) in done:
                continue
            r = run(photos, variant, split, a)
            results.append(r)
            results_path.write_text(json.dumps(results, indent=1))
            print(f"RESULT {variant} split {split}: best epoch {r['outcome']['best_epoch'] + 1} "
                  f"of {r['outcome']['epochs_run']} {json.dumps(r['best'])}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
