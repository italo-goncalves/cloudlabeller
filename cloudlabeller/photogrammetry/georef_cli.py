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

"""Subprocess entry point for georeferencing (align the model to EXIF GPS).

Runs COLMAP ``model_aligner`` (alignment_type=enu: local East-North-Up frame,
metres, true north) against the images' EXIF GPS, then applies the resulting
similarity transform to EVERY product in place: sparse cloud + cameras, dense
cloud (points + normals), and mesh. Label arrays are per-point/per-vertex and
follow their geometry unchanged. The COLMAP model on disk is replaced by the
aligned one (original backed up to ``sparse_prealigned/``) so later stages
(MVS, confidence cleaning) stay consistent.

Usage::

    python -m cloudlabeller.photogrammetry.georef_cli PROJECT_ROOT
        --colmap-binary PATH [--max-error 5.0]

Emits ``PROGRESS <frac> <msg>``; prints ``RESULT <json>`` on success
(scale = metres per model unit, ENU origin as mean camera GPS).
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from cloudlabeller.logging_setup import setup_logging

log = logging.getLogger("cloudlabeller.georef")


def _parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="georef_cli")
    p.add_argument("project_root")
    p.add_argument("--colmap-binary", required=True)
    p.add_argument("--max-error", type=float, default=5.0,
                   help="RANSAC threshold in metres (GPS accuracy)")
    return p.parse_args(argv[1:])


def main(argv: list[str]) -> int:
    args = _parse(argv)
    setup_logging(logfile=None, console=True)
    from cloudlabeller.workers.resources import limit_subprocess_resources
    limit_subprocess_resources()

    def progress(frac: float, msg: str = "") -> None:
        print(f"PROGRESS {frac:.3f} {msg}", flush=True)

    import pycolmap

    from cloudlabeller.io.geometry import (
        load_cloud_npz,
        load_mesh_npz,
        save_cloud_npz,
        save_mesh_npz,
    )
    from cloudlabeller.photogrammetry.extract import best_model_dir
    from cloudlabeller.photogrammetry.georef import (
        exif_gps,
        rotate_normals,
        transform_camera,
        transform_points,
        umeyama_similarity,
    )
    from cloudlabeller.photogrammetry.mvs import _colmap
    from cloudlabeller.photogrammetry.pipeline import (
        ReconstructResult,
        load_result,
        save_result,
    )

    root = Path(args.project_root)
    rec_dir = root / "reconstruction"
    backup_dir = rec_dir / "sparse_prealigned"
    if backup_dir.exists():
        log.error("this project is already georeferenced (backup exists at %s) "
                  "— aligning twice would double-transform", backup_dir)
        return 2

    # 1. Gather EXIF GPS for the solved images.
    progress(0.02, "Reading EXIF GPS from the image store…")
    result = load_result(rec_dir, root / "images")
    gps: dict[str, tuple[float, float, float]] = {}
    for record in result.images:
        lla = exif_gps(record.path)
        if lla is not None:
            gps[Path(record.path).name] = lla
    if len(gps) < 3:
        log.error("georeferencing needs GPS EXIF in at least 3 solved images "
                  "(found %d)", len(gps))
        return 2
    origin = np.mean(np.array(list(gps.values()), np.float64), axis=0)
    progress(0.08, f"{len(gps)} images have GPS")

    # 2. COLMAP model_aligner -> aligned model in a temp dir.
    model_dir = best_model_dir(rec_dir / "sparse")
    with tempfile.TemporaryDirectory(prefix="cl_georef_") as tmp:
        ref_path = Path(tmp) / "ref_images.txt"
        ref_path.write_text(
            "\n".join(f"{name} {lat} {lon} {alt}"
                      for name, (lat, lon, alt) in gps.items()) + "\n",
            encoding="utf-8")
        aligned_dir = Path(tmp) / "aligned"
        aligned_dir.mkdir()
        progress(0.1, "Aligning model to GPS (COLMAP model_aligner, ENU)…")
        _colmap(str(args.colmap_binary), [
            "model_aligner",
            "--input_path", str(model_dir),
            "--output_path", str(aligned_dir),
            "--ref_images_path", str(ref_path),
            "--ref_is_gps", "1",
            "--alignment_type", "enu",
            "--alignment_max_error", str(args.max_error),
        ])

        # 3. Similarity transform from matched camera centres (exact: the
        #    aligner applied a Sim3; Umeyama recovers it to machine precision).
        progress(0.4, "Computing the similarity transform…")
        orig = pycolmap.Reconstruction(str(model_dir))
        aligned = pycolmap.Reconstruction(str(aligned_dir))
        orig_centers, new_centers = [], []
        aligned_by_name = {im.name: im for im in aligned.images.values()}
        for im in orig.images.values():
            other = aligned_by_name.get(im.name)
            if other is not None:
                orig_centers.append(im.projection_center())
                new_centers.append(other.projection_center())
        if len(orig_centers) < 3:
            log.error("aligned model shares <3 cameras with the original")
            return 2
        s, R, t = umeyama_similarity(np.array(orig_centers), np.array(new_centers))
        fit = transform_points(np.array(orig_centers), s, R, t) - np.array(new_centers)
        rms = float(np.sqrt((fit ** 2).sum(1).mean()))
        log.info("similarity: scale=%.6f (m/unit), fit rms=%.2e m", s, rms)
        if rms > 0.01 * s * float(np.ptp(np.array(orig_centers), axis=0).max()):
            log.error("transform fit rms %.3g is not a rigid similarity — "
                      "the aligner may have failed (too few GPS inliers?)", rms)
            return 2

        # 4. Transform every product in place.
        progress(0.5, "Transforming sparse cloud + cameras…")
        result.cloud.xyz = transform_points(result.cloud.xyz, s, R, t)
        for record in result.images:
            if record.camera is not None:
                record.camera = transform_camera(record.camera, s, R, t)
        save_result(ReconstructResult(cloud=result.cloud, images=result.images),
                    rec_dir)

        dense_path = rec_dir / "dense.npz"
        if dense_path.exists():
            progress(0.6, "Transforming dense cloud…")
            dense = load_cloud_npz(dense_path)
            dense.xyz = transform_points(dense.xyz, s, R, t)
            if dense.normals is not None:
                dense.normals = rotate_normals(dense.normals, R)
            save_cloud_npz(dense, dense_path)

        mesh_path = root / "products" / "mesh.npz"
        if mesh_path.exists():
            progress(0.8, "Transforming mesh…")
            mesh = load_mesh_npz(mesh_path)
            mesh.vertices = transform_points(mesh.vertices, s, R, t)
            save_mesh_npz(mesh, mesh_path)

        # 5. Swap in the aligned COLMAP model (backup the original) so future
        #    MVS runs / confidence cleaning match the new coordinates.
        progress(0.9, "Replacing the COLMAP model (original backed up)…")
        backup_dir.mkdir(parents=True)
        shutil.move(str(model_dir), str(backup_dir / model_dir.name))
        shutil.move(str(aligned_dir), str(model_dir))

    # 6. Old visibility cache entries can never match again — free the space.
    pmap = rec_dir / "pmap"
    if pmap.exists():
        for f in pmap.glob("*.npz"):
            f.unlink(missing_ok=True)

    print("RESULT " + json.dumps({
        "scale_m_per_unit": s,
        "origin_lla": [float(v) for v in origin],
        "n_gps": len(gps),
        "fit_rms_m": rms,
    }), flush=True)
    progress(1.0, "Georeferencing complete")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
