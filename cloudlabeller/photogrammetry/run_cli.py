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

"""Subprocess entry point for reconstruction.

Run in its own process so COLMAP's native threads / OpenGL (GPU SIFT) context
cannot collide with the GUI's VTK OpenGL context, and so a native crash fails the
job instead of taking down the app.

Usage::

    python -m cloudlabeller.photogrammetry.run_cli IMAGE_DIR WORKSPACE \
        [--matcher sequential|exhaustive] [--gpu] [--single-camera] \
        [--camera-model SIMPLE_RADIAL] [--max-image-size 3200]

Emits ``PROGRESS <frac> <message>`` lines on stdout; writes ``cloud.npz`` and
``cameras.json`` into WORKSPACE; prints ``RESULT <n_points> <n_cameras>`` on
success. Python logs + COLMAP's native (glog) log go to stderr (the parent
captures and persists them). Exit code is non-zero on failure.
"""

from __future__ import annotations

import argparse
import logging
import sys

from cloudlabeller.logging_setup import setup_logging
from cloudlabeller.photogrammetry.options import CAMERA_MODELS, MATCHERS
from cloudlabeller.photogrammetry.pipeline import reconstruct, save_result

log = logging.getLogger("cloudlabeller.sfm")


def _parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="run_cli")
    p.add_argument("image_dir", help="project image store (SfM runs on this)")
    p.add_argument("workspace")
    p.add_argument("--ingest-from", default=None,
                   help="copy image files from this source folder into image_dir first")
    p.add_argument("--ingest-only", action="store_true",
                   help="ingest then exit (no reconstruction)")
    p.add_argument("--on-conflict", choices=("skip", "overwrite"), default="skip",
                   help="what to do when a filename already exists in the store")
    p.add_argument("--matcher", choices=MATCHERS, default="spatial")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--single-camera", action="store_true")
    p.add_argument("--camera-model", choices=CAMERA_MODELS, default="SIMPLE_RADIAL")
    p.add_argument("--max-image-size", type=int, default=3200)
    p.add_argument("--colmap-binary", default=None,
                   help="CUDA colmap.exe for GPU SIFT/matching (with --gpu)")
    return p.parse_args(argv[1:])


def main(argv: list[str]) -> int:
    args = _parse(argv)

    # Log to stderr so our lines interleave with COLMAP's native (glog) output;
    # the parent ProcessJob captures the combined stream into the run log file.
    setup_logging(logfile=None, console=True)
    from cloudlabeller.workers.resources import limit_subprocess_resources
    limit_subprocess_resources()

    def progress(frac: float, msg: str = "") -> None:
        print(f"PROGRESS {frac:.3f} {msg}", flush=True)
        if msg:
            log.info("[%3d%%] %s", int(frac * 100), msg)

    # Ingest: copy source images into the project store (always a real copy).
    if args.ingest_from:
        from cloudlabeller.core.images import ingest_images
        log.info("Ingesting images from %s -> %s (on_conflict=%s)",
                 args.ingest_from, args.image_dir, args.on_conflict)
        res = ingest_images(args.ingest_from, args.image_dir,
                            on_conflict=args.on_conflict, progress=progress)
        log.info("Ingest done: %d copied, %d overwritten, %d skipped",
                 len(res.copied), len(res.overwritten), len(res.skipped))
        # changed=1 means the store changed -> a prior reconstruction is stale.
        print(f"RESULT_INGEST copied={len(res.copied)} "
              f"overwritten={len(res.overwritten)} changed={int(res.changed)}", flush=True)

    if args.ingest_only:
        return 0

    log.info("Reconstruction starting: store=%s workspace=%s matcher=%s gpu=%s "
             "single_camera=%s camera_model=%s max_image_size=%s",
             args.image_dir, args.workspace, args.matcher, args.gpu,
             args.single_camera, args.camera_model, args.max_image_size)
    try:
        result = reconstruct(
            args.image_dir, args.workspace,
            matcher=args.matcher, progress=progress, use_gpu=args.gpu,
            single_camera=args.single_camera, camera_model=args.camera_model,
            max_image_size=args.max_image_size, colmap_binary=args.colmap_binary,
        )
    except Exception:
        log.exception("Reconstruction failed")
        return 1

    save_result(result, args.workspace)
    log.info("Reconstruction done: %s", result.summary())
    print(f"RESULT {result.cloud.n_points} {len(result.images)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
