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

"""Subprocess entry point for dense MVS.

Runs out-of-process (like SfM) for crash isolation + GPU/OpenGL isolation. Writes
``dense.npz`` (and ``mesh.npz`` if requested) into the workspace and prints
``RESULT_DENSE <n_points>``.

Usage::

    python -m cloudlabeller.photogrammetry.mvs_cli WORKSPACE IMAGES_DIR \
        [--max-image-size 2000] [--mesh]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from cloudlabeller.io.geometry import save_cloud_npz, save_mesh_npz
from cloudlabeller.logging_setup import setup_logging
from cloudlabeller.photogrammetry.mvs import run_mvs

log = logging.getLogger("cloudlabeller.mvs")

DENSE_FILE = "dense.npz"
MESH_FILE = "mesh.npz"


def _parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="mvs_cli")
    p.add_argument("workspace", help="reconstruction dir (contains sparse/)")
    p.add_argument("images_dir", help="project image store")
    p.add_argument("--max-image-size", type=int, default=2000)
    p.add_argument("--quality", choices=("standard", "draft"), default="standard",
                   help="draft = ~3-4x faster stereo, noisier cloud")
    p.add_argument("--mesh", action="store_true", help="also build a Poisson mesh")
    p.add_argument("--mesh-out", default=None, help="where to write mesh.npz")
    p.add_argument("--colmap-binary", default=None, help="path to colmap.exe (CUDA)")
    return p.parse_args(argv[1:])


def main(argv: list[str]) -> int:
    args = _parse(argv)
    setup_logging(logfile=None, console=True)
    # Below-normal priority is inherited by the colmap.exe children.
    from cloudlabeller.workers.resources import limit_subprocess_resources
    limit_subprocess_resources()

    def progress(frac: float, msg: str = "") -> None:
        print(f"PROGRESS {frac:.3f} {msg}", flush=True)
        if msg:
            log.info("[%3d%%] %s", int(frac * 100), msg)

    log.info("Dense MVS starting: workspace=%s images=%s max_image_size=%s mesh=%s",
             args.workspace, args.images_dir, args.max_image_size, args.mesh)
    try:
        cloud, mesh = run_mvs(args.workspace, args.images_dir,
                              max_image_size=args.max_image_size,
                              build_mesh=args.mesh, colmap_binary=args.colmap_binary,
                              quality=args.quality, progress=progress)
    except Exception:
        log.exception("Dense MVS failed")
        return 1

    save_cloud_npz(cloud, Path(args.workspace) / DENSE_FILE)
    if mesh is not None:
        mesh_out = args.mesh_out or str(Path(args.workspace) / MESH_FILE)
        save_mesh_npz(mesh, mesh_out)
    log.info("Dense MVS done: %d points%s", cloud.n_points,
             "" if mesh is None else f", mesh {len(mesh.faces)} faces")
    print(f"RESULT_DENSE {cloud.n_points}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
