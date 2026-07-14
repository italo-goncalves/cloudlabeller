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

"""Subprocess entry point for dense-cloud meshing (crash isolation + live logs).

Usage::

    python -m cloudlabeller.photogrammetry.mesh_cli DENSE_NPZ MESH_OUT_NPZ \
        [--method poisson|delaunay] [--depth N] [--trim 10] \
        [--point-weight 1.0] [--workspace DENSE_DIR] [--colmap-binary PATH]

Reads the dense cloud, meshes it via colmap.exe (Screened Poisson by default;
``--depth`` omitted = matched to the cloud's density), writes ``mesh.npz`` and
prints ``RESULT_MESH <n_vertices> <n_faces>``. ``--method delaunay`` needs
``--workspace`` (the dense MVS folder with ``fused.ply`` + ``fused.ply.vis``).
"""

from __future__ import annotations

import argparse
import logging
import sys

from cloudlabeller.io.geometry import load_cloud_npz, save_mesh_npz
from cloudlabeller.logging_setup import setup_logging
from cloudlabeller.photogrammetry.meshing import create_mesh, create_mesh_delaunay

log = logging.getLogger("cloudlabeller.mesh")


def _parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="mesh_cli")
    p.add_argument("dense_npz")
    p.add_argument("mesh_out")
    p.add_argument("--method", choices=("poisson", "delaunay"), default="poisson")
    p.add_argument("--depth", type=int, default=None,
                   help="Poisson octree depth (omit to match the cloud density)")
    p.add_argument("--trim", type=float, default=10.0)
    p.add_argument("--point-weight", type=float, default=1.0)
    p.add_argument("--workspace", default=None,
                   help="dense MVS workspace (required for --method delaunay)")
    p.add_argument("--colmap-binary", default=None)
    return p.parse_args(argv[1:])


def main(argv: list[str]) -> int:
    args = _parse(argv)
    setup_logging(logfile=None, console=True)
    from cloudlabeller.workers.resources import limit_subprocess_resources
    limit_subprocess_resources()

    def progress(frac: float, msg: str = "") -> None:
        print(f"PROGRESS {frac:.3f} {msg}", flush=True)
        if msg:
            log.info("[%3d%%] %s", int(frac * 100), msg)

    try:
        cloud = load_cloud_npz(args.dense_npz)
        if args.method == "delaunay":
            mesh = create_mesh_delaunay(args.workspace, cloud,
                                        colmap_binary=args.colmap_binary,
                                        progress=progress)
        else:
            mesh = create_mesh(cloud, depth=args.depth, trim=args.trim,
                               point_weight=args.point_weight,
                               colmap_binary=args.colmap_binary, progress=progress)
    except Exception:
        log.exception("Meshing failed")
        return 1

    save_mesh_npz(mesh, args.mesh_out)
    log.info("Mesh written: %s", args.mesh_out)
    print(f"RESULT_MESH {len(mesh.vertices)} {len(mesh.faces)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
