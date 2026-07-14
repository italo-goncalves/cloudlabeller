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

"""Triangulated mesh from the dense cloud (COLMAP Screened Poisson / Delaunay).

The mesh is a *product of the dense cloud*: it is (re)built whenever the dense
cloud changes (after Dense MVS, after importing a dense cloud) and on demand via
Photogrammetry → Create Mesh. Vertex colours come from the mesher's colour
interpolation when available, else are sampled from the nearest dense point —
the viewer interpolates them across each triangle in RGB space.

Two methods:
  * **Poisson** (default) — watertight, smooths noise; detail is capped by the
    octree ``depth`` (cell edge = bounding-cube side / 2**depth). By default
    the depth is chosen so the cell size matches the cloud's point spacing —
    the mesh carries roughly the same detail as the dense cloud.
  * **Delaunay** — interpolates the fused points exactly (max detail, no
    smoothing, noisier). Needs the MVS *workspace* (``fused.ply`` + its
    ``.vis`` visibility file), so it is unavailable for imported clouds.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np

from cloudlabeller.core.dataset import Mesh, PointCloud
from cloudlabeller.io.geometry import load_mesh, save_cloud_ply
from cloudlabeller.photogrammetry.mvs import _colmap, find_colmap_binary

ProgressFn = Callable[[float, str], None]

# Poisson octree depth bounds: below 8 the mesh is a blob; above 13 (COLMAP's
# own default) time/memory explode with little visible gain over the data.
MIN_DEPTH, MAX_DEPTH = 8, 13


def estimate_point_spacing(cloud: PointCloud, sample: int = 200_000,
                           seed: int = 0) -> float:
    """Median nearest-neighbour distance of the cloud (its "resolution").

    Measured on a random subset for speed; the subset's NN distances are
    scaled by ``sqrt(sample/n)`` — for points spread over a 2D surface,
    thinning to a fraction f stretches neighbour spacing by 1/sqrt(f).
    """
    from scipy.spatial import cKDTree

    n = cloud.n_points
    if n < 2:
        return 0.0
    if n <= sample:
        pts = cloud.xyz
        correction = 1.0
    else:
        idx = np.random.default_rng(seed).choice(n, size=sample, replace=False)
        pts = cloud.xyz[idx]
        correction = math.sqrt(sample / n)
    dist, _ = cKDTree(pts).query(pts, k=2)
    return float(np.median(dist[:, 1])) * correction


def suggest_poisson_depth(cloud: PointCloud) -> int:
    """Octree depth whose cell size ≈ the cloud's point spacing.

    Poisson reconstructs on a 2**depth grid over the bounding cube, so this
    is the depth at which the mesh resolves roughly the same detail as the
    dense cloud — more depth adds cost but no information; less discards it.
    """
    spacing = estimate_point_spacing(cloud)
    if spacing <= 0:
        return 11
    side = float(np.max(cloud.xyz.max(axis=0) - cloud.xyz.min(axis=0)))
    if side <= 0:
        return MIN_DEPTH
    return int(np.clip(round(math.log2(side / spacing)), MIN_DEPTH, MAX_DEPTH))


def estimate_mesh_cost(n_points: int, depth: int, suggested_depth: int,
                       method: str = "poisson") -> dict:
    """VERY rough {minutes, ram_gb, vertices} for a meshing run.

    Calibrated to order-of-magnitude on survey-scale clouds (~100M points on
    a desktop CPU); real times vary with hardware and surface complexity.
    Poisson cost doubles per depth step; vertices quarter per step below the
    density-matched depth (surface cells shrink 4x per step). Delaunay
    tetrahedralises every fused point regardless of a depth setting.
    """
    m = n_points / 1e6
    if method == "delaunay":
        return {"minutes": max(1.0, m * 0.5), "ram_gb": 1.0 + m * 0.35,
                "vertices": float(n_points)}
    scale = 2.0 ** (depth - 11)
    return {"minutes": max(0.5, m * 0.1 * scale),
            "ram_gb": 1.0 + m * 0.08 * scale,
            "vertices": float(n_points) * min(1.0, 4.0 ** (depth - suggested_depth))}


def delaunay_available(workspace: str | Path | None) -> bool:
    """Whether the MVS workspace still has what ``delaunay_mesher`` needs:
    the fused cloud plus its per-point visibility (``fused.ply.vis``)."""
    if workspace is None:
        return False
    ws = Path(workspace)
    return (ws / "fused.ply").exists() and (ws / "fused.ply.vis").exists()


def _colourise(mesh: Mesh, cloud: PointCloud, progress: ProgressFn) -> Mesh:
    """Fill missing vertex colours from the nearest dense point."""
    if mesh.vertex_colors is None and cloud.rgb is not None:
        progress(0.92, "Sampling vertex colours from the dense cloud…")
        from scipy.spatial import cKDTree

        _, idx = cKDTree(cloud.xyz).query(mesh.vertices, k=1)
        mesh.vertex_colors = cloud.rgb[idx]
    return mesh


def create_mesh(
    cloud: PointCloud,
    depth: int | None = None,
    trim: float = 10.0,
    point_weight: float = 1.0,
    colmap_binary: str | None = None,
    progress: ProgressFn = lambda f, m="": None,
) -> Mesh:
    """Screened-Poisson mesh of ``cloud``; returns a colourised :class:`Mesh`.

    ``depth`` bounds the octree resolution; None (default) picks the depth
    whose cells match the cloud's point spacing (:func:`suggest_poisson_depth`)
    — a mesh roughly as detailed as the dense cloud. ``trim`` removes
    low-support bubbles Poisson grows over unobserved regions;
    ``point_weight`` > 1 makes the surface hug the samples more tightly
    (sharper, less smoothing).
    """
    if cloud.normals is None:
        raise RuntimeError(
            "Poisson meshing needs per-point normals. The dense cloud produced by "
            "Run Dense (MVS) has them; imported clouds may not — re-run MVS or "
            "import a cloud that includes normals.")
    exe = find_colmap_binary(colmap_binary)
    if exe is None:
        raise RuntimeError("COLMAP executable not found — dense meshing uses "
                           "colmap.exe (poisson_mesher).")

    progress(0.02, "Estimating cloud density…")
    suggested = suggest_poisson_depth(cloud)
    matched = depth is None
    if matched:
        depth = suggested
    est = estimate_mesh_cost(cloud.n_points, depth, suggested)
    progress(0.04, f"Poisson depth {depth}"
                   + (" (matches the cloud density)" if matched else
                      f" (density-matched depth would be {suggested})")
                   + f" — rough estimate: ~{est['minutes']:.0f} min, "
                     f"~{est['ram_gb']:.0f} GB RAM, "
                     f"~{est['vertices'] / 1e6:.1f} M vertices")

    with tempfile.TemporaryDirectory(prefix="cl_mesh_") as tmp:
        cloud_ply = Path(tmp) / "cloud.ply"
        mesh_ply = Path(tmp) / "mesh.ply"
        progress(0.05, f"Writing {cloud.n_points:,} points for meshing…")
        save_cloud_ply(cloud, cloud_ply)

        progress(0.15, f"Poisson meshing (depth {depth})…")
        from cloudlabeller.workers.resources import worker_threads
        args = ["poisson_mesher",
                "--input_path", str(cloud_ply),
                "--output_path", str(mesh_ply),
                "--PoissonMeshing.depth", str(depth),
                "--PoissonMeshing.trim", str(trim),
                "--PoissonMeshing.num_threads", str(worker_threads())]
        if point_weight != 1.0:
            args += ["--PoissonMeshing.point_weight", str(point_weight)]
        _colmap(exe, args)

        progress(0.85, "Loading mesh…")
        mesh = load_mesh(mesh_ply)

    mesh = _colourise(mesh, cloud, progress)
    progress(1.0, f"Mesh complete: {len(mesh.vertices):,} vertices, "
                  f"{len(mesh.faces):,} faces")
    return mesh


def create_mesh_delaunay(
    workspace: str | Path,
    cloud: PointCloud,
    colmap_binary: str | None = None,
    progress: ProgressFn = lambda f, m="": None,
) -> Mesh:
    """Delaunay mesh from the MVS workspace (max detail, no smoothing).

    ``workspace`` is the dense MVS folder (``reconstruction/dense``) holding
    ``fused.ply`` + ``fused.ply.vis``; ``cloud`` is only used to colourise the
    vertices. Delaunay interpolates the fused points exactly, so the mesh is
    as detailed — and as noisy — as the dense cloud itself.
    """
    ws = Path(workspace)
    if not delaunay_available(ws):
        raise RuntimeError(
            "Delaunay meshing needs the dense MVS workspace (fused.ply and "
            "fused.ply.vis) — it is unavailable for imported dense clouds. "
            "Run Dense (MVS) first, or use Poisson meshing.")
    exe = find_colmap_binary(colmap_binary)
    if exe is None:
        raise RuntimeError("COLMAP executable not found — dense meshing uses "
                           "colmap.exe (delaunay_mesher).")

    est = estimate_mesh_cost(cloud.n_points, 0, 0, method="delaunay")
    progress(0.05, f"Delaunay meshing {cloud.n_points:,} fused points "
                   f"(rough estimate: ~{est['minutes']:.0f} min, "
                   f"~{est['ram_gb']:.0f} GB RAM)…")
    from cloudlabeller.workers.resources import worker_threads

    with tempfile.TemporaryDirectory(prefix="cl_mesh_") as tmp:
        mesh_ply = Path(tmp) / "mesh.ply"
        _colmap(exe, ["delaunay_mesher",
                      "--input_path", str(ws),
                      "--input_type", "dense",
                      "--output_path", str(mesh_ply),
                      "--DelaunayMeshing.num_threads", str(worker_threads())])
        progress(0.85, "Loading mesh…")
        mesh = load_mesh(mesh_ply)

    mesh = _colourise(mesh, cloud, progress)
    progress(1.0, f"Mesh complete: {len(mesh.vertices):,} vertices, "
                  f"{len(mesh.faces):,} faces")
    return mesh
