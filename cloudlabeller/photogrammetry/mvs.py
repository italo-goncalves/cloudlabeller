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

"""Multi-View Stereo: dense cloud (+ optional mesh) from a solved SfM model.

COLMAP's patch-match stereo is **CUDA-only**. Since the PyPI ``pycolmap`` wheel is
CPU-only (``has_cuda == False``), dense MVS is driven by the **COLMAP CUDA
executable** (``colmap.exe``) when available — undistort -> patch-match stereo ->
stereo fusion -> (optional) Poisson meshing. If no CUDA path exists at all, a
clear error points the user to Photogrammetry -> Download COLMAP.
"""

from __future__ import annotations

import collections
import re
import shutil
import subprocess
from pathlib import Path
from typing import Callable

from cloudlabeller.core.dataset import Mesh, PointCloud
from cloudlabeller.io.geometry import load_cloud, load_mesh

ProgressFn = Callable[[float, str], None]

# COLMAP logs "Processing view 12 / 57", "Fusing image 3 / 57", etc.
_COUNTER_RE = re.compile(r"(\d+)\s*/\s*(\d+)")

# Default location the app downloads the CUDA COLMAP binary to.
_DEFAULT_COLMAP = Path.home() / ".cloudlabeller" / "colmap" / "bin" / "colmap.exe"

# Records the ``max_image_size`` + quality the dense workspace was built at.
UNDISTORT_MARKER = "undistort_size.txt"
# Records the sparse-model fingerprint the images were undistorted from;
# written only after undistortion COMPLETES (a resumed run may skip it).
UNDISTORT_DONE = "undistort_done.txt"

# Draft quality: fewer patch-match source images and a lighter solve. Roughly
# 3-4x faster stereo; noisier cloud — for a first look, not the final product.
DRAFT_SRC_IMAGES = 10


def sparse_model_fingerprint(sparse: Path) -> str:
    """Cheap identity of a COLMAP sparse model (file sizes + mtimes): a re-run
    of SfM produces different files, so undistorted images derived from the
    old model must not be reused."""
    parts = []
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        f = Path(sparse) / name
        parts.append(f"{name}:{f.stat().st_size}:{f.stat().st_mtime_ns}"
                     if f.exists() else f"{name}:absent")
    return "|".join(parts)


def prepare_dense_workspace(dense: Path, max_image_size: int,
                            quality: str = "standard",
                            progress: ProgressFn = lambda f, m="": None) -> bool:
    """Create ``dense/``, clearing stale outputs when the settings changed.

    Patch-match stereo resumes any depth maps already in ``dense/stereo`` —
    valuable after a crash, but maps computed at a different ``max_image_size``
    (or quality preset) no longer match the current run and fail stereo/fusion
    with a size-check error (or silently mix qualities). A marker file records
    the settings of the previous run; on mismatch — or when it is missing,
    e.g. a workspace from an older app version — every previous output is
    wiped. The marker is written before undistortion so an interrupted run
    still resumes with the same settings.

    Returns True when the existing workspace was KEPT (same settings — the
    caller may reuse completed stages), False for a fresh/cleared one.
    """
    marker = dense / UNDISTORT_MARKER
    current = f"{max(0, int(max_image_size))}|{quality}"   # 0/negative = full res
    has_outputs = any((dense / d).exists() for d in ("images", "sparse", "stereo"))
    kept = False
    if has_outputs:
        previous = marker.read_text(encoding="utf-8").strip() if marker.exists() else None
        if previous != current:
            progress(0.02, "Detail/quality changed — clearing previous dense outputs…")
            shutil.rmtree(dense)
        else:
            kept = True
    dense.mkdir(parents=True, exist_ok=True)
    marker.write_text(current, encoding="utf-8")
    return kept


def set_patch_match_sources(dense: Path, n_sources: int) -> None:
    """Cap the source images per depth map in ``stereo/patch-match.cfg``.

    The undistorter writes ``__auto__, 20`` — every depth map is estimated
    against up to 20 covisible neighbours. Well-overlapped surveys are nearly
    indistinguishable at 10, for about half the stereo time.
    """
    cfg = dense / "stereo" / "patch-match.cfg"
    if not cfg.exists():
        return
    lines = [f"__auto__, {n_sources}" if line.strip().startswith("__auto__") else line
             for line in cfg.read_text(encoding="utf-8").splitlines()]
    cfg.write_text("\n".join(lines) + "\n", encoding="utf-8")


def find_colmap_binary(override: str | None = None) -> str | None:
    """Locate a COLMAP executable: explicit override, portable bundle (next to
    the interpreter), app install, then PATH."""
    import sys

    portable = Path(sys.prefix) / "colmap" / "bin" / "colmap.exe"
    candidates = [override, portable, _DEFAULT_COLMAP, shutil.which("colmap")]
    for c in candidates:
        if c and Path(c).exists():
            return str(c)
    # Downloaded bundle whose inner layout is not bin/colmap.exe (a release
    # zip restructure): search the app-managed directory before giving up.
    managed = _DEFAULT_COLMAP.parent.parent
    if managed.is_dir():
        hit = next(managed.rglob("colmap.exe"), None)
        if hit:
            return str(hit)
    return None


def find_sparse_model(sparse_dir: str | Path) -> Path:
    """Locate the COLMAP sparse model directory (``sparse/0`` etc.)."""
    root = Path(sparse_dir)
    if (root / "points3D.bin").exists() or (root / "points3D.txt").exists():
        return root
    candidates = [
        p for p in (root.iterdir() if root.is_dir() else [])
        if (p / "points3D.bin").exists() or (p / "points3D.txt").exists()
    ]
    if not candidates:
        raise RuntimeError(f"No sparse model found under {root}; run SfM first.")
    return max(candidates, key=lambda p: (p / "points3D.bin").stat().st_size
               if (p / "points3D.bin").exists() else 0)


def run_mvs(
    workspace: str | Path,
    images_dir: str | Path,
    max_image_size: int = 2000,
    build_mesh: bool = False,
    colmap_binary: str | None = None,
    quality: str = "standard",         # "standard" | "draft" (~3-4x faster stereo)
    progress: ProgressFn = lambda f, m="": None,
) -> tuple[PointCloud, Mesh | None]:
    """Run dense MVS; return (dense cloud, mesh-or-None)."""
    ws = Path(workspace)
    images = Path(images_dir)
    sparse = find_sparse_model(ws / "sparse")
    dense = ws / "dense"
    kept = prepare_dense_workspace(dense, max_image_size, quality, progress)

    exe = find_colmap_binary(colmap_binary)
    if exe is not None:
        return _run_mvs_cli(exe, images, sparse, dense, max_image_size, build_mesh,
                            quality, kept, progress)

    # No CUDA executable found — try in-process pycolmap (also CUDA-gated).
    import pycolmap
    if not pycolmap.has_cuda:
        raise RuntimeError(
            "Dense MVS needs CUDA. No COLMAP CUDA executable was found and the "
            "installed pycolmap is CPU-only. Install a CUDA build of COLMAP "
            "(colmap.exe) and set its path in the app config.")
    return _run_mvs_pycolmap(pycolmap, images, sparse, dense, max_image_size,
                             build_mesh, progress)


# -- COLMAP CUDA executable backend ---------------------------------------
def _colmap(exe: str, args: list[str], progress: ProgressFn | None = None,
            stage: tuple[str, float, float] | None = None) -> None:
    """Run a colmap subcommand, streaming its output line-by-line.

    Each line is printed immediately (so the parent ProcessJob relays it to the
    Log panel in real time). If ``stage`` = (label, frac_start, frac_end) is
    given, COLMAP's "X / Y" counters drive ``progress`` within that band. Raises
    with the output tail on a non-zero exit.
    """
    proc = subprocess.Popen([exe, *args], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail: collections.deque[str] = collections.deque(maxlen=15)
    while True:                                  # readline avoids iterator read-ahead
        raw = proc.stdout.readline()
        if not raw:
            break
        line = raw.rstrip()
        print(line, flush=True)                  # -> mvs_cli stdout -> Log panel
        tail.append(line)
        if progress and stage:
            m = _COUNTER_RE.search(line)
            if m and int(m.group(2)) > 0:
                label, a, b = stage
                cur, tot = int(m.group(1)), int(m.group(2))
                progress(a + (b - a) * min(cur / tot, 1.0), f"{label} {cur}/{tot}")
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(
            f"colmap {args[0]} failed (exit {proc.returncode}):\n" + "\n".join(tail))


def _run_mvs_cli(exe, images, sparse, dense, max_image_size, build_mesh,
                 quality, kept, progress):
    """Dense MVS through the COLMAP executable: undistort (skipped on a
    same-settings resume) -> patch-match stereo -> fusion [-> Poisson mesh].
    Returns ``(cloud, mesh_or_None)``."""
    # Cap every CPU-bound stage at cores-1 so the OS/UI stays responsive
    # (flags verified against the bundled colmap.exe --help).
    from cloudlabeller.workers.resources import worker_threads
    n = str(worker_threads())

    # Undistortion re-writes every image (~20% of the run) yet is a pure
    # function of (sparse model, max_image_size): on a same-settings resume
    # with the SAME sparse model it can be skipped wholesale. The done-marker
    # is written only after the stage completes, so an interrupted
    # undistortion is never mistaken for a finished one.
    done = dense / UNDISTORT_DONE
    fingerprint = sparse_model_fingerprint(sparse)
    reusable = (kept and done.exists()
                and done.read_text(encoding="utf-8") == fingerprint)
    if reusable:
        progress(0.25, "Reusing undistorted images from the previous run")
    else:
        done.unlink(missing_ok=True)
        progress(0.05, "Undistorting images…")
        undistort = ["image_undistorter", "--image_path", str(images),
                     "--input_path", str(sparse), "--output_path", str(dense),
                     "--output_type", "COLMAP", "--num_threads", n]
        if max_image_size and max_image_size > 0:
            undistort += ["--max_image_size", str(max_image_size)]
        _colmap(exe, undistort, progress, ("Undistorting", 0.05, 0.25))
        done.write_text(fingerprint, encoding="utf-8")

    stereo = ["patch_match_stereo", "--workspace_path", str(dense),
              "--workspace_format", "COLMAP",
              "--PatchMatchStereo.num_threads", n]
    fusion_input = "geometric"
    if quality == "draft":
        set_patch_match_sources(dense, DRAFT_SRC_IMAGES)
        # Skip the second (geometric-consistency) stereo pass and lighten the
        # per-pixel solve. Without geometric maps, fusion reads photometric.
        stereo += ["--PatchMatchStereo.geom_consistency", "0",
                   "--PatchMatchStereo.window_radius", "3",
                   "--PatchMatchStereo.num_iterations", "3"]
        fusion_input = "photometric"
        progress(0.25, f"Draft quality: {DRAFT_SRC_IMAGES} source images, "
                       "single stereo pass")
    _colmap(exe, stereo, progress, ("Patch-match stereo (GPU)", 0.25, 0.78))

    fused = dense / "fused.ply"
    _colmap(exe, ["stereo_fusion", "--workspace_path", str(dense),
                  "--workspace_format", "COLMAP", "--input_type", fusion_input,
                  "--output_path", str(fused),
                  "--StereoFusion.num_threads", n],
            progress, ("Stereo fusion", 0.78, 0.95))
    cloud = load_cloud(fused)

    mesh = None
    if build_mesh:
        progress(0.96, "Poisson meshing…")
        meshed = dense / "meshed.ply"
        _colmap(exe, ["poisson_mesher", "--input_path", str(fused),
                      "--output_path", str(meshed),
                      "--PoissonMeshing.num_threads", n])
        mesh = load_mesh(meshed)

    progress(1.0, f"Dense complete: {cloud.n_points} points")
    return cloud, mesh


# -- in-process pycolmap backend (only if it ever ships with CUDA) --------
def _run_mvs_pycolmap(pycolmap, images, sparse, dense, max_image_size, build_mesh, progress):
    """Same MVS chain through pycolmap's in-process bindings (used only when
    no CUDA executable is available AND pycolmap ships CUDA support)."""
    progress(0.05, "Undistorting images…")
    pycolmap.undistort_images(dense, sparse, images)
    progress(0.30, "Patch-match stereo (GPU)…")
    opts = pycolmap.PatchMatchOptions()
    if max_image_size and max_image_size > 0:
        opts.max_image_size = max_image_size
    from cloudlabeller.workers.resources import worker_threads
    opts.num_threads = worker_threads()
    pycolmap.patch_match_stereo(dense, options=opts)
    progress(0.80, "Stereo fusion…")
    fused = dense / "fused.ply"
    pycolmap.stereo_fusion(fused, dense)
    cloud = load_cloud(fused)
    mesh = None
    if build_mesh:
        progress(0.95, "Poisson meshing…")
        meshed = dense / "meshed.ply"
        pycolmap.poisson_meshing(fused, meshed)
        mesh = load_mesh(meshed)
    progress(1.0, f"Dense complete: {cloud.n_points} points")
    return cloud, mesh
