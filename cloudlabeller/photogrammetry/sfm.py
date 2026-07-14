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

"""Structure-from-Motion wrapper around pycolmap (verified against pycolmap 4.0.4).

A single SfM run yields BOTH a sparse point cloud and the solved camera poses,
so no dense/MVS step (and no CUDA) is required to get a first cloud + cameras.

Heavy stages call ``progress(fraction, message)`` so a worker thread can relay
status to the UI without this module importing Qt.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from cloudlabeller.core.images import IMAGE_EXTENSIONS, list_image_files

ProgressFn = Callable[[float, str], None]

__all__ = ["run_sfm", "SfMResult", "IMAGE_EXTENSIONS", "list_image_files"]


def _noop(_f: float, _m: str = "") -> None:  # default progress sink
    pass


class SfMResult:
    """Thin holder for COLMAP outputs."""

    def __init__(self, database: Path, sparse_dir: Path, reconstruction) -> None:
        self.database = database
        self.sparse_dir = sparse_dir
        self.reconstruction = reconstruction  # pycolmap.Reconstruction


# Matcher name -> colmap.exe subcommand (the GPU extraction/matching path).
_EXE_MATCHERS = {"spatial": "spatial_matcher", "sequential": "sequential_matcher",
                 "exhaustive": "exhaustive_matcher"}


def _extract_and_match_exe(exe: str, database: Path, image_dir: Path,
                           image_names: list[str], workspace: Path,
                           matcher: str, camera_model: str, single_camera: bool,
                           max_image_size: int, n_threads: int,
                           progress: ProgressFn) -> None:
    """GPU feature extraction + matching via the CUDA ``colmap.exe``.

    Writes into the same ``database.db`` the pycolmap mapper reads — the
    database format is shared — so only these two (dominant) stages move to
    the GPU; incremental mapping stays in pycolmap. Raises on any failure;
    the caller falls back to the CPU (pycolmap) path.
    """
    from cloudlabeller.photogrammetry.mvs import _colmap

    subcommand = _EXE_MATCHERS.get(matcher)
    if subcommand is None:
        raise RuntimeError(f"matcher {matcher!r} has no colmap.exe equivalent")

    # Failsafe shared with the CPU path: only real image files reach COLMAP.
    image_list = workspace / "image_list.txt"
    image_list.write_text("\n".join(image_names), encoding="utf-8")

    # Option-group names verified against the bundled colmap.exe 4.1 (the 4.x
    # CLI renamed SiftExtraction/SiftMatching to FeatureExtraction/-Matching).
    extract = ["feature_extractor",
               "--database_path", str(database),
               "--image_path", str(image_dir),
               "--image_list_path", str(image_list),
               "--ImageReader.camera_model", camera_model,
               "--FeatureExtraction.use_gpu", "1",
               "--FeatureExtraction.num_threads", str(n_threads)]
    if single_camera:
        extract += ["--ImageReader.single_camera", "1"]
    if max_image_size and max_image_size > 0:
        extract += ["--FeatureExtraction.max_image_size", str(max_image_size)]
    progress(0.05, f"Extracting features from {len(image_names)} images (GPU)…")
    _colmap(exe, extract, progress, ("Extracting", 0.05, 0.35))

    progress(0.35, "Matching features (GPU)…")
    _colmap(exe, [subcommand, "--database_path", str(database),
                  "--FeatureMatching.use_gpu", "1"],
            progress, ("Matching", 0.35, 0.70))


def run_sfm(
    image_dir: str | Path,
    workspace: str | Path,
    matcher: str = "spatial",   # "spatial" | "sequential" | "exhaustive" | "vocab_tree"
    progress: ProgressFn = _noop,
    use_gpu: bool = False,
    single_camera: bool = True,
    camera_model: str = "SIMPLE_RADIAL",
    max_image_size: int = 3200,
    colmap_binary: str | None = None,
) -> SfMResult:
    """Feature extraction → matching → incremental mapping.

    Returns an :class:`SfMResult` wrapping the largest reconstruction (most
    points3D). Convert it with :func:`cloudlabeller.photogrammetry.pipeline.reconstruct`.

    Options:
      * ``use_gpu`` — run SIFT extraction + matching on the GPU through the
        CUDA ``colmap.exe`` (``colmap_binary``, auto-located when None); the
        mapper stays in pycolmap on the shared database. Falls back to CPU
        pycolmap when the executable is missing or fails. Defaults to
        **False**; the caller must be isolated in a subprocess — GPU SIFT
        creates its own GL/CUDA context, which crashes next to a live
        in-process VTK context.
      * ``single_camera`` — share one intrinsic across all images (right for a
        single physical camera, e.g. one drone).
      * ``camera_model`` — COLMAP camera model name (distortion handling).
      * ``max_image_size`` — downscale long edge to this many px before feature
        extraction (0 = full resolution).
    """
    import pycolmap  # lazy import of the optional heavy dependency

    image_dir = Path(image_dir)
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    database = workspace / "database.db"
    sparse_dir = workspace / "sparse"
    sparse_dir.mkdir(exist_ok=True)
    if database.exists():
        database.unlink()  # COLMAP refuses to overwrite an existing database

    # Leave at least one core for the OS/UI — heavy stages otherwise saturate
    # every core and make the whole machine sluggish.
    from cloudlabeller.workers.resources import worker_threads
    n_threads = worker_threads()

    # Failsafe: only feed COLMAP real image files, so sidecar files in the
    # folder (DJI PPK .nav/.obs/.bin/.MRK, etc.) are never read as bitmaps.
    image_names = list_image_files(image_dir)
    if not image_names:
        raise RuntimeError(
            f"No image files found in {image_dir} "
            f"(looked for: {', '.join(sorted(IMAGE_EXTENSIONS))})"
        )

    # GPU path: extraction + matching through the CUDA colmap.exe (the two
    # stages that dominate SfM wall time); on any failure fall back to CPU.
    extracted = False
    if use_gpu:
        from cloudlabeller.photogrammetry.mvs import find_colmap_binary

        exe = find_colmap_binary(colmap_binary)
        if exe is not None:
            try:
                _extract_and_match_exe(exe, database, image_dir, image_names,
                                       workspace, matcher, camera_model,
                                       single_camera, max_image_size,
                                       n_threads, progress)
                extracted = True
            except Exception as exc:
                progress(0.05, f"GPU extraction failed ({exc}) — "
                               "falling back to CPU SIFT…")
                database.unlink(missing_ok=True)   # no half-written database
                use_gpu = False                    # pycolmap GPU would fail too
        else:
            progress(0.05, "No CUDA COLMAP executable found — using CPU SIFT…")
            use_gpu = False

    if not extracted:
        reader_options = pycolmap.ImageReaderOptions()
        reader_options.camera_model = camera_model
        camera_mode = (pycolmap.CameraMode.SINGLE if single_camera
                       else pycolmap.CameraMode.AUTO)

        extraction_options = pycolmap.FeatureExtractionOptions()
        extraction_options.use_gpu = use_gpu
        if max_image_size and max_image_size > 0:
            extraction_options.max_image_size = max_image_size
        extraction_options.num_threads = n_threads
        # CPU SIFT runs one extractor per thread, each preallocating buffers
        # sized to the (possibly full-res) image — many cores × large images
        # can exhaust RAM and crash. Cap threads above the default size to
        # bound peak memory.
        large = max_image_size == 0 or max_image_size > 3200
        if not use_gpu and large:
            extraction_options.num_threads = min(4, n_threads)
        matching_options = pycolmap.FeatureMatchingOptions()
        matching_options.use_gpu = use_gpu
        matching_options.num_threads = n_threads

        note = (" (capped at 4 threads for large images)"
                if (not use_gpu and large) else "")
        progress(0.05, f"Extracting features from {len(image_names)} images{note}…")
        pycolmap.extract_features(database, image_dir, image_names=image_names,
                                  camera_mode=camera_mode,
                                  reader_options=reader_options,
                                  extraction_options=extraction_options)

        progress(0.35, "Matching features…")
        if matcher == "exhaustive":
            pycolmap.match_exhaustive(database, matching_options=matching_options)
        elif matcher == "vocab_tree":
            pycolmap.match_vocabtree(database, matching_options=matching_options)
        elif matcher == "spatial":
            pycolmap.match_spatial(database, matching_options=matching_options)
        else:
            pycolmap.match_sequential(database, matching_options=matching_options)

    progress(0.70, "Incremental mapping… (global bundle adjustment is silent "
                   "while solving — large models can take many minutes)")
    mapping_options = pycolmap.IncrementalPipelineOptions()
    mapping_options.num_threads = n_threads
    # Global BA dominates wall time on large image sets: by default it reruns
    # after every ~10% model growth, each time up to 5 refinement rounds of 50
    # iterations with NO convergence tolerance — 20+ silent minutes at ~1000
    # images. Relax to a large-project schedule: rerun every ~20% growth,
    # fewer rounds, and a function tolerance so converged solves exit early.
    mapping_options.ba_global_frames_ratio = 1.2
    mapping_options.ba_global_points_ratio = 1.2
    mapping_options.ba_global_max_refinements = 3
    mapping_options.ba_global_function_tolerance = 1e-5
    maps = pycolmap.incremental_mapping(database, image_dir, sparse_dir,
                                        options=mapping_options)
    recs = list(maps.values()) if isinstance(maps, dict) else list(maps)
    if not recs:
        raise RuntimeError(
            "SfM produced no reconstruction — too few/poor-overlap images?"
        )
    reconstruction = max(recs, key=lambda r: r.num_points3D())

    progress(1.0, f"SfM complete: {reconstruction.num_points3D()} points, "
                  f"{reconstruction.num_reg_images()} cameras")
    return SfMResult(database=database, sparse_dir=sparse_dir, reconstruction=reconstruction)
