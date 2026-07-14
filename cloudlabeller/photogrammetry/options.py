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

"""Reconstruction options shared by the GUI dialog and the subprocess CLI."""

from __future__ import annotations

from dataclasses import dataclass

MATCHERS = ("spatial", "sequential", "exhaustive")
CAMERA_MODELS = ("SIMPLE_RADIAL", "RADIAL", "OPENCV", "FULL_OPENCV", "PINHOLE", "SIMPLE_PINHOLE")


@dataclass
class SfmOptions:
    matcher: str = "spatial"   # uses EXIF GPS priors; ideal for drone surveys
    use_gpu: bool = False
    single_camera: bool = True
    camera_model: str = "SIMPLE_RADIAL"
    max_image_size: int = 3200          # 0 = full resolution (no downscale)

    def to_cli_args(self) -> list[str]:
        """Flags for ``run_cli`` (image_dir/workspace are prepended separately)."""
        args = [
            "--matcher", self.matcher,
            "--camera-model", self.camera_model,
            "--max-image-size", str(self.max_image_size),
        ]
        if self.use_gpu:
            args.append("--gpu")
        if self.single_camera:
            args.append("--single-camera")
        return args
