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

"""UI-agnostic domain model. No Qt imports allowed in this package."""

from cloudlabeller.core.dataset import Camera, Dataset, ImageRecord, Mesh, PointCloud
from cloudlabeller.core.label_schema import LabelClass, LabelSchema
from cloudlabeller.core.labels import LabelStore, Modality
from cloudlabeller.core.project import Project

__all__ = [
    "Project",
    "Dataset",
    "ImageRecord",
    "Camera",
    "PointCloud",
    "Mesh",
    "LabelSchema",
    "LabelClass",
    "LabelStore",
    "Modality",
]
