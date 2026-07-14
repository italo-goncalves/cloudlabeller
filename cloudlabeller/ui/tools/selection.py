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

"""Concrete selection tools built on VTK pickers.

Each is a PLACEHOLDER body with the intended VTK mechanism documented, since the
exact wiring depends on the PolyData/actor created in ``Viewer3D._refresh_cloud``.
Implement against that actor's point ids.
"""

from __future__ import annotations

from cloudlabeller.ui.tools.base import OnSelect


class SinglePickTool:
    """Click a single point. Uses ``vtkPointPicker`` / ``enable_point_picking``."""

    name = "single"

    def activate(self, plotter, on_select: OnSelect) -> None:
        # PLACEHOLDER:
        #   plotter.enable_point_picking(
        #       callback=lambda pt, idx: on_select(np.array([idx])),
        #       use_picker=True, show_message=False)
        raise NotImplementedError

    def deactivate(self, plotter) -> None:
        plotter.disable_picking()


class LassoTool:
    """Freehand lasso -> frustum selection of enclosed points.

    Capture a screen-space polygon, build a frustum with ``vtkAreaPicker`` /
    ``vtkExtractSelectedFrustum``, and return the contained point ids. Optionally
    depth-gate to front-facing points only for precision.
    """

    name = "lasso"

    def activate(self, plotter, on_select: OnSelect) -> None:
        raise NotImplementedError  # PLACEHOLDER

    def deactivate(self, plotter) -> None:
        plotter.disable_picking()


class BoxSelectTool:
    """Rubber-band rectangle -> frustum selection (``vtkRubberBandPick``)."""

    name = "box"

    def activate(self, plotter, on_select: OnSelect) -> None:
        raise NotImplementedError  # PLACEHOLDER

    def deactivate(self, plotter) -> None:
        plotter.disable_picking()


class BrushTool:
    """Spherical brush: paint points within a world-space radius of the cursor's
    surface hit. Radius adjustable; supports drag-painting and snap-to-surface."""

    name = "brush"

    def __init__(self, radius: float = 0.05) -> None:
        self.radius = radius

    def activate(self, plotter, on_select: OnSelect) -> None:
        # PLACEHOLDER: on each drag event, pick the surface point under the
        # cursor, query a KD-tree of cloud points within ``self.radius``, and
        # emit those ids via on_select.
        raise NotImplementedError

    def deactivate(self, plotter) -> None:
        plotter.disable_picking()
