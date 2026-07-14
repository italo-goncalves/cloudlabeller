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

"""Common interface for 3D selection tools.

A tool wires itself to the PyVista plotter on :meth:`activate`, and reports the
indices of selected cloud points through the ``on_select`` callback. The viewer
turns those indices into a paint command. Keeping the contract this small lets
new tools (region-grow, plane-cut, …) drop in without touching the viewer.
"""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np

OnSelect = Callable[[np.ndarray], None]


class SelectionTool(Protocol):
    name: str

    def activate(self, plotter, on_select: OnSelect) -> None:
        """Attach interactor observers / picking widgets to the plotter."""
        ...

    def deactivate(self, plotter) -> None:
        """Detach observers and remove any widgets."""
        ...
