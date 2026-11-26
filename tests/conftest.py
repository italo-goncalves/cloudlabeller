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

"""Shared test setup.

The suite runs in the desktop app's Environment as well as the dev env, so
nothing here may need PySide6, pyvista or vtk.
"""


def product_dir(proj, slot: str, sid: str = "s1"):
    """The folder a test writes a step's product into: the active
    Reconstruction's ``slot`` product of Survey ``sid``, made (with the
    Reconstruction) when missing, as running the step would."""
    survey = proj.survey(sid)
    rec = survey.ensure_active()
    if rec.products.get(slot) is None:
        rec.products[slot] = f"{slot}-1"
    d = survey.products_dir / rec.products[slot]
    d.mkdir(parents=True, exist_ok=True)
    survey.save_manifest()
    return d
