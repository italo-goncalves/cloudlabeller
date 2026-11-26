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

The desktop app's Environment installs CloudLabeller without the ``qt`` extra
(PySide6, pyvista, pyvistaqt). The suite runs there too: a test that fails
only because one of those modules is missing is reported as skipped, so
every other test still has to pass.
"""

from __future__ import annotations

import pytest

QT_EXTRA = frozenset({"PySide6", "shiboken6", "pyvista", "pyvistaqt", "vtk",
                      "vtkmodules"})


def _missing_qt_module(excinfo) -> str | None:
    exc = excinfo.value if excinfo is not None else None
    while exc is not None:
        if isinstance(exc, ModuleNotFoundError) and exc.name:
            root = exc.name.split(".")[0]
            if root in QT_EXTRA:
                return root
        exc = exc.__cause__ or exc.__context__
    return None


class missing_module:
    """Stand-in for an optional module: any attribute access raises the
    ``ModuleNotFoundError`` the hook below turns into a skip."""

    def __init__(self, name: str):
        self._name = name

    def __getattr__(self, attr):
        raise ModuleNotFoundError(f"No module named {self._name!r}", name=self._name)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.failed and call.excinfo is not None:
        name = _missing_qt_module(call.excinfo)
        if name is not None:
            report.outcome = "skipped"
            report.longrepr = (str(item.path), item.location[1] or 0,
                               f"needs the qt extra ({name} not installed)")
