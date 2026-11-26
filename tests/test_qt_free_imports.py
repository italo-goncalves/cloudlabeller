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

"""The stage CLIs and the Engine import without Qt or VTK.

The desktop app's Environment has no PySide6, pyvista or vtk (they are the
``qt`` extra). Each import runs in a fresh interpreter with those modules
blocked through ``sys.modules`` (a ``None`` entry makes ``import`` fail), so
the check holds even on the dev env, where they are installed.
"""

import subprocess
import sys
from pathlib import Path

import pytest

import cloudlabeller

PKG = Path(cloudlabeller.__file__).parent
BLOCKED = ("PySide6", "shiboken6", "pyvista", "pyvistaqt", "vtk", "vtkmodules")
CLI_MODULES = sorted(
    "cloudlabeller." + ".".join(p.relative_to(PKG).with_suffix("").parts)
    for p in PKG.rglob("*_cli.py")
)
ENGINE_MODULES = ["cloudlabeller.engine", "cloudlabeller.engine.server",
                  "cloudlabeller.engine.methods", "cloudlabeller.engine.jobs",
                  "cloudlabeller.engine.scene", "cloudlabeller.engine.geo",
                  "cloudlabeller.engine.__main__"]
# What the CLIs import inside main() (so a plain import of them misses it).
RUNTIME_MODULES = ["cloudlabeller.workers.resources", "cloudlabeller.io.geometry",
                   "cloudlabeller.io.export", "cloudlabeller.photogrammetry.mvs",
                   "cloudlabeller.photogrammetry.meshing",
                   "cloudlabeller.photogrammetry.pipeline",
                   "cloudlabeller.transfer",
                   # what the Engine's methods import when called
                   "cloudlabeller.core.selection", "cloudlabeller.core.images",
                   "cloudlabeller.photogrammetry.crs", "cloudlabeller.photogrammetry.extract",
                   "cloudlabeller.photogrammetry.cameras", "cloudlabeller.photogrammetry.georef",
                   "cloudlabeller.photogrammetry.colmap_fetch",
                   "cloudlabeller.photogrammetry.options", "cloudlabeller.ml.model_store"]

_PROBE = """
import importlib, sys
for name in {blocked!r}:
    sys.modules[name] = None
importlib.import_module({module!r})
"""


def test_every_stage_cli_is_found():
    names = {m.rsplit(".", 1)[1] for m in CLI_MODULES}
    assert {"run_cli", "mvs_cli", "mesh_cli", "georef_cli", "reproject_cli",
            "train_cli", "predict_cli", "geoid_cli", "colmap_cli"} <= names


@pytest.mark.parametrize("module", CLI_MODULES + ENGINE_MODULES + RUNTIME_MODULES)
def test_imports_without_qt(module):
    code = _PROBE.format(blocked=BLOCKED, module=module)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
