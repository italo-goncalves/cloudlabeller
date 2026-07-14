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

"""Create-Mesh options dialog: method, detail and cost estimates.

The detail choices map to Poisson octree depths; the default ("match cloud
density") picks the depth whose cells are the size of the dense cloud's point
spacing, so the mesh carries roughly the same detail as the cloud. Every
choice shows a rough time / RAM / vertex estimate, with a warning when the
estimate approaches this machine's RAM.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QRadioButton,
    QVBoxLayout,
)

from cloudlabeller.photogrammetry.meshing import (
    delaunay_available,
    estimate_mesh_cost,
    suggest_poisson_depth,
)

# (label template, depth). None = match the cloud density (auto).
DETAIL_CHOICES: list[tuple[str, int | None]] = [
    ("Match cloud density — depth {d} (recommended)", None),
    ("Maximum (depth 13)", 13),
    ("High (depth 12)", 12),
    ("Standard (depth 11)", 11),
    ("Draft (depth 10)", 10),
]


def total_ram_gb() -> float | None:
    """This machine's physical RAM in GB (None if undeterminable)."""
    try:
        import ctypes

        class _MemStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_uint64),
                        ("ullAvailPhys", ctypes.c_uint64),
                        ("ullTotalPageFile", ctypes.c_uint64),
                        ("ullAvailPageFile", ctypes.c_uint64),
                        ("ullTotalVirtual", ctypes.c_uint64),
                        ("ullAvailVirtual", ctypes.c_uint64),
                        ("ullAvailExtendedVirtual", ctypes.c_uint64)]

        stat = _MemStatus()
        stat.dwLength = ctypes.sizeof(stat)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return stat.ullTotalPhys / 2 ** 30
    except Exception:
        return None


def format_estimate(est: dict, ram_gb: float | None) -> str:
    """One-line rough-cost text for an :func:`estimate_mesh_cost` result;
    appends a red warning when the RAM estimate nears the machine's total."""
    minutes = est["minutes"]
    time_txt = f"~{minutes:.0f} min" if minutes < 90 else f"~{minutes / 60:.1f} h"
    verts = est["vertices"]
    verts_txt = f"{verts / 1e6:.0f} M" if verts >= 1e6 else f"{verts / 1e3:.0f} k"
    text = (f"Rough estimate: {time_txt}, ~{est['ram_gb']:.0f} GB RAM, "
            f"~{verts_txt} vertices")
    if ram_gb is not None and est["ram_gb"] > 0.8 * ram_gb:
        text += (f"<br><span style='color:#d33'>⚠ may exceed this machine's "
                 f"{ram_gb:.0f} GB RAM — consider a lower detail.</span>")
    return text


class CreateMeshDialog(QDialog):
    """Options for (re)building the mesh from the dense cloud."""

    def __init__(self, cloud, workspace=None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Create Mesh")
        self.setMinimumWidth(520)
        self._n_points = cloud.n_points
        self._suggested = suggest_poisson_depth(cloud)
        self._ram = total_ram_gb()

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            f"Mesh the dense cloud ({self._n_points:,} points)."))

        self.rb_poisson = QRadioButton("Poisson — watertight, smooths noise (recommended)")
        self.rb_delaunay = QRadioButton("Delaunay — maximum detail, keeps the cloud's noise")
        self.rb_poisson.setChecked(True)
        group = QButtonGroup(self)
        group.addButton(self.rb_poisson)
        group.addButton(self.rb_delaunay)
        layout.addWidget(self.rb_poisson)
        layout.addWidget(self.rb_delaunay)
        if not delaunay_available(workspace):
            self.rb_delaunay.setEnabled(False)
            self.rb_delaunay.setToolTip(
                "Needs the dense MVS workspace (fused.ply + fused.ply.vis) — "
                "unavailable for imported dense clouds.")

        form = QFormLayout()
        self.cmb_detail = QComboBox()
        for label, depth in DETAIL_CHOICES:
            self.cmb_detail.addItem(label.format(d=self._suggested), depth)
        self.cmb_detail.setToolTip(
            "Poisson octree depth: each step doubles the mesh resolution and "
            "roughly doubles time and memory.")
        form.addRow("Detail:", self.cmb_detail)

        self.spin_trim = QDoubleSpinBox()
        self.spin_trim.setRange(0.0, 20.0)
        self.spin_trim.setValue(10.0)
        self.spin_trim.setToolTip("Removes low-support bubbles Poisson grows "
                                  "over unobserved regions (0 = keep everything).")
        form.addRow("Trim:", self.spin_trim)

        self.spin_weight = QDoubleSpinBox()
        self.spin_weight.setRange(0.5, 20.0)
        self.spin_weight.setValue(1.0)
        self.spin_weight.setSingleStep(0.5)
        self.spin_weight.setToolTip("Screening weight: higher hugs the points "
                                    "more tightly (sharper, less smoothing).")
        form.addRow("Point weight:", self.spin_weight)
        layout.addLayout(form)

        self.lbl_estimate = QLabel()
        self.lbl_estimate.setWordWrap(True)
        layout.addWidget(self.lbl_estimate)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.rb_poisson.toggled.connect(self._refresh)
        self.cmb_detail.currentIndexChanged.connect(self._refresh)
        self._refresh()

    # -- state ---------------------------------------------------------------
    def _refresh(self) -> None:
        poisson = self.rb_poisson.isChecked()
        for w in (self.cmb_detail, self.spin_trim, self.spin_weight):
            w.setEnabled(poisson)
        opts = self.options()
        est = estimate_mesh_cost(self._n_points,
                                 opts["depth"] if opts["depth"] is not None
                                 else self._suggested,
                                 self._suggested, method=opts["method"])
        self.lbl_estimate.setText(format_estimate(est, self._ram))

    def options(self) -> dict:
        """{'method', 'depth' (None = match density), 'trim', 'point_weight'}."""
        return {
            "method": "delaunay" if self.rb_delaunay.isChecked() else "poisson",
            "depth": self.cmb_detail.currentData(),
            "trim": float(self.spin_trim.value()),
            "point_weight": float(self.spin_weight.value()),
        }
