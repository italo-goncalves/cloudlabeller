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


"""How long training runs and which weights it keeps (``ml.trainer``), and
the Tile size the photos call for. No TensorFlow needed."""

import numpy as np
import pytest

from cloudlabeller.ml.model_spec import ModelSpec, default_tile_size
from cloudlabeller.ml.tile_dataset import n_batches
from cloudlabeller.ml.trainer import BestEpoch, epochs_for_steps


def test_steps_become_whole_epochs_rounded_up():
    per_epoch = n_batches(957, 5, 2)                 # FIDA: 957 Tiles, 10 windows each
    assert per_epoch == 4785
    assert epochs_for_steps(50000, per_epoch) == 11  # 10.45 -> 11
    assert epochs_for_steps(4785, per_epoch) == 1
    assert epochs_for_steps(4786, per_epoch) == 2
    assert epochs_for_steps(1, per_epoch) == 1
    assert epochs_for_steps(10, 0) == 10             # a guard, never a division by zero


def test_the_best_epoch_is_the_lowest_loss_and_patience_stops_after_it():
    """The FIDA run: val_loss was lowest at epoch 45 and rose after; with a
    patience of 20 it stops at epoch 65 and keeps epoch 45."""
    losses = list(np.linspace(0.3, 0.0681, 45)) + list(np.linspace(0.07, 0.11, 55))
    tracker = BestEpoch(patience=20)
    stopped = None
    for epoch, loss in enumerate(losses):
        _, stop = tracker.update(epoch, loss)
        if stop:
            stopped = epoch
            break
    assert tracker.epoch == 44 and tracker.best == pytest.approx(0.0681)
    assert stopped == 64


def test_without_patience_training_runs_on_and_a_nan_is_never_best():
    tracker = BestEpoch(patience=0)
    assert tracker.update(0, 1.0) == (True, False)
    assert tracker.update(1, float("nan")) == (False, False)
    assert tracker.update(2, None) == (False, False)
    assert tracker.update(3, 2.0) == (False, False)
    assert tracker.update(4, 0.5) == (True, False)
    assert tracker.epoch == 4


@pytest.mark.parametrize("w, h, tile", [
    (5472, 3648, 2048),     # DJI: a quarter of 5472 is 1368
    (4000, 3000, 1024),     # 1000
    (4096, 3072, 2048),     # 1024 is not greater than 1024
    (100, 80, 32),          # never below 32
])
def test_the_default_tile_is_the_first_power_of_two_above_a_quarter_photo(w, h, tile):
    assert default_tile_size(w, h) == tile == default_tile_size(h, w)
    assert ModelSpec(tile_size=tile).validate() is None


def test_a_project_without_model_settings_takes_the_tile_its_photos_call_for(tmp_path):
    from cloudlabeller.core.project import Project

    proj = Project.create(tmp_path / "p.clproj")
    assert proj.model_spec().tile_size == ModelSpec().tile_size     # no photo size yet
    proj.surveys["s1"].settings["source_resolution"] = [5472, 3648]
    proj.add_survey().settings["source_resolution"] = [4000, 3000]   # the larger one counts
    assert proj.default_tile_size() == 2048 and proj.model_spec().tile_size == 2048
    proj.set_model_spec(ModelSpec(tile_size=512))
    assert proj.model_spec().tile_size == 512                       # saved settings win
