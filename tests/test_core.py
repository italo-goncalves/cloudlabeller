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

"""Tests for the pure-Python core (no Qt, no heavy deps).

These exercise the label store, schema, project round-trip and mesh<->cloud
transfer that are fully implemented in the scaffold.
"""

import numpy as np
import pytest

from cloudlabeller.core.dataset import Mesh, PointCloud
from cloudlabeller.core.label_schema import UNLABELLED_ID, LabelSchema
from cloudlabeller.core.labels import LabelStore
from cloudlabeller.core.project import Project
from cloudlabeller.transfer.mesh_cloud import cloud_to_mesh


class TestResampleLabelMask:
    """Upscaling a prediction to the photo resolution must smooth class
    boundaries without ever averaging ids into a class that was not there.
    Regression for the nearest-neighbour prediction upscale (2026-07-21)."""

    def _diag(self, h, w, a=0, b=2):
        """h×w mask split by a diagonal: class ``a`` above, ``b`` below."""
        m = np.full((h, w), a, np.int32)
        for r in range(h):
            m[r, r * w // h:] = b
        return m

    def test_upscale_introduces_no_spurious_class(self):
        from cloudlabeller.core.raster import resample_label_mask

        small = self._diag(30, 40, a=0, b=2)          # 0 and 2 adjacent
        big = resample_label_mask(small, (400, 300))  # 10x enlarge
        assert set(np.unique(big).tolist()) == {0, 2}   # never a spurious 1

    def test_upscale_is_smoother_than_nearest(self):
        import cv2

        from cloudlabeller.core.raster import resample_label_mask

        small = self._diag(30, 40)
        size = (400, 300)
        smooth = resample_label_mask(small, size)
        nn = cv2.resize(small.astype(np.float32), size,
                        interpolation=cv2.INTER_NEAREST).astype(np.int32)

        def edge_var(m):                              # jaggedness of the 0|2 seam
            cols = [np.argmax(row == 2) for row in m]
            return float(np.var(np.diff(cols)))

        assert edge_var(smooth) < edge_var(nn) / 2    # markedly less staircased

    def test_dense_mask_stays_dense(self):
        from cloudlabeller.core.raster import resample_label_mask

        small = self._diag(20, 20, a=1, b=3)
        big = resample_label_mask(small, (200, 200))
        assert (big >= 0).all()                       # no -1 introduced (dense)

    def test_all_unlabelled_and_size_convention(self):
        from cloudlabeller.core.raster import resample_label_mask

        out = resample_label_mask(np.full((10, 12), -1, np.int32), (24, 20))
        assert out.shape == (20, 24) and np.all(out == -1)


def test_prediction_mask_upscales_smoothly(tmp_path):
    from cloudlabeller.core.dataset import Camera, ImageRecord

    proj = Project.create(tmp_path / "p.clproj")
    small = np.zeros((30, 40), np.int32)              # 0 above, 2 below diagonal
    for r in range(30):
        small[r, r * 40 // 30:] = 2
    proj.predictions_dir.mkdir(parents=True, exist_ok=True)
    np.save(proj.predictions_dir / "a.jpg.npy", small)
    cam = Camera(K=np.eye(3), R=np.eye(3), t=np.zeros(3), width=400, height=300)
    proj.dataset.images = [ImageRecord(0, tmp_path / "a.jpg", camera=cam)]
    proj.labels.mark_ml_labeled("a.jpg")

    mask = proj.prediction_mask("a.jpg")
    assert mask.shape == (300, 400)
    assert set(np.unique(mask).tolist()) == {0, 2}    # no averaged-in class 1


def test_schema_starts_empty_and_numbers_from_zero():
    schema = LabelSchema()
    assert schema.classes == []                       # no reserved unlabelled class
    assert UNLABELLED_ID == -1
    rock = schema.add("rock", "#ff0000")
    veg = schema.add("vegetation", "#00ff00")
    assert rock.id == 0 and veg.id == 1               # contiguous from 0
    assert schema.by_id(0).name == "rock"
    assert schema.lookup_table() == {0: (255, 0, 0), 1: (0, 255, 0)}


def test_schema_remove_renumbers():
    schema = LabelSchema()
    for n in ("a", "b", "c", "d"):
        schema.add(n)
    schema.remove(1)                                  # delete "b"
    assert [c.id for c in schema.classes] == [0, 1, 2]
    assert [c.name for c in schema.classes] == ["a", "c", "d"]


def test_schema_rename_and_color():
    schema = LabelSchema()
    schema.add("a")
    schema.add("b")
    schema.rename(0, "stone")
    schema.set_color(1, "#0000ff")
    assert schema.by_id(0).name == "stone"
    assert schema.lookup_table()[1] == (0, 0, 255)


def test_label_store_default_is_unlabelled():
    store = LabelStore()
    store.init_cloud(4)
    assert list(store.cloud_labels) == [-1, -1, -1, -1]


def test_label_store_paint_and_undo():
    store = LabelStore()
    store.init_cloud(10)
    store.paint_cloud(np.array([0, 1, 2]), class_id=3)
    assert list(store.cloud_labels[:3]) == [3, 3, 3]
    assert store.can_undo()
    store.undo()
    assert list(store.cloud_labels) == [-1] * 10      # back to unlabelled
    store.redo()
    assert store.cloud_labels[1] == 3


def test_label_status_provenance():
    store = LabelStore()
    assert store.status_of("a.jpg") == "none"
    store.mark_auto_labeled("a.jpg")
    assert store.status_of("a.jpg") == "auto"
    store.mark_user_labeled("a.jpg")
    assert store.status_of("a.jpg") == "user"
    store.mark_auto_labeled("a.jpg")                  # user is sticky
    assert store.status_of("a.jpg") == "user"


def test_dense_cloud_labels_persist(tmp_path):
    proj = Project.create(tmp_path / "p.clproj")
    proj.labels.cloud_labels = np.array([0, 1, -1], np.int32)
    proj.labels.dense_cloud_labels = np.array([1, 1, 0, -1], np.int32)
    proj.save()

    reopened = Project.open(proj.root)
    assert list(reopened.labels.cloud_labels) == [0, 1, -1]
    assert list(reopened.labels.dense_cloud_labels) == [1, 1, 0, -1]


def test_save_manifest_persists_status_without_full_save(tmp_path):
    proj = Project.create(tmp_path / "p.clproj")
    proj.labels.mark_user_labeled("a.jpg")
    proj.save_manifest()                              # cheap, manifest only
    assert Project.open(proj.root).labels.status_of("a.jpg") == "user"


def test_auto_mask_source_prefers_labelled_dense(tmp_path):
    from cloudlabeller.core.dataset import PointCloud

    proj = Project.create(tmp_path / "p.clproj")
    proj.dataset.cloud = PointCloud(xyz=np.zeros((4, 3), np.float32))
    proj.dataset.dense_cloud = PointCloud(xyz=np.zeros((6, 3), np.float32))
    assert proj.auto_mask_source() is None            # nothing labelled

    proj.labels.cloud_labels = np.array([0, -1, -1, -1], np.int32)
    cloud, labels = proj.auto_mask_source()
    assert cloud is proj.dataset.cloud                # sparse fallback

    proj.labels.dense_cloud_labels = np.array([1, -1, -1, -1, -1, -1], np.int32)
    cloud, labels = proj.auto_mask_source()
    assert cloud is proj.dataset.dense_cloud          # labelled dense wins

    proj.labels.dense_cloud_labels = np.full(3, -1, np.int32)   # stale length
    cloud, labels = proj.auto_mask_source()
    assert cloud is proj.dataset.cloud                # mismatched dense ignored


def test_stale_auto_masks_purged_on_open(tmp_path):
    """Masks saved for auto-status images are derived data: skipped and removed
    on open (older versions materialised them, exhausting RAM)."""
    proj = Project.create(tmp_path / "p.clproj")
    proj.labels.image_masks["user.JPG"] = np.zeros((4, 4), np.int32)
    proj.labels.mark_user_labeled("user.JPG")
    proj.labels.image_masks["auto.JPG"] = np.ones((4, 4), np.int32)
    proj.labels.mark_auto_labeled("auto.JPG")
    proj.save()
    assert (proj.labels_dir / "images" / "auto.JPG.npy").exists()

    reopened = Project.open(proj.root)
    assert "user.JPG" in reopened.labels.image_masks           # kept
    assert "auto.JPG" not in reopened.labels.image_masks       # derived: not loaded
    assert not (proj.labels_dir / "images" / "auto.JPG.npy").exists()   # purged


def test_project_persists_image_status(tmp_path):
    proj = Project.create(tmp_path / "p.clproj")
    proj.labels.mark_user_labeled("img1.JPG")
    proj.labels.mark_auto_labeled("img2.JPG")
    proj.save()

    reopened = Project.open(proj.root)
    assert reopened.labels.status_of("img1.JPG") == "user"
    assert reopened.labels.status_of("img2.JPG") == "auto"
    assert reopened.labels.status_of("img3.JPG") == "none"


def test_set_cloud_labels_undo_redo_roundtrip():
    """Regression: redo of a bulk transfer must restore the transferred labels
    (a dummy scalar class_id used to zero everything on redo)."""
    store = LabelStore()
    store.init_cloud(5)
    transferred = np.array([2, 1, 0, 1, 2], np.int32)
    store.set_cloud_labels(transferred)
    store.undo()
    assert list(store.cloud_labels) == [-1] * 5
    store.redo()
    assert list(store.cloud_labels) == [2, 1, 0, 1, 2]


def test_undo_redo_report_what_changed():
    store = LabelStore()
    store.init_cloud(4)
    store.init_image("a.jpg", 2, 2)
    store.paint_cloud(np.array([0]), 1)
    store.paint_image("a.jpg", np.array([0, 1]), 1)

    from cloudlabeller.core.labels import Modality
    assert store.undo() == (Modality.IMAGE, "a.jpg")   # last edit was the image
    assert store.undo() == (Modality.CLOUD, None)
    assert store.undo() is None                        # stack empty
    assert store.redo() == (Modality.CLOUD, None)


def test_merge_cloud_labels_preserves_lasso_work():
    """Regression: Images→Cloud must merge, not replace — points without image
    votes keep their lasso-assigned labels."""
    store = LabelStore()
    store.init_dense_cloud(6)
    store.paint_dense_cloud(np.array([0, 1]), 2)          # lasso: pts 0,1 -> class 2

    transferred = np.array([-1, -1, 3, 3, -1, -1], np.int32)   # votes on pts 2,3 only
    n = store.merge_cloud_labels(transferred, dense=True)
    assert n == 2
    assert list(store.dense_cloud_labels) == [2, 2, 3, 3, -1, -1]   # lasso kept

    # the merge is one undoable step: undo removes the transfer, keeps the lasso
    store.undo()
    assert list(store.dense_cloud_labels) == [2, 2, -1, -1, -1, -1]

    # sparse variant, including allocation from None
    n = store.merge_cloud_labels(np.array([5, -1, -1], np.int32))
    assert n == 1 and list(store.cloud_labels) == [5, -1, -1]


def test_remap_after_delete_includes_mesh():
    store = LabelStore()
    store.mesh_labels = np.array([0, 1, 2, -1], np.int32)
    store.remap_after_delete(1)
    assert list(store.mesh_labels) == [0, -1, 1, -1]


def test_mesh_labels_persist(tmp_path):
    proj = Project.create(tmp_path / "p.clproj")
    proj.labels.mesh_labels = np.array([1, -1, 0], np.int32)
    proj.save()
    reopened = Project.open(proj.root)
    assert list(reopened.labels.mesh_labels) == [1, -1, 0]


def test_remap_after_delete_includes_dense():
    """Regression: dense-cloud labels must be remapped on class deletion too."""
    store = LabelStore()
    store.init_cloud(3)
    store.cloud_labels[:] = [0, 1, 2]
    store.dense_cloud_labels = np.array([0, 1, 2, 2], np.int32)
    store.remap_after_delete(1)
    assert list(store.cloud_labels) == [0, -1, 1]
    assert list(store.dense_cloud_labels) == [0, -1, 1, 1]


def test_label_store_remap_after_delete():
    store = LabelStore()
    store.init_cloud(6)
    store.cloud_labels[:] = np.array([-1, 0, 1, 2, 1, 3])
    store.remap_after_delete(1)                       # delete class 1
    # class 1 -> unlabelled; classes above shift down (2->1, 3->2)
    assert list(store.cloud_labels) == [-1, 0, -1, 1, -1, 2]


def test_project_delete_label_remaps_data(tmp_path):
    proj = Project.create(tmp_path / "p.clproj")
    for n in ("a", "b", "c"):
        proj.add_label(n)
    proj.labels.init_cloud(3)
    proj.labels.cloud_labels[:] = np.array([0, 1, 2])
    proj.delete_label(1)
    assert [c.name for c in proj.schema.classes] == ["a", "c"]
    assert list(proj.labels.cloud_labels) == [0, -1, 1]


def test_label_store_notifies_listeners():
    store = LabelStore()
    store.init_cloud(5)
    seen = []
    store.subscribe(lambda modality, key: seen.append((modality.value, key)))
    store.paint_cloud(np.array([0]), 1)
    assert seen == [("cloud", None)]


def test_cloud_to_mesh_nearest_vertex():
    cloud = PointCloud(xyz=np.array([[0, 0, 0], [10, 0, 0]], dtype=float))
    labels = np.array([1, 2], dtype=np.int32)
    mesh = Mesh(
        vertices=np.array([[0.1, 0, 0], [9.9, 0, 0]], dtype=float),
        faces=np.array([[0, 1, 0]], dtype=np.int32),
    )
    out = cloud_to_mesh(cloud, labels, mesh)
    assert list(out) == [1, 2]


def test_project_loads_reconstruction(tmp_path):
    """Opening a project must restore the sparse cloud + cameras saved by SfM."""
    from pathlib import Path

    from cloudlabeller.core.dataset import Camera, ImageRecord, PointCloud
    from cloudlabeller.photogrammetry.pipeline import ReconstructResult, save_result

    proj = Project.create(tmp_path / "p.clproj")
    cloud = PointCloud(xyz=np.zeros((10, 3), np.float32), rgb=np.zeros((10, 3), np.uint8))
    cam = Camera(K=np.eye(3), R=np.eye(3), t=np.zeros(3), width=4, height=3)
    images = [ImageRecord(image_id=1, path=Path("/imgs/a.jpg"), camera=cam)]
    save_result(ReconstructResult(cloud, images), proj.reconstruction_dir)

    reopened = Project.open(proj.root)
    assert reopened.dataset.cloud is not None
    assert reopened.dataset.cloud.n_points == 10
    assert len(reopened.dataset.images) == 1
    assert reopened.dataset.images[0].camera is not None
    # label array is initialised to match the loaded cloud
    assert reopened.labels.cloud_labels is not None
    assert len(reopened.labels.cloud_labels) == 10


def test_ingest_images_copies_only_images(tmp_path):
    from cloudlabeller.core.images import ingest_images, list_image_files

    src = tmp_path / "src"
    src.mkdir()
    for n in ["a.JPG", "b.png", "x.nav", "y.MRK", "notes.txt"]:
        (src / n).write_bytes(b"data-" + n.encode())
    store = tmp_path / "store"

    r1 = ingest_images(src, store)
    assert set(r1.copied) == {"a.JPG", "b.png"}            # images only
    assert (store / "a.JPG").read_bytes() == b"data-a.JPG"  # real byte copy
    assert set(list_image_files(store)) == {"a.JPG", "b.png"}

    r2 = ingest_images(src, store)                          # idempotent
    assert r2.copied == [] and set(r2.skipped) == {"a.JPG", "b.png"}


def test_ingest_conflict_skip_vs_overwrite(tmp_path):
    from cloudlabeller.core.images import find_conflicts, ingest_images

    src = tmp_path / "src"
    src.mkdir()
    (src / "a.JPG").write_bytes(b"NEW")
    store = tmp_path / "store"
    store.mkdir()
    (store / "a.JPG").write_bytes(b"OLD")

    assert find_conflicts(src, store) == ["a.JPG"]

    r = ingest_images(src, store, on_conflict="skip")
    assert r.skipped == ["a.JPG"] and not r.changed
    assert (store / "a.JPG").read_bytes() == b"OLD"          # kept existing

    r2 = ingest_images(src, store, on_conflict="overwrite")
    assert r2.overwritten == ["a.JPG"] and r2.changed
    assert (store / "a.JPG").read_bytes() == b"NEW"          # replaced


def test_ingest_dedupes_duplicate_basenames_within_source(tmp_path):
    from cloudlabeller.core.images import ingest_images

    src = tmp_path / "src"
    (src / "sub1").mkdir(parents=True)
    (src / "sub2").mkdir()
    (src / "sub1" / "dup.JPG").write_bytes(b"first")
    (src / "sub2" / "dup.JPG").write_bytes(b"second")
    store = tmp_path / "store"

    r = ingest_images(src, store)
    assert r.copied == ["dup.JPG"]                           # first one wins
    assert r.skipped == ["dup.JPG"]                          # duplicate skipped
    assert (store / "dup.JPG").read_bytes() == b"first"


def test_reconstruction_status(tmp_path):
    from cloudlabeller.core.dataset import PointCloud

    proj = Project.create(tmp_path / "p.clproj")
    assert proj.reconstruction_status() == "none"            # no cloud yet
    proj.dataset.cloud = PointCloud(xyz=np.zeros((2, 3), np.float32))
    assert proj.reconstruction_status() == "current"
    proj.mark_reconstruction_outdated()
    assert proj.reconstruction_status() == "outdated"
    proj.mark_reconstruction_current()
    assert proj.reconstruction_status() == "current"


def test_project_dataset_unions_store_and_cameras(tmp_path):
    from cloudlabeller.core.dataset import Camera, ImageRecord, PointCloud
    from cloudlabeller.core.images import ingest_images
    from cloudlabeller.photogrammetry.pipeline import ReconstructResult, save_result

    proj = Project.create(tmp_path / "p.clproj")
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.JPG").write_bytes(b"x")
    (src / "b.JPG").write_bytes(b"x")
    ingest_images(src, proj.images_dir)                    # both in the store

    cloud = PointCloud(xyz=np.zeros((3, 3), np.float32))
    cam = Camera(K=np.eye(3), R=np.eye(3), t=np.zeros(3), width=4, height=3)
    save_result(ReconstructResult(cloud, [ImageRecord(1, proj.images_dir / "a.JPG", cam)]),
                proj.reconstruction_dir)                   # only a.JPG was solved

    reopened = Project.open(proj.root)
    by_name = {im.path.name: im for im in reopened.dataset.images}
    assert set(by_name) == {"a.JPG", "b.JPG"}              # union of store + solved
    assert by_name["a.JPG"].camera is not None             # solved
    assert by_name["b.JPG"].camera is None                 # present but unsolved
    assert by_name["a.JPG"].path.exists()                  # resolves to store copy


def test_geometry_npz_roundtrip(tmp_path):
    from cloudlabeller.core.dataset import Mesh, PointCloud
    from cloudlabeller.io.geometry import (
        load_cloud_npz, load_mesh_npz, save_cloud_npz, save_mesh_npz,
    )

    cloud = PointCloud(xyz=np.random.rand(5, 3).astype(np.float32),
                       rgb=np.random.randint(0, 255, (5, 3)).astype(np.uint8))
    cp = tmp_path / "d.npz"
    save_cloud_npz(cloud, cp)
    back = load_cloud_npz(cp)
    assert back.n_points == 5 and np.array_equal(back.rgb, cloud.rgb)

    mesh = Mesh(vertices=np.random.rand(4, 3).astype(np.float32),
                faces=np.array([[0, 1, 2], [1, 2, 3]], np.int32))
    mp = tmp_path / "m.npz"
    save_mesh_npz(mesh, mp)
    bm = load_mesh_npz(mp)
    assert bm.vertices.shape == (4, 3) and bm.faces.shape == (2, 3)
    assert bm.vertex_colors is None


def test_project_loads_dense_and_mesh(tmp_path):
    from cloudlabeller.core.dataset import Mesh, PointCloud
    from cloudlabeller.io.geometry import save_cloud_npz, save_mesh_npz

    proj = Project.create(tmp_path / "p.clproj")
    save_cloud_npz(PointCloud(xyz=np.zeros((7, 3), np.float32)), proj.dense_cloud_path)
    save_mesh_npz(Mesh(vertices=np.zeros((3, 3), np.float32),
                       faces=np.array([[0, 1, 2]], np.int32)), proj.mesh_path)

    reopened = Project.open(proj.root)
    assert reopened.dataset.dense_cloud is not None
    assert reopened.dataset.dense_cloud.n_points == 7
    assert reopened.dataset.has("dense") and reopened.dataset.has("mesh")


def _vis_scene():
    from cloudlabeller.core.dataset import Camera

    rng = np.random.default_rng(0)
    xyz = np.column_stack([rng.uniform(-2, 2, 800), rng.uniform(-1.5, 1.5, 800),
                           rng.uniform(4, 9, 800)]).astype(np.float32)
    cloud = PointCloud(xyz=xyz)
    K = np.array([[800., 0, 320], [0, 800., 240], [0, 0, 1]])
    cam = Camera(K=K, R=np.eye(3), t=np.zeros(3), width=640, height=480)
    return cloud, cam


def test_visibility_index_caches_and_matches(tmp_path, monkeypatch):
    import cloudlabeller.transfer.visibility as vismod
    from cloudlabeller.transfer.hylite_bridge import visible_point_pixels
    from cloudlabeller.transfer.visibility import VisibilityIndex

    cloud, cam = _vis_scene()
    direct_idx, direct_pix = visible_point_pixels(cloud, cam)

    calls = {"n": 0}
    real = vismod.visible_point_pixels

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(vismod, "visible_point_pixels", counting)

    index = VisibilityIndex(tmp_path / "pmap")
    idx1, pix1 = index.get("a.jpg", cloud, cam)
    assert np.array_equal(idx1, direct_idx)            # identical to the direct path
    assert np.array_equal(pix1, direct_pix)
    assert calls["n"] == 1

    index.get("a.jpg", cloud, cam)                     # RAM hit
    assert calls["n"] == 1

    fresh = VisibilityIndex(tmp_path / "pmap")         # new instance: disk hit
    idx2, pix2 = fresh.get("a.jpg", cloud, cam)
    assert calls["n"] == 1
    assert np.array_equal(idx2, direct_idx) and np.array_equal(pix2, direct_pix)

    moved = PointCloud(xyz=cloud.xyz + 0.5)            # new cloud -> recompute
    fresh.get("a.jpg", moved, cam)
    assert calls["n"] == 2

    cam2 = type(cam)(K=cam.K, R=cam.R, t=np.array([0.3, 0, 0.0]),
                     width=cam.width, height=cam.height)
    fresh.get("a.jpg", cloud, cam2)                    # new pose -> recompute
    assert calls["n"] == 3


def test_transfer_with_visibility_matches_without(tmp_path):
    from pathlib import Path

    from cloudlabeller.core.dataset import Dataset, ImageRecord
    from cloudlabeller.transfer import cloud_to_image, images_to_cloud
    from cloudlabeller.transfer.visibility import VisibilityIndex

    cloud, cam = _vis_scene()
    labels = (cloud.xyz[:, 0] > 0).astype(np.int32)
    vis = VisibilityIndex(tmp_path / "pmap")

    plain = cloud_to_image(cloud, labels, cam)
    cached = cloud_to_image(cloud, labels, cam, name="a.jpg", visibility=vis)
    assert np.array_equal(plain, cached)

    dataset = Dataset(images=[ImageRecord(0, Path("a.jpg"), cam)])
    back_plain = images_to_cloud(cloud, dataset, {"a.jpg": plain}, 2)
    back_cached = images_to_cloud(cloud, dataset, {"a.jpg": plain}, 2, visibility=vis)
    assert np.array_equal(back_plain, back_cached)


def test_mesh_nn_cache(tmp_path, monkeypatch):
    from cloudlabeller.core.dataset import Mesh
    from cloudlabeller.transfer.mesh_cloud import cloud_to_mesh

    rng = np.random.default_rng(1)
    cloud = PointCloud(xyz=rng.normal(size=(500, 3)).astype(np.float32))
    mesh = Mesh(vertices=cloud.xyz[::2] + 0.01,
                faces=np.array([[0, 1, 2]], np.int32))
    labels = rng.integers(-1, 3, 500).astype(np.int32)
    cache = tmp_path / "mesh_nn.npz"

    first = cloud_to_mesh(cloud, labels, mesh, cache_path=cache)
    assert cache.exists()

    # Same geometry, new labels: must NOT rebuild the KD-tree (cache hit).
    import scipy.spatial

    def boom(*a, **k):
        raise AssertionError("KD-tree rebuilt despite valid cache")

    monkeypatch.setattr(scipy.spatial, "cKDTree", boom)
    labels2 = (labels + 1).clip(-1, 2)
    second = cloud_to_mesh(cloud, labels2, mesh, cache_path=cache)
    assert second.shape == first.shape and not np.array_equal(first, second)
    monkeypatch.undo()

    # Changed mesh geometry -> fingerprint mismatch -> recompute (no error).
    mesh2 = Mesh(vertices=mesh.vertices + 1.0, faces=mesh.faces)
    third = cloud_to_mesh(cloud, labels, mesh2, cache_path=cache)
    assert third.shape[0] == len(mesh2.vertices)


def test_rasterize_polygon_square():
    from cloudlabeller.core.raster import rasterize_polygon

    mask = rasterize_polygon([(2, 2), (6, 2), (6, 6), (2, 6)], 10, 10)
    assert mask.dtype == bool and mask.shape == (10, 10)
    assert mask[4, 4] and not mask[0, 0]
    assert rasterize_polygon([(1, 1)], 5, 5).sum() == 0   # <3 points -> empty


def test_mask_to_indexed():
    from cloudlabeller.core.raster import mask_to_indexed

    mask = np.array([[-1, 0], [1, 7]], np.int32)      # 7 = stray id (old schema)
    index, table = mask_to_indexed(mask, {0: (255, 0, 0), 1: (0, 255, 0)})
    assert index.dtype == np.uint8
    assert index[0, 0] == 0                            # unlabelled -> row 0
    assert table[0] == (0, 0, 0, 0)                    # ... which is transparent
    assert table[index[0, 1]] == (255, 0, 0, 255)      # class 0 -> red
    assert table[index[1, 0]] == (0, 255, 0, 255)      # class 1 -> green
    assert table[index[1, 1]] == (0, 0, 0, 0)          # stray id -> transparent


def test_mask_to_rgba():
    from cloudlabeller.core.raster import mask_to_rgba

    mask = np.array([[-1, 0], [1, -1]], np.int32)
    rgba = mask_to_rgba(mask, {0: (255, 0, 0), 1: (0, 255, 0)})
    assert rgba.shape == (2, 2, 4)
    assert list(rgba[0, 1]) == [255, 0, 0, 255]    # class 0
    assert list(rgba[1, 0]) == [0, 255, 0, 255]    # class 1
    assert list(rgba[0, 0]) == [0, 0, 0, 0]        # unlabelled -> transparent


def test_project_roundtrip(tmp_path):
    proj = Project.create(tmp_path / "demo.clproj")
    proj.schema.add("rock", "#ff0000")          # class 0
    proj.schema.add("vegetation", "#00ff00")    # class 1
    proj.labels.init_cloud(4)
    proj.labels.paint_cloud(np.array([0, 1]), class_id=1)
    proj.save()

    reopened = Project.open(proj.root)
    assert [c.name for c in reopened.schema.classes] == ["rock", "vegetation"]
    assert reopened.labels.cloud_labels is not None
    assert list(reopened.labels.cloud_labels) == [1, 1, -1, -1]   # rest unlabelled


class TestMaskFromFile:
    """External mask import: .jpg = highest intensity, .png = highest alpha."""

    def test_jpg_selects_highest_intensity(self, tmp_path):
        from PIL import Image

        from cloudlabeller.core.raster import mask_from_file

        arr = np.zeros((20, 30, 3), np.uint8)
        arr[5:10, 8:20] = 255                       # bright region = selection
        p = tmp_path / "m.jpg"
        Image.fromarray(arr).save(p, quality=95)
        sel = mask_from_file(p, expected_shape=(20, 30))
        assert sel.shape == (20, 30)
        assert sel[7, 10] and not sel[0, 0]
        # JPEG noise tolerated: selection stays confined to the bright block
        ys, xs = np.nonzero(sel)
        assert ys.min() >= 4 and ys.max() <= 10

    def test_png_selects_highest_alpha(self, tmp_path):
        from PIL import Image

        from cloudlabeller.core.raster import mask_from_file

        rgba = np.zeros((10, 12, 4), np.uint8)
        rgba[..., :3] = 200                         # colour is irrelevant
        rgba[2:5, 3:7, 3] = 255                     # opaque region = selection
        p = tmp_path / "m.png"
        Image.fromarray(rgba).save(p)
        sel = mask_from_file(p)
        assert sel[3, 4] and not sel[0, 0]
        assert sel.sum() == 3 * 4

    def test_png_without_alpha_rejected(self, tmp_path):
        from PIL import Image

        from cloudlabeller.core.raster import mask_from_file

        p = tmp_path / "m.png"
        Image.fromarray(np.zeros((5, 5, 3), np.uint8)).save(p)
        with pytest.raises(ValueError, match="alpha"):
            mask_from_file(p)

    def test_dimension_mismatch_rejected(self, tmp_path):
        from PIL import Image

        from cloudlabeller.core.raster import mask_from_file

        p = tmp_path / "m.jpg"
        Image.fromarray(np.zeros((5, 6, 3), np.uint8)).save(p)
        with pytest.raises(ValueError, match=r"must\s+match|match the image"):
            mask_from_file(p, expected_shape=(100, 200))

    def test_unsupported_format_rejected(self, tmp_path):
        from cloudlabeller.core.raster import mask_from_file

        p = tmp_path / "m.tif"
        p.write_bytes(b"xx")
        with pytest.raises(ValueError, match="Unsupported"):
            mask_from_file(p)


def test_worker_threads_leaves_one_core():
    """Heavy subprocesses must leave at least one core for the OS/UI."""
    import os

    from cloudlabeller.workers.resources import worker_threads

    assert worker_threads() == max(1, (os.cpu_count() or 2) - 1)


class TestExport:
    """Cloud/mesh export round-trips (labels included in every cloud format)."""

    def _cloud(self):
        rng = np.random.default_rng(0)
        return PointCloud(
            xyz=rng.normal(size=(100, 3)).astype(np.float32),
            rgb=rng.integers(0, 255, (100, 3), dtype=np.uint8),
            normals=rng.normal(size=(100, 3)).astype(np.float32),
        ), rng.integers(-1, 4, 100).astype(np.int32)

    def test_csv_roundtrip(self, tmp_path):
        from cloudlabeller.io.export import export_cloud

        cloud, labels = self._cloud()
        p = tmp_path / "c.csv"
        export_cloud(cloud, labels, p)
        back = np.loadtxt(p, delimiter=",", skiprows=1)
        assert back.shape == (100, 7)
        assert np.allclose(back[:, :3], cloud.xyz, atol=1e-5)
        assert np.array_equal(back[:, 6].astype(np.int32), labels)

    def test_las_roundtrip(self, tmp_path):
        import laspy

        from cloudlabeller.io.export import export_cloud

        cloud, labels = self._cloud()
        p = tmp_path / "c.las"
        export_cloud(cloud, labels, p)
        las = laspy.read(str(p))
        assert len(las.points) == 100
        xyz = np.column_stack([las.x, las.y, las.z])
        assert np.allclose(xyz, cloud.xyz, atol=1e-4)
        assert np.array_equal(np.asarray(las["label"]), labels)
        assert np.array_equal(np.asarray(las.red) // 257, cloud.rgb[:, 0])

    def test_ply_roundtrip(self, tmp_path):
        from plyfile import PlyData

        from cloudlabeller.io.export import export_cloud

        cloud, labels = self._cloud()
        p = tmp_path / "c.ply"
        export_cloud(cloud, labels, p)
        v = PlyData.read(str(p))["vertex"]
        assert len(v) == 100
        assert np.allclose(v["x"], cloud.xyz[:, 0], atol=1e-5)
        assert np.array_equal(np.asarray(v["label"]), labels)
        assert np.array_equal(np.asarray(v["red"]), cloud.rgb[:, 0])

    def test_unlabelled_cloud_exports_minus_one(self, tmp_path):
        from cloudlabeller.io.export import export_cloud

        cloud, _ = self._cloud()
        p = tmp_path / "c.csv"
        export_cloud(cloud, None, p)                  # no labels at all
        back = np.loadtxt(p, delimiter=",", skiprows=1)
        assert (back[:, 6] == -1).all()

    def test_unsupported_format_rejected(self, tmp_path):
        from cloudlabeller.io.export import export_cloud

        cloud, labels = self._cloud()
        with pytest.raises(ValueError, match="Unsupported"):
            export_cloud(cloud, labels, tmp_path / "c.xyz")

    def test_mesh_formats(self, tmp_path):
        import trimesh

        from cloudlabeller.io.export import export_mesh

        mesh = Mesh(vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1]],
                                      np.float32),
                    faces=np.array([[0, 1, 2], [1, 3, 2]], np.int32),
                    vertex_colors=np.array([[255, 0, 0]] * 4, np.uint8))
        for ext in (".ply", ".obj", ".stl"):
            p = tmp_path / f"m{ext}"
            export_mesh(mesh, p)
            back = trimesh.load(str(p))
            assert len(back.faces) == 2, ext
        ply = trimesh.load(str(tmp_path / "m.ply"))
        assert np.array_equal(np.asarray(ply.visual.vertex_colors)[:, 0],
                              [255] * 4)              # colours survive PLY


def test_project_summary_info(tmp_path):
    proj = Project.create(tmp_path / "p.clproj")
    proj.schema.add("rock")
    proj.schema.add("soil")
    proj.dataset.cloud = PointCloud(xyz=np.zeros((1234, 3), np.float32))
    rows = dict(proj.summary_info())
    assert "1,234 points" in rows["Sparse cloud"]
    assert "not georeferenced" in rows["Coordinates"]
    assert "0: rock, 1: soil" == rows["Classes"]

    proj.settings["georeferenced"] = {
        "frame": "ENU", "origin_lla": [-30.5539871, -53.4158972, 275.7],
        "scale_m_per_unit": 36.3597, "n_gps": 57}
    rows = dict(proj.summary_info())
    assert "metres, true north" in rows["Coordinates"]
    assert "-30.5539871°, -53.4158972°, 275.7 m" == rows["Origin (lat, lon, alt)"]
    assert "36.3597 m" in rows["Alignment scale"] and "57 GPS" in rows["Alignment scale"]
