# COLMAP's depth and normal maps as U-Net inputs

*Experiment of 2026-11-27/28, on the Lomito project (MVA Petrobras). Script:
[`tools/experiments/colmap_maps_as_features.py`](../../tools/experiments/colmap_maps_as_features.py).
Follows [normals-as-features.md](normals-as-features.md) (FIDA).*

## Question

COLMAP's MVS step leaves a depth map and a normal map per photo in
`dense/stereo/`, which take most of the MVS workspace's disk space. Are they
worth keeping as training inputs, at reduced resolution and made independent
of the camera's position, before the workspace is freed? And do they beat the
dense cloud's normals splatted into the photo, which the app already uses
(`ModelSpec.normals`)?

## Method

- **Site.** Lomito: 879 photos at 8192 × 5460, 29 painted by hand,
  10 classes. MVS maps at 1024 × 678 (⅛ of the photos): 1,758 depth maps
  (4.6 GB) and 1,758 normal maps (14 GB). Dense cloud of 7.9 M points.
- **COLMAP's maps into the photos.** The geometric maps are in the undistorted
  pinhole frame, in SfM units (scale 17.885 m per unit). Each pixel is lifted to
  a 3D point in metres and projected with the original camera, distortion
  included. A z-buffer keeps the nearest point, and one-pixel holes are filled.
- **Four inputs, on the same Tiles:**
  - `rgb`: the photo alone;
  - `cloud`: RGB + the dense cloud's normals splatted into the photo, in the
    camera frame (what the app does now);
  - `colmap`: RGB + COLMAP's geometric normal map, in the camera frame;
  - `invariant`: RGB + a validity flag, the slope (angle of the world-frame
    normal to the vertical) and the relief at 0.1 m and 0.5 m (smoothed depth
    minus depth, in metres, so what sticks out towards the camera is positive).
    None of these depend on where the camera was. There is no mirror flip in
    augmentation, since the channels don't change under a mirror.
- **Resolution.** Everything at ⅛ of the photos (the MVS resolution; about
  1 cm per pixel at Lomito's ~10 m distance). Tiles of 2048 px, so 256 px for
  the network.
- **Training and validation.** Only the 29 painted photos, split by whole
  photos: 25 % held out per split, two splits (507/156 and 512/151 Tiles).
  Every input trains on each split (a paired comparison). The model is the
  same as on FIDA (16 channels, 4 blocks, dropout 0.1, 5 × 5 filters), with
  batch 2, 30,000 steps and patience 10, keeping the epoch with the lowest
  validation loss.

Coverage: the splatted normals cover 94 % of a photo and COLMAP's 96 %. The
two agree closely (median cosine 0.997).

## Results

Best epoch of each run (accuracy counts labelled pixels only):

| Input | Split 0 val_acc / val_loss | Split 1 val_acc / val_loss | Mean val_acc | Mean val_loss |
|---|---|---|---|---|
| RGB | 0.665 / 0.966 | 0.604 / 1.144 | 0.635 | 1.055 |
| + cloud normals | 0.677 / 0.951 | 0.669 / 1.035 | 0.673 | **0.993** |
| + COLMAP normals | 0.680 / 1.027 | 0.675 / 1.022 | **0.677** | 1.025 |
| + slope + relief | **0.705 / 0.916** | 0.615 / 1.097 | 0.660 | 1.006 |

The best epochs all came early (2–4 of 12), and training accuracy kept rising
(to 0.76–0.84), so every input overfits 22 training photos.

What this says:

- **Normals help on a second site.** Both kinds of normals beat RGB on both
  splits: about 4 points of accuracy on average, and a lower validation loss.
  The gain is smaller than on FIDA (where errors halved), but it points the
  same way.
- **COLMAP's normals add nothing over the cloud's.** The two are almost the
  same map (cosine 0.997) and score the same. The splatted cloud normals
  need no extra storage.
- **Slope and relief are not proven.** They were the best input on split 0 and
  close to RGB on split 1. With 7 validation photos per split, that spread is
  within the noise.

## Caveats

- **Very small data.** 29 painted photos, 7 held out per split, and 10 classes,
  some of them rare. A swing of one photo moves a score by several points.
- **Only painted photos.** Lomito has no cloud-made masks, so this
  is a different setup from FIDA's (training on cloud-made masks, validating on
  painted ones).
- **Overfitting dominates.** The early best epochs suggest more labelled
  photos would matter more than any extra input.

## Recommendation

- **Don't keep COLMAP's maps for training.** The dense cloud's normals do as
  well. Storing the maps at ¼ resolution is not needed for this.
- **Freeing the MVS workspace's stereo maps is safe for training.** The app's
  normals come from the dense cloud (`dense.npz`), not from `dense/stereo/`.
  A "Free the MVS workspace" action still needs its own check: re-running
  fusion or meshing reads that workspace. Nothing is deleted without the
  user's explicit OK.
- **Keep slope and relief as an idea.** Test them again on a site with more
  painted photos, or on FIDA with cloud-made training masks, before adding them
  to the app.

## Reproducing

In the WSL Environment (the experiment only reads the Project, which must be in
the 0.3.0 layout):

```
PYTHONPATH=/mnt/c/Repos/RockAnalytica/cloudlabeller \
  ~/.local/share/cloudlabeller/engine/bin/python -u \
  /mnt/c/Repos/RockAnalytica/cloudlabeller/tools/experiments/colmap_maps_as_features.py \
  "/mnt/c/Big data/CloudLabeller_projects/MVA Petrobras/Lomito" ~/cl-experiments/lomito \
  --scales 0.1 0.5
```

The per-photo channels (`photos.npz`) and `results.json`, with every epoch's
metrics, stay in `~/cl-experiments/lomito` in WSL, outside the repo. Finished
runs are skipped when it is run again.
