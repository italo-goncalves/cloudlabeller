# Cloud normals as extra U-Net inputs

*Experiment of 2026-11-27, on the FIDA project. Script:
[`tools/experiments/normals_as_features.py`](../../tools/experiments/normals_as_features.py),
tests: [`tests/test_normal_maps.py`](../../tests/test_normal_maps.py) (the normal maps and the
augmentation are the app's own since).*

## Question

Does giving the U-Net the dense cloud's surface normals, as three more input
channels beside RGB, make it label photos better?

## Method

- **Normal map per photo.** The dense-cloud points visible in a photo (from
  the Survey's visibility cache, `cache/pmap`, read without writing) give
  their normals. Each normal is rotated into the camera's frame (x right,
  y down, z forward) and splatted into its pixel (5 × 5, as auto masks splat
  labels), encoded as uint8 `round(127.5 (n + 1))`; 128 means no normal.
- **Same Tiles for both.** Every photo is cut once into 6-channel Tiles
  (2048 px, shrink 8, so 256 × 256 for the network; overlap 0.35). RGB alone
  uses the first three channels of the same Tiles.
- **Augmentation.** The app's mirror/slide/gamma windows (5 rolls), with the
  gamma on RGB only. Mirroring a Tile left to right turns the normals' x
  component around, so the mirrored half gets `x → −x`
  (now `ml.tile_dataset.augment` itself; with RGB alone it is unchanged, which a
  test checks).
- **Training and validation sets.** Training on the 372 photos labelled
  from the cloud (1,153 Tiles); validation on the 4 photos painted by hand
  (39 Tiles). A cloud-made mask and its normal map share their edges, so
  validating on cloud-made masks would flatter the normals; the painted
  masks are ground truth the cloud didn't make.
- **Runs.** Same model settings as the app's last FIDA run (16 channels,
  4 blocks, dropout 0.1, 5 × 5 filters, 4 classes), batch 2, 50,000 steps
  (5,765 per epoch, so 9 epochs), keeping the epoch with the lowest
  validation loss. Two seeds per variant. RTX 4060 Ti, about 12.5 minutes
  per run.

Coverage: on average the normals cover 69 % of a training photo and 89 % of
a validation photo.

## Results

Best epoch of each run (validation is the 4 painted photos; accuracy counts
labelled pixels only):

| Input | Seed | Best epoch | val_loss | val_accuracy | train accuracy |
|---|---|---|---|---|---|
| RGB | 0 | 5 | 0.211 | 0.958 | 0.952 |
| RGB | 1 | 7 | 0.280 | 0.949 | 0.965 |
| RGB + normals | 0 | 6 | 0.090 | 0.975 | 0.970 |
| RGB + normals | 1 | 5 | **0.074** | **0.976** | 0.962 |
| **mean RGB** | | | 0.245 | 0.954 | |
| **mean RGB + normals** | | | **0.082** | **0.976** | |

Validation loss per epoch:

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| RGB s0 | 0.684 | 0.554 | 0.310 | 0.305 | 0.211 | 0.378 | 0.575 | 0.329 | 0.313 |
| RGB s1 | 0.459 | 0.404 | 0.362 | 0.772 | 0.564 | 0.329 | 0.280 | 0.325 | 0.351 |
| +normals s0 | 0.357 | 0.251 | 0.169 | 0.151 | 0.094 | 0.090 | 0.121 | 0.093 | 0.098 |
| +normals s1 | 0.280 | 0.291 | 0.166 | 0.110 | 0.074 | 0.084 | 0.082 | 0.091 | 0.146 |

What this says:

- **Errors roughly halve.** The share of labelled validation pixels labelled
  wrong falls from 4.6 % to 2.4 %, and the validation loss falls by about
  three times (0.245 → 0.082).
- **It is not overfitting.** Training accuracy is about the same in both
  (0.95–0.97). The gain is on photos the model never saw.
- **Training is steadier.** With RGB alone the validation loss swings from
  epoch to epoch (0.211 → 0.575 → 0.329 for seed 0). With normals it falls
  smoothly and stays below 0.15 from epoch 4 on.
- **The two seeds agree.** Each run with normals beats each run without them.

## Caveats

- **Small validation set.** It is 4 photos (39 Tiles) of one site and one
  date. The gap is large and the two seeds agree, but the size of the gain
  is uncertain.
- **Coverage is not separated out.** These runs didn't measure accuracy
  separately on pixels with a normal and on the 11 % without one. Where the
  cloud has no points (water, sky, thin vegetation), the "no normal" pattern
  itself carries information. That is fair, but it means the model leans on
  how well MVS covered the photo.
- **Training labels come from the same cloud.** The masks and the normals
  share every edge, so the model may partly learn "the class changes where
  the normal map changes". Validation on painted masks shows this still
  helps on real boundaries, but a model trained only on painted masks hasn't
  been tested.

## The cost in the app

- **A photo must belong to a dense Reconstruction to be predicted.** Today a
  model can label the photos of a new Survey before any 3D work. With
  normals, that Survey needs SfM and MVS first, and only covered pixels get
  them.
- **Slower cutting.** It went from about 5 minutes to about 17 for 372
  photos: the normal maps and the 6-channel shrink. Tiles take twice the
  memory (625 MB for FIDA's).
- **The model has to say what it takes.** A model trained on normals has
  6 input channels and can't predict on RGB alone, and the reverse. The
  manifest must record its inputs, and Predict must check them.
- **Nothing locked changes.** `build_model` already takes `in_channels`.

## Recommendation

*Done on 2026-11-27: Model Settings has the experimental option "Also learn
from the cloud's normals" (`ModelSpec.normals`), and the 3D view and the
Image view can colour by normals.*


**Add normals as an option.** Off by default, it would be a Model Settings
checkbox "Use the cloud's normals (needs a dense Reconstruction)". The
evidence for it is strong, but the cost is a real change to the workflow:
predicting needs MVS first. A user who reconstructs every Survey anyway,
as in FIDA, gains a lot.

Before building it:

1. **Accuracy split by coverage.** Score with and without normals on pixels
   that have a normal and on pixels that don't. If the gain sits only on
   covered pixels, predicting the uncovered ones from RGB alone (two models,
   or a fallback) may be worth it.
2. **Another Survey or site.** Repeat on another Survey or site, so the
   gain isn't from FIDA alone.
3. **Variants worth trying.**
   - World-frame normals instead of camera-frame normals: the slope and
     aspect of the surface, which are geologically meaningful and don't
     depend on the view.
   - Normals alone, without the coverage pattern, by filling the holes.

## Reproducing

In the WSL Environment (the experiment only reads the Project):

```
PYTHONPATH=/mnt/c/Repos/RockAnalytica/cloudlabeller \
  ~/.local/share/cloudlabeller/engine/bin/python -u \
  /mnt/c/Repos/RockAnalytica/cloudlabeller/tools/experiments/normals_as_features.py \
  "/mnt/c/Big data/CloudLabeller_projects/FIDA" ~/cl-experiments/normals --seeds 2
```

The Tiles (`tiles.npz`, 625 MB) and `results.json`, with every epoch's
metrics, stay in `~/cl-experiments/normals` in WSL, outside the repo.
