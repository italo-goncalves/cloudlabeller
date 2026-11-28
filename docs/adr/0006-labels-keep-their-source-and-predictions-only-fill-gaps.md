# Labels keep their source, and predictions only fill gaps

Status: accepted (2026-11-28).

Until now each Photo has one status (`user`, `auto` or `ml`), and the status
decides which layer is the Photo's whole Mask. That lets the network's output
reach its own training data in three ways (traced 2026-11-28):

- **Through the cloud.** Predicted Photos vote on the cloud
  (`transfer.images_to_cloud`, `include_auto` on by default). Cloud → images
  then marks every Photo `auto`, and training with "include auto" learns
  from them. Nothing records that those cloud labels came from the network.
- **Through painting.** The mask editor starts from the overlay it shows,
  and `image.save_mask` stores the whole mask as `user`. A few strokes on a
  predicted Photo turn millions of predicted pixels into hand-drawn ground
  truth, for training and for validation.
- **Over cloud-made labels.** Predict replaces every Photo that isn't
  `user`. A Photo labelled from the cloud then shows only the prediction.

From now on, every label has a **source**, ranked from most to least
reliable:

1. **Hand-drawn**: painted by the user on a Photo or on the cloud.
2. **From the cloud**: projected into a Photo from cloud points whose labels
   came from hand-drawn sources.
3. **Predicted**: the network's output, or projected from cloud points whose
   labels came only from predictions.

The rules:

1. **Labels are kept in layers and composed per pixel**, not chosen per
   Photo. A Photo's label at a pixel comes from the best source that labels
   it. The layers are already stored apart: the Mask, the cloud and the
   prediction. Photos need no separate quality mask.
2. **A Mask holds only what the user painted.** The editor shows the
   composed labels but saves only the pixels the user's strokes touched.
   Painting the class a pixel already shows still makes it hand-drawn.
   Keeping a prediction as ground truth is an explicit action ("Accept the
   prediction" over a region). The eraser removes the user's own labels, so
   the layer below shows again.
3. **Every cloud point keeps its source and an Uncertainty**: one `uint8`
   and one `float16` per point. The vote counts behind them are not kept.
4. **Votes are taken per source, best first.** A point that gets any vote
   from a better source is decided by those votes alone; worse sources only
   fill points no better source reached. A point painted directly on the
   cloud is hand-drawn and isn't changed by votes: the stroke was made on
   the points themselves.
5. **The Uncertainty of a cloud point comes from the votes that decided
   it**, as the normalised entropy of their Class shares smoothed by a
   Dirichlet prior (α = 1/K, about one vote of doubt), with the amount of
   evidence set to the number of **independent views** rather than of
   Photos. Votes from cameras in the same direction cell (64 cells, about
   25° each) as seen from the point count once. A point painted on the
   cloud has an Uncertainty of 0.
6. **Projected labels don't vote on the cloud they came from.** That would
   repeat the cloud's own labels and drive the Uncertainty towards 0 with no
   new evidence. They still vote on another cloud: the dense cloud from the
   sparse one, or one Reconstruction from another.
7. **Predictions only fill gaps.** Predict runs on every Photo, hand-drawn
   ones included, and its labels show only where no better source reached.
8. **Training never learns from predictions.** Pixels are weighted by
   source: hand-drawn 1; from the cloud 0.5 by default, optionally scaled by
   (1 − Uncertainty); predicted 0. Validation scores hand-drawn pixels only.

## Considered Options

- **A quality mask stored beside each Photo's labels** (the first idea): it
  records the same thing as the layers but must be kept in step with three
  writers, and it costs a `uint8` per pixel per Photo. Composing the layers
  gives the source for free.
- **Keep a status per Photo and only block `ml` Photos better**: this closes
  the direct routes but not the cloud route. It also can't let a prediction
  fill the gaps of a hand-drawn Photo.
- **Keep the vote counts per point**: an N × K `int32` array is 316 MB on
  Lomito's 7.9 M-point dense cloud with 10 Classes, and nothing needs it
  after the Uncertainty is computed.
- **Raw vote counts as the evidence**: 30 overlapping Photos of one flight
  line would look almost certain (Uncertainty 0.08 against 0.55 with
  independent views), although they share occlusions, lighting and mask
  edges.

## Consequences

- The status per Photo stays only as a summary for the Dataset pane: which
  sources a Photo has. It no longer decides what is shown, trained or voted.
- The Engine's `image.mask` returns the composed labels and each pixel's
  source; the Image view shows the source by opacity and with a "Show source"
  toggle. Both views can colour by Uncertainty, one 0..1 scale for cloud
  and predicted labels.
- Masks saved before this change can't be split into strokes and copied
  overlay. They stay hand-drawn. A per-Photo action removes the pixels equal
  to the current prediction or projection, with a preview, and the user
  decides where to run it.
- Training needs per-pixel weights, in the trainer's own code; the LOCKED
  block of `ml/unet.py` doesn't change. Validation scores drop, because they
  stop counting the network's own output as ground truth. Models trained
  before can't be compared with later ones.
- The contract changes for `image.mask`, `image.save_mask`,
  `transfer.images_to_cloud` and `transfer.cloud_to_images` go into
  `docs/engine.md` in the same change.
