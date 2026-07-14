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

"""U-Net model — Ítalo's TensorFlow/Keras architecture (LOCKED block below).

``build_model`` / ``build_model_from_spec`` wrap the locked ``u_net`` into a
compiled ``tf.keras.Model``; the hyper-parameters come from
:class:`~cloudlabeller.ml.model_spec.ModelSpec` (Model → Model Settings…).
"""

from __future__ import annotations
import tensorflow as tf

# === LOCKED: Ítalo ===
def encoder_block(input_layer, out_channels, filter_size=3,
                  depth=2, residual=False,
                  dropout_prob=0.3, max_pooling=True):
    conv = input_layer

    conv_out = conv
    for _ in range(depth):
        conv_out = tf.keras.layers.Conv2D(
            filters=out_channels,
            kernel_size=filter_size,
            activation='relu',
            padding="same",
            kernel_initializer='HeNormal')(conv_out)
    if residual:
        conv = tf.keras.layers.Conv2D(
            filters=out_channels,
            kernel_size=1,
            activation='relu',
            padding="same",
            kernel_initializer='HeNormal')(conv)
        conv_out = tf.keras.layers.Add()([conv_out, conv])
    conv = conv_out

    if dropout_prob > 0:
        conv = tf.keras.layers.Dropout(dropout_prob)(conv)

    if max_pooling:
        next_layer = tf.keras.layers.MaxPool2D()(conv)
        return next_layer, conv
    else:
        return conv


def decoder_block(down_layer, out_channels, skip_layer=None, filter_size=3,
                  depth=2, residual=False,
                  dropout_prob=0.3):
    up = tf.keras.layers.UpSampling2D()(down_layer)
    up = tf.keras.layers.Conv2D(
        filters=out_channels,
        kernel_size=2,
        activation='linear',
        padding="same",
        kernel_initializer='HeNormal')(up)

    conv = up
    if skip_layer is not None:
        # conv = tf.keras.layers.Concatenate()([skip_layer, up])
        conv = tf.keras.layers.Add()([skip_layer, up])

    conv_out = conv
    for _ in range(depth):
        conv_out = tf.keras.layers.Conv2D(
            filters=out_channels,
            kernel_size=filter_size,
            activation='relu',
            padding="same",
            kernel_initializer='HeNormal')(conv_out)

    if residual:
        conv = tf.keras.layers.Conv2D(
            filters=out_channels,
            kernel_size=1,
            activation='relu',
            padding="same",
            kernel_initializer='HeNormal')(conv)
        conv_out = tf.keras.layers.Add()([conv_out, conv])

    if dropout_prob > 0:
        conv_out = tf.keras.layers.Dropout(dropout_prob)(conv_out)

    return conv_out


def dice_loss(y_true, y_pred):
    return 1 - 2 * tf.reduce_sum(y_true * y_pred) / (
            tf.reduce_sum(y_true) + tf.reduce_sum(y_pred))


def focal_loss(y_true, y_pred):
    cross_entropy = - (1 - y_pred) ** 2 * y_true * tf.math.log(
        y_pred * 0.9999 + 0.0001)
    return tf.reduce_mean(cross_entropy)


def u_net(input_layer, output_size, channels=16, blocks=4,
          block_depth=2, residual=False,
          dropout_prob=0.1, filter_size=5,
          end_activation=None):
    out_layer = input_layer
    skips = []
    for i in range(blocks):
        out_layer, skip = encoder_block(out_layer, channels * 2 ** i,
                                        dropout_prob=dropout_prob,
                                        filter_size=filter_size,
                                        depth=block_depth,
                                        residual=residual)
        skips.append(skip)

    out_layer = encoder_block(out_layer, channels * 2 ** blocks,
                              max_pooling=False)

    for i in range(blocks):
        out_layer = decoder_block(out_layer,
                                  channels * 2 ** (blocks - i - 1),
                                  skip_layer=skips[-i - 1],
                                  dropout_prob=dropout_prob,
                                  filter_size=filter_size,
                                  depth=block_depth,
                                  residual=residual)

    out_layer = tf.keras.layers.Conv2D(
        filters=output_size,
        kernel_size=1,
        activation=end_activation,
        padding="same",
        kernel_initializer='HeNormal')(out_layer)

    return out_layer
# === END LOCKED ===


def build_model(n_classes: int, image_width: int, image_height: int,
                in_channels: int = 3, **kwargs):
    """Build and compile the U-Net. Extra ``kwargs`` (channels, blocks,
    dropout_prob, filter_size, …) flow straight into ``u_net``."""
    # Keras is channels-last: Conv2D input is (height, width, channels).
    net_input = tf.keras.layers.Input([image_height, image_width, in_channels])
    # u_net returns the output *layer*; wrap input/output into a Model.
    output = u_net(net_input, n_classes, end_activation='softmax', **kwargs)
    model = tf.keras.Model(net_input, output)
    model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            # SparseCategoricalCrossentropy(ignore_class=-1) skips unlabelled
            # pixels (label -1), matching the project's UNLABELLED_ID convention.
            loss=tf.keras.losses.SparseCategoricalCrossentropy(ignore_class=-1),
            metrics=['accuracy']
    )
    return model


def build_model_from_spec(spec, n_classes: int,
                          source_resolution: tuple[int, int],
                          in_channels: int = 3):
    """Build the U-Net from a :class:`~cloudlabeller.ml.model_spec.ModelSpec`
    (the Model Settings dialog): the training size comes from
    ``spec.training_size`` and the architecture from ``spec.unet_kwargs()``.
    This is the single seam between the dialog's values and ``build_model``."""
    width, height = spec.training_size(*source_resolution)
    return build_model(n_classes, width, height,
                       in_channels=in_channels, **spec.unet_kwargs())

def load_model(weights_path: str, n_classes: int, image_width: int,
               image_height: int, in_channels: int = 3, **kwargs):
    """Rebuild the architecture and restore weights from ``weights_path``.

    Named models saved by the app carry their parameters in a manifest — use
    ``ml.model_store.load_model(ml_dir, name)`` instead of calling this directly.
    """
    model = build_model(n_classes, image_width, image_height,
                        in_channels=in_channels, **kwargs)
    model.load_weights(weights_path)
    return model
