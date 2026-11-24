"""MobileNetV2 transfer-learning classifier (TensorFlow / Keras 3).

Architecture::

    Input (224,224,3) raw 0-255 float
      -> augmentation (training only)
      -> Rescaling(1/127.5, offset=-1)         # MobileNetV2 expects [-1, 1]
      -> MobileNetV2 backbone (ImageNet weights, include_top=False)
      -> GlobalAveragePooling2D -> Dropout -> Dense(5, softmax, L2)

The scaling and augmentation live INSIDE the model, so the saved file
is self-contained and every caller feeds identical 0-255 arrays.
"""
from __future__ import annotations

import numpy as np
from tensorflow import keras

from . import config
from .preprocessing import build_augmentation

BACKBONE_NAME = "mobilenetv2_backbone"
SCALING_NAME = "mobilenet_v2_scaling"
GAP_NAME = "gap"
DROPOUT_NAME = "head_dropout"
OUTPUT_NAME = "predictions"


def build_model(dropout: float = 0.3, l2: float = 1e-4, weights: str | None = "imagenet",
                allow_weight_fallback: bool = False, augment: bool = True,
                num_classes: int = config.NUM_CLASSES, seed: int = config.SEED):
    """Build the (uncompiled) model. Returns ``(model, info)``.

    ``info["pretrained_weights"]`` states what was *actually* loaded.
    If ImageNet weights cannot be downloaded this raises, unless
    ``allow_weight_fallback=True``, in which case the backbone is randomly
    initialised and ``info`` says so (never silently).
    """
    info = {"architecture": "MobileNetV2 (alpha=1.0) + GAP + Dropout + Dense(softmax)",
            "pretrained_weights": weights or "none (random init)", "weight_fallback_used": False}
    try:
        backbone = keras.applications.MobileNetV2(
            input_shape=config.IMG_SHAPE, include_top=False, weights=weights, name=BACKBONE_NAME)
    except Exception as exc:
        if weights is None or not allow_weight_fallback:
            raise RuntimeError(
                f"Could not load MobileNetV2 weights '{weights}': {exc}. "
                "Check internet access or pass allow_weight_fallback=True "
                "(results will then NOT use pretrained weights).") from exc
        backbone = keras.applications.MobileNetV2(
            input_shape=config.IMG_SHAPE, include_top=False, weights=None, name=BACKBONE_NAME)
        info.update({"pretrained_weights": "none (random init, fallback)", "weight_fallback_used": True,
                     "fallback_reason": str(exc)})

    inputs = keras.Input(shape=config.IMG_SHAPE, name="image")
    x = build_augmentation(seed)(inputs) if augment else inputs
    x = keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name=SCALING_NAME)(x)
    x = backbone(x, training=False)           # keep BatchNorm in inference mode
    x = keras.layers.GlobalAveragePooling2D(name=GAP_NAME)(x)
    x = keras.layers.Dropout(dropout, name=DROPOUT_NAME)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name=OUTPUT_NAME,
                                 kernel_regularizer=keras.regularizers.l2(l2))(x)
    model = keras.Model(inputs, outputs, name="dr_mobilenetv2")
    info["dropout"], info["l2"] = dropout, l2
    return model, info


def set_backbone_trainable(model, n_top_layers: int) -> dict:
    """Freeze the backbone (``n_top_layers=0``) or unfreeze its last N layers.

    BatchNormalization layers always stay frozen: with ~1k images and small
    batches, updating BN statistics tends to hurt.
    """
    backbone = model.get_layer(BACKBONE_NAME)
    if n_top_layers <= 0:
        backbone.trainable = False
    else:
        backbone.trainable = True
        for layer in backbone.layers[:-n_top_layers]:
            layer.trainable = False
        for layer in backbone.layers:
            if isinstance(layer, keras.layers.BatchNormalization):
                layer.trainable = False
    trainable = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    total = model.count_params()
    return {"trainable_params": trainable, "total_params": int(total),
            "frozen_params": int(total - trainable), "unfrozen_backbone_layers": max(n_top_layers, 0)}


def compile_model(model, learning_rate: float):
    model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"])
    return model
