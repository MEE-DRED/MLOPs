"""Grad-CAM for the MobileNetV2 classifier.

IMPORTANT: Grad-CAM highlights image regions that most increased the model's
score for a class. It is a model-attention visual, NOT a lesion detector and
NOT a clinical explanation. Always present it with that caveat.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from .model import BACKBONE_NAME, GAP_NAME, OUTPUT_NAME, SCALING_NAME
from .preprocessing import to_model_input


def grad_cam(model, image_uint8: np.ndarray, class_index: int | None = None):
    """Return ``(heatmap (224,224) in [0,1], class_index, probabilities)``.

    The backbone is a nested model, so its feature map is obtained by calling
    the layers manually inside a GradientTape.
    """
    x = tf.convert_to_tensor(to_model_input(image_uint8))
    scaling, backbone = model.get_layer(SCALING_NAME), model.get_layer(BACKBONE_NAME)
    gap, head = model.get_layer(GAP_NAME), model.get_layer(OUTPUT_NAME)
    with tf.GradientTape() as tape:
        conv = backbone(scaling(x), training=False)          # (1, 7, 7, 1280)
        tape.watch(conv)
        probs = head(gap(conv))
        idx = int(tf.argmax(probs[0])) if class_index is None else int(class_index)
        score = probs[:, idx]
    grads = tape.gradient(score, conv)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(weights * conv, axis=-1))[0]
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    cam = tf.image.resize(cam[..., None], (224, 224), method="bilinear")[..., 0]
    return cam.numpy(), idx, probs[0].numpy()
