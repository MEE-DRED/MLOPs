"""Interpretable image-quality statistics for the dataset-insights dashboard.

Brightness, contrast and sharpness are computed on the *retinal field only*
(non-black pixels, with the circular border excluded for sharpness) so the
black background of fundus photographs does not dominate the numbers.

These are descriptive image properties. They are NOT causal explanations of
disease and should not be presented as such.
"""
from __future__ import annotations

import numpy as np

BACKGROUND_THRESHOLD = 10.0   # gray level below which a pixel counts as background


def to_gray(images: np.ndarray) -> np.ndarray:
    """ITU-R BT.601 luma, float32, accepts (H,W,3) or (N,H,W,3)."""
    x = np.asarray(images, dtype=np.float32)
    return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]


def _features_one(gray: np.ndarray) -> tuple[float, float, float, float]:
    mask = gray > BACKGROUND_THRESHOLD
    frac = float(mask.mean())
    if mask.sum() < 10:                      # (nearly) blank image
        return 0.0, 0.0, 0.0, frac
    vals = gray[mask]
    brightness = float(vals.mean())
    contrast = float(vals.std())

    # variance of the Laplacian over pixels whose whole 5-point stencil is
    # inside the retina (avoids the huge response at the circular border)
    c = gray[1:-1, 1:-1]
    lap = (gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4.0 * c)
    interior = (mask[1:-1, 1:-1] & mask[:-2, 1:-1] & mask[2:, 1:-1]
                & mask[1:-1, :-2] & mask[1:-1, 2:])
    sharpness = float(lap[interior].var()) if interior.sum() > 10 else 0.0
    return brightness, contrast, sharpness, frac


def compute_image_features(images: np.ndarray) -> dict[str, np.ndarray]:
    """Per-image brightness, contrast, sharpness and retina-area fraction."""
    images = np.asarray(images)
    if images.ndim == 3:
        images = images[None]
    out = np.zeros((len(images), 4), dtype=np.float64)
    for i, img in enumerate(images):
        out[i] = _features_one(to_gray(img))
    return {"brightness": out[:, 0], "contrast": out[:, 1],
            "sharpness": out[:, 2], "retina_fraction": out[:, 3]}


def features_dataframe(data, class_names):
    """Feature table for every image in every split (pandas DataFrame)."""
    import pandas as pd

    frames = []
    for split, (images, labels) in data.items():
        feats = compute_image_features(images)
        df = pd.DataFrame(feats)
        df["split"] = split
        df["label"] = labels
        df["class"] = [class_names[i] for i in labels]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


FEATURE_EXPLANATIONS = {
    "brightness": "Average light level inside the retinal field (0 = black, 255 = white). "
                  "Reflects camera exposure, pupil size and media clarity, not disease directly.",
    "contrast": "Spread of light levels inside the retinal field. Low contrast looks washed-out "
                "or hazy; high contrast shows vessels and bright/dark structures more clearly.",
    "sharpness": "Variance of the Laplacian (a standard focus measure). Higher means finer, "
                 "crisper detail; blurry or out-of-focus images score low. It also rises with noise.",
}
