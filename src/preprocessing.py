"""Image loading, validation and tf.data pipelines.

Design decision (avoids double normalisation and train/serve skew):
the saved Keras model contains its own ``Rescaling`` layer that maps
0-255 pixels to the [-1, 1] range MobileNetV2 expects, and its own
augmentation layers (active only while training). Everything outside the
model - training, API, retraining - therefore feeds the *same* thing:
RGB float32 arrays in the 0-255 range, shape (N, 224, 224, 3).
Do not divide by 255 anywhere else.

Only ``make_tf_dataset`` and ``build_augmentation`` import TensorFlow
(lazily), so the rest can be tested without it.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable, Union

import numpy as np
from PIL import Image, UnidentifiedImageError

from . import config

ImageSource = Union[str, Path, bytes, bytearray, io.IOBase]


class InvalidImageError(ValueError):
    """The supplied file is not an acceptable retinal image."""


class InvalidLabelError(ValueError):
    """The supplied class label is not an integer in 0..4."""


# --------------------------------------------------------------- loading
def _read_bytes(source: ImageSource) -> bytes:
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    if isinstance(source, (str, Path)):
        p = Path(source)
        if not p.is_file():
            raise InvalidImageError(f"File not found: {p}")
        return p.read_bytes()
    if hasattr(source, "read"):
        return source.read()
    raise InvalidImageError(f"Unsupported image source type: {type(source).__name__}")


def load_image(source: ImageSource, size: int = config.IMG_SIZE,
               non_square: str = "pad") -> np.ndarray:
    """Decode, validate and convert one image to ``uint8`` RGB (size, size, 3).

    * Only JPEG and PNG are accepted (checked from file *content*, not name).
    * Corrupt/truncated files, tiny images and decompression bombs are rejected.
    * Grayscale/RGBA/palette images are converted to RGB.
    * Non-square images are padded with black (``non_square="pad"``) or
      centre-cropped (``"crop"``) to a square before resizing, so the aspect
      ratio is not distorted. Note: the MedMNIST preparation of the original
      photographs may not match this exactly; see docs/architecture.md.
    """
    raw = _read_bytes(source)
    if not raw:
        raise InvalidImageError("Empty file")
    if len(raw) > config.MAX_UPLOAD_BYTES:
        raise InvalidImageError(
            f"File is {len(raw) / 1e6:.1f} MB; the limit is {config.MAX_UPLOAD_BYTES / 1e6:.0f} MB")
    try:
        probe = Image.open(io.BytesIO(raw))
        fmt = probe.format
        w, h = probe.size
        probe.verify()                      # detects many corrupt files
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise InvalidImageError(f"Not a readable image: {exc}") from exc
    if fmt not in config.ALLOWED_IMAGE_FORMATS:
        raise InvalidImageError(f"Unsupported format {fmt}; allowed: {config.ALLOWED_IMAGE_FORMATS}")
    if min(w, h) < config.MIN_IMAGE_SIDE:
        raise InvalidImageError(f"Image too small ({w}x{h}); minimum side {config.MIN_IMAGE_SIDE}px")
    if w * h > config.MAX_IMAGE_PIXELS:
        raise InvalidImageError(f"Image too large ({w}x{h})")
    try:
        img = Image.open(io.BytesIO(raw))   # verify() leaves the first handle unusable
        img = img.convert("RGB")            # full decode; truncated data raises here
    except (OSError, SyntaxError, ValueError) as exc:
        raise InvalidImageError(f"Image data is corrupt or truncated: {exc}") from exc

    if w != h:
        if non_square == "crop":
            side = min(w, h)
            left, top = (w - side) // 2, (h - side) // 2
            img = img.crop((left, top, left + side, top + side))
        elif non_square == "pad":
            side = max(w, h)
            canvas = Image.new("RGB", (side, side), (0, 0, 0))
            canvas.paste(img, ((side - w) // 2, (side - h) // 2))
            img = canvas
        else:
            raise ValueError("non_square must be 'pad' or 'crop'")
    if img.size != (size, size):
        img = img.resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def validate_label(label) -> int:
    """Return ``label`` as an int in 0..4 or raise ``InvalidLabelError``.

    Accepts ints and digit strings ("3"). Rejects bools, floats with a
    fractional part, blanks, class names and out-of-range values - an
    unlabelled or mis-labelled image must never reach supervised retraining.
    """
    if isinstance(label, bool) or label is None:
        raise InvalidLabelError(f"Invalid label: {label!r}")
    if isinstance(label, (int, np.integer)):
        value = int(label)
    elif isinstance(label, (float, np.floating)):
        if not float(label).is_integer():
            raise InvalidLabelError(f"Label must be a whole number, got {label!r}")
        value = int(label)
    elif isinstance(label, str):
        text = label.strip()
        if not text.lstrip("+-").isdigit():
            raise InvalidLabelError(f"Label must be an integer 0-{config.NUM_CLASSES - 1}, got {label!r}")
        value = int(text)
    else:
        raise InvalidLabelError(f"Unsupported label type {type(label).__name__}")
    if not 0 <= value < config.NUM_CLASSES:
        raise InvalidLabelError(f"Label {value} outside 0-{config.NUM_CLASSES - 1}")
    return value


# ------------------------------------------------------- model inputs
def to_model_input(images: Union[np.ndarray, Iterable[np.ndarray]]) -> np.ndarray:
    """uint8 image(s) -> float32 batch in the 0-255 range the model expects.

    Accepts (H,W,3), (N,H,W,3) or an iterable of (H,W,3) arrays.
    """
    arr = np.asarray(images if isinstance(images, np.ndarray) else list(images))
    if arr.ndim == 3:
        arr = arr[None]
    if arr.ndim != 4 or arr.shape[1:] != config.IMG_SHAPE:
        raise InvalidImageError(f"Expected (N,{config.IMG_SHAPE}) images, got {arr.shape}")
    return arr.astype(np.float32)


def mobilenet_v2_reference_scaling(x: np.ndarray) -> np.ndarray:
    """NumPy reference of the in-model Rescaling layer (x / 127.5 - 1).

    Used only to unit-test that the model's scaling matches the
    ``tf.keras.applications.mobilenet_v2.preprocess_input`` convention.
    """
    return np.asarray(x, dtype=np.float32) / 127.5 - 1.0


# ------------------------------------------------ tf.data (needs TF)
def build_augmentation(seed: int = config.SEED):
    """Keras augmentation block (active only when ``training=True``).

    Deliberately mild: horizontal flip, rotation up to about +/-14 degrees,
    10 % zoom and moderate brightness/contrast jitter. No vertical flips,
    shear or heavy warps that could distort retinal structures.
    """
    from tensorflow import keras
    layers = keras.layers
    return keras.Sequential([
        layers.RandomFlip("horizontal", seed=seed),
        layers.RandomRotation(0.04, fill_mode="constant", fill_value=0.0, seed=seed),
        layers.RandomZoom(0.1, fill_mode="constant", fill_value=0.0, seed=seed),
        layers.RandomBrightness(0.1, value_range=(0, 255), seed=seed),
        layers.RandomContrast(0.1, seed=seed),
    ], name="augmentation")


def make_tf_dataset(images: np.ndarray, labels: np.ndarray, batch_size: int = 32,
                    shuffle: bool = False, seed: int = config.SEED):
    """Batched ``tf.data`` pipeline of (float32 0-255 image, int label).

    Arrays stay uint8 in memory (about 160 MB for the 1,080 training images)
    and are cast per batch. Augmentation lives inside the model, so
    validation/test pipelines are never augmented.
    """
    import tensorflow as tf

    images = np.asarray(images)
    labels = np.asarray(labels).reshape(-1).astype(np.int32)
    if len(images) != len(labels):
        raise ValueError("images and labels differ in length")
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    if shuffle:
        ds = ds.shuffle(len(images), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).map(
        lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)
