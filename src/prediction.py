"""Single-image inference shared by the notebook, the API and the dashboard."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

import numpy as np

from . import config
from .model_registry import ModelRegistry
from .preprocessing import InvalidImageError, load_image, to_model_input


class ModelNotLoadedError(RuntimeError):
    pass


class Predictor:
    """Wraps a Keras model (or any object with ``predict``) for inference.

    A model can be injected (used by the unit tests); otherwise use
    ``Predictor.from_registry`` which loads the *current accepted* version.
    """

    def __init__(self, model=None, version: Optional[int] = None,
                 class_names=config.CLASS_NAMES):
        self.model = model
        self.version = version
        self.class_names = tuple(class_names)
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    @classmethod
    def from_registry(cls, registry: Optional[ModelRegistry] = None) -> "Predictor":
        registry = registry or ModelRegistry()
        version = registry.current_version()
        if version is None:
            raise ModelNotLoadedError("No current model in the registry")
        return cls.load_path(registry.model_path(version), version)

    @classmethod
    def load_path(cls, path: Path | str, version: Optional[int] = None) -> "Predictor":
        from tensorflow import keras          # lazy: keeps this module importable without TF
        model = keras.models.load_model(str(path), compile=False)
        return cls(model=model, version=version)

    def predict_array(self, image_uint8: np.ndarray) -> dict:
        if not self.is_loaded:
            raise ModelNotLoadedError("Model is not loaded")
        batch = to_model_input(image_uint8)
        with self._lock:
            probs = np.asarray(self.model.predict(batch, verbose=0))[0].astype(float)
        total = probs.sum()
        probs = probs / total if total > 0 else probs
        idx = int(probs.argmax())
        return {
            "predicted_class": idx,
            "predicted_class_name": self.class_names[idx],
            "confidence": float(probs[idx]),
            "probabilities": {name: float(p) for name, p in zip(self.class_names, probs)},
            "model_version": self.version,
        }

    def predict_bytes(self, data: bytes) -> dict:
        """Validate + preprocess an uploaded file, then predict.

        Raises ``InvalidImageError`` for anything that is not a usable image.
        """
        return self.predict_array(load_image(data))
