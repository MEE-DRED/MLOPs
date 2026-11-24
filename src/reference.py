"""Access to the ORIGINAL RetinaMNIST data for retraining (replay + validation) and leakage checks.

``ArrayReference`` wraps in-memory arrays (used by tests). ``NpzReference``
loads the real file lazily. Neither exposes the test images to training:
the test split is only ever *hashed* so uploads identical to a test image
can be rejected.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import numpy as np

from . import config
from .data_acquisition import load_retinamnist
from .image_analysis import compute_image_features


def array_hash(img: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(img, dtype=np.uint8).tobytes()).hexdigest()


class ArrayReference:
    def __init__(self, train, val, test=None):
        self._train, self._val, self._test = train, val, test
        self._index: Optional[dict] = None
        self._features: Optional[dict] = None

    def available(self) -> bool:
        return True

    def train(self):
        return self._train

    def val(self):
        return self._val

    def hash_index(self) -> dict[str, str]:
        """md5(pixel bytes) -> split name, for train/val/test."""
        if self._index is None:
            idx: dict[str, str] = {}
            for name, part in (("train", self._train), ("val", self._val), ("test", self._test)):
                if part is not None:
                    for img in part[0]:
                        idx[array_hash(img)] = name
            self._index = idx
        return self._index

    def train_class_counts(self) -> list[int]:
        return np.bincount(self._train[1], minlength=config.NUM_CLASSES).tolist()

    def reference_features(self) -> dict[str, np.ndarray]:
        if self._features is None:
            self._features = compute_image_features(self._train[0])
        return self._features


class NpzReference(ArrayReference):
    """Lazily reads ``retinamnist_224.npz``. ``available()`` is False if the file is missing."""

    def __init__(self, path: Path | str = config.DATASET_NPZ):
        super().__init__(None, None, None)
        self.path = Path(path)
        self._loaded = False
        self._test_hashes: Optional[dict] = None

    def available(self) -> bool:
        return self.path.is_file()

    def _load(self):
        if not self._loaded:
            data = load_retinamnist(self.path, splits=("train", "val"))
            self._train, self._val = data["train"], data["val"]
            self._loaded = True

    def train(self):
        self._load()
        return self._train

    def val(self):
        self._load()
        return self._val

    def hash_index(self) -> dict[str, str]:
        if self._index is None:
            self._load()
            idx = {array_hash(i): "train" for i in self._train[0]}
            idx.update({array_hash(i): "val" for i in self._val[0]})
            test_x = load_retinamnist(self.path, splits=("test",))["test"][0]
            idx.update({array_hash(i): "test" for i in test_x})
            del test_x                                   # never kept in memory
            self._index = idx
        return self._index

    def train_class_counts(self) -> list[int]:
        self._load()
        return super().train_class_counts()

    def reference_features(self):
        self._load()
        return super().reference_features()
