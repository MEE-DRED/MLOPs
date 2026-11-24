"""Dataset acquisition and validation for RetinaMNIST-224 (MedMNIST).

Nothing here needs TensorFlow, so it can be unit-tested anywhere.

The official file is an ``.npz`` archive with the keys
``{train,val,test}_{images,labels}``. Images are ``uint8`` arrays of shape
(N, 224, 224, 3); labels have shape (N, 1) with values 0-4.
"""
from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from . import config

SplitData = Dict[str, Tuple[np.ndarray, np.ndarray]]

# Direct download from the MedMNIST Zenodo record. NOTE: this URL was written
# from memory and has NOT been verified from the build sandbox (no network).
# The ``medmnist`` package route below is the officially supported one.
ZENODO_URL = "https://zenodo.org/records/10519652/files/retinamnist_224.npz?download=1"


class DatasetError(Exception):
    """Raised when the dataset file is missing or structurally invalid."""


# ------------------------------------------------------------- loading
def load_retinamnist(path: Path | str = config.DATASET_NPZ,
                     splits: Iterable[str] = config.SPLITS) -> SplitData:
    """Load the requested splits. Labels are flattened to 1-D ``int64``.

    ``np.load`` on an ``.npz`` is lazy per key, so requesting only
    ``("test",)`` does not read the training images into memory.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset file not found: {path}. See data/README.md or run "
            "src.data_acquisition.ensure_dataset()."
        )
    data: SplitData = {}
    try:
        with np.load(path) as archive:
            for split in splits:
                for suffix in ("images", "labels"):
                    key = f"{split}_{suffix}"
                    if key not in archive.files:
                        raise DatasetError(
                            f"Key '{key}' missing from {path.name}; found {archive.files}")
                images = archive[f"{split}_images"]
                labels = archive[f"{split}_labels"].reshape(-1).astype(np.int64)
                data[split] = (images, labels)
    except (ValueError, OSError) as exc:      # corrupt / not an npz
        raise DatasetError(f"Could not read {path}: {exc}") from exc
    return data


def file_sha256(path: Path | str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------- validation
def _hashes(images: np.ndarray) -> list[str]:
    return [hashlib.md5(img.tobytes()).hexdigest() for img in images]


def validate_dataset(data: SplitData, num_classes: int = config.NUM_CLASSES,
                     expected_shape: tuple = config.IMG_SHAPE) -> dict:
    """Check structure, value ranges, labels, blanks and duplicates/leakage.

    Returns a JSON-serialisable report. ``report["ok"]`` is False if any
    *blocking* issue was found; non-blocking observations go in ``warnings``.
    """
    issues: list[str] = []
    warnings: list[str] = []
    report: dict = {"splits": {}, "cross_split_duplicates": {}}
    hashes: dict[str, list[str]] = {}

    for split, (images, labels) in data.items():
        info: dict = {"n_images": int(len(images)), "n_labels": int(len(labels))}
        if images.ndim != 4 or tuple(images.shape[1:]) != tuple(expected_shape):
            issues.append(f"{split}: image shape {images.shape} != (N,{expected_shape})")
        if images.dtype != np.uint8:
            issues.append(f"{split}: dtype {images.dtype} != uint8")
        if len(images) != len(labels):
            issues.append(f"{split}: {len(images)} images but {len(labels)} labels")
        if len(labels) and (labels.min() < 0 or labels.max() >= num_classes):
            issues.append(f"{split}: labels outside 0..{num_classes - 1}")
        counts = np.bincount(np.clip(labels, 0, num_classes - 1), minlength=num_classes)
        info["class_counts"] = counts.tolist()
        absent = [i for i, c in enumerate(counts) if c == 0]
        if absent:
            warnings.append(f"{split}: no samples for classes {absent}")
        if images.ndim == 4 and len(images):
            flat_max = images.reshape(len(images), -1).max(axis=1)
            info["blank_images"] = int((flat_max == 0).sum())
            if info["blank_images"]:
                issues.append(f"{split}: {info['blank_images']} completely black images")
            hashes[split] = _hashes(images)
            info["duplicates_within_split"] = len(hashes[split]) - len(set(hashes[split]))
            if info["duplicates_within_split"]:
                warnings.append(f"{split}: {info['duplicates_within_split']} duplicate images")
        report["splits"][split] = info

    names = list(hashes)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = len(set(hashes[a]) & set(hashes[b]))
            report["cross_split_duplicates"][f"{a}&{b}"] = overlap
            if overlap:
                issues.append(f"DATA LEAKAGE: {overlap} identical images in {a} and {b}")

    report["issues"] = issues
    report["warnings"] = warnings
    report["ok"] = not issues
    return report


def class_distribution_table(data: SplitData, class_names=config.CLASS_NAMES):
    """Counts and percentages per class and split as a pandas DataFrame."""
    import pandas as pd

    rows = []
    for split, (_, labels) in data.items():
        counts = np.bincount(labels, minlength=len(class_names))
        for idx, name in enumerate(class_names):
            rows.append({"split": split, "class_index": idx, "class": name,
                         "count": int(counts[idx]),
                         "percent": round(100 * counts[idx] / max(len(labels), 1), 2)})
    return pd.DataFrame(rows)


# --------------------------------------------------------- acquisition
def download_with_medmnist(root: Path | str, size: int = 224) -> Path:
    """Official route: let the ``medmnist`` package download and verify the file."""
    try:
        from medmnist import RetinaMNIST
    except ImportError as exc:
        raise DatasetError("`pip install medmnist` first") from exc
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    RetinaMNIST(split="train", download=True, size=size, root=str(root))
    path = root / f"retinamnist_{size}.npz"
    if not path.is_file():
        raise DatasetError(f"medmnist finished but {path} does not exist")
    return path


def _looks_valid(path: Path) -> bool:
    try:
        with np.load(path) as z:
            return all(f"{s}_{k}" in z.files for s in config.SPLITS for k in ("images", "labels"))
    except Exception:
        return False


def ensure_dataset(dest: Path | str = config.DATASET_NPZ,
                   search_paths: Iterable[Path | str] = (),
                   try_medmnist: bool = True,
                   try_direct_url: bool = True,
                   log=print) -> Path:
    """Make sure ``dest`` exists, trying several sources in order.

    1. ``dest`` itself.
    2. Any file in ``search_paths`` (e.g. a copy on Google Drive) is copied.
    3. The ``medmnist`` package (official).
    4. Direct download from Zenodo (URL unverified, see ``ZENODO_URL``).

    Raises ``DatasetError`` with manual instructions if all fail.
    """
    dest = Path(dest)
    if dest.is_file() and _looks_valid(dest):
        log(f"Dataset found: {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)

    for candidate in map(Path, search_paths):
        if candidate.is_file() and _looks_valid(candidate):
            log(f"Copying dataset from {candidate}")
            shutil.copy2(candidate, dest)
            return dest

    if try_medmnist:
        try:
            log("Trying the official medmnist package ...")
            got = download_with_medmnist(dest.parent)
            if got != dest:
                shutil.copy2(got, dest)
            if _looks_valid(dest):
                return dest
        except Exception as exc:                       # network / not installed
            log(f"  medmnist route failed: {exc}")

    if try_direct_url:
        tmp = dest.with_suffix(".part")
        try:
            log(f"Trying direct download: {ZENODO_URL}")
            urllib.request.urlretrieve(ZENODO_URL, tmp)
            if _looks_valid(tmp):
                tmp.replace(dest)
                return dest
            log("  downloaded file is not a valid RetinaMNIST npz")
        except Exception as exc:
            log(f"  direct download failed: {exc}")
        finally:
            tmp.unlink(missing_ok=True)

    raise DatasetError(
        f"Could not obtain {dest.name}. Put the file at {dest} (or in your Google "
        "Drive project folder) manually; download it from https://medmnist.com/ "
        "(RetinaMNIST, size 224).")
