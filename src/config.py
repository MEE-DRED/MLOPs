"""Central configuration.

Every path and tunable value lives here and can be overridden with an
environment variable, so the same code runs on a laptop, in Google Colab
(paths pointed at Google Drive) and inside a Docker container.

IMPORTANT: this module reads environment variables at import time. In the
Colab notebook the DR_* variables are therefore set *before* ``import src.config``.
No TensorFlow import happens here.
"""
from __future__ import annotations

import os
from pathlib import Path


def _path(env_name: str, default: Path) -> Path:
    return Path(os.environ.get(env_name, str(default))).expanduser()


def _float(env_name: str, default: float) -> float:
    return float(os.environ.get(env_name, default))


def _int(env_name: str, default: int) -> int:
    return int(os.environ.get(env_name, default))


# ----------------------------------------------------------------- paths
PROJECT_ROOT = _path("DR_PROJECT_ROOT", Path(__file__).resolve().parents[1])
DATA_DIR = _path("DR_DATA_DIR", PROJECT_ROOT / "data")
DATASET_NPZ = _path("DR_DATASET_NPZ", DATA_DIR / "retinamnist_224.npz")
RETRAIN_UPLOAD_DIR = _path("DR_UPLOAD_DIR", DATA_DIR / "retraining_uploads")
MODELS_DIR = _path("DR_MODELS_DIR", PROJECT_ROOT / "models")
METADATA_DIR = MODELS_DIR / "metadata"
DB_PATH = _path("DR_DB_PATH", PROJECT_ROOT / "database" / "dr_mlops.sqlite3")
OUTPUTS_DIR = _path("DR_OUTPUTS_DIR", PROJECT_ROOT / "outputs")
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"
TABLES_DIR = OUTPUTS_DIR / "tables"

# ----------------------------------------------------------------- data
IMG_SIZE = 224
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
NUM_CLASSES = 5
CLASS_NAMES = (
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR",
)
SPLITS = ("train", "val", "test")
SEED = _int("DR_SEED", 42)

ALLOWED_IMAGE_FORMATS = ("JPEG", "PNG")      # PIL format names
ALLOWED_EXTENSIONS = (".jpg", ".jpeg", ".png")
MAX_UPLOAD_BYTES = _int("DR_MAX_UPLOAD_BYTES", 10 * 1024 * 1024)   # 10 MB / file
MIN_IMAGE_SIDE = 32
MAX_IMAGE_PIXELS = 50_000_000

# --------------------------------------------- model acceptance policy
# See src/evaluate.py::AcceptancePolicy and docs/architecture.md.
ACCEPT_MIN_DELTA_MACRO_F1 = _float("DR_ACCEPT_MIN_DELTA_F1", 0.0)
ACCEPT_MAX_ACCURACY_DROP = _float("DR_ACCEPT_MAX_ACC_DROP", 0.02)
ACCEPT_MAX_CLASS_RECALL_DROP = _float("DR_ACCEPT_MAX_RECALL_DROP", 0.35)
ACCEPT_MIN_CLASS_SUPPORT = _int("DR_ACCEPT_MIN_SUPPORT", 5)
ACCEPT_MIN_EVAL_SAMPLES = _int("DR_ACCEPT_MIN_EVAL", 30)

# ------------------------------- retraining trigger (used from Phase 3)
AUTO_RETRAIN_ENABLED = os.environ.get("DR_AUTO_RETRAIN", "false").lower() == "true"
AUTO_RETRAIN_MIN_NEW_IMAGES = _int("DR_AUTO_RETRAIN_MIN_NEW", 50)
AUTO_RETRAIN_COOLDOWN_MINUTES = _int("DR_AUTO_RETRAIN_COOLDOWN_MIN", 60)


# --------------------------------------------------- retraining settings
RETRAIN_MIN_IMAGES = _int("DR_RETRAIN_MIN_IMAGES", 20)        # valid labelled uploads needed
RETRAIN_MIN_CLASSES = _int("DR_RETRAIN_MIN_CLASSES", 2)       # distinct classes needed
RETRAIN_REPLAY_SIZE = _int("DR_RETRAIN_REPLAY_SIZE", 300)     # original training images mixed in
RETRAIN_EPOCHS = _int("DR_RETRAIN_EPOCHS", 8)
RETRAIN_LEARNING_RATE = _float("DR_RETRAIN_LR", 1e-5)
RETRAIN_UNFREEZE_LAYERS = _int("DR_RETRAIN_UNFREEZE", 20)
RETRAIN_BATCH_SIZE = _int("DR_RETRAIN_BATCH", 16)
RETRAIN_PATIENCE = _int("DR_RETRAIN_PATIENCE", 3)

# Optional shared secret for state-changing endpoints (upload/retrain/rollback).
API_KEY = os.environ.get("DR_API_KEY", "")

DISCLAIMER = ("Experimental academic demonstration. Not a medical device and not a diagnosis. "
              "Do not use for any medical decision.")


def class_name(index: int) -> str:
    """Human-readable name for a class index."""
    return CLASS_NAMES[int(index)]
