"""Container/Colab start-up bootstrap (pure Python, no TensorFlow).

* Creates the state folders (models, database, uploads) under DR_STATE_DIR.
* Seeds the registry from the model files baked into the image (or DR_BAKED_MODELS_DIR) the FIRST time only:
  existing versions in the state dir are never overwritten, so persistent storage keeps retrained versions.
* Reports - but never crashes on - a missing dataset: predictions work without it; retraining/shift indicators need it.

Usage:  python -m scripts.prepare_runtime   (prints a JSON status; exit code is always 0)
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def prepare(state_dir: Path, baked_models_dir: Path, dataset_path: Path, auto_download: bool = False,
            log=print) -> dict:
    state_dir, baked_models_dir, dataset_path = Path(state_dir), Path(baked_models_dir), Path(dataset_path)
    models = state_dir / "models"
    for d in (models / "metadata", state_dir / "database", state_dir / "uploads"):
        d.mkdir(parents=True, exist_ok=True)

    pointer = models / "current_model.json"
    seeded = []
    if not pointer.exists() and (baked_models_dir / "current_model.json").exists():
        for f in sorted(baked_models_dir.glob("model_v*.keras")):
            if not (models / f.name).exists():
                shutil.copy2(f, models / f.name); seeded.append(f.name)
        for f in sorted((baked_models_dir / "metadata").glob("model_v*.json")):
            if not (models / "metadata" / f.name).exists():
                shutil.copy2(f, models / "metadata" / f.name)
        shutil.copy2(baked_models_dir / "current_model.json", pointer)      # pointer last
        log(f"Seeded registry from {baked_models_dir}: {seeded}")

    dataset_message = "present"
    if not dataset_path.is_file():
        dataset_message = "missing"
        if auto_download:
            try:
                from src.data_acquisition import ensure_dataset
                ensure_dataset(dataset_path, try_medmnist=False, try_direct_url=True, log=log)
                dataset_message = "downloaded"
            except Exception as exc:
                dataset_message = f"missing (download failed: {exc})"
    if not dataset_path.is_file():
        log("WARNING: original dataset not found. Predictions work; retraining and shift indicators are disabled "
            f"until {dataset_path} exists.")

    return {"state_dir": str(state_dir), "models_seeded": seeded,
            "current_model_present": pointer.exists(),
            "model_files": sorted(p.name for p in models.glob("model_v*.keras")),
            "dataset_present": dataset_path.is_file(), "dataset": dataset_message}


def main() -> None:
    state = Path(os.environ.get("DR_STATE_DIR", ROOT / "state"))
    baked = Path(os.environ.get("DR_BAKED_MODELS_DIR", ROOT / "models"))
    dataset = Path(os.environ.get("DR_DATASET_NPZ", ROOT / "data" / "retinamnist_224.npz"))
    auto = os.environ.get("DR_AUTO_DOWNLOAD_DATASET", "false").lower() == "true"
    status = prepare(state, baked, dataset, auto)
    print(json.dumps(status, indent=2))
    if not status["current_model_present"]:
        print("WARNING: no model in the registry. The API will start in 'degraded' mode. "
              "Copy models/ (model_v*.keras, metadata/, current_model.json) from your Colab/Drive run into the repo.")


if __name__ == "__main__":
    main()
