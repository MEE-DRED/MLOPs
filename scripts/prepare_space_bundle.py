"""Assemble a clean folder to push to a Hugging Face Docker Space (git repo of the Space).

    python -m scripts.prepare_space_bundle --out ../dr-space --dataset /path/to/retinamnist_224.npz

Includes only what the container needs (not notebooks/tests/docs), the trained models/, the dataset (if given - it is NOT
committed to GitHub, only to the Space repo) and a Space README with the required YAML header. Large files are tracked
with Git LFS via .gitattributes. Refuses to build without a registered model unless --allow-no-model.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ["Dockerfile", "requirements-docker.txt"]
DIRS = ["src", "app", "scripts"]
SPACE_DOCKERIGNORE = ".git\n**/__pycache__\n**/*.pyc\ndata/retraining_uploads/*\nstate/\n.env\n"
GITATTRIBUTES = "*.npz filter=lfs diff=lfs merge=lfs -text\n*.keras filter=lfs diff=lfs merge=lfs -text\n"
_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "*.sqlite3*")


class BundleError(Exception):
    pass


def build_bundle(out: Path, dataset: Path | None = None, repo: Path = ROOT, allow_no_model: bool = False) -> dict:
    out = Path(out)
    models = repo / "models"
    if not (models / "current_model.json").is_file() and not allow_no_model:
        raise BundleError(
            "No trained model found: models/current_model.json is missing. Copy models/ (model_v*.keras, metadata/, "
            "current_model.json) from your Colab/Drive run into the repo first (or pass --allow-no-model).")
    if not (repo / "outputs" / "tables" / "image_features.csv").is_file():
        raise BundleError("outputs/tables/image_features.csv is missing (needed for the dashboard's dataset insights).")
    if out.exists() and any(out.iterdir()):
        raise BundleError(f"{out} already exists and is not empty; choose a new folder.")
    out.mkdir(parents=True, exist_ok=True)

    for f in FILES:
        shutil.copy2(repo / f, out / f)
    for d in DIRS:
        shutil.copytree(repo / d, out / d, ignore=_IGNORE)
    (out / "outputs" / "tables").mkdir(parents=True)
    shutil.copy2(repo / "outputs" / "tables" / "image_features.csv", out / "outputs" / "tables" / "image_features.csv")
    shutil.copytree(models, out / "models", ignore=_IGNORE)
    (out / "data").mkdir()
    (out / "data" / "README.md").write_text("Original RetinaMNIST-224 data for validation/replay during retraining.\n")
    if dataset is not None:
        dataset = Path(dataset)
        if not dataset.is_file():
            raise BundleError(f"dataset file not found: {dataset}")
        shutil.copy2(dataset, out / "data" / "retinamnist_224.npz")
    shutil.copy2(repo / "deploy" / "huggingface" / "README_SPACE.md", out / "README.md")
    (out / ".dockerignore").write_text(SPACE_DOCKERIGNORE)
    (out / ".gitattributes").write_text(GITATTRIBUTES)

    files = sorted(p for p in out.rglob("*") if p.is_file())
    manifest = {"out": str(out), "n_files": len(files),
                "total_mb": round(sum(p.stat().st_size for p in files) / 1e6, 1),
                "large_files_mb": {str(p.relative_to(out)): round(p.stat().st_size / 1e6, 1)
                                   for p in files if p.stat().st_size > 5e6},
                "has_model": (out / "models" / "current_model.json").is_file(),
                "has_dataset": (out / "data" / "retinamnist_224.npz").is_file()}
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset")
    ap.add_argument("--allow-no-model", action="store_true")
    a = ap.parse_args()
    try:
        print(json.dumps(build_bundle(Path(a.out), Path(a.dataset) if a.dataset else None,
                                      allow_no_model=a.allow_no_model), indent=2))
    except BundleError as exc:
        raise SystemExit(f"ERROR: {exc}")


if __name__ == "__main__":
    main()
