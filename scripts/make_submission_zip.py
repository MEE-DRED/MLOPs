"""Build the submission ZIP of the code repository.

    python -m scripts.make_submission_zip --out /path/Diabetic-Retinopathy-MLOps_submission.zip

Guarantees:
* the trained model is included (refuses otherwise - the rubric requires "script + model file"; use --allow-no-model only for drafts);
* the raw dataset, databases, caches, state, checkpoints, uploads, credentials (.env) are NOT included;
* text files are scanned for credential-looking strings; any hit aborts the build;
* the layout required by the assignment brief is checked.
"""
from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE_DIRS = {"__pycache__", ".git", ".ipynb_checkpoints", ".venv", "state", "checkpoints", "staging", "dr_mlops_workdir", ".pytest_cache"}
EXCLUDE_SUFFIX = (".pyc", ".npz", ".sqlite3", ".sqlite3-wal", ".sqlite3-shm", ".tmp", ".part")
EXCLUDE_NAMES = {".env", ".DS_Store"}
SECRET_PATTERNS = {
    "Hugging Face token": re.compile(r"hf_[A-Za-z0-9]{20,}"),
    "GitHub token": re.compile(r"(ghp|gho|ghs|github_pat)_[A-Za-z0-9_]{20,}"),
    "AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "hard-coded secret": re.compile(r"""(?i)\b(api[_-]?key|secret|password|token)\b\s*[:=]\s*['"][A-Za-z0-9_\-]{12,}['"]"""),
}
REQUIRED = ["README.md", "src/preprocessing.py", "src/model.py", "src/prediction.py", "src/retrain.py", "app/dashboard.py", "src/api.py",
            "Dockerfile", "requirements.txt", "docs/deployment.md", "data"]     # models/ is enforced by the model check above


class SubmissionError(Exception):
    pass


def _included(repo: Path):
    for p in sorted(repo.rglob("*")):
        rel = p.relative_to(repo)
        if not p.is_file() or EXCLUDE_DIRS & set(rel.parts):
            continue
        if p.name in EXCLUDE_NAMES or p.name.endswith(EXCLUDE_SUFFIX) or p.name == ".env":
            continue
        if rel.parts[:2] == ("data", "retraining_uploads") and p.name != ".gitkeep":
            continue
        yield p, rel


def _scan_secrets(repo: Path, files) -> list[str]:
    hits = []
    for p, rel in files:
        if p.suffix in (".png", ".jpg", ".jpeg", ".keras", ".csv", ".zip") or p.stat().st_size > 3_000_000:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for name, pat in SECRET_PATTERNS.items():
            m = pat.search(text)
            if m:
                hits.append(f"{rel}: looks like a {name}")
    return hits


def build_submission_zip(out_zip: Path, repo: Path = ROOT, allow_no_model: bool = False, top_name: str = "Diabetic-Retinopathy-MLOps") -> dict:
    repo, out_zip = Path(repo), Path(out_zip)
    files = list(_included(repo))
    rels = {str(r).replace("\\", "/") for _, r in files}

    # model present and consistent
    model_ok, model_msg = False, "models/current_model.json missing"
    ptr = repo / "models" / "current_model.json"
    if ptr.is_file():
        try:
            v = json.loads(ptr.read_text())["current_version"]
            model_ok = (repo / "models" / f"model_v{v}.keras").is_file() and (repo / "models" / "metadata" / f"model_v{v}.json").is_file()
            model_msg = f"current version v{v}" if model_ok else f"current_model.json points to v{v} but its .keras/metadata file is missing"
        except Exception as exc:
            model_msg = f"current_model.json unreadable: {exc}"
    if not model_ok and not allow_no_model:
        raise SubmissionError(f"No usable trained model in the repository ({model_msg}). Copy models/ (model_v*.keras, metadata/, current_model.json) "
                              "from Google Drive first - the rubric needs the model file. (--allow-no-model is for drafts only.)")

    missing = [r for r in REQUIRED if not any(x == r or x.startswith(r + "/") for x in rels)]
    if not any(x.startswith("notebook/") and x.endswith(".ipynb") for x in rels):
        missing.append("notebook/*.ipynb")
    if missing:
        raise SubmissionError(f"Required files/folders missing: {missing}")
    hits = _scan_secrets(repo, files)
    if hits:
        raise SubmissionError("Possible credentials found - remove them before submitting:\n  " + "\n  ".join(hits))

    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for p, rel in files:
            z.write(p, f"{top_name}/{rel}")
        for d in ("data/train", "data/validation", "data/test", "models", "models/metadata"):   # keep empty-folder layout
            if not any(str(r).replace('\\', '/').startswith(d + "/") for _, r in files):
                z.writestr(f"{top_name}/{d}/.gitkeep", "")
    size = out_zip.stat().st_size
    return {"zip": str(out_zip), "n_files": len(files), "size_mb": round(size / 1e6, 2), "model": model_msg if model_ok else "NOT INCLUDED (draft)",
            "largest": sorted(((round(p.stat().st_size / 1e6, 1), str(r)) for p, r in files), reverse=True)[:5]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default=str(ROOT))
    ap.add_argument("--allow-no-model", action="store_true")
    a = ap.parse_args()
    try:
        print(json.dumps(build_submission_zip(Path(a.out), Path(a.repo), a.allow_no_model), indent=2))
    except SubmissionError as exc:
        raise SystemExit(f"ERROR: {exc}")


if __name__ == "__main__":
    main()
