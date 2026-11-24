"""Validation and storage of labelled images uploaded for retraining.

Rules (supervised learning needs correct labels):
* an image without a valid label 0-4 is rejected, never stored;
* unreadable / unsupported / corrupt files are rejected;
* byte-identical re-uploads are rejected (duplicates are not new data);
* images identical to a validation or test image are rejected (no leakage);
  images identical to an original training image are rejected (not new data).
Every outcome - accepted or rejected - is recorded in the database.
"""
from __future__ import annotations

import hashlib
import os
import re
import uuid
from pathlib import Path
from typing import Optional, Sequence

from . import config
from .database import Database, DuplicateUploadError
from .image_analysis import compute_image_features
from .preprocessing import InvalidImageError, InvalidLabelError, load_image, validate_label
from .reference import array_hash


def safe_filename(name: Optional[str]) -> str:
    """Display-only name: no paths, no control characters, bounded length."""
    base = os.path.basename(str(name or "unnamed").replace("\\", "/"))
    base = re.sub(r"[^\w.\- ]", "_", base)[:120]
    return base or "unnamed"


def _extension(data: bytes) -> str:
    return ".png" if data[:8] == b"\x89PNG\r\n\x1a\n" else ".jpg"


def ingest_upload(db: Database, *, data: bytes, filename: str, label, batch_id: str,
                  upload_dir: Path | str = config.RETRAIN_UPLOAD_DIR,
                  hash_index: Optional[dict] = None) -> dict:
    """Validate and store one labelled image. Returns a per-file result dict."""
    name = safe_filename(filename)

    def reject(reason: str, lab=None, sha=None) -> dict:
        db.add_rejected_upload(batch_id=batch_id, filename=name, label=lab, reason=reason, sha256=sha)
        return {"filename": name, "status": "rejected", "reason": reason, "label": lab}

    try:
        lab = validate_label(label)
    except InvalidLabelError as exc:
        return reject(f"invalid label: {exc}")
    try:
        img = load_image(data)
    except InvalidImageError as exc:
        return reject(f"invalid image: {exc}", lab)

    sha = hashlib.sha256(data).hexdigest()
    if hash_index is not None:
        where = hash_index.get(array_hash(img))
        if where in ("val", "test"):
            return reject(f"identical to an original {'validation' if where == 'val' else 'test'} image "
                          "(not allowed in retraining data)", lab, sha)
        if where == "train":
            return reject("identical to an image already in the original training data", lab, sha)

    feats = {k: float(v[0]) for k, v in compute_image_features(img).items()}
    folder = Path(upload_dir) / f"class_{lab}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{sha[:32]}{_extension(data)}"
    try:
        tmp = path.with_name(path.name + f".{uuid.uuid4().hex}.part")
        tmp.write_bytes(data)
        os.replace(tmp, path)
        upload_id = db.add_upload(batch_id=batch_id, filename=name, stored_path=path, sha256=sha,
                                  label=lab, features=feats)
    except DuplicateUploadError:
        return reject("duplicate: this exact image was already uploaded", lab, sha)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return {"filename": name, "status": "accepted", "label": lab, "id": upload_id}


def parse_labels(raw, n_files: int) -> list:
    """Accept a JSON list, a comma-separated string or a python list; must match ``n_files``."""
    import json
    if isinstance(raw, (list, tuple)):
        labels = list(raw)
    else:
        text = (raw or "").strip()
        if text.startswith("["):
            try:
                labels = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"labels is not valid JSON: {exc}") from exc
        else:
            labels = [t.strip() for t in text.split(",")] if text else []
    if len(labels) != n_files:
        raise ValueError(f"Got {n_files} file(s) but {len(labels)} label(s); every image needs exactly one label")
    return labels


def ingest_batch(db: Database, files: Sequence[tuple[str, bytes]], labels: Sequence, *,
                 upload_dir: Path | str = config.RETRAIN_UPLOAD_DIR, hash_index: Optional[dict] = None) -> dict:
    batch_id = uuid.uuid4().hex[:12]
    results = [ingest_upload(db, data=data, filename=fn, label=lab, batch_id=batch_id,
                             upload_dir=upload_dir, hash_index=hash_index)
               for (fn, data), lab in zip(files, labels)]
    return {"batch_id": batch_id, "n_files": len(results),
            "n_accepted": sum(r["status"] == "accepted" for r in results),
            "n_rejected": sum(r["status"] == "rejected" for r in results), "results": results}
