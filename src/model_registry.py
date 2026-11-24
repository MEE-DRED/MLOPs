"""File-based model registry with versioning, acceptance status and rollback.

Layout (under ``config.MODELS_DIR``)::

    model_v1.keras  model_v2.keras ...
    metadata/model_v1.json  metadata/model_v2.json ...
    current_model.json      <- pointer to the version the API serves

Rules:
* Version numbers only ever increase; files are never overwritten or deleted.
* Rejected candidates are kept (status "rejected") for the audit trail but
  can never become current.
* Files are written to a temp name then atomically renamed, so a crash cannot
  leave a half-written model or pointer.
* ``current_model.json`` keeps a history stack that ``rollback`` pops.

No TensorFlow import: validation of a model's *contents* is done by the
caller (see ``evaluate.model_integrity_check``) and passed in as a validator.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from . import config

SERVABLE = ("accepted",)
_VERSION_RE = re.compile(r"model_v(\d+)\.keras$")


class RegistryError(Exception):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


class ModelRegistry:
    def __init__(self, models_dir: Path | str = config.MODELS_DIR):
        self.models_dir = Path(models_dir)
        self.metadata_dir = self.models_dir / "metadata"
        self.pointer_path = self.models_dir / "current_model.json"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    # ------------------------------------------------------------ paths
    def model_path(self, version: int) -> Path:
        return self.models_dir / f"model_v{int(version)}.keras"

    def metadata_path(self, version: int) -> Path:
        return self.metadata_dir / f"model_v{int(version)}.json"

    # --------------------------------------------------------- queries
    def _existing_numbers(self) -> list[int]:
        nums = set()
        for p in self.models_dir.glob("model_v*.keras"):
            m = _VERSION_RE.search(p.name)
            if m:
                nums.add(int(m.group(1)))
        for p in self.metadata_dir.glob("model_v*.json"):
            m = re.search(r"model_v(\d+)\.json$", p.name)
            if m:
                nums.add(int(m.group(1)))
        return sorted(nums)

    def next_version(self) -> int:
        nums = self._existing_numbers()
        return (nums[-1] + 1) if nums else 1

    def get_metadata(self, version: int) -> dict:
        path = self.metadata_path(version)
        if not path.is_file():
            raise RegistryError(f"No metadata for version {version}")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RegistryError(f"Metadata for v{version} is corrupt: {exc}") from exc

    def list_versions(self) -> list[dict]:
        out = []
        for n in self._existing_numbers():
            try:
                out.append(self.get_metadata(n))
            except RegistryError:
                out.append({"version": n, "status": "unknown", "error": "metadata missing or corrupt"})
        return out

    def _read_pointer(self) -> Optional[dict]:
        if not self.pointer_path.is_file():
            return None
        try:
            return json.loads(self.pointer_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RegistryError(f"current_model.json is corrupt: {exc}") from exc

    def current_version(self) -> Optional[int]:
        ptr = self._read_pointer()
        return int(ptr["current_version"]) if ptr else None

    def current_model_path(self) -> Optional[Path]:
        v = self.current_version()
        return self.model_path(v) if v is not None else None

    def get_current_metadata(self) -> Optional[dict]:
        v = self.current_version()
        return self.get_metadata(v) if v is not None else None

    # ------------------------------------------------------- mutation
    def register_version(self, model_file: Path | str, metadata: dict,
                         status: str, make_current: bool = False,
                         reason: str = "") -> dict:
        """Store a trained model as the next version.

        ``status`` is "accepted" or "rejected". Only accepted versions may be
        made current. The source file is copied, not moved.
        """
        if status not in ("accepted", "rejected"):
            raise RegistryError("status must be 'accepted' or 'rejected'")
        if make_current and status != "accepted":
            raise RegistryError("A rejected model can never become current")
        model_file = Path(model_file)
        if not model_file.is_file() or model_file.stat().st_size == 0:
            raise RegistryError(f"Model file missing or empty: {model_file}")
        with self._lock:
            version = self.next_version()
            final = self.model_path(version)
            tmp = final.with_name(final.name + ".tmp")
            shutil.copyfile(model_file, tmp)
            os.replace(tmp, final)
            meta = dict(metadata)
            meta.update({
                "version": version,
                "created_at": _now(),
                "status": status,
                "model_file": final.name,
                "model_size_bytes": final.stat().st_size,
            })
            _atomic_write_json(self.metadata_path(version), meta)
            if make_current:
                self.set_current(version, reason=reason or "registered as current")
            return meta

    def set_current(self, version: int, reason: str = "",
                    validator: Optional[Callable[[Path], tuple[bool, str]]] = None) -> None:
        """Point the service at ``version`` (must be accepted and present).

        ``validator(model_path) -> (ok, message)`` may load the model and run
        an integrity check *before* the pointer is switched.
        """
        with self._lock:
            path = self.model_path(version)
            if not path.is_file() or path.stat().st_size == 0:
                raise RegistryError(f"Model file for v{version} not found")
            meta = self.get_metadata(version)
            if meta.get("status") not in SERVABLE:
                raise RegistryError(f"v{version} has status '{meta.get('status')}' and cannot be served")
            if validator is not None:
                ok, msg = validator(path)
                if not ok:
                    raise RegistryError(f"v{version} failed validation: {msg}")
            ptr = self._read_pointer() or {"current_version": None, "history": [], "events": []}
            previous = ptr.get("current_version")
            if previous is not None and previous != version:
                ptr["history"].append(previous)
            ptr["current_version"] = int(version)
            ptr["updated_at"] = _now()
            ptr["events"].append({"time": _now(), "action": "set_current",
                                  "from": previous, "to": int(version), "reason": reason})
            _atomic_write_json(self.pointer_path, ptr)

    def rollback(self, validator: Optional[Callable[[Path], tuple[bool, str]]] = None,
                 reason: str = "manual rollback") -> int:
        """Switch back to the previously served version. Returns its number."""
        with self._lock:
            ptr = self._read_pointer()
            if not ptr or not ptr.get("history"):
                raise RegistryError("No earlier version to roll back to")
            history = list(ptr["history"])
            while history:
                target = history.pop()
                try:
                    meta = self.get_metadata(target)
                    path = self.model_path(target)
                    if meta.get("status") not in SERVABLE or not path.is_file():
                        continue
                    if validator is not None:
                        ok, _ = validator(path)
                        if not ok:
                            continue
                except RegistryError:
                    continue
                from_version = ptr["current_version"]
                ptr.update({"current_version": target, "history": history, "updated_at": _now()})
                ptr["events"].append({"time": _now(), "action": "rollback",
                                      "from": from_version, "to": target, "reason": reason})
                _atomic_write_json(self.pointer_path, ptr)
                return target
            raise RegistryError("No valid earlier version available for rollback")


def build_metadata(*, architecture: str, training_dataset: dict, training_config: dict,
                   metrics: dict, acceptance: dict | None = None, parent_version: int | None = None,
                   pretrained_weights: str = "unknown", environment: dict | None = None,
                   notes: str = "") -> dict:
    """Assemble the metadata record stored with every model version.

    ``metrics`` should be a dict such as ``{"validation": {...}, "test": {...}}``
    so it is always clear which split each number came from.
    """
    return {
        "architecture": architecture,
        "pretrained_weights": pretrained_weights,
        "input": {"shape": list(config.IMG_SHAPE), "dtype": "float32 RGB 0-255",
                  "scaling": "inside model (Rescaling 1/127.5, offset -1)"},
        "classes": list(config.CLASS_NAMES),
        "training_dataset": training_dataset,
        "training_config": training_config,
        "metrics": metrics,
        "acceptance": acceptance,
        "parent_version": parent_version,
        "environment": environment or {},
        "notes": notes,
        "disclaimer": "Experimental academic model. Not a medical device or diagnostic tool.",
    }
