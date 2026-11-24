"""Application service layer used by the FastAPI app (and directly by tests).

Keeps all behaviour out of the HTTP layer so it can be unit-tested without FastAPI.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

from . import config
from .database import Database
from .evaluate import model_integrity_check
from .image_analysis import compute_image_features
from .model_registry import ModelRegistry, RegistryError
from .monitoring import ServiceClock, class_distribution_shift, feature_drift, format_duration
from .prediction import ModelNotLoadedError, Predictor
from .preprocessing import InvalidImageError, load_image
from .reference import NpzReference
from .retrain import KerasTrainer, RetrainingBusy, RetrainingManager, RetrainingNotPossible
from .uploads import ingest_batch, parse_labels


def _default_validator(path: Path) -> tuple:
    """Load a model file and run the integrity check (used before rollback switches the pointer)."""
    try:
        return model_integrity_check(Predictor.load_path(path).model)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


class AppService:
    def __init__(self, db: Optional[Database] = None, registry: Optional[ModelRegistry] = None,
                 reference=None, trainer=None, predictor_factory: Optional[Callable[[ModelRegistry], Predictor]] = None,
                 upload_dir: Path | str = config.RETRAIN_UPLOAD_DIR, manager: Optional[RetrainingManager] = None,
                 model_validator: Optional[Callable[[Path], tuple]] = None):
        self.db = db or Database()
        self.registry = registry or ModelRegistry()
        self.reference = reference or NpzReference()
        self.upload_dir = Path(upload_dir)
        self._predictor_factory = predictor_factory or Predictor.from_registry
        self._model_validator = model_validator or _default_validator
        self.clock = ServiceClock()
        self.predictor: Optional[Predictor] = None
        self.model_error: Optional[str] = None
        self._reload_lock = threading.Lock()
        self.manager = manager or RetrainingManager(self.db, self.registry, self.reference,
                                                    trainer or KerasTrainer(), on_model_changed=self.reload_model)

    # ------------------------------------------------------------ model
    def reload_model(self) -> None:
        """(Re)load the registry's current model. The old predictor keeps serving if loading fails."""
        with self._reload_lock:
            try:
                new = self._predictor_factory(self.registry)
                ok, msg = model_integrity_check(new.model)
                if not ok:
                    raise RuntimeError(f"integrity check failed: {msg}")
                self.predictor, self.model_error = new, None
            except Exception as exc:
                self.model_error = f"{type(exc).__name__}: {exc}"
                if self.predictor is None:
                    return
                raise

    def startup(self) -> None:
        try:
            self.reload_model()
        except Exception:
            pass                                    # model_error is already recorded; /health reports it

    def health(self) -> dict:
        loaded = self.predictor is not None and self.predictor.is_loaded
        return {"status": "ok" if loaded else "degraded", "api": "up", "model_loaded": loaded,
                "model_version": self.predictor.version if loaded else None,
                "model_error": None if loaded else (self.model_error or "no model registered yet"),
                "uptime_seconds": round(self.clock.uptime_seconds(), 1),
                "retraining_running": self.manager.is_running(), "disclaimer": config.DISCLAIMER}

    # ------------------------------------------------------- prediction
    def predict(self, data: bytes) -> dict:
        t0 = time.perf_counter()
        try:
            if self.predictor is None:
                raise ModelNotLoadedError("No model is loaded")
            img = load_image(data)
            result = self.predictor.predict_array(img)
        except InvalidImageError as exc:
            self.db.log_prediction(success=False, error="invalid_image")
            raise
        except ModelNotLoadedError:
            self.db.log_prediction(success=False, error="model_not_loaded")
            raise
        except Exception as exc:
            self.db.log_prediction(success=False, error=f"internal_error:{type(exc).__name__}")
            raise
        feats = {k: float(v[0]) for k, v in compute_image_features(img).items()}
        self.db.log_prediction(success=True, predicted_class=result["predicted_class"],
                               confidence=result["confidence"], model_version=result["model_version"],
                               latency_ms=(time.perf_counter() - t0) * 1000, features=feats)
        result["disclaimer"] = config.DISCLAIMER
        return result

    # ---------------------------------------------------------- uploads
    def upload_retraining_data(self, files: Sequence[tuple[str, bytes]], labels_raw) -> dict:
        labels = parse_labels(labels_raw, len(files))             # ValueError -> HTTP 422
        index = self.reference.hash_index() if self.reference.available() else None
        report = ingest_batch(self.db, files, labels, upload_dir=self.upload_dir, hash_index=index)
        report["leakage_check"] = "performed" if index is not None else "skipped (original dataset unavailable)"
        report["total_available_images"] = self.db.count_uploads("accepted")
        auto = self.manager.maybe_auto_trigger() if report["n_accepted"] else None
        report["auto_retraining_started"] = auto
        return report

    def retraining_data_summary(self) -> dict:
        return {**self.manager.eligibility(), "invalid_uploads": self.db.count_uploads("rejected"),
                "excluded_uploads": self.db.count_uploads("excluded"),
                "recent_uploads": self.db.recent_uploads(20)}

    # ------------------------------------------------------- retraining
    def retrain(self, wait: bool = False) -> dict:
        return self.manager.start("manual", wait=wait)

    def retrain_status(self, event_id: Optional[int] = None) -> dict:
        ev = self.db.get_retrain_event(event_id) if event_id else self.db.last_retrain_event()
        return {"running": self.manager.is_running(), "event": ev}

    # ---------------------------------------------------------- models
    def models(self) -> dict:
        return {"current_version": self.registry.current_version(), "versions": self.registry.list_versions()}

    def rollback(self) -> dict:
        if self.manager.is_running():
            raise RetrainingBusy("Cannot roll back while retraining is running")
        target = self.registry.rollback(validator=self._model_validator)       # RegistryError -> HTTP 409
        self.reload_model()
        return {"rolled_back_to": target}

    # ------------------------------------------------------- monitoring
    def monitoring(self) -> dict:
        stats = self.db.prediction_stats()
        up_counts = self.db.upload_class_counts("accepted")
        drift = {"class_distribution": None, "image_statistics": None}
        if self.reference.available():
            try:
                drift["class_distribution"] = class_distribution_shift(up_counts, self.reference.train_class_counts())
                obs = {k: v for k, v in self.db.prediction_features().items()}
                up_f = self.db.upload_features()
                combined = {k: list(obs[k]) + list(up_f[k]) for k in obs}
                ref_f = self.reference.reference_features()
                drift["image_statistics"] = feature_drift(ref_f, combined)
            except Exception as exc:
                drift["error"] = f"{type(exc).__name__}: {exc}"
        h = self.health()
        return {
            "service": {"status": h["status"], "uptime_seconds": h["uptime_seconds"],
                        "uptime": format_duration(h["uptime_seconds"]),
                        "started_at_unix": self.clock.started_at,
                        "model_loaded": h["model_loaded"], "model_version": h["model_version"]},
            "predictions": {**stats, "recent": self.db.recent_predictions(15)},
            "retraining_data": {"available_images": self.db.count_uploads("accepted"),
                                "new_since_last_retraining": self.db.count_uploads("accepted", only_new=True),
                                "class_counts": up_counts, "invalid_uploads": self.db.count_uploads("rejected"),
                                "excluded_uploads": self.db.count_uploads("excluded"),
                                "recent_uploads": self.db.recent_uploads(15)},
            "retraining": {"running": self.manager.is_running(), "recent_events": self.db.recent_retrain_events(5),
                           "auto_retraining_enabled": self.manager.auto_enabled,
                           "auto_threshold_new_images": self.manager.auto_min_new},
            "drift": drift,
            "model_versions": self.db.list_model_versions(),
        }
