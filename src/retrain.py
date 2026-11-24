"""Retraining: fine-tune the CURRENT registry model on uploaded labelled images.

Safeguards against making the model worse (small-data retraining):
* starts from the current accepted model's weights - never a fresh network;
* only the top backbone layers train, at a very small learning rate, with early stopping;
* uploaded images are mixed with a stratified REPLAY sample of the original training
  data so a few new images cannot wipe out earlier learning;
* the candidate AND the current model are scored on the ORIGINAL validation split and
  the candidate is accepted only if it passes ``evaluate.AcceptancePolicy``;
* the TEST split is never read, used for training, or used for acceptance;
* the current model keeps serving until a validated replacement is registered;
* one job at a time (single-flight lock); automatic triggering has a threshold + cool-down.
"""
from __future__ import annotations

import gc
import hashlib
import shutil
import tempfile
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Protocol

import numpy as np

from . import config
from .database import Database
from .evaluate import (AcceptancePolicy, compute_metrics, evaluate_acceptance, model_integrity_check,
                       predict_probabilities)
from .model_registry import ModelRegistry, build_metadata
from .preprocessing import InvalidImageError, load_image
from .reference import array_hash


class RetrainingBusy(Exception):
    """A retraining job is already running."""


class RetrainingNotPossible(Exception):
    def __init__(self, message: str, detail: Optional[dict] = None):
        super().__init__(message)
        self.detail = detail or {}


@dataclass(frozen=True)
class RetrainSettings:
    min_images: int = config.RETRAIN_MIN_IMAGES
    min_classes: int = config.RETRAIN_MIN_CLASSES
    replay_size: int = config.RETRAIN_REPLAY_SIZE
    epochs: int = config.RETRAIN_EPOCHS
    learning_rate: float = config.RETRAIN_LEARNING_RATE
    unfreeze_layers: int = config.RETRAIN_UNFREEZE_LAYERS
    batch_size: int = config.RETRAIN_BATCH_SIZE
    patience: int = config.RETRAIN_PATIENCE
    seed: int = config.SEED

    def to_dict(self) -> dict:
        return asdict(self)


def stratified_sample(labels: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Indices of ~``n`` samples preserving class proportions (>=1 per present class)."""
    labels = np.asarray(labels)
    n = min(n, len(labels))
    rng = np.random.default_rng(seed)
    picks = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        k = max(1, int(round(n * len(idx) / len(labels))))
        picks.append(rng.choice(idx, size=min(k, len(idx)), replace=False))
    return np.sort(np.concatenate(picks))


# ----------------------------------------------------------------- trainers
class Trainer(Protocol):
    def load(self, path: Path): ...
    def predict_probs(self, model, images: np.ndarray) -> np.ndarray: ...
    def finetune(self, model, new: tuple, replay: tuple, val: tuple, settings: RetrainSettings) -> dict: ...
    def save(self, model, path: Path) -> None: ...


class KerasTrainer:
    """The real trainer (needs TensorFlow). Written but not executable in the build sandbox."""

    def load(self, path: Path):
        from tensorflow import keras
        return keras.models.load_model(str(path), compile=False)

    def predict_probs(self, model, images):
        return predict_probabilities(model, images, 32)

    def finetune(self, model, new, replay, val, settings: RetrainSettings) -> dict:
        from tensorflow import keras
        from .model import compile_model, set_backbone_trainable
        from .preprocessing import make_tf_dataset
        from .train import MacroF1Callback, balanced_class_weights, set_seed

        set_seed(settings.seed)
        x = np.concatenate([new[0], replay[0]]); y = np.concatenate([new[1], replay[1]])
        stats = set_backbone_trainable(model, settings.unfreeze_layers)
        compile_model(model, settings.learning_rate)
        ds = make_tf_dataset(x, y, settings.batch_size, shuffle=True, seed=settings.seed)
        val_ds = make_tf_dataset(val[0], val[1], 32, shuffle=False)
        f1 = MacroF1Callback(val[0], val[1])
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "best.weights.h5"
            cbs = [f1,
                   keras.callbacks.EarlyStopping(monitor="val_macro_f1", mode="max", patience=settings.patience),
                   keras.callbacks.ModelCheckpoint(str(ckpt), monitor="val_macro_f1", mode="max",
                                                   save_best_only=True, save_weights_only=True)]
            hist = model.fit(ds, validation_data=val_ds, epochs=settings.epochs,
                             class_weight=balanced_class_weights(y), callbacks=cbs, verbose=0).history
            if ckpt.is_file():
                model.load_weights(str(ckpt))          # best epoch by validation macro-F1
        return {"epochs_run": len(hist["loss"]), "val_macro_f1_per_epoch": f1.values,
                "train_loss": [float(v) for v in hist["loss"]], **stats}

    def save(self, model, path: Path) -> None:
        model.save(str(path))


# ------------------------------------------------------------------ manager
class RetrainingManager:
    def __init__(self, db: Database, registry: ModelRegistry, reference, trainer: Trainer,
                 settings: Optional[RetrainSettings] = None, policy: Optional[AcceptancePolicy] = None,
                 on_model_changed: Optional[Callable[[], None]] = None,
                 auto_enabled: bool = config.AUTO_RETRAIN_ENABLED,
                 auto_min_new: int = config.AUTO_RETRAIN_MIN_NEW_IMAGES,
                 cooldown_minutes: float = config.AUTO_RETRAIN_COOLDOWN_MINUTES):
        self.db, self.registry, self.reference, self.trainer = db, registry, reference, trainer
        self.settings = settings or RetrainSettings()
        self.policy = policy or AcceptancePolicy()
        self.on_model_changed = on_model_changed
        self.auto_enabled, self.auto_min_new, self.cooldown_minutes = auto_enabled, auto_min_new, cooldown_minutes
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------ status
    def is_running(self) -> bool:
        return self._lock.locked()

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def eligibility(self) -> dict:
        n_acc = self.db.count_uploads("accepted")
        counts = self.db.upload_class_counts("accepted")
        n_classes = sum(c > 0 for c in counts)
        has_model = self.registry.current_version() is not None
        ds = self.reference.available()
        reasons = []
        if not has_model:
            reasons.append("there is no current model to start from (train and register version 1 first)")
        if not ds:
            reasons.append("the original RetinaMNIST data is not available (needed for validation and replay)")
        if n_acc < self.settings.min_images:
            reasons.append(f"only {n_acc} valid labelled image(s) uploaded; at least {self.settings.min_images} are required")
        if n_classes < self.settings.min_classes:
            reasons.append(f"images cover {n_classes} class(es); at least {self.settings.min_classes} different classes are required")
        return {"eligible": not reasons,
                "message": "Ready to retrain." if not reasons else "Cannot retrain: " + "; ".join(reasons) + ".",
                "n_accepted_images": n_acc, "n_new_images": self.db.count_uploads("accepted", only_new=True),
                "class_counts": counts, "n_classes": n_classes, "min_images": self.settings.min_images,
                "min_classes": self.settings.min_classes, "has_current_model": has_model,
                "dataset_available": ds, "running": self.is_running()}

    # ---------------------------------------------------------- starting
    def start(self, trigger: str = "manual", wait: bool = False) -> dict:
        """Begin retraining. Raises ``RetrainingNotPossible`` / ``RetrainingBusy``.

        With ``wait=False`` the job runs in a background thread and this returns
        ``{"event_id", "status": "running"}`` immediately.
        """
        elig = self.eligibility()
        if not elig["eligible"]:
            raise RetrainingNotPossible(elig["message"], elig)
        if not self._lock.acquire(blocking=False):
            raise RetrainingBusy("A retraining job is already running")
        try:
            event_id = self.db.create_retrain_event(trigger, self.registry.current_version(), self.settings.to_dict())
        except Exception:
            self._lock.release()
            raise
        if wait:
            return self._execute(event_id)
        self._thread = threading.Thread(target=self._execute, args=(event_id,), daemon=True,
                                        name=f"retrain-{event_id}")
        self._thread.start()
        return {"event_id": event_id, "status": "running"}

    def maybe_auto_trigger(self) -> Optional[dict]:
        """Start a job automatically if (and only if) it is safe and warranted.

        Conditions: auto mode on; no job running; enough NEW valid images that
        have never been part of an attempt; eligible; and the last job of any
        kind started more than ``cooldown_minutes`` ago. Images are marked as
        attempted by every job that trains (even a failed or rejected one), so a
        failure can never cause a retry loop.
        """
        if not self.auto_enabled or self.is_running():
            return None
        if self.db.count_uploads("accepted", only_new=True) < self.auto_min_new:
            return None
        last = self.db.last_retrain_event()
        if last and self.cooldown_minutes > 0:
            started = datetime.fromisoformat(last["started_at"])
            age = (datetime.now(timezone.utc) - started).total_seconds() / 60
            if age < self.cooldown_minutes:
                return None
        try:
            return self.start(trigger="auto")
        except (RetrainingBusy, RetrainingNotPossible):
            return None

    # --------------------------------------------------------- execution
    def _execute(self, event_id: int) -> dict:
        try:
            self._run(event_id)
        except Exception as exc:                       # never leave a 'running' row behind
            self.db.finish_retrain_event(event_id, status="failed",
                                         message=f"Retraining failed: {type(exc).__name__}: {exc}. "
                                                 "The current model was not changed.")
        finally:
            gc.collect()
            self._lock.release()
        return self.db.get_retrain_event(event_id)

    def _load_valid_uploads(self):
        hash_index = self.reference.hash_index()
        xs, ys, ids, shas = [], [], [], []
        for row in self.db.accepted_uploads():
            try:
                img = load_image(row["stored_path"])                    # the shared preprocessing path
            except InvalidImageError as exc:
                self.db.mark_excluded(row["id"], f"file unreadable at retraining time: {exc}")
                continue
            if hash_index.get(array_hash(img)) in ("val", "test", "train"):
                self.db.mark_excluded(row["id"], "identical to an original dataset image")
                continue
            xs.append(img); ys.append(int(row["label"])); ids.append(row["id"]); shas.append(row["sha256"])
        return (np.stack(xs) if xs else np.zeros((0, *config.IMG_SHAPE), np.uint8),
                np.array(ys, dtype=np.int64), ids, shas)

    def _run(self, event_id: int) -> None:
        s = self.settings
        parent = self.registry.current_version()
        if parent is None:
            raise RuntimeError("No current model in the registry")

        # 1-2. load + preprocess uploaded data
        x_new, y_new, ids, shas = self._load_valid_uploads()
        if len(ids) < s.min_images or len(set(y_new.tolist())) < s.min_classes:
            self.db.finish_retrain_event(
                event_id, status="skipped", n_uploaded=len(ids),
                message=f"Not enough valid labelled data after re-validation ({len(ids)} images, "
                        f"{len(set(y_new.tolist()))} classes; need {s.min_images} images, {s.min_classes} classes). "
                        "No training was performed.")
            return
        self.db.mark_uploads_attempted(ids, event_id)

        # 3. replay sample + ORIGINAL validation split (test split is never touched)
        tx, ty = self.reference.train()
        sel = stratified_sample(ty, s.replay_size, s.seed + event_id)
        x_rep, y_rep = tx[sel], ty[sel]
        val = self.reference.val()

        # 4. load the CURRENT model and score it before it is modified in place
        model = self.trainer.load(self.registry.model_path(parent))
        current_metrics = compute_metrics(val[1], self.trainer.predict_probs(model, val[0]))

        # 5. fine-tune from the current weights
        history = self.trainer.finetune(model, (x_new, y_new), (x_rep, y_rep), val, s)

        # 6. candidate evaluation + integrity
        ok, msg = model_integrity_check(model)
        cand_metrics = compute_metrics(val[1], self.trainer.predict_probs(model, val[0])) if ok else None

        # 7. save candidate, re-load it and re-check (round trip) before it can serve
        staging = self.registry.models_dir / "staging"
        staging.mkdir(parents=True, exist_ok=True)
        staged = staging / f"candidate_event{event_id}.keras"
        try:
            if ok:
                self.trainer.save(model, staged)
                ok, msg = model_integrity_check(self.trainer.load(staged))
                if not ok:
                    msg = f"saved candidate failed re-load check: {msg}"
            acceptance = (evaluate_acceptance(cand_metrics, current_metrics, ok, msg, self.policy) if ok else
                          {"accepted": False, "reasons": [f"Integrity check failed: {msg}"],
                           "checks": {"integrity": False}, "policy": self.policy.to_dict()})
            accepted = bool(acceptance["accepted"]) and cand_metrics is not None

            parent_meta = self.registry.get_current_metadata() or {}
            meta = build_metadata(
                architecture=parent_meta.get("architecture", "MobileNetV2"),
                pretrained_weights=parent_meta.get("pretrained_weights", "unknown"),
                training_dataset={
                    "base": "RetinaMNIST-224 stratified replay sample + user uploads",
                    "n_uploaded_images": len(ids), "n_replay_images": int(len(sel)),
                    "uploaded_class_counts": np.bincount(y_new, minlength=config.NUM_CLASSES).tolist(),
                    "uploads_fingerprint": hashlib.sha256("".join(sorted(shas)).encode()).hexdigest(),
                    "retraining_event_id": event_id},
                training_config={**s.to_dict(), **{k: v for k, v in history.items()}},
                metrics={"validation": cand_metrics,
                         "current_model_validation_at_comparison": current_metrics,
                         "test": None},
                acceptance=acceptance, parent_version=parent,
                notes="Fine-tuned from the previous version on uploaded labelled images. "
                      "Test metrics intentionally not computed: the test split is reserved.")
            status = "accepted" if accepted else "rejected"
            meta_out = self.registry.register_version(staged, meta, status, make_current=False)
            if accepted:
                self.registry.set_current(meta_out["version"], reason=f"retraining event {event_id}")
            self.db.upsert_model_version(meta_out, len(ids))
        finally:
            staged.unlink(missing_ok=True)
            del model

        extra = ""
        if accepted and self.on_model_changed is not None:
            try:
                self.on_model_changed()
            except Exception as exc:
                extra = f" WARNING: version registered but the prediction service failed to reload it: {exc}"
        if accepted:
            text = (f"Candidate accepted and registered as version {meta_out['version']} "
                    f"(validation macro-F1 {current_metrics['f1_macro']:.3f} -> {cand_metrics['f1_macro']:.3f}).")
        else:
            text = (f"Candidate registered as version {meta_out['version']} but REJECTED; version {parent} "
                    f"keeps serving. Reasons: {'; '.join(acceptance['reasons'])}")
        self.db.finish_retrain_event(
            event_id, status=status, message=text + extra, candidate_version=meta_out["version"],
            n_uploaded=len(ids), n_replay=int(len(sel)), current_metrics=current_metrics,
            candidate_metrics=cand_metrics, acceptance=acceptance)
