"""Metrics, model integrity checks and the model-acceptance policy.

Pure NumPy / scikit-learn: no TensorFlow import, so it is fully testable
without a GPU. Any object with ``predict(batch) -> (N, 5)`` works as a model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (classification_report, cohen_kappa_score, confusion_matrix,
                             log_loss, precision_recall_fscore_support, roc_auc_score)

from . import config
from .preprocessing import to_model_input


# ------------------------------------------------------------ inference
def predict_probabilities(model, images_uint8: np.ndarray, batch_size: int = 32) -> np.ndarray:
    """Softmax probabilities (N, 5) for uint8 images, batched to limit memory."""
    out = []
    for start in range(0, len(images_uint8), batch_size):
        batch = to_model_input(images_uint8[start:start + batch_size])
        out.append(np.asarray(model.predict(batch, verbose=0), dtype=np.float64))
    return np.concatenate(out) if out else np.zeros((0, config.NUM_CLASSES))


# -------------------------------------------------------------- metrics
def compute_metrics(y_true, probs, class_names=config.CLASS_NAMES) -> dict:
    """All evaluation metrics as a JSON-safe dict.

    Averaging: ``macro`` = unweighted mean over the 5 classes (treats the
    rare severe/proliferative classes equally - the primary selection metric);
    ``weighted`` = mean weighted by class support (dominated by "No DR").
    ``loss`` is the multiclass cross-entropy of the predicted probabilities.
    """
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    probs = np.asarray(probs, dtype=np.float64)
    k = len(class_names)
    labels = list(range(k))
    if probs.ndim != 2 or probs.shape[1] != k or len(probs) != len(y_true):
        raise ValueError(f"probs {probs.shape} incompatible with {len(y_true)} labels / {k} classes")
    sums = probs.sum(axis=1, keepdims=True)
    probs = probs / np.where(sums == 0, 1.0, sums)
    y_pred = probs.argmax(axis=1)

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    macro = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    auc_per_class = {}
    for c in labels:
        pos = (y_true == c)
        if 0 < pos.sum() < len(pos):
            auc_per_class[class_names[c]] = float(roc_auc_score(pos, probs[:, c]))
    auc_macro = float(np.mean(list(auc_per_class.values()))) if len(auc_per_class) == k else None

    return {
        "n_samples": int(len(y_true)),
        "accuracy": float((y_pred == y_true).mean()),
        "loss": float(log_loss(y_true, np.clip(probs, 1e-7, 1.0), labels=labels)),
        "precision_macro": float(macro[0]), "recall_macro": float(macro[1]), "f1_macro": float(macro[2]),
        "precision_weighted": float(weighted[0]), "recall_weighted": float(weighted[1]),
        "f1_weighted": float(weighted[2]),
        "per_class": {class_names[i]: {"precision": float(p[i]), "recall": float(r[i]),
                                       "f1": float(f[i]), "support": int(s[i])} for i in labels},
        "confusion_matrix": cm.tolist(),
        "roc_auc_ovr_macro": auc_macro,
        "roc_auc_per_class": auc_per_class,
        "quadratic_weighted_kappa": float(cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=labels)),
        "within_one_grade_accuracy": float((np.abs(y_true - y_pred) <= 1).mean()),
        "majority_class_baseline_accuracy": float(np.bincount(y_true, minlength=k).max() / len(y_true)),
    }


def text_report(y_true, probs, class_names=config.CLASS_NAMES) -> str:
    y_pred = np.asarray(probs).argmax(axis=1)
    return classification_report(np.asarray(y_true).reshape(-1), y_pred,
                                 labels=list(range(len(class_names))),
                                 target_names=list(class_names), zero_division=0, digits=3)


def misclassified_indices(y_true, probs) -> np.ndarray:
    """Indices of wrong predictions, most confidently wrong first."""
    y_true = np.asarray(y_true).reshape(-1)
    probs = np.asarray(probs)
    y_pred = probs.argmax(axis=1)
    wrong = np.where(y_pred != y_true)[0]
    return wrong[np.argsort(-probs[wrong, y_pred[wrong]])]


# ------------------------------------------------------ model integrity
def model_integrity_check(model, n_probe: int = 2) -> tuple[bool, str]:
    """Cheap sanity checks before a model may serve or be promoted."""
    try:
        rng = np.random.default_rng(0)
        probe = rng.integers(0, 256, size=(n_probe, *config.IMG_SHAPE)).astype(np.float32)
        out = np.asarray(model.predict(probe, verbose=0))
    except Exception as exc:
        return False, f"model.predict failed: {exc}"
    if out.shape != (n_probe, config.NUM_CLASSES):
        return False, f"output shape {out.shape}, expected ({n_probe}, {config.NUM_CLASSES})"
    if not np.all(np.isfinite(out)):
        return False, "output contains NaN or infinity"
    if np.any(out < -1e-6) or not np.allclose(out.sum(axis=1), 1.0, atol=1e-3):
        return False, "outputs are not a probability distribution"
    return True, "ok"


# ------------------------------------------------------ acceptance policy
@dataclass(frozen=True)
class AcceptancePolicy:
    """Rules a candidate model must pass to replace the current one.

    All comparisons use the ORIGINAL validation split, never the test split.
    1. The candidate passes ``model_integrity_check``.
    2. Evaluated on at least ``min_eval_samples`` images.
    3. macro-F1 >= current macro-F1 + ``min_delta_macro_f1``.
    4. Accuracy has not dropped by more than ``max_accuracy_drop``.
    5. No class with >= ``min_class_support`` validation samples lost more than
       ``max_class_recall_drop`` recall (stops a model that trades away a
       rare severe class for a small macro-F1 gain).
    With no current model, only rules 1-2 apply.
    """
    min_delta_macro_f1: float = config.ACCEPT_MIN_DELTA_MACRO_F1
    max_accuracy_drop: float = config.ACCEPT_MAX_ACCURACY_DROP
    max_class_recall_drop: float = config.ACCEPT_MAX_CLASS_RECALL_DROP
    min_class_support: int = config.ACCEPT_MIN_CLASS_SUPPORT
    min_eval_samples: int = config.ACCEPT_MIN_EVAL_SAMPLES

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_acceptance(candidate: dict, current: dict | None,
                        integrity_ok: bool = True, integrity_message: str = "ok",
                        policy: AcceptancePolicy | None = None) -> dict:
    """Apply the policy. Returns {accepted, reasons, checks, policy}."""
    policy = policy or AcceptancePolicy()
    checks: dict[str, bool] = {}
    reasons: list[str] = []

    checks["integrity"] = bool(integrity_ok)
    if not integrity_ok:
        reasons.append(f"Integrity check failed: {integrity_message}")
    checks["min_eval_samples"] = candidate.get("n_samples", 0) >= policy.min_eval_samples
    if not checks["min_eval_samples"]:
        reasons.append(f"Only {candidate.get('n_samples', 0)} evaluation samples "
                       f"(< {policy.min_eval_samples})")

    if current is not None:
        d_f1 = candidate["f1_macro"] - current["f1_macro"]
        checks["macro_f1_not_worse"] = d_f1 >= policy.min_delta_macro_f1 - 1e-12
        if not checks["macro_f1_not_worse"]:
            reasons.append(f"Validation macro-F1 changed by {d_f1:+.4f} "
                           f"(required >= {policy.min_delta_macro_f1:+.4f})")
        d_acc = candidate["accuracy"] - current["accuracy"]
        checks["accuracy_drop_ok"] = d_acc >= -policy.max_accuracy_drop - 1e-12
        if not checks["accuracy_drop_ok"]:
            reasons.append(f"Accuracy dropped by {-d_acc:.4f} (max {policy.max_accuracy_drop})")
        bad = []
        for name, cur in current.get("per_class", {}).items():
            cand = candidate.get("per_class", {}).get(name)
            if cand is None or cur["support"] < policy.min_class_support:
                continue
            if cur["recall"] - cand["recall"] > policy.max_class_recall_drop + 1e-12:
                bad.append(f"{name} (recall {cur['recall']:.2f} -> {cand['recall']:.2f})")
        checks["no_class_collapse"] = not bad
        if bad:
            reasons.append("Recall dropped too far for: " + ", ".join(bad))

    accepted = all(checks.values())
    if accepted:
        reasons.append("All acceptance checks passed" if current is not None
                       else "No current model; integrity and size checks passed")
    return {"accepted": accepted, "reasons": reasons, "checks": checks, "policy": policy.to_dict()}
