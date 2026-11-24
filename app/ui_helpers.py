"""Pure helper functions for the dashboard (no Streamlit import, so they are unit-tested)."""
from __future__ import annotations

from typing import Optional

import pandas as pd

from src import config
from src.monitoring import format_duration

LABEL_PLACEHOLDER = "- choose a label -"


def label_choices() -> list[str]:
    return [f"{i} - {name}" for i, name in enumerate(config.CLASS_NAMES)]


def choice_to_label(choice: Optional[str]) -> Optional[int]:
    """'3 - Severe' -> 3 ; placeholder / None -> None."""
    if not choice or choice == LABEL_PLACEHOLDER:
        return None
    head = choice.split(" ", 1)[0]
    return int(head) if head.isdigit() and int(head) < config.NUM_CLASSES else None


def probabilities_frame(result: dict) -> pd.DataFrame:
    return pd.DataFrame({"probability": result["probabilities"]}).loc[list(config.CLASS_NAMES)]


def friendly_error(exc) -> str:
    """Readable text for an ApiError/ApiUnavailable (duck-typed to avoid importing the client)."""
    status, detail = getattr(exc, "status", None), getattr(exc, "detail", str(exc))
    if isinstance(detail, dict):
        detail = detail.get("message") or str(detail)
    if status is None:
        return f"The API could not be reached. {detail}"
    return f"{detail} (HTTP {status})"


def uptime_text(seconds: float) -> str:
    return format_duration(seconds)


# ------------------------------------------------------------ model metadata
def current_meta(models: dict) -> Optional[dict]:
    cur = models.get("current_version")
    return next((m for m in models.get("versions", []) if m.get("version") == cur), None)


def dataset_counts(models: dict) -> Optional[dict]:
    """Class counts per split of the ORIGINAL dataset, taken from the model metadata that recorded them."""
    for m in sorted(models.get("versions", []), key=lambda m: m.get("version", 0)):
        counts = (m.get("training_dataset") or {}).get("class_counts")
        if counts:
            return counts
    return None


def performance_view(models: dict) -> Optional[dict]:
    """Metrics to display for the current model.

    Prefers TEST metrics. Retrained versions deliberately have no test metrics (the test split is reserved),
    so then the validation metrics are shown and the most recent test metrics are taken from an ancestor
    version and labelled as such.
    """
    cur = current_meta(models)
    if cur is None:
        return None
    metrics = cur.get("metrics") or {}
    if metrics.get("test"):
        return {"version": cur["version"], "split": "test", "metrics": metrics["test"], "note": None,
                "validation": metrics.get("validation")}
    for m in sorted(models.get("versions", []), key=lambda m: -m.get("version", 0)):
        t = (m.get("metrics") or {}).get("test")
        if t:
            return {"version": m["version"], "split": "test", "metrics": t, "validation": metrics.get("validation"),
                    "note": (f"Version {cur['version']} was retrained, so the test set was never evaluated for it. "
                             f"The test metrics shown belong to version {m['version']}; version {cur['version']}'s "
                             "validation metrics are in the table below.")}
    return {"version": cur["version"], "split": "validation", "metrics": metrics.get("validation"),
            "note": "No test metrics are recorded for this model.", "validation": metrics.get("validation")}


def headline_table(metrics: dict) -> pd.DataFrame:
    rows = [("Accuracy", "accuracy"), ("Loss (cross-entropy)", "loss"), ("Precision (macro)", "precision_macro"),
            ("Recall (macro)", "recall_macro"), ("F1 (macro)", "f1_macro"), ("F1 (weighted)", "f1_weighted"),
            ("ROC-AUC (one-vs-rest, macro)", "roc_auc_ovr_macro"), ("Quadratic weighted kappa", "quadratic_weighted_kappa"),
            ("Within-one-grade accuracy", "within_one_grade_accuracy"),
            ("Majority-class accuracy (reference)", "majority_class_baseline_accuracy")]
    return pd.DataFrame({"value": {label: metrics.get(key) for label, key in rows}}).round(3)


def per_class_table(metrics: dict) -> pd.DataFrame:
    return pd.DataFrame(metrics["per_class"]).T.loc[list(config.CLASS_NAMES)].round(3)


def versions_table(models: dict) -> pd.DataFrame:
    rows = []
    for m in sorted(models.get("versions", []), key=lambda m: -m.get("version", 0)):
        val = (m.get("metrics") or {}).get("validation") or {}
        rows.append({"version": m.get("version"), "status": m.get("status"),
                     "serving": "yes" if m.get("version") == models.get("current_version") else "",
                     "created": m.get("created_at"), "parent": m.get("parent_version"),
                     "val accuracy": _r(val.get("accuracy")), "val macro-F1": _r(val.get("f1_macro")),
                     "uploaded images used": (m.get("training_dataset") or {}).get("n_uploaded_images")})
    return pd.DataFrame(rows)


def _r(x):
    return None if x is None else round(float(x), 3)


# ------------------------------------------------------- image characteristics
def feature_summaries(df: pd.DataFrame) -> dict:
    """Median per class and Spearman correlation with grade, computed from the real feature table."""
    cols = ["brightness", "contrast", "sharpness", "retina_fraction"]
    order = list(config.CLASS_NAMES)
    medians = df.groupby("class")[cols].median().loc[order].round(2)
    rho = df[cols].corrwith(df["label"], method="spearman").round(3).to_frame("Spearman rho vs DR grade")
    narrow = (df["retina_fraction"] < 0.69)
    framing = (narrow.groupby(df["class"]).mean().loc[order] * 100).round(1).to_frame("% narrow-framing images")
    return {"medians": medians, "rho": rho, "framing": framing}


# --------------------------------------------------------- uploads / retraining
def upload_results_table(report: dict) -> pd.DataFrame:
    return pd.DataFrame([{"file": r["filename"], "label": r.get("label"), "result": r["status"],
                          "reason": r.get("reason", "")} for r in report["results"]])


STATUS_BADGES = {"running": ("⏳", "Running"), "accepted": ("✅", "Accepted - new model version is live"),
                 "rejected": ("🛑", "Rejected - previous model keeps serving"),
                 "failed": ("❌", "Failed - model unchanged"), "skipped": ("⚠️", "Skipped - not enough valid data")}


def status_badge(status: Optional[str]) -> str:
    icon, text = STATUS_BADGES.get(status or "", ("", "No retraining has been run yet"))
    return f"{icon} {text}".strip()


def metrics_comparison(event: dict) -> Optional[pd.DataFrame]:
    cur, cand = event.get("current_metrics"), event.get("candidate_metrics")
    if not cur or not cand:
        return None
    keys = [("Accuracy", "accuracy"), ("Macro F1", "f1_macro"), ("Weighted F1", "f1_weighted"),
            ("Macro precision", "precision_macro"), ("Macro recall", "recall_macro"), ("Loss", "loss"),
            ("Quadratic weighted kappa", "quadratic_weighted_kappa")]
    rows = {label: {"current model": cur.get(k), "candidate": cand.get(k),
                    "change": None if cur.get(k) is None or cand.get(k) is None else cand[k] - cur[k]}
            for label, k in keys}
    return pd.DataFrame(rows).T.round(4).rename_axis("validation metric")


def events_table(events: list) -> pd.DataFrame:
    rows = [{"id": e["id"], "trigger": e["trigger"], "status": e["status"], "started": e["started_at"],
             "finished": e.get("finished_at"), "parent": e.get("parent_version"),
             "candidate": e.get("candidate_version"), "uploaded images": e.get("n_uploaded_images")} for e in events]
    return pd.DataFrame(rows)


def class_count_frame(counts: list) -> pd.DataFrame:
    return pd.DataFrame({"images": counts}, index=list(config.CLASS_NAMES))
