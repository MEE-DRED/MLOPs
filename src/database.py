"""SQLite persistence: uploads, predictions, retraining events, model versions.

Images are stored on disk (see ``uploads.py``); only metadata lives here.
Every method opens a short-lived connection (WAL mode), so it is safe to call
from the API threads and the background retraining thread.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from . import config

FEATURES = ("brightness", "contrast", "sharpness", "retina_fraction")

SCHEMA = """
CREATE TABLE IF NOT EXISTS uploads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id TEXT,
    original_filename TEXT,
    stored_path TEXT,
    sha256 TEXT,
    label INTEGER,
    status TEXT NOT NULL CHECK (status IN ('accepted','rejected','excluded')),
    reason TEXT,
    uploaded_at TEXT NOT NULL,
    retrain_event_id INTEGER,
    brightness REAL, contrast REAL, sharpness REAL, retina_fraction REAL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_uploads_unique_accepted ON uploads(sha256) WHERE status = 'accepted';
CREATE INDEX IF NOT EXISTS idx_uploads_status ON uploads(status);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    success INTEGER NOT NULL,
    predicted_class INTEGER,
    confidence REAL,
    model_version INTEGER,
    latency_ms REAL,
    error TEXT,
    brightness REAL, contrast REAL, sharpness REAL, retina_fraction REAL
);

CREATE TABLE IF NOT EXISTS retraining_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trigger TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('running','accepted','rejected','failed','skipped')),
    started_at TEXT NOT NULL,
    finished_at TEXT,
    parent_version INTEGER,
    candidate_version INTEGER,
    n_uploaded_images INTEGER,
    n_replay_images INTEGER,
    message TEXT,
    current_metrics TEXT,
    candidate_metrics TEXT,
    acceptance TEXT,
    config TEXT
);

CREATE TABLE IF NOT EXISTS model_versions (
    version INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    parent_version INTEGER,
    val_accuracy REAL,
    val_macro_f1 REAL,
    n_uploaded_images INTEGER,
    metadata_file TEXT
);
"""


class DatabaseError(Exception):
    """Wraps low-level sqlite errors."""


class DuplicateUploadError(DatabaseError):
    """An accepted upload with the same content hash already exists."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json(obj: Any) -> Optional[str]:
    return None if obj is None else json.dumps(obj, default=str)


class Database:
    def __init__(self, path: Path | str = config.DB_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    # ------------------------------------------------------------ plumbing
    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except sqlite3.Error as exc:
            conn.rollback()
            raise DatabaseError(str(exc)) from exc
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init(self) -> None:
        with self._conn() as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.executescript(SCHEMA)
            # a restart while training was running leaves a stale 'running' row
            c.execute("UPDATE retraining_events SET status='failed', finished_at=?, "
                      "message='Interrupted: the service restarted while this job was running' "
                      "WHERE status='running'", (now(),))

    def _all(self, sql: str, args: Iterable = ()) -> list[dict]:
        with self._conn() as c:
            return [dict(r) for r in c.execute(sql, tuple(args)).fetchall()]

    def _one(self, sql: str, args: Iterable = ()) -> Optional[dict]:
        rows = self._all(sql, args)
        return rows[0] if rows else None

    # -------------------------------------------------------------- uploads
    def add_upload(self, *, batch_id, filename, stored_path, sha256, label, features: dict) -> int:
        try:
            with self._conn() as c:
                try:
                    cur = c.execute(
                        "INSERT INTO uploads (batch_id, original_filename, stored_path, sha256, label, status, "
                        "uploaded_at, brightness, contrast, sharpness, retina_fraction) "
                        "VALUES (?,?,?,?,?,'accepted',?,?,?,?,?)",
                        (batch_id, filename, str(stored_path), sha256, int(label), now(),
                         *(features.get(k) for k in FEATURES)))
                except sqlite3.IntegrityError as exc:
                    raise DuplicateUploadError("identical image already uploaded") from exc
                return int(cur.lastrowid)
        except DuplicateUploadError:
            raise

    def add_rejected_upload(self, *, batch_id, filename, label, reason, sha256=None) -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO uploads (batch_id, original_filename, sha256, label, status, reason, uploaded_at) "
                "VALUES (?,?,?,?,'rejected',?,?)",
                (batch_id, filename, sha256, label, reason, now()))
            return int(cur.lastrowid)

    def accepted_uploads(self) -> list[dict]:
        return self._all("SELECT * FROM uploads WHERE status='accepted' ORDER BY id")

    def count_uploads(self, status: str = "accepted", only_new: bool = False) -> int:
        sql = "SELECT COUNT(*) AS n FROM uploads WHERE status=?"
        if only_new:
            sql += " AND retrain_event_id IS NULL"
        return int(self._one(sql, (status,))["n"])

    def upload_class_counts(self, status: str = "accepted") -> list[int]:
        counts = [0] * config.NUM_CLASSES
        for r in self._all("SELECT label, COUNT(*) AS n FROM uploads WHERE status=? AND label IS NOT NULL "
                           "GROUP BY label", (status,)):
            if 0 <= r["label"] < config.NUM_CLASSES:
                counts[r["label"]] = r["n"]
        return counts

    def recent_uploads(self, limit: int = 20) -> list[dict]:
        return self._all("SELECT id, original_filename, label, status, reason, uploaded_at, retrain_event_id "
                         "FROM uploads ORDER BY id DESC LIMIT ?", (limit,))

    def mark_uploads_attempted(self, ids: Iterable[int], event_id: int) -> None:
        with self._conn() as c:
            c.executemany("UPDATE uploads SET retrain_event_id=? WHERE id=? AND retrain_event_id IS NULL",
                          [(event_id, i) for i in ids])

    def mark_excluded(self, upload_id: int, reason: str) -> None:
        with self._conn() as c:
            c.execute("UPDATE uploads SET status='excluded', reason=? WHERE id=?", (reason, upload_id))

    def upload_features(self) -> dict[str, list[float]]:
        rows = self._all("SELECT brightness, contrast, sharpness, retina_fraction FROM uploads "
                         "WHERE status='accepted' AND brightness IS NOT NULL")
        return {k: [r[k] for r in rows] for k in FEATURES}

    # ---------------------------------------------------------- predictions
    def log_prediction(self, *, success: bool, predicted_class=None, confidence=None, model_version=None,
                       latency_ms=None, error=None, features: Optional[dict] = None) -> int:
        features = features or {}
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO predictions (created_at, success, predicted_class, confidence, model_version, "
                "latency_ms, error, brightness, contrast, sharpness, retina_fraction) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (now(), int(success), predicted_class, confidence, model_version, latency_ms, error,
                 *(features.get(k) for k in FEATURES)))
            return int(cur.lastrowid)

    def prediction_stats(self) -> dict:
        row = self._one("SELECT COUNT(*) AS total, COALESCE(SUM(success),0) AS ok, "
                        "AVG(CASE WHEN success=1 THEN latency_ms END) AS avg_latency FROM predictions")
        by_class = [0] * config.NUM_CLASSES
        for r in self._all("SELECT predicted_class, COUNT(*) AS n FROM predictions "
                           "WHERE success=1 GROUP BY predicted_class"):
            by_class[r["predicted_class"]] = r["n"]
        errors = {r["error"]: r["n"] for r in self._all(
            "SELECT error, COUNT(*) AS n FROM predictions WHERE success=0 GROUP BY error")}
        return {"total": int(row["total"]), "successful": int(row["ok"]),
                "failed": int(row["total"] - row["ok"]), "avg_latency_ms": row["avg_latency"],
                "by_class": by_class, "errors": errors}

    def recent_predictions(self, limit: int = 20) -> list[dict]:
        return self._all("SELECT id, created_at, success, predicted_class, confidence, model_version, "
                         "latency_ms, error FROM predictions ORDER BY id DESC LIMIT ?", (limit,))

    def prediction_features(self, limit: int = 500) -> dict[str, list[float]]:
        rows = self._all("SELECT brightness, contrast, sharpness, retina_fraction FROM predictions "
                         "WHERE success=1 AND brightness IS NOT NULL ORDER BY id DESC LIMIT ?", (limit,))
        return {k: [r[k] for r in rows] for k in FEATURES}

    # ----------------------------------------------------- retraining events
    def create_retrain_event(self, trigger: str, parent_version, config_dict: dict) -> int:
        with self._conn() as c:
            cur = c.execute("INSERT INTO retraining_events (trigger, status, started_at, parent_version, config) "
                            "VALUES (?, 'running', ?, ?, ?)", (trigger, now(), parent_version, _json(config_dict)))
            return int(cur.lastrowid)

    def finish_retrain_event(self, event_id: int, *, status: str, message: str, candidate_version=None,
                             n_uploaded=None, n_replay=None, current_metrics=None, candidate_metrics=None,
                             acceptance=None) -> None:
        with self._conn() as c:
            c.execute("UPDATE retraining_events SET status=?, finished_at=?, message=?, candidate_version=?, "
                      "n_uploaded_images=?, n_replay_images=?, current_metrics=?, candidate_metrics=?, acceptance=? "
                      "WHERE id=?", (status, now(), message, candidate_version, n_uploaded, n_replay,
                                     _json(_slim(current_metrics)), _json(_slim(candidate_metrics)),
                                     _json(acceptance), event_id))

    @staticmethod
    def _decode(row: Optional[dict]) -> Optional[dict]:
        if row:
            for k in ("current_metrics", "candidate_metrics", "acceptance", "config"):
                if row.get(k):
                    row[k] = json.loads(row[k])
        return row

    def get_retrain_event(self, event_id: int) -> Optional[dict]:
        return self._decode(self._one("SELECT * FROM retraining_events WHERE id=?", (event_id,)))

    def last_retrain_event(self, trigger: Optional[str] = None) -> Optional[dict]:
        if trigger:
            return self._decode(self._one("SELECT * FROM retraining_events WHERE trigger=? "
                                          "ORDER BY id DESC LIMIT 1", (trigger,)))
        return self._decode(self._one("SELECT * FROM retraining_events ORDER BY id DESC LIMIT 1"))

    def recent_retrain_events(self, limit: int = 10) -> list[dict]:
        return [self._decode(r) for r in self._all(
            "SELECT * FROM retraining_events ORDER BY id DESC LIMIT ?", (limit,))]

    # ------------------------------------------------------- model versions
    def upsert_model_version(self, meta: dict, n_uploaded: int = 0) -> None:
        val = (meta.get("metrics") or {}).get("validation") or {}
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO model_versions (version, created_at, status, parent_version, "
                      "val_accuracy, val_macro_f1, n_uploaded_images, metadata_file) VALUES (?,?,?,?,?,?,?,?)",
                      (meta["version"], meta["created_at"], meta["status"], meta.get("parent_version"),
                       val.get("accuracy"), val.get("f1_macro"), n_uploaded, f"model_v{meta['version']}.json"))

    def list_model_versions(self) -> list[dict]:
        return self._all("SELECT * FROM model_versions ORDER BY version DESC")


def _slim(metrics: Optional[dict]) -> Optional[dict]:
    """Keep the headline numbers (not the full confusion matrix) in the event row."""
    if not metrics:
        return None
    keep = ("n_samples", "accuracy", "loss", "f1_macro", "f1_weighted", "precision_macro", "recall_macro",
            "quadratic_weighted_kappa", "per_class")
    return {k: metrics[k] for k in keep if k in metrics}
