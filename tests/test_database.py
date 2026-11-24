import tempfile
import unittest
from pathlib import Path

from src.database import Database, DatabaseError, DuplicateUploadError

F = {"brightness": 1.0, "contrast": 2.0, "sharpness": 3.0, "retina_fraction": 0.7}


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.path = Path(tempfile.mkdtemp()) / "sub" / "t.sqlite3"
        self.db = Database(self.path)

    def add(self, sha="a", label=1, name="x.png"):
        return self.db.add_upload(batch_id="b", filename=name, stored_path="/tmp/x", sha256=sha, label=label, features=F)

    def test_initialisation_creates_tables_and_dirs(self):
        self.assertTrue(self.path.is_file())
        import sqlite3
        names = {r[0] for r in sqlite3.connect(self.path).execute("select name from sqlite_master where type='table'")}
        self.assertTrue({"uploads", "predictions", "retraining_events", "model_versions"} <= names)

    def test_upload_record_fields(self):
        self.add()
        row = self.db.accepted_uploads()[0]
        self.assertEqual((row["original_filename"], row["label"], row["status"]), ("x.png", 1, "accepted"))
        self.assertTrue(row["uploaded_at"]); self.assertIsNone(row["retrain_event_id"])
        self.assertEqual(row["brightness"], 1.0)

    def test_persists_across_restart(self):
        self.add(); self.db.log_prediction(success=True, predicted_class=2, confidence=.9, model_version=1)
        again = Database(self.path)
        self.assertEqual(again.count_uploads(), 1); self.assertEqual(again.prediction_stats()["total"], 1)

    def test_duplicate_accepted_rejected_but_rejected_rows_do_not_block(self):
        self.add("same")
        with self.assertRaises(DuplicateUploadError):
            self.add("same")
        self.db.add_rejected_upload(batch_id="b", filename="r.png", label=1, reason="bad", sha256="same")
        self.assertEqual(self.db.count_uploads("accepted"), 1); self.assertEqual(self.db.count_uploads("rejected"), 1)

    def test_class_counts(self):
        self.add("a", 0); self.add("b", 0); self.add("c", 4)
        self.assertEqual(self.db.upload_class_counts(), [2, 0, 0, 0, 1])

    def test_check_constraint_blocks_bad_status(self):
        import sqlite3
        with self.assertRaises(sqlite3.IntegrityError):
            sqlite3.connect(self.path).execute("insert into uploads(status, uploaded_at) values ('weird','now')")

    def test_prediction_stats(self):
        self.db.log_prediction(success=True, predicted_class=3, confidence=.8, model_version=1, latency_ms=10)
        self.db.log_prediction(success=True, predicted_class=3, confidence=.7, model_version=1, latency_ms=30)
        self.db.log_prediction(success=False, error="invalid_image")
        s = self.db.prediction_stats()
        self.assertEqual((s["total"], s["successful"], s["failed"]), (3, 2, 1))
        self.assertEqual(s["by_class"], [0, 0, 0, 2, 0]); self.assertEqual(s["errors"], {"invalid_image": 1})
        self.assertAlmostEqual(s["avg_latency_ms"], 20.0)
        self.assertEqual(len(self.db.recent_predictions(2)), 2)

    def test_retrain_event_lifecycle_and_json(self):
        eid = self.db.create_retrain_event("manual", 1, {"lr": 1e-5})
        self.assertEqual(self.db.get_retrain_event(eid)["status"], "running")
        self.db.finish_retrain_event(eid, status="accepted", message="ok", candidate_version=2, n_uploaded=30,
                                     candidate_metrics={"accuracy": .8, "confusion_matrix": [[1]], "f1_macro": .7})
        ev = self.db.get_retrain_event(eid)
        self.assertEqual(ev["status"], "accepted"); self.assertEqual(ev["config"], {"lr": 1e-5})
        self.assertEqual(ev["candidate_metrics"]["accuracy"], .8)
        self.assertNotIn("confusion_matrix", ev["candidate_metrics"])           # slimmed
        self.assertEqual(self.db.last_retrain_event()["id"], eid)

    def test_stale_running_event_marked_failed_on_restart(self):
        eid = self.db.create_retrain_event("auto", 1, {})
        again = Database(self.path)
        ev = again.get_retrain_event(eid)
        self.assertEqual(ev["status"], "failed"); self.assertIn("restarted", ev["message"])

    def test_mark_attempted_is_idempotent(self):
        i = self.add()
        self.db.mark_uploads_attempted([i], 1); self.db.mark_uploads_attempted([i], 2)
        self.assertEqual(self.db.accepted_uploads()[0]["retrain_event_id"], 1)
        self.assertEqual(self.db.count_uploads(only_new=True), 0)

    def test_model_versions_mirror(self):
        self.db.upsert_model_version({"version": 2, "created_at": "t", "status": "accepted", "parent_version": 1,
                                      "metrics": {"validation": {"accuracy": .5, "f1_macro": .4}}}, 25)
        v = self.db.list_model_versions()[0]
        self.assertEqual((v["version"], v["val_macro_f1"], v["n_uploaded_images"]), (2, .4, 25))

    def test_bad_sql_raises_database_error(self):
        with self.assertRaises(DatabaseError):
            self.db._all("select * from no_such_table")


if __name__ == "__main__":
    unittest.main()
