"""Runs every dashboard page against a stand-in `streamlit` and the real client/service/database/registry stack.
Verifies our logic and wiring - NOT real Streamlit rendering."""
import importlib
import sys
import threading
import unittest

from app.api_client import ApiClient
from tests.dashboard_fakes import FakeStreamlit, FakeUpload, LocalTransport, StopApp
from tests.helpers import array_png, fake_fundus, make_env, png_bytes, upload_images
from tests.test_service import make_service


class DashboardBase(unittest.TestCase):
    def setUp(self, **env_kw):
        self.st = FakeStreamlit()
        self._orig = sys.modules.get("streamlit")
        sys.modules["streamlit"] = self.st
        sys.modules.pop("app.dashboard", None)
        self.dash = importlib.import_module("app.dashboard")
        self.env = make_env(**env_kw)
        self.svc, _ = make_service(self.env)
        self.svc.startup()
        self.transport = LocalTransport(self.svc)
        self.client = ApiClient("http://api:8000", session=self.transport)
        self.dash.get_client = lambda: self.client

    def tearDown(self):
        sys.modules.pop("app.dashboard", None)
        if self._orig is not None:
            sys.modules["streamlit"] = self._orig
        else:
            sys.modules.pop("streamlit", None)

    def show(self, page):
        self.st.page = page
        self.st.calls.clear()
        self.st.button_states.clear()
        self.st.fragments.clear()
        self.dash.main()

    def metrics(self):
        return {c[1][0]: c[1][1] for c in self.st.of("metric")}


class OverviewAndHealthTests(DashboardBase):
    def test_overview_shows_real_values(self):
        self.client.predict(png_bytes(), "a.png")
        upload_images(self.env, 4)
        self.show("Overview")
        m = self.metrics()
        self.assertEqual((m["API status"], m["Current model"], m["Total predictions"]), ("Online", "v1", 1))
        self.assertEqual((m["Retraining images uploaded"], m["Last retraining"]), (4, "never run"))
        self.assertEqual(self.st.of("error"), [])
        self.assertTrue(any("not a medical device" in t.lower() for t in self.st.texts("warning")))

    def test_api_down_shows_error_and_stops_without_claiming_online(self):
        self.transport.down = True
        with self.assertRaises(StopApp):
            self.show("Overview")
        self.assertTrue(any("not reachable" in t for t in self.st.texts("error")))
        self.assertEqual(self.st.of("metric"), [])
        self.assertTrue(self.st.sidebar.__dict__["_st"] is self.st)
        self.assertTrue(any("API unreachable" in t for t in self.st.texts("error")))

    def test_no_model_loaded_is_reported(self):
        self.svc.predictor = None
        self.show("Overview")
        self.assertEqual(self.metrics()["Current model"], "none")
        self.assertTrue(any("No model is loaded" in t for t in self.st.texts("error")))
        self.assertTrue(any("NO model loaded" in t for t in self.st.texts("warning")))


class PredictPageTests(DashboardBase):
    def test_prediction_flow(self):
        self.st.uploads = FakeUpload("eye.png", png_bytes((300, 250)))
        self.st.click = {"Classify"}
        self.show("Predict")
        self.assertEqual(self.st.of("error"), [])
        self.assertTrue(any("3 - Severe" in t for t in self.st.texts("subheader")))
        self.assertEqual(self.metrics()["Confidence"], "80.0%")
        frame = self.st.of("bar_chart")[0][1][0]
        self.assertEqual(len(frame), 5); self.assertAlmostEqual(frame["probability"].sum(), 1.0, places=5)
        self.assertEqual(self.svc.db.prediction_stats()["successful"], 1)
        self.assertTrue(any("not a medical device" in t.lower() for t in self.st.texts("warning")))

    def test_result_not_shown_before_click(self):
        self.st.uploads = FakeUpload("eye.png", png_bytes())
        self.show("Predict")
        self.assertEqual(self.st.of("bar_chart"), []); self.assertEqual(self.svc.db.prediction_stats()["total"], 0)

    def test_no_file(self):
        self.show("Predict")
        self.assertTrue(self.st.of("info"))

    def test_invalid_file_shows_error_and_is_logged(self):
        self.st.uploads = FakeUpload("bad.png", b"not an image")
        self.st.click = {"Classify"}
        self.show("Predict")
        self.assertTrue(any("Invalid image" in t for t in self.st.texts("error")))
        self.assertEqual(self.svc.db.prediction_stats()["errors"], {"invalid_image": 1})


class InsightsPageTests(DashboardBase):
    def test_all_tabs_render(self):
        self.show("Dataset insights")
        self.assertEqual(self.st.of("error"), [])
        self.assertGreaterEqual(len(self.st.of("pyplot")), 3)            # feature figure + 2 confusion matrices
        self.assertTrue(any("Rank correlation" in t for t in self.st.texts("markdown")))
        self.assertTrue(any("not causes" in t or "not causes or signs" in t for t in self.st.texts("info")))
        self.assertGreaterEqual(len(self.st.of("dataframe")), 6)
        self.assertIn("Accuracy", self.metrics())

    def test_missing_feature_table_is_explained(self):
        self.dash.FEATURES_CSV = self.env["root"] / "nope.csv"
        self.dash.load_features = lambda: None
        self.show("Dataset insights")
        self.assertTrue(any("not found" in t for t in self.st.texts("info")))


class UploadPageTests(DashboardBase):
    def uploads(self, n, seed=70):
        imgs = fake_fundus(n, seed=seed)
        return [FakeUpload(f"f{i}.png", array_png(imgs[i])) for i in range(n)]

    def test_upload_is_blocked_until_every_image_has_a_label(self):
        self.st.uploads = self.uploads(3)
        self.show("Bulk upload")
        self.assertTrue(any("still need a label" in t for t in self.st.texts("warning")))
        self.assertIn(("Upload 3 labelled image(s)", True), self.st.button_states)      # disabled
        self.assertEqual(self.env["db"].count_uploads("accepted"), 0)

    def test_labelled_upload_reaches_database_and_shows_results(self):
        self.st.uploads = self.uploads(4)
        self.st.selectbox_values = {"label_": "2 - Moderate"}
        self.st.click = {"Upload 4"}
        self.show("Bulk upload")
        self.assertEqual(self.env["db"].upload_class_counts(), [0, 0, 4, 0, 0])
        a = self.metrics()
        self.assertEqual((a["Accepted"], a["Rejected"], a["Total stored images"]), (4, 0, 4))
        self.assertTrue(any("All files were accepted" in t for t in self.st.texts("success")))
        frame = self.st.of("dataframe")[-1][1][0]
        self.assertEqual(list(frame["result"]), ["accepted"] * 4)

    def test_rejected_files_reported_with_reason(self):
        files = self.uploads(2, seed=71) + [FakeUpload("bad.png", b"junk")]
        self.st.uploads = files
        self.st.selectbox_values = {"label_": "1 - Mild"}
        self.st.click = {"Upload 3"}
        self.show("Bulk upload")
        self.assertEqual((self.metrics()["Accepted"], self.metrics()["Rejected"]), (2, 1))
        frame = self.st.of("dataframe")[-1][1][0]
        self.assertIn("invalid image", frame.loc[frame["file"] == "bad.png", "reason"].iloc[0])
        self.assertTrue(self.st.texts("error"))

    def test_apply_to_all_shortcut(self):
        self.st.uploads = self.uploads(3, seed=72)
        self.st.click = {"Apply to all"}
        self.st.session_state["set_all_choice"] = "4 - Proliferative DR"
        self.show("Bulk upload")
        self.assertEqual(self.st.session_state["label_0_f0.png_%d" % self.st.uploads[0].size], "4 - Proliferative DR")


class RetrainPageTests(DashboardBase):
    def setUp(self):
        self.gate = threading.Event()
        super().setUp(gate=self.gate)

    def tearDown(self):
        self.gate.set(); self.svc.manager.join(10)
        super().tearDown()

    def test_button_disabled_and_reason_shown_without_data(self):
        self.show("Retraining")
        self.assertIn(("🔁 Retrain model now", True), self.st.button_states)
        self.assertTrue(any("Cannot retrain" in t for t in self.st.texts("warning")))
        self.assertEqual(self.env["db"].last_retrain_event(), None)

    def test_click_starts_real_job_polls_and_shows_result(self):
        upload_images(self.env, 25)
        self.svc.manager.settings = self.svc.manager.settings.__class__(min_images=10, min_classes=2, replay_size=30, epochs=1)
        # --- ready: enabled button
        self.show("Retraining")
        self.assertIn(("🔁 Retrain model now", False), self.st.button_states)
        self.assertEqual(self.st.fragments, [None])                       # idle: no polling
        # --- click: real backend job starts (held open by the gate) and the app reruns
        self.st.click = {"Retrain model now"}
        self.show("Retraining")
        self.assertTrue(self.st.of("rerun"))
        self.assertTrue(self.svc.manager.is_running())
        self.assertEqual(self.env["db"].last_retrain_event()["status"], "running")
        # --- rerun while running: button disabled (no double start), polling every 3 s, progress text
        self.st.click = {"Retrain model now"}
        self.show("Retraining")
        self.assertIn(("🔁 Retrain model now", True), self.st.button_states)
        self.assertEqual(self.st.fragments, [3])
        self.assertTrue(any("Training in progress" in t for t in self.st.texts("info")))
        self.assertEqual(len(self.env["db"].recent_retrain_events(5)), 1)        # the second click did NOT start a job
        self.st.click = set()                                             # real clicks are one-shot
        # --- job finishes: panel detects the transition, triggers a rerun and shows the outcome
        self.gate.set(); self.svc.manager.join(10)
        self.show("Retraining")
        self.assertTrue(self.st.of("rerun"))
        self.show("Retraining")
        self.assertTrue(any("accepted" in t.lower() for t in self.st.texts("success") + self.st.texts("write") + self.st.texts("markdown")))
        table = [c[1][0] for c in self.st.of("dataframe") if "current model" in getattr(c[1][0], "columns", [])]
        self.assertTrue(table)                                            # candidate vs current metrics table
        self.assertEqual(self.svc.health()["model_version"], 2)
        self.assertIn(("🔁 Retrain model now", False), self.st.button_states)

    def test_busy_conflict_is_reported_not_crashed(self):
        upload_images(self.env, 25)
        self.svc.manager.settings = self.svc.manager.settings.__class__(min_images=10, min_classes=2, replay_size=30, epochs=1)
        self.client.retrain()                                             # someone else started a job
        self.st.session_state["retrain_clicked"] = False
        self.st.click = {"Retrain model now"}
        self.show("Retraining")                                           # button is disabled while running
        self.assertEqual(len(self.env["db"].recent_retrain_events(5)), 1)

    def test_rollback_control(self):
        upload_images(self.env, 25)
        self.svc.manager.settings = self.svc.manager.settings.__class__(min_images=10, min_classes=2, replay_size=30, epochs=1)
        self.gate.set(); self.svc.retrain(wait=True)
        self.assertEqual(self.svc.health()["model_version"], 2)
        self.st.checkbox_values = {"I understand this changes the model used for predictions": True}
        self.st.click = {"Roll back"}
        self.show("Retraining")
        self.assertEqual(self.svc.health()["model_version"], 1)
        self.assertTrue(any("Now serving version 1" in t for t in self.st.texts("success")))


class MonitoringPageTests(DashboardBase):
    def test_numbers_come_from_recorded_activity(self):
        self.client.predict(png_bytes(), "a.png"); self.client.predict(png_bytes(), "b.png")
        try: self.client.predict(b"junk", "c.png")
        except Exception: pass
        upload_images(self.env, 5, labels=[0, 1, 2, "bad", 4])
        self.show("Monitoring")
        m = self.metrics()
        self.assertEqual((m["Total requests"], m["Successful"], m["Failed"]), (3, 2, 1))
        self.assertEqual((m["Stored images"], m["Invalid uploads"]), (4, 1))
        self.assertEqual((m["API status"], m["Model available"], m["Model version"]), ("Online", "Yes", "v1"))
        self.assertTrue(any("NOT image drift" in t for t in self.st.texts("markdown")))
        self.assertTrue(any("Not enough" in t or "Need at least" in t for t in self.st.texts("caption")))
        self.assertEqual(self.st.fragments, [None])

    def test_auto_refresh_polls(self):
        self.st.checkbox_values = {"Auto-refresh every 5 seconds": True}
        self.show("Monitoring")
        self.assertEqual(self.st.fragments, [5])


if __name__ == "__main__":
    unittest.main()
