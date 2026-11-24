import json
import unittest

import numpy as np
import pandas as pd

from app import ui_helpers as H
from app.api_client import ApiClient, ApiError, ApiUnavailable
from tests.dashboard_fakes import LocalTransport
from tests.helpers import array_png, fake_fundus, png_bytes
from tests.test_service import make_service


class HelperTests(unittest.TestCase):
    def test_label_choices_and_parsing(self):
        ch = H.label_choices()
        self.assertEqual(len(ch), 5); self.assertEqual(ch[3], "3 - Severe")
        self.assertEqual([H.choice_to_label(c) for c in ch], [0, 1, 2, 3, 4])
        for bad in (None, "", H.LABEL_PLACEHOLDER, "9 - Nope", "x"):
            self.assertIsNone(H.choice_to_label(bad))

    def test_probabilities_frame_order(self):
        r = {"probabilities": {"Severe": .1, "No DR": .6, "Mild": .1, "Moderate": .1, "Proliferative DR": .1}}
        self.assertEqual(list(H.probabilities_frame(r).index)[0], "No DR")

    def test_friendly_error(self):
        self.assertIn("HTTP 422", H.friendly_error(ApiError(422, {"message": "need 20 images"})))
        self.assertIn("need 20 images", H.friendly_error(ApiError(422, {"message": "need 20 images"})))
        self.assertIn("could not be reached", H.friendly_error(ApiUnavailable("refused")))

    def test_status_badge(self):
        self.assertIn("Rejected", H.status_badge("rejected")); self.assertIn("No retraining", H.status_badge(None))

    def test_performance_view_prefers_test_then_ancestor_then_validation(self):
        v = lambda n, test: {"version": n, "metrics": {"validation": {"accuracy": .5}, "test": test}}
        self.assertEqual(H.performance_view({"current_version": 1, "versions": [v(1, {"accuracy": .4})]})["split"], "test")
        r = H.performance_view({"current_version": 2, "versions": [v(1, {"accuracy": .4}), v(2, None)]})
        self.assertEqual((r["version"], r["split"]), (1, "test")); self.assertIn("never evaluated", r["note"])
        r = H.performance_view({"current_version": 1, "versions": [v(1, None)]})
        self.assertEqual(r["split"], "validation")
        self.assertIsNone(H.performance_view({"current_version": None, "versions": []}))

    def test_dataset_counts_from_first_version_that_has_them(self):
        models = {"versions": [{"version": 2, "training_dataset": {"base": "x"}},
                               {"version": 1, "training_dataset": {"class_counts": {"train": [1] * 5}}}]}
        self.assertEqual(H.dataset_counts(models), {"train": [1] * 5})
        self.assertIsNone(H.dataset_counts({"versions": []}))

    def test_metrics_comparison(self):
        ev = {"current_metrics": {"accuracy": .5, "f1_macro": .4}, "candidate_metrics": {"accuracy": .6, "f1_macro": .45}}
        t = H.metrics_comparison(ev)
        self.assertAlmostEqual(t.loc["Accuracy", "change"], 0.1); self.assertIsNone(H.metrics_comparison({}))

    def test_feature_summaries_on_real_table(self):
        from src import config
        df = pd.read_csv(config.TABLES_DIR / "image_features.csv")
        s = H.feature_summaries(df)
        self.assertEqual(list(s["medians"].index), list(config.CLASS_NAMES))
        self.assertLess(s["rho"].loc["brightness"].iloc[0], -0.25)        # real data: brighter = healthier
        fr = s["framing"].iloc[:, 0]
        self.assertLess(fr["No DR"], 20); self.assertGreater(fr["Proliferative DR"], 50)


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.svc, self.env = make_service(); self.svc.startup()
        self.tr = LocalTransport(self.svc)
        self.client = ApiClient("http://api:8000", session=self.tr)

    def test_health_predict_models_monitoring(self):
        self.assertTrue(self.client.health()["model_loaded"])
        r = self.client.predict(png_bytes(), "x.png")
        self.assertEqual(r["predicted_class_name"], "Severe")
        self.assertEqual(self.client.models()["current_version"], 1)
        self.assertEqual(self.client.monitoring()["predictions"]["total"], 1)

    def test_errors_become_apierror(self):
        with self.assertRaises(ApiError) as cm:
            self.client.predict(b"junk", "x.png")
        self.assertEqual(cm.exception.status, 422); self.assertIn("Invalid image", str(cm.exception.detail))
        with self.assertRaises(ApiError) as cm:
            self.client.retrain()
        self.assertEqual(cm.exception.status, 422); self.assertIn("message", cm.exception.detail)

    def test_unreachable(self):
        self.tr.down = True
        with self.assertRaises(ApiUnavailable):
            self.client.health()

    def test_upload_sends_labels_as_json_list(self):
        imgs = fake_fundus(3, seed=61)
        rep = self.client.upload_retraining_data([(f"{i}.png", array_png(imgs[i])) for i in range(3)], [0, 1, 4])
        self.assertEqual(rep["n_accepted"], 3)
        self.assertEqual(self.env["db"].upload_class_counts(), [1, 1, 0, 0, 1])
        with self.assertRaises(ValueError):
            self.client.upload_retraining_data([("a.png", b"x")], [1, 2])

    def test_api_key(self):
        tr = LocalTransport(self.svc, api_key="s3cret")
        with self.assertRaises(ApiError) as cm:
            ApiClient("http://api:8000", session=tr).retrain()
        self.assertEqual(cm.exception.status, 401)
        with self.assertRaises(ApiError) as cm:                           # authorised: now fails for a different reason
            ApiClient("http://api:8000", api_key="s3cret", session=tr).retrain()
        self.assertEqual(cm.exception.status, 422)
        self.assertTrue(ApiClient("http://api:8000", session=tr).health())     # health is open


if __name__ == "__main__":
    unittest.main()
