import unittest

import numpy as np

from src.model_registry import RegistryError
from src.prediction import ModelNotLoadedError, Predictor
from src.preprocessing import InvalidImageError
from src.retrain import RetrainSettings, RetrainingBusy, RetrainingManager, RetrainingNotPossible
from src.service import AppService
from tests.helpers import StubModel, array_png, fake_fundus, make_env, png_bytes, upload_images


def make_service(env=None, **kw):
    env = env or make_env()
    svc = AppService(
        db=env["db"], registry=env["registry"], reference=env["reference"], upload_dir=env["upload_dir"],
        predictor_factory=lambda reg: Predictor(model=StubModel(favourite=3), version=reg.current_version()),
        model_validator=lambda p: (True, "ok"),
        manager=RetrainingManager(env["db"], env["registry"], env["reference"], env["trainer"],
                                  settings=RetrainSettings(min_images=10, min_classes=2, replay_size=30, epochs=1),
                                  cooldown_minutes=0, **kw))
    svc.manager.on_model_changed = svc.reload_model
    return svc, env


class HealthTests(unittest.TestCase):
    def test_health_before_and_after_startup(self):
        svc, _ = make_service()
        h = svc.health()
        self.assertEqual((h["status"], h["model_loaded"], h["model_version"]), ("degraded", False, None))
        svc.startup()
        h = svc.health()
        self.assertEqual((h["status"], h["model_loaded"], h["model_version"], h["api"]), ("ok", True, 1, "up"))
        self.assertGreaterEqual(h["uptime_seconds"], 0); self.assertIn("Not a medical device", h["disclaimer"])

    def test_health_when_no_model_registered(self):
        env = make_env(); env["registry"].pointer_path.unlink()
        svc = AppService(db=env["db"], registry=env["registry"], reference=env["reference"],
                         upload_dir=env["upload_dir"], trainer=env["trainer"])
        svc.startup()
        h = svc.health()
        self.assertFalse(h["model_loaded"]); self.assertIn("No current model", h["model_error"])


class PredictionServiceTests(unittest.TestCase):
    def setUp(self):
        self.svc, self.env = make_service(); self.svc.startup()

    def test_successful_prediction_is_logged(self):
        r = self.svc.predict(png_bytes((300, 250)))
        self.assertEqual((r["predicted_class"], r["model_version"]), (3, 1)); self.assertIn("disclaimer", r)
        s = self.env["db"].prediction_stats()
        self.assertEqual((s["total"], s["successful"], s["failed"]), (1, 1, 0)); self.assertEqual(s["by_class"][3], 1)
        self.assertGreater(self.env["db"].prediction_features()["brightness"][0], -1)

    def test_invalid_image_is_logged_as_failure(self):
        with self.assertRaises(InvalidImageError):
            self.svc.predict(b"junk")
        s = self.env["db"].prediction_stats()
        self.assertEqual((s["total"], s["failed"], s["errors"]), (1, 1, {"invalid_image": 1}))

    def test_no_model_loaded(self):
        svc, env = make_service()                                     # startup() not called
        with self.assertRaises(ModelNotLoadedError):
            svc.predict(png_bytes())
        self.assertEqual(env["db"].prediction_stats()["errors"], {"model_not_loaded": 1})

    def test_reload_failure_keeps_old_predictor(self):
        old = self.svc.predictor
        self.svc._predictor_factory = lambda reg: (_ for _ in ()).throw(RuntimeError("bad file"))
        with self.assertRaises(RuntimeError):
            self.svc.reload_model()
        self.assertIs(self.svc.predictor, old); self.assertTrue(self.svc.health()["model_loaded"])


class UploadAndRetrainServiceTests(unittest.TestCase):
    def setUp(self):
        self.svc, self.env = make_service(); self.svc.startup()

    def files(self, n, seed=700):
        imgs = fake_fundus(n, seed=seed)
        return [(f"f{i}.png", array_png(imgs[i])) for i in range(n)]

    def test_upload_with_comma_labels(self):
        rep = self.svc.upload_retraining_data(self.files(4), "0,1,2,9")
        self.assertEqual((rep["n_accepted"], rep["n_rejected"], rep["total_available_images"]), (3, 1, 3))
        self.assertEqual(rep["leakage_check"], "performed")

    def test_label_count_mismatch_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.svc.upload_retraining_data(self.files(3), "0,1")
        self.assertEqual(self.env["db"].count_uploads("accepted") + self.env["db"].count_uploads("rejected"), 0)

    def test_full_cycle_upload_retrain_and_prediction_service_reloads(self):
        with self.assertRaises(RetrainingNotPossible):
            self.svc.retrain(wait=True)
        self.svc.upload_retraining_data(self.files(15), ",".join(str(i % 3) for i in range(15)))
        ev = self.svc.retrain(wait=True)
        self.assertEqual(ev["status"], "accepted")
        self.assertEqual(self.svc.health()["model_version"], 2)           # prediction service switched
        self.assertEqual(self.svc.predict(png_bytes())["model_version"], 2)
        st = self.svc.retrain_status()
        self.assertFalse(st["running"]); self.assertEqual(st["event"]["id"], ev["id"])
        self.assertEqual(self.svc.models()["current_version"], 2)

    def test_rollback(self):
        self.svc.upload_retraining_data(self.files(15), ",".join(str(i % 3) for i in range(15)))
        self.svc.retrain(wait=True)
        self.assertEqual(self.svc.rollback(), {"rolled_back_to": 1})
        self.assertEqual(self.svc.health()["model_version"], 1)
        with self.assertRaises(RegistryError):
            self.svc.rollback()

    def test_auto_retraining_started_from_upload(self):
        svc, env = make_service(auto_enabled=True, auto_min_new=10); svc.startup()
        rep = svc.upload_retraining_data(self.files(12, seed=710), ",".join(str(i % 3) for i in range(12)))
        self.assertEqual(rep["auto_retraining_started"]["status"], "running")
        svc.manager.join(10)
        self.assertEqual(env["db"].last_retrain_event()["trigger"], "auto")


class MonitoringSnapshotTests(unittest.TestCase):
    def test_snapshot_uses_recorded_data(self):
        svc, env = make_service(); svc.startup()
        svc.predict(png_bytes())
        try:
            svc.predict(b"bad")
        except InvalidImageError:
            pass
        files = [(f"f{i}.png", array_png(fake_fundus(1, seed=800 + i)[0])) for i in range(12)]
        svc.upload_retraining_data(files, ",".join(str(i % 3) for i in range(12)))
        svc.upload_retraining_data([("bad.png", b"junk")], "1")
        svc.retrain(wait=True)
        m = svc.monitoring()
        self.assertEqual(m["service"]["status"], "ok"); self.assertEqual(m["service"]["model_version"], 2)
        p = m["predictions"]
        self.assertEqual((p["total"], p["successful"], p["failed"]), (2, 1, 1))
        d = m["retraining_data"]
        self.assertEqual((d["available_images"], d["invalid_uploads"], d["class_counts"][:3]), (12, 1, [4, 4, 4]))
        self.assertEqual(d["new_since_last_retraining"], 0)
        self.assertEqual(m["retraining"]["recent_events"][0]["status"], "accepted")
        self.assertEqual(len(m["model_versions"]), 1)
        self.assertFalse(m["drift"]["class_distribution"]["assessable"])      # only 12 uploads: honest
        self.assertIn("NOT image drift", m["drift"]["class_distribution"]["kind"])
        import json; json.dumps(m, default=str)                               # serialisable


if __name__ == "__main__":
    unittest.main()
