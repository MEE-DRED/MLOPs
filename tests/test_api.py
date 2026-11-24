"""HTTP-layer tests. They need fastapi + httpx and are SKIPPED when those are not installed
(as in the build sandbox). Run them in Colab/Docker: python -m unittest tests.test_api -v"""
import unittest

try:
    from fastapi.testclient import TestClient
    HAVE = True
except Exception:                      # pragma: no cover
    HAVE = False

from tests.helpers import array_png, fake_fundus, png_bytes
from tests.test_service import make_service


@unittest.skipUnless(HAVE, "fastapi / httpx not installed in this environment")
class ApiTests(unittest.TestCase):
    def setUp(self):
        from src import config
        from src.api import create_app
        config.API_KEY = ""
        self.svc, self.env = make_service()
        self.client = TestClient(create_app(self.svc)); self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)

    def test_health(self):
        r = self.client.get("/health"); self.assertEqual(r.status_code, 200)
        self.assertEqual({"status", "model_loaded", "model_version"} <= set(r.json()), True)
        self.assertTrue(r.json()["model_loaded"])

    def test_predict_ok_and_invalid(self):
        r = self.client.post("/predict", files={"file": ("a.png", png_bytes(), "image/png")})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(set(r.json()) >= {"predicted_class", "predicted_class_name", "probabilities", "model_version"}, True)
        self.assertEqual(self.client.post("/predict", files={"file": ("a.png", b"junk", "image/png")}).status_code, 422)

    def test_upload_and_validation(self):
        imgs = fake_fundus(2, seed=33)
        files = [("files", (f"{i}.png", array_png(imgs[i]), "image/png")) for i in range(2)]
        r = self.client.post("/upload-retraining-data", files=files, data={"labels": "1,9"})
        self.assertEqual(r.status_code, 200); self.assertEqual((r.json()["n_accepted"], r.json()["n_rejected"]), (1, 1))
        bad = self.client.post("/upload-retraining-data", files=files, data={"labels": "1"})
        self.assertEqual(bad.status_code, 422)

    def test_retrain_without_data_explains(self):
        r = self.client.post("/retrain")
        self.assertEqual(r.status_code, 422); self.assertIn("message", r.json()["detail"])

    def test_monitoring(self):
        r = self.client.get("/monitoring"); self.assertEqual(r.status_code, 200)
        self.assertIn("predictions", r.json())

    def test_api_key_enforced_when_configured(self):
        from src import config
        config.API_KEY = "secret"
        try:
            self.assertEqual(self.client.post("/retrain").status_code, 401)
            self.assertNotEqual(self.client.post("/retrain", headers={"X-API-Key": "secret"}).status_code, 401)
        finally:
            config.API_KEY = ""


if __name__ == "__main__":
    unittest.main()
