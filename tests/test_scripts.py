"""Tests for the deployment/verification scripts (bootstrap, Space bundle, flood engine, e2e engine) and the admin gate."""
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from app.api_client import ApiClient
from scripts import e2e_check, flood_test
from scripts.prepare_runtime import prepare
from scripts.prepare_space_bundle import BundleError, build_bundle
from tests.dashboard_fakes import LocalTransport
from tests.helpers import array_png, fake_fundus, png_bytes
from tests.test_dashboard import DashboardBase
from tests.test_service import make_service

REPO = Path(__file__).resolve().parents[1]


def fake_models(root: Path, versions=(1, 2), current=2) -> Path:
    (root / "metadata").mkdir(parents=True)
    for v in versions:
        (root / f"model_v{v}.keras").write_bytes(f"model-{v}".encode())
        (root / "metadata" / f"model_v{v}.json").write_text(json.dumps({"version": v, "status": "accepted"}))
    (root / "current_model.json").write_text(json.dumps({"current_version": current, "history": [1], "events": []}))
    return root


class PrepareRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.baked = fake_models(self.tmp / "baked")

    def run_prepare(self, **kw):
        return prepare(self.tmp / "state", self.baked, kw.pop("dataset", self.tmp / "none.npz"), log=lambda *_: None, **kw)

    def test_first_start_seeds_registry_and_creates_folders(self):
        s = self.run_prepare()
        self.assertEqual(s["models_seeded"], ["model_v1.keras", "model_v2.keras"]); self.assertTrue(s["current_model_present"])
        for d in ("models/metadata", "database", "uploads"):
            self.assertTrue((self.tmp / "state" / d).is_dir())
        self.assertEqual((self.tmp / "state/models/model_v2.keras").read_bytes(), b"model-2")
        self.assertTrue((self.tmp / "state/models/metadata/model_v1.json").is_file())

    def test_restart_never_overwrites_persisted_state(self):
        self.run_prepare()
        st = self.tmp / "state" / "models"
        (st / "model_v3.keras").write_bytes(b"retrained"); (st / "model_v2.keras").write_bytes(b"CHANGED")
        (st / "current_model.json").write_text(json.dumps({"current_version": 3, "history": [2], "events": []}))
        s = self.run_prepare()
        self.assertEqual(s["models_seeded"], [])
        self.assertEqual(json.loads((st / "current_model.json").read_text())["current_version"], 3)
        self.assertEqual((st / "model_v2.keras").read_bytes(), b"CHANGED"); self.assertTrue((st / "model_v3.keras").is_file())

    def test_no_baked_model_reported_not_fatal(self):
        s = prepare(self.tmp / "s2", self.tmp / "empty", self.tmp / "none.npz", log=lambda *_: None)
        self.assertFalse(s["current_model_present"]); self.assertEqual(s["model_files"], [])

    def test_missing_dataset_is_reported_and_download_failure_is_not_fatal(self):
        self.assertFalse(self.run_prepare()["dataset_present"])
        with mock.patch("src.data_acquisition.urllib.request.urlretrieve", side_effect=OSError("offline")):
            s = self.run_prepare(auto_download=True)
        self.assertFalse(s["dataset_present"]); self.assertIn("download failed", s["dataset"])

    def test_dataset_present(self):
        ds = self.tmp / "d.npz"; ds.write_bytes(b"x")
        self.assertTrue(self.run_prepare(dataset=ds)["dataset_present"])


class SpaceBundleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.repo = self.tmp / "repo"
        for d in ("src", "app", "scripts", "deploy/huggingface", "outputs/tables"):
            (self.repo / d).mkdir(parents=True)
        for f in ("Dockerfile", "requirements-docker.txt"):
            (self.repo / f).write_text(f)
        (self.repo / "src/a.py").write_text("x"); (self.repo / "src/__pycache__").mkdir(); (self.repo / "src/__pycache__/a.pyc").write_text("x")
        (self.repo / "app/b.py").write_text("x"); (self.repo / "scripts/c.py").write_text("x")
        (self.repo / "outputs/tables/image_features.csv").write_text("a,b")
        shutil.copy2(REPO / "deploy/huggingface/README_SPACE.md", self.repo / "deploy/huggingface/README_SPACE.md")

    def test_refuses_without_model(self):
        with self.assertRaises(BundleError) as cm:
            build_bundle(self.tmp / "out", repo=self.repo)
        self.assertIn("No trained model", str(cm.exception))

    def test_bundle_contents(self):
        fake_models(self.repo / "models")
        ds = self.tmp / "retinamnist_224.npz"; ds.write_bytes(b"d" * 1000)
        m = build_bundle(self.tmp / "out", ds, repo=self.repo)
        out = self.tmp / "out"
        for p in ("Dockerfile", "requirements-docker.txt", "src/a.py", "app/b.py", "scripts/c.py", "models/current_model.json",
                  "models/model_v2.keras", "data/retinamnist_224.npz", ".gitattributes", ".dockerignore"):
            self.assertTrue((out / p).exists(), p)
        self.assertFalse((out / "src/__pycache__").exists())
        for excluded in ("tests", "notebook", "docs", "tools"):
            self.assertFalse((out / excluded).exists())
        readme = (out / "README.md").read_text()
        self.assertTrue(readme.startswith("---")); self.assertIn("sdk: docker", readme); self.assertIn("app_port: 7860", readme)
        self.assertIn("*.keras filter=lfs", (out / ".gitattributes").read_text())
        self.assertTrue(m["has_model"] and m["has_dataset"])

    def test_refuses_non_empty_output_and_missing_dataset(self):
        fake_models(self.repo / "models")
        (self.tmp / "out").mkdir(); (self.tmp / "out" / "x").write_text("x")
        with self.assertRaises(BundleError):
            build_bundle(self.tmp / "out", repo=self.repo)
        with self.assertRaises(BundleError):
            build_bundle(self.tmp / "out2", self.tmp / "missing.npz", repo=self.repo)

    def test_real_repo_has_everything_the_dockerfile_copies(self):
        text = (REPO / "Dockerfile").read_text()
        for src in [l.split()[2] for l in text.splitlines() if l.startswith("COPY")]:
            self.assertTrue((REPO / src).exists(), f"Dockerfile COPY source missing: {src}")
        self.assertTrue((REPO / "scripts/start_space.sh").stat().st_mode & 0o111, "start script must be executable")
        self.assertIn("scripts.prepare_runtime", (REPO / "scripts/start_space.sh").read_text())


class FloodEngineTests(unittest.TestCase):
    def test_counts_and_percentiles(self):
        calls = {"n": 0}
        def send():
            calls["n"] += 1
            return (200 if calls["n"] % 5 else 503), 0.0
        r = flood_test.run_flood(send, 50, 8)
        self.assertEqual((r["requests"], r["succeeded"], r["failed"]), (50, 40, 10))
        self.assertEqual(r["status_codes"], {"200": 40, "503": 10}); self.assertEqual(r["success_rate_percent"], 80.0)
        self.assertGreater(r["throughput_rps"], 0); self.assertLessEqual(r["latency_ms_successful"]["p50"], r["latency_ms_successful"]["p99"])

    def test_exceptions_counted_as_failures(self):
        def send(): raise ConnectionError("down")
        r = flood_test.run_flood(send, 10, 3)
        self.assertEqual((r["succeeded"], r["failed"], r["status_codes"]), (0, 10, {"ConnectionError": 10}))

    def test_runs_requests_concurrently(self):
        import threading, time
        active, peak, lock = [0], [0], threading.Lock()
        def send():
            with lock: active[0] += 1; peak[0] = max(peak[0], active[0])
            time.sleep(0.05)
            with lock: active[0] -= 1
            return 200, 0.0
        flood_test.run_flood(send, 20, 5)
        self.assertGreater(peak[0], 1)

    def test_markdown_row(self):
        r = flood_test.run_flood(lambda: (200, 0.0), 5, 2)
        self.assertIn("| POST /predict | 5 | 2 | 5/5 (100.0%)", flood_test.markdown_table(r, "POST /predict"))

    def test_synthetic_png_is_valid(self):
        from src.preprocessing import load_image
        self.assertEqual(load_image(flood_test._synthetic_png()).shape, (224, 224, 3))


class E2EEngineTests(unittest.TestCase):
    def setUp(self):
        self.svc, self.env = make_service(); self.svc.startup()
        self.client = ApiClient("http://api:8000", session=LocalTransport(self.svc))
        self.samples = [(f"true3_Severe_test{i}.png", array_png(fake_fundus(1, seed=40 + i)[0]), 3) for i in range(3)]

    def test_all_checks_pass_against_healthy_service(self):
        res = e2e_check.run_checks(self.client, self.samples)
        failed = [r for r in res if not r["ok"]]
        self.assertEqual(failed, [], failed)
        self.assertGreaterEqual(len(res), 10)
        self.assertEqual(res[-1]["detail"], "3/3")                      # stub model always says class 3
        self.assertEqual(self.env["db"].count_uploads("accepted"), 0)   # the bad-label probe stored nothing

    def test_api_down_fails_fast(self):
        self.client.session.down = True
        res = e2e_check.run_checks(self.client, self.samples)
        self.assertEqual((len(res), res[0]["ok"]), (1, False))

    def test_no_model_loaded_fails_health_check(self):
        self.svc.predictor = None
        res = e2e_check.run_checks(self.client, self.samples)
        self.assertFalse(res[0]["ok"])

    def test_detects_broken_server_behaviour(self):
        orig = self.svc.predict
        def wrong_version(data):
            r = orig(data); r["model_version"] = 99; return r
        self.svc.predict = wrong_version
        res = e2e_check.run_checks(self.client, self.samples)
        bad = next(r for r in res if "valid images" in r["name"])
        self.assertFalse(bad["ok"]); self.assertIn("v99", bad["detail"])

    def test_load_samples_parses_expected_class(self):
        d = Path(tempfile.mkdtemp())
        (d / "true2_Moderate_test5.png").write_bytes(png_bytes()); (d / "WRONG_true1_pred0_test9.png").write_bytes(png_bytes())
        got = {n: e for n, _, e in e2e_check.load_samples(d)}
        self.assertEqual(got, {"true2_Moderate_test5.png": 2, "WRONG_true1_pred0_test9.png": None})


class AdminGateTests(DashboardBase):
    def test_no_password_means_open(self):
        with mock.patch.dict(os.environ, {"DR_DASHBOARD_PASSWORD": ""}):
            self.show("Retraining")
        self.assertTrue(self.st.of("metric")); self.assertEqual(self.st.of("text_input"), [])

    def test_protected_pages_ask_for_password_and_do_nothing_else(self):
        with mock.patch.dict(os.environ, {"DR_DASHBOARD_PASSWORD": "pw123"}):
            for page in ("Retraining", "Bulk upload"):
                self.show(page)
                self.assertTrue(self.st.of("text_input"), page)
                self.assertEqual(self.st.of("metric"), []); self.assertEqual(self.st.button_states, [])
            self.show("Predict")                                         # public pages stay open
            self.assertEqual(self.st.of("text_input"), [])

    def test_wrong_and_right_password(self):
        with mock.patch.dict(os.environ, {"DR_DASHBOARD_PASSWORD": "pw123"}):
            self.st.text_inputs = {"admin_pw": "nope"}
            self.show("Retraining")
            self.assertTrue(any("Incorrect" in t for t in self.st.texts("error"))); self.assertFalse(self.st.session_state.get("admin_ok"))
            self.st.text_inputs = {"admin_pw": "pw123"}
            self.show("Retraining")
            self.assertTrue(self.st.session_state.get("admin_ok")); self.assertTrue(self.st.of("rerun"))
            self.show("Retraining")                                      # after the rerun the page renders
            self.assertTrue(self.st.of("metric"))


if __name__ == "__main__":
    unittest.main()
