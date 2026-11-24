import threading
import unittest

import numpy as np

from src.reference import array_hash
from src.retrain import (RetrainSettings, RetrainingBusy, RetrainingManager, RetrainingNotPossible,
                         stratified_sample)
from tests.helpers import array_png, fake_fundus, make_env, upload_images


def manager(env, **kw):
    return RetrainingManager(env["db"], env["registry"], env["reference"], env["trainer"],
                             settings=RetrainSettings(min_images=10, min_classes=2, replay_size=30, epochs=1),
                             cooldown_minutes=kw.pop("cooldown_minutes", 0), **kw)


class EligibilityTests(unittest.TestCase):
    def test_no_data_gives_clear_explanation(self):
        env = make_env(); m = manager(env)
        e = m.eligibility()
        self.assertFalse(e["eligible"]); self.assertIn("at least 10", e["message"])
        with self.assertRaises(RetrainingNotPossible) as cm:
            m.start(wait=True)
        self.assertIn("only 0 valid labelled", str(cm.exception))
        self.assertIsNone(env["db"].last_retrain_event())          # nothing pretended
        self.assertEqual(env["registry"].current_version(), 1)

    def test_single_class_not_enough(self):
        env = make_env(); upload_images(env, 12, labels=[1] * 12)
        e = manager(env).eligibility()
        self.assertFalse(e["eligible"]); self.assertIn("different classes", e["message"])

    def test_no_current_model(self):
        env = make_env(); env["registry"].pointer_path.unlink(); upload_images(env, 12)
        self.assertIn("no current model", manager(env).eligibility()["message"])

    def test_dataset_unavailable(self):
        env = make_env(); upload_images(env, 12)
        env["reference"].available = lambda: False
        self.assertIn("not available", manager(env).eligibility()["message"])

    def test_ready(self):
        env = make_env(); upload_images(env, 12)
        e = manager(env).eligibility(); self.assertTrue(e["eligible"]); self.assertEqual(e["n_accepted_images"], 12)


class RetrainingFlowTests(unittest.TestCase):
    def test_accepted_candidate_becomes_new_current_version(self):
        env = make_env(current_acc=0.60, candidate_acc=0.90)
        upload_images(env, 25)
        changed = []
        m = manager(env, on_model_changed=lambda: changed.append(env["registry"].current_version()))
        ev = m.start(wait=True)
        self.assertEqual(ev["status"], "accepted", ev["message"])
        self.assertEqual((ev["parent_version"], ev["candidate_version"]), (1, 2))
        self.assertEqual(env["registry"].current_version(), 2); self.assertEqual(changed, [2])
        self.assertGreater(ev["candidate_metrics"]["f1_macro"], ev["current_metrics"]["f1_macro"])
        self.assertTrue(ev["acceptance"]["accepted"])
        meta = env["registry"].get_metadata(2)
        self.assertEqual((meta["status"], meta["parent_version"]), ("accepted", 1))
        self.assertEqual(meta["training_dataset"]["n_uploaded_images"], 25)
        self.assertIsNone(meta["metrics"]["test"])                       # test split never evaluated
        self.assertEqual(env["registry"].model_path(1).read_text(), '{"acc": 0.6}')   # v1 untouched
        self.assertEqual(env["db"].list_model_versions()[0]["version"], 2)
        self.assertFalse(list((env["registry"].models_dir / "staging").glob("*")))    # staging cleaned
        self.assertFalse(m.is_running())

    def test_starts_from_current_model_not_fresh(self):
        env = make_env(current_acc=0.60, candidate_acc=0.90); upload_images(env, 15)
        seen = {}
        orig = env["trainer"].finetune
        def spy(model, new, replay, val, settings):
            seen["acc_before"] = model.acc                   # weights loaded from the registry file
            return orig(model, new, replay, val, settings)
        env["trainer"].finetune = spy
        manager(env).start(wait=True)
        self.assertEqual(seen["acc_before"], 0.60)

    def test_inputs_to_training_never_include_test_images(self):
        env = make_env(); upload_images(env, 15); manager(env).start(wait=True)
        call = env["trainer"].finetune_calls[0]
        test_hashes = {array_hash(i) for i in env["reference"]._test[0]}
        for arrays in (call["new"][0], call["replay"][0], call["val"][0]):
            self.assertFalse(test_hashes & {array_hash(i) for i in arrays})
        # validation passed to the trainer is the ORIGINAL validation split
        np.testing.assert_array_equal(call["val"][1], env["reference"].val()[1])
        self.assertEqual(len(call["new"][0]), 15); self.assertLessEqual(len(call["replay"][0]), 40)

    def test_replay_sample_is_stratified(self):
        y = np.array([0] * 80 + [1] * 15 + [4] * 5)
        idx = stratified_sample(y, 20, seed=1)
        self.assertEqual(set(y[idx]), {0, 1, 4})
        self.assertGreaterEqual((y[idx] == 4).sum(), 1)
        np.testing.assert_array_equal(idx, stratified_sample(y, 20, seed=1))          # reproducible

    def test_worse_candidate_rejected_and_current_keeps_serving(self):
        env = make_env(current_acc=0.80, candidate_acc=0.30); upload_images(env, 15)
        changed = []
        ev = manager(env, on_model_changed=lambda: changed.append(1)).start(wait=True)
        self.assertEqual(ev["status"], "rejected"); self.assertIn("REJECTED", ev["message"])
        self.assertFalse(ev["acceptance"]["accepted"]); self.assertTrue(ev["acceptance"]["reasons"])
        self.assertEqual(env["registry"].current_version(), 1); self.assertEqual(changed, [])
        self.assertEqual(env["registry"].get_metadata(2)["status"], "rejected")       # kept for audit
        self.assertTrue(env["registry"].model_path(2).is_file())

    def test_equal_candidate_accepted_under_default_policy(self):
        env = make_env(current_acc=0.70, candidate_acc=0.70); upload_images(env, 15)
        self.assertEqual(manager(env).start(wait=True)["status"], "accepted")

    def test_training_crash_reports_failure_and_changes_nothing(self):
        env = make_env(fail=True); upload_images(env, 15)
        m = manager(env); ev = m.start(wait=True)
        self.assertEqual(ev["status"], "failed"); self.assertIn("simulated training crash", ev["message"])
        self.assertIn("current model was not changed", ev["message"])
        self.assertEqual(env["registry"].current_version(), 1); self.assertEqual(len(env["registry"].list_versions()), 1)
        self.assertFalse(m.is_running())

    def test_corrupt_saved_candidate_is_rejected(self):
        env = make_env(); upload_images(env, 15)
        orig_load = env["trainer"].load
        calls = {"n": 0}
        def load(path):
            calls["n"] += 1
            if calls["n"] == 2:                               # the re-load check of the saved candidate
                class Bad:
                    def predict(self, b, verbose=0): return np.full((len(b), 5), np.nan)
                return Bad()
            return orig_load(path)
        env["trainer"].load = load
        ev = manager(env).start(wait=True)
        self.assertEqual(ev["status"], "rejected"); self.assertIn("re-load", ev["message"])
        self.assertEqual(env["registry"].current_version(), 1)

    def test_reload_failure_is_reported_but_version_stays_registered(self):
        env = make_env(); upload_images(env, 15)
        def boom(): raise RuntimeError("cannot reload")
        ev = manager(env, on_model_changed=boom).start(wait=True)
        self.assertEqual(ev["status"], "accepted"); self.assertIn("failed to reload", ev["message"])

    def test_leaked_image_in_db_is_excluded_at_retraining_time(self):
        env = make_env(); upload_images(env, 12)
        from src.uploads import ingest_upload
        # bypass the upload-time check (no hash index) to simulate an old/unchecked record
        ingest_upload(env["db"], data=array_png(env["reference"]._test[0][0]), filename="leak.png", label=1,
                      batch_id="x", upload_dir=env["upload_dir"], hash_index=None)
        self.assertEqual(env["db"].count_uploads("accepted"), 13)
        manager(env).start(wait=True)
        self.assertEqual(env["db"].count_uploads("excluded"), 1)
        self.assertEqual(len(env["trainer"].finetune_calls[0]["new"][0]), 12)

    def test_insufficient_after_revalidation_is_skipped_not_trained(self):
        env = make_env(); upload_images(env, 12)
        for row in env["db"].accepted_uploads()[:5]:
            __import__("pathlib").Path(row["stored_path"]).unlink()
        ev = manager(env).start(wait=True)
        self.assertEqual(ev["status"], "skipped"); self.assertEqual(env["trainer"].finetune_calls, [])
        self.assertEqual(env["db"].count_uploads(only_new=True), 7)       # not marked as attempted

    def test_uploads_marked_as_attempted_and_reused_cumulatively(self):
        env = make_env(); upload_images(env, 12); m = manager(env); m.start(wait=True)
        self.assertEqual(env["db"].count_uploads(only_new=True), 0)
        upload_images(env, 3, seed=950)
        self.assertEqual(env["db"].count_uploads(only_new=True), 3)
        m.start(wait=True)
        self.assertEqual(len(env["trainer"].finetune_calls[1]["new"][0]), 15)   # old + new


class ConcurrencyAndAutoTriggerTests(unittest.TestCase):
    def test_second_job_refused_while_one_runs(self):
        gate = threading.Event()
        env = make_env(gate=gate); upload_images(env, 15)
        m = manager(env)
        first = m.start(wait=False)
        self.assertEqual(first["status"], "running"); self.assertTrue(m.is_running())
        self.assertTrue(m.eligibility()["running"])
        with self.assertRaises(RetrainingBusy):
            m.start(wait=False)
        self.assertIsNone(m.maybe_auto_trigger())
        gate.set(); m.join(10)
        self.assertFalse(m.is_running())
        self.assertEqual(env["db"].get_retrain_event(first["event_id"])["status"], "accepted")
        self.assertEqual(len(env["trainer"].finetune_calls), 1)

    def test_auto_disabled_never_triggers(self):
        env = make_env(); upload_images(env, 20)
        self.assertIsNone(manager(env, auto_enabled=False, auto_min_new=5).maybe_auto_trigger())

    def test_auto_waits_for_threshold(self):
        env = make_env(); m = manager(env, auto_enabled=True, auto_min_new=15)
        upload_images(env, 12); self.assertIsNone(m.maybe_auto_trigger())
        upload_images(env, 5, seed=951)
        r = m.maybe_auto_trigger(); self.assertIsNotNone(r); m.join(10)
        self.assertEqual(env["db"].last_retrain_event()["trigger"], "auto")

    def test_auto_does_not_refire_without_new_images(self):
        env = make_env(); m = manager(env, auto_enabled=True, auto_min_new=10)
        upload_images(env, 12); m.maybe_auto_trigger(); m.join(10)
        self.assertIsNone(m.maybe_auto_trigger())
        self.assertEqual(len(env["trainer"].finetune_calls), 1)

    def test_auto_does_not_retry_after_failure(self):
        env = make_env(fail=True); m = manager(env, auto_enabled=True, auto_min_new=10)
        upload_images(env, 12); m.maybe_auto_trigger(); m.join(10)
        self.assertEqual(env["db"].last_retrain_event()["status"], "failed")
        self.assertIsNone(m.maybe_auto_trigger()); self.assertIsNone(m.maybe_auto_trigger())
        self.assertEqual(len(env["trainer"].finetune_calls), 1)

    def test_cooldown_blocks_back_to_back_jobs(self):
        env = make_env(); m = manager(env, auto_enabled=True, auto_min_new=5, cooldown_minutes=60)
        upload_images(env, 12); m.maybe_auto_trigger(); m.join(10)
        upload_images(env, 8, seed=952)                          # plenty of new images, but too soon
        self.assertIsNone(m.maybe_auto_trigger())
        self.assertEqual(len(env["trainer"].finetune_calls), 1)

    def test_manual_button_still_works_with_auto_on(self):
        env = make_env(); m = manager(env, auto_enabled=True, auto_min_new=1000)
        upload_images(env, 12)
        self.assertEqual(m.start("manual", wait=True)["trigger"], "manual")


if __name__ == "__main__":
    unittest.main()
