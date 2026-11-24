import unittest

import numpy as np

from src import config
from src.evaluate import (AcceptancePolicy, compute_metrics, evaluate_acceptance,
                           misclassified_indices, model_integrity_check, predict_probabilities)
from tests.helpers import StubModel, fake_fundus


def onehot(y, k=5, p=0.9):
    out = np.full((len(y), k), (1 - p) / (k - 1))
    out[np.arange(len(y)), y] = p
    return out


class MetricsTests(unittest.TestCase):
    def test_perfect(self):
        y = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        m = compute_metrics(y, onehot(y))
        for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted",
                  "quadratic_weighted_kappa", "within_one_grade_accuracy", "roc_auc_ovr_macro"):
            self.assertAlmostEqual(m[k], 1.0, places=6, msg=k)
        self.assertEqual(m["confusion_matrix"], (2 * np.eye(5, dtype=int)).tolist())
        self.assertLess(m["loss"], 0.2)

    def test_known_values(self):
        y = np.array([0, 0, 0, 0, 1, 1, 2, 2, 3, 4])
        pred = np.array([0, 0, 0, 1, 1, 2, 2, 2, 3, 3])   # 7/10 correct
        m = compute_metrics(y, onehot(pred))
        self.assertAlmostEqual(m["accuracy"], 0.7)
        self.assertAlmostEqual(m["per_class"]["No DR"]["recall"], 0.75)
        self.assertAlmostEqual(m["per_class"]["No DR"]["precision"], 1.0)
        self.assertAlmostEqual(m["per_class"]["Proliferative DR"]["recall"], 0.0)
        self.assertEqual(m["per_class"]["Mild"]["support"], 2)
        self.assertEqual(m["confusion_matrix"][0], [3, 1, 0, 0, 0])
        self.assertAlmostEqual(m["within_one_grade_accuracy"], 1.0)       # all errors adjacent
        self.assertAlmostEqual(m["majority_class_baseline_accuracy"], 0.4)

    def test_macro_vs_weighted_differ_on_imbalance(self):
        y = np.array([0] * 8 + [4] * 2)
        pred = np.zeros(10, int)
        m = compute_metrics(y, onehot(pred))
        self.assertGreater(m["f1_weighted"], m["f1_macro"])

    def test_missing_class_gives_no_macro_auc(self):
        y = np.array([0, 1, 2, 3])
        self.assertIsNone(compute_metrics(y, onehot(y))["roc_auc_ovr_macro"])

    def test_bad_shapes(self):
        with self.assertRaises(ValueError):
            compute_metrics([0, 1], np.ones((2, 3)))

    def test_json_safe(self):
        import json
        y = np.arange(10) % 5
        json.dumps(compute_metrics(y, onehot(y)))

    def test_misclassified_sorted_by_confidence(self):
        y = np.array([0, 0, 0])
        probs = np.array([[.9, .1, 0, 0, 0], [.1, .6, .1, .1, .1], [.05, .05, .05, .05, .8]])
        self.assertEqual(list(misclassified_indices(y, probs)), [2, 1])

    def test_predict_probabilities_batches(self):
        m = StubModel(); x = fake_fundus(5)
        p = predict_probabilities(m, x, batch_size=2)
        self.assertEqual(p.shape, (5, 5)); self.assertEqual(m.calls, 3)


class IntegrityTests(unittest.TestCase):
    def test_good(self):
        self.assertTrue(model_integrity_check(StubModel())[0])

    def test_nan(self):
        ok, msg = model_integrity_check(StubModel(bad="nan")); self.assertFalse(ok); self.assertIn("NaN", msg)

    def test_shape(self):
        self.assertFalse(model_integrity_check(StubModel(bad="shape"))[0])

    def test_exception(self):
        class Boom:
            def predict(self, *a, **k): raise RuntimeError("x")
        self.assertFalse(model_integrity_check(Boom())[0])


def metrics(f1, acc, recalls, n=120, support=(54, 12, 28, 20, 6)):
    return {"n_samples": n, "f1_macro": f1, "accuracy": acc,
            "per_class": {name: {"recall": r, "support": s}
                          for name, r, s in zip(config.CLASS_NAMES, recalls, support)}}


class AcceptanceTests(unittest.TestCase):
    cur = metrics(0.50, 0.60, [0.9, 0.3, 0.5, 0.5, 0.5])

    def test_first_model_accepted(self):
        self.assertTrue(evaluate_acceptance(self.cur, None)["accepted"])

    def test_first_model_failing_integrity(self):
        r = evaluate_acceptance(self.cur, None, integrity_ok=False, integrity_message="bad")
        self.assertFalse(r["accepted"]); self.assertIn("bad", r["reasons"][0])

    def test_better_accepted(self):
        cand = metrics(0.55, 0.62, [0.9, 0.4, 0.5, 0.5, 0.5])
        self.assertTrue(evaluate_acceptance(cand, self.cur)["accepted"])

    def test_worse_f1_rejected(self):
        cand = metrics(0.45, 0.61, [0.9, 0.3, 0.5, 0.5, 0.5])
        r = evaluate_acceptance(cand, self.cur)
        self.assertFalse(r["accepted"]); self.assertFalse(r["checks"]["macro_f1_not_worse"])

    def test_equal_f1_accepted_with_default_delta(self):
        self.assertTrue(evaluate_acceptance(dict(self.cur), self.cur)["accepted"])

    def test_min_delta_enforced(self):
        pol = AcceptancePolicy(min_delta_macro_f1=0.02)
        cand = metrics(0.51, 0.60, [0.9, 0.3, 0.5, 0.5, 0.5])
        self.assertFalse(evaluate_acceptance(cand, self.cur, policy=pol)["accepted"])

    def test_accuracy_collapse_rejected(self):
        cand = metrics(0.52, 0.50, [0.9, 0.3, 0.5, 0.5, 0.5])
        r = evaluate_acceptance(cand, self.cur)
        self.assertFalse(r["accepted"]); self.assertFalse(r["checks"]["accuracy_drop_ok"])

    def test_rare_class_collapse_rejected_despite_f1_gain(self):
        cand = metrics(0.60, 0.65, [0.95, 0.5, 0.6, 0.6, 0.0])      # class 4 recall 0.5 -> 0.0
        r = evaluate_acceptance(cand, self.cur)
        self.assertFalse(r["accepted"]); self.assertFalse(r["checks"]["no_class_collapse"])
        self.assertIn("Proliferative DR", " ".join(r["reasons"]))

    def test_tiny_eval_set_rejected(self):
        cand = metrics(0.9, 0.9, [1] * 5, n=10)
        self.assertFalse(evaluate_acceptance(cand, self.cur)["accepted"])

    def test_policy_recorded(self):
        self.assertIn("min_delta_macro_f1", evaluate_acceptance(self.cur, None)["policy"])


if __name__ == "__main__":
    unittest.main()
