import unittest

import numpy as np

from src.monitoring import class_distribution_shift, feature_drift, format_duration


class MonitoringTests(unittest.TestCase):
    def test_class_shift_not_assessable_when_small(self):
        r = class_distribution_shift([3, 2, 0, 0, 0], [486, 128, 206, 194, 66])
        self.assertFalse(r["assessable"]); self.assertIn("NOT image drift", r["kind"])

    def test_class_shift_values(self):
        r = class_distribution_shift([40, 0, 0, 0, 0], [20, 20, 20, 20, 20])
        self.assertTrue(r["assessable"]); self.assertAlmostEqual(r["total_variation_distance"], 0.8)
        same = class_distribution_shift([20, 20, 20, 20, 20], [10, 10, 10, 10, 10])
        self.assertAlmostEqual(same["total_variation_distance"], 0.0)

    def test_feature_drift_needs_enough_samples(self):
        r = feature_drift({"brightness": np.arange(100.0)}, {"brightness": [1.0] * 5})
        self.assertFalse(r["assessable"])

    def test_no_flag_for_same_distribution_and_flag_for_shifted(self):
        rng = np.random.default_rng(0)
        ref = {"brightness": rng.normal(90, 20, 1000), "contrast": rng.normal(32, 8, 1000)}
        same = {"brightness": rng.normal(90, 20, 200), "contrast": rng.normal(32, 8, 200)}
        shifted = {"brightness": rng.normal(140, 20, 200), "contrast": rng.normal(32, 8, 200)}
        self.assertFalse(feature_drift(ref, same)["any_flagged"])
        d = feature_drift(ref, shifted)
        self.assertTrue(d["any_flagged"]); self.assertTrue(d["features"]["brightness"]["shift_flagged"])
        self.assertFalse(d["features"]["contrast"]["shift_flagged"])
        self.assertIn("does not mean accuracy has dropped", d["note"])

    def test_format_duration(self):
        self.assertEqual(format_duration(3725), "01:02:05"); self.assertEqual(format_duration(90061), "1d 01:01:01")


if __name__ == "__main__":
    unittest.main()
