import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src import data_acquisition as da
from tests.helpers import fake_fundus


def _write_npz(path, mutate=None):
    d = {}
    for i, s in enumerate(("train", "val", "test")):
        d[f"{s}_images"] = fake_fundus(5, seed=i + 1)
        d[f"{s}_labels"] = np.arange(5).reshape(-1, 1).astype(np.uint8)
    if mutate:
        mutate(d)
    np.savez(path, **d)


class SyntheticDatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_load_and_validate_ok(self):
        p = self.tmp / "ok.npz"; _write_npz(p)
        data = da.load_retinamnist(p)
        self.assertEqual(set(data), {"train", "val", "test"})
        self.assertEqual(data["train"][1].ndim, 1)            # labels flattened
        report = da.validate_dataset(data)
        self.assertTrue(report["ok"], report["issues"])
        self.assertEqual(report["splits"]["train"]["class_counts"], [1, 1, 1, 1, 1])

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            da.load_retinamnist(self.tmp / "nope.npz")

    def test_missing_key(self):
        p = self.tmp / "bad.npz"
        _write_npz(p, lambda d: d.pop("val_labels"))
        with self.assertRaises(da.DatasetError):
            da.load_retinamnist(p)

    def test_not_an_npz(self):
        p = self.tmp / "junk.npz"; p.write_bytes(b"this is not a zip file")
        with self.assertRaises(da.DatasetError):
            da.load_retinamnist(p)

    def test_invalid_label_detected(self):
        p = self.tmp / "lab.npz"
        _write_npz(p, lambda d: d["test_labels"].__setitem__(0, 9))
        report = da.validate_dataset(da.load_retinamnist(p))
        self.assertFalse(report["ok"])
        self.assertTrue(any("labels outside" in i for i in report["issues"]))

    def test_leakage_detected(self):
        p = self.tmp / "leak.npz"
        _write_npz(p, lambda d: d["test_images"].__setitem__(0, d["train_images"][0]))
        report = da.validate_dataset(da.load_retinamnist(p))
        self.assertFalse(report["ok"])
        self.assertEqual(report["cross_split_duplicates"]["train&test"], 1)

    def test_blank_image_detected(self):
        p = self.tmp / "blank.npz"
        _write_npz(p, lambda d: d["train_images"].__setitem__(0, 0))
        self.assertFalse(da.validate_dataset(da.load_retinamnist(p))["ok"])

    def test_wrong_dtype_detected(self):
        p = self.tmp / "dt.npz"
        _write_npz(p, lambda d: d.update(val_images=d["val_images"].astype(np.float32)))
        self.assertFalse(da.validate_dataset(da.load_retinamnist(p))["ok"])

    def test_ensure_dataset_copies_from_search_path(self):
        src = self.tmp / "drive" / "retinamnist_224.npz"; src.parent.mkdir()
        _write_npz(src)
        dest = self.tmp / "local" / "retinamnist_224.npz"
        got = da.ensure_dataset(dest, search_paths=[src], try_medmnist=False,
                                try_direct_url=False, log=lambda *_: None)
        self.assertEqual(got, dest); self.assertTrue(dest.is_file())

    def test_ensure_dataset_failure_message(self):
        with self.assertRaises(da.DatasetError):
            da.ensure_dataset(self.tmp / "x" / "retinamnist_224.npz", try_medmnist=False,
                              try_direct_url=False, log=lambda *_: None)

    def test_distribution_table(self):
        p = self.tmp / "t.npz"; _write_npz(p)
        df = da.class_distribution_table(da.load_retinamnist(p))
        self.assertEqual(len(df), 15); self.assertEqual(df["count"].sum(), 15)


REAL = Path(os.environ.get("DR_TEST_NPZ", "/mnt/user-data/uploads/retinamnist_224.npz"))


@unittest.skipUnless(REAL.is_file(), "real RetinaMNIST file not available")
class RealDatasetTests(unittest.TestCase):
    """Runs against the actual supplied dataset."""

    @classmethod
    def setUpClass(cls):
        cls.data = da.load_retinamnist(REAL)

    def test_split_sizes(self):
        self.assertEqual({s: len(v[1]) for s, v in self.data.items()},
                         {"train": 1080, "val": 120, "test": 400})

    def test_validation_report_clean(self):
        report = da.validate_dataset(self.data)
        self.assertTrue(report["ok"], report["issues"])
        self.assertEqual(sum(report["cross_split_duplicates"].values()), 0)

    def test_expected_class_counts(self):
        self.assertEqual(da.validate_dataset(self.data)["splits"]["train"]["class_counts"],
                         [486, 128, 206, 194, 66])


if __name__ == "__main__":
    unittest.main()
