import unittest

import numpy as np

from src.uploads import ingest_upload, parse_labels, safe_filename
from tests.helpers import array_png, fake_fundus, make_env, png_bytes, upload_images


class UploadTests(unittest.TestCase):
    def setUp(self):
        self.env = make_env()
        self.db = self.env["db"]

    def one(self, data, label, name="a.png", index=True):
        return ingest_upload(self.db, data=data, filename=name, label=label, batch_id="t",
                             upload_dir=self.env["upload_dir"],
                             hash_index=self.env["reference"].hash_index() if index else None)

    def test_valid_upload_saved_to_disk_and_db(self):
        img = fake_fundus(1, seed=7)[0]
        r = self.one(array_png(img), 3)
        self.assertEqual(r["status"], "accepted")
        row = self.db.accepted_uploads()[0]
        self.assertEqual(row["label"], 3); self.assertEqual(row["original_filename"], "a.png")
        self.assertTrue(row["uploaded_at"])
        from pathlib import Path
        self.assertTrue(Path(row["stored_path"]).is_file())
        self.assertIn("class_3", row["stored_path"])

    def test_invalid_labels_rejected_and_recorded(self):
        for bad in [5, -1, "x", None, "", 2.5]:
            r = self.one(array_png(fake_fundus(1, seed=8)[0]), bad)
            self.assertEqual(r["status"], "rejected", bad); self.assertIn("invalid label", r["reason"])
        self.assertEqual(self.db.count_uploads("accepted"), 0)
        self.assertEqual(self.db.count_uploads("rejected"), 6)
        self.assertFalse(list((self.env["upload_dir"]).rglob("*.png")))     # nothing stored

    def test_unlabelled_image_never_accepted(self):
        self.assertEqual(self.one(array_png(fake_fundus(1, seed=9)[0]), None)["status"], "rejected")

    def test_invalid_images_rejected(self):
        for name, blob in {"text": b"hello", "empty": b"", "gif": png_bytes((64, 64), fmt="GIF"),
                           "truncated": png_bytes((200, 200))[:-30]}.items():
            r = self.one(blob, 1, name + ".png")
            self.assertEqual(r["status"], "rejected", name); self.assertIn("invalid image", r["reason"])

    def test_duplicate_rejected(self):
        data = array_png(fake_fundus(1, seed=11)[0])
        self.assertEqual(self.one(data, 1)["status"], "accepted")
        r = self.one(data, 2, "again.png")
        self.assertEqual(r["status"], "rejected"); self.assertIn("duplicate", r["reason"])
        self.assertEqual(self.db.count_uploads("accepted"), 1)

    def test_leakage_with_test_and_val_and_train_rejected(self):
        ref = self.env["reference"]
        for split, arr in (("test", ref._test[0][0]), ("validation", ref.val()[0][0]), ("training", ref.train()[0][0])):
            r = self.one(array_png(arr), 0)
            self.assertEqual(r["status"], "rejected", split); self.assertIn(split, r["reason"])
        self.assertEqual(self.db.count_uploads("accepted"), 0)

    def test_hostile_filename_cannot_escape_upload_dir(self):
        r = self.one(array_png(fake_fundus(1, seed=12)[0]), 1, "../../../etc/passwd")
        self.assertEqual(r["filename"], "passwd")
        from pathlib import Path
        stored = Path(self.db.accepted_uploads()[0]["stored_path"]).resolve()
        self.assertTrue(str(stored).startswith(str(self.env["upload_dir"].resolve())))
        self.assertEqual(safe_filename("a b\x00<>.png"), "a b___.png"); self.assertEqual(safe_filename(None), "unnamed")

    def test_batch_report_counts(self):
        rep = upload_images(self.env, 6, labels=[0, 1, 2, 9, "z", 4])
        self.assertEqual((rep["n_files"], rep["n_accepted"], rep["n_rejected"]), (6, 4, 2))
        self.assertEqual([r["status"] for r in rep["results"]], ["accepted"] * 3 + ["rejected"] * 2 + ["accepted"])

    def test_parse_labels(self):
        self.assertEqual(parse_labels("0, 3,1", 3), ["0", "3", "1"])
        self.assertEqual(parse_labels("[0,3,1]", 3), [0, 3, 1])
        self.assertEqual(parse_labels([1, 2], 2), [1, 2])
        for raw, n in [("0,1", 3), ("", 1), ("[1,", 1), ("0,1,2", 2)]:
            with self.assertRaises(ValueError):
                parse_labels(raw, n)

    def test_features_recorded(self):
        self.one(array_png(fake_fundus(1, seed=13)[0]), 2)
        self.assertGreater(self.db.upload_features()["brightness"][0], 0)


if __name__ == "__main__":
    unittest.main()
