import json
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.make_submission_zip import SubmissionError, build_submission_zip

REPO = Path(__file__).resolve().parents[1]


class SubmissionZipTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.repo = self.tmp / "repo"
        shutil.copytree(REPO, self.repo, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git", "state"))
        shutil.rmtree(self.repo / "models", ignore_errors=True)

    def add_model(self, version=2):
        m = self.repo / "models"; (m / "metadata").mkdir(parents=True)
        (m / f"model_v{version}.keras").write_bytes(b"weights"); (m / "metadata" / f"model_v{version}.json").write_text("{}")
        (m / "current_model.json").write_text(json.dumps({"current_version": version}))

    def names(self, z):
        return zipfile.ZipFile(z).namelist()

    def test_refuses_without_model(self):
        with self.assertRaises(SubmissionError) as cm:
            build_submission_zip(self.tmp / "o.zip", self.repo)
        self.assertIn("model file", str(cm.exception)); self.assertFalse((self.tmp / "o.zip").exists())

    def test_refuses_when_pointer_target_missing(self):
        self.add_model(2); (self.repo / "models/model_v2.keras").unlink()
        with self.assertRaises(SubmissionError) as cm:
            build_submission_zip(self.tmp / "o.zip", self.repo)
        self.assertIn("missing", str(cm.exception))

    def test_complete_zip_has_model_layout_and_no_forbidden_files(self):
        self.add_model()
        (self.repo / "data" / "retinamnist_224.npz").write_bytes(b"huge dataset")
        (self.repo / "database").mkdir(exist_ok=True); (self.repo / "database" / "x.sqlite3").write_bytes(b"db")
        (self.repo / ".env").write_text("DR_API" + "_KEY=abcdefghijklmnop")
        (self.repo / "src" / "__pycache__").mkdir(exist_ok=True); (self.repo / "src/__pycache__/a.pyc").write_bytes(b"x")
        (self.repo / "data/retraining_uploads/class_1").mkdir(parents=True, exist_ok=True)
        (self.repo / "data/retraining_uploads/class_1/secret_patient.png").write_bytes(b"x")
        info = build_submission_zip(self.tmp / "o.zip", self.repo)
        names = self.names(self.tmp / "o.zip"); top = "Diabetic-Retinopathy-MLOps/"
        for must in ("README.md", "notebook/diabetic_retinopathy_image_classification.ipynb", "src/preprocessing.py", "src/model.py", "src/prediction.py",
                     "models/model_v2.keras", "models/current_model.json", "data/train/.gitkeep", "data/test/.gitkeep", "Dockerfile", "docs/deployment.md"):
            self.assertIn(top + must, names, must)
        joined = "\n".join(names)
        for bad in (".npz", ".sqlite3", ".pyc", "secret_patient"):
            self.assertNotIn(bad, joined, bad)
        self.assertFalse([n for n in names if n.endswith("/.env")], "real .env must never be zipped")
        self.assertIn(top + ".env.example", names)
        self.assertIn("v2", info["model"]); self.assertLess(info["size_mb"], 50)

    def test_secret_in_source_aborts(self):
        self.add_model()
        (self.repo / "src" / "oops.py").write_text("TOKEN = " + chr(34) + "hf" + "_" + "a" * 30 + chr(34) + "\n")
        with self.assertRaises(SubmissionError) as cm:
            build_submission_zip(self.tmp / "o.zip", self.repo)
        self.assertIn("oops.py", str(cm.exception))
        (self.repo / "src" / "oops.py").write_text("api" + "_key = " + chr(34) + "supersecret" + "value123" + chr(34) + "\n")
        with self.assertRaises(SubmissionError):
            build_submission_zip(self.tmp / "o.zip", self.repo)

    def test_env_variable_names_without_values_are_fine(self):
        self.add_model()                                        # the real repo mentions DR_API_KEY etc. only as names/empty values
        build_submission_zip(self.tmp / "o.zip", self.repo)

    def test_missing_required_file_aborts(self):
        self.add_model(); (self.repo / "src/prediction.py").unlink()
        with self.assertRaises(SubmissionError) as cm:
            build_submission_zip(self.tmp / "o.zip", self.repo)
        self.assertIn("src/prediction.py", str(cm.exception))

    def test_draft_without_model_allowed_explicitly(self):
        info = build_submission_zip(self.tmp / "d.zip", self.repo, allow_no_model=True)
        self.assertIn("NOT INCLUDED", info["model"])


if __name__ == "__main__":
    unittest.main()
