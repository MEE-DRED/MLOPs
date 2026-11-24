"""Static guards for bugs that only show up with real TensorFlow (which the build sandbox lacks)."""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class SourceGuards(unittest.TestCase):
    def test_no_keras_saving_namespace(self):
        # `from tensorflow import keras` has no `.saving` in Colab's TF 2.21 / Keras 3.13 (this crashed a real run).
        # Use keras.models.load_model / model.save instead.
        bad = [str(p.relative_to(ROOT)) for p in list((ROOT / "src").glob("*.py")) + list((ROOT / "app").glob("*.py"))
               if re.search(r"keras\.saving\.", p.read_text())]
        self.assertEqual(bad, [])

    def test_notebook_does_not_use_keras_saving(self):
        nb = (ROOT / "notebook" / "diabetic_retinopathy_image_classification.ipynb").read_text()
        self.assertNotIn("keras.saving.", nb)


if __name__ == "__main__":
    unittest.main()
