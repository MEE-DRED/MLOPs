import io
import unittest

import numpy as np
from PIL import Image

from src import config
from src.prediction import ModelNotLoadedError, Predictor
from src.preprocessing import InvalidImageError
from tests.helpers import StubModel, fake_fundus, png_bytes


class PredictorTests(unittest.TestCase):
    def setUp(self):
        self.stub = StubModel(favourite=3)
        self.p = Predictor(model=self.stub, version=7)

    def test_output_structure(self):
        r = self.p.predict_bytes(png_bytes((300, 250)))
        self.assertEqual(set(r), {"predicted_class", "predicted_class_name", "confidence",
                                  "probabilities", "model_version"})
        self.assertEqual(r["predicted_class"], 3)
        self.assertEqual(r["predicted_class_name"], "Severe")
        self.assertEqual(r["model_version"], 7)
        self.assertEqual(list(r["probabilities"]), list(config.CLASS_NAMES))
        self.assertAlmostEqual(sum(r["probabilities"].values()), 1.0, places=6)
        self.assertAlmostEqual(r["confidence"], r["probabilities"]["Severe"])

    def test_model_receives_0_255_float32_batch(self):
        seen = {}
        class Spy(StubModel):
            def predict(self, batch, verbose=0):
                seen["shape"], seen["dtype"], seen["max"] = batch.shape, batch.dtype, batch.max()
                return super().predict(batch)
        img = fake_fundus(1)[0]
        buf = io.BytesIO(); Image.fromarray(img).save(buf, "PNG")
        Predictor(model=Spy(), version=1).predict_bytes(buf.getvalue())
        self.assertEqual(seen["shape"], (1, 224, 224, 3)); self.assertEqual(seen["dtype"], np.float32)
        self.assertEqual(seen["max"], float(img.max()))             # not rescaled outside the model

    def test_invalid_bytes_rejected_before_model_is_called(self):
        with self.assertRaises(InvalidImageError):
            self.p.predict_bytes(b"not an image")
        self.assertEqual(self.stub.calls, 0)

    def test_not_loaded(self):
        with self.assertRaises(ModelNotLoadedError):
            Predictor().predict_bytes(png_bytes())
        self.assertFalse(Predictor().is_loaded); self.assertTrue(self.p.is_loaded)

    def test_from_registry_without_model(self):
        import tempfile
        from src.model_registry import ModelRegistry
        with self.assertRaises(ModelNotLoadedError):
            Predictor.from_registry(ModelRegistry(tempfile.mkdtemp()))


if __name__ == "__main__":
    unittest.main()
