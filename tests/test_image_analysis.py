import unittest

import numpy as np
from PIL import Image, ImageFilter

from src.image_analysis import compute_image_features
from tests.helpers import fake_fundus


def disc(value, noise=0.0, seed=0):
    yy, xx = np.mgrid[:224, :224]
    img = np.zeros((224, 224, 3), np.float32)
    inside = ((yy - 112) ** 2 + (xx - 112) ** 2) <= 100 ** 2
    img[inside] = value
    if noise:
        img[inside] += np.random.default_rng(seed).normal(0, noise, (inside.sum(), 3))
    return np.clip(img, 0, 255).astype(np.uint8)


class FeatureTests(unittest.TestCase):
    def test_brightness_ignores_black_border(self):
        f = compute_image_features(disc(100))
        self.assertAlmostEqual(f["brightness"][0], 100, delta=1)
        self.assertAlmostEqual(f["contrast"][0], 0, delta=1)
        self.assertAlmostEqual(f["sharpness"][0], 0, delta=1)        # flat disc: border excluded
        self.assertGreater(f["retina_fraction"][0], 0.5)

    def test_brighter_is_brighter(self):
        f = compute_image_features(np.stack([disc(60), disc(160)]))
        self.assertLess(f["brightness"][0], f["brightness"][1])

    def test_noise_raises_contrast_and_sharpness(self):
        f = compute_image_features(np.stack([disc(100), disc(100, noise=25)]))
        self.assertLess(f["contrast"][0], f["contrast"][1])
        self.assertLess(f["sharpness"][0], f["sharpness"][1])

    def test_blur_lowers_sharpness(self):
        sharp = fake_fundus(1)[0]
        blurred = np.asarray(Image.fromarray(sharp).filter(ImageFilter.GaussianBlur(3)))
        f = compute_image_features(np.stack([sharp, blurred]))
        self.assertGreater(f["sharpness"][0], f["sharpness"][1])

    def test_blank_image_is_safe(self):
        f = compute_image_features(np.zeros((224, 224, 3), np.uint8))
        self.assertEqual(f["brightness"][0], 0.0); self.assertEqual(f["sharpness"][0], 0.0)

    def test_batch_shape(self):
        f = compute_image_features(fake_fundus(4))
        self.assertTrue(all(len(v) == 4 for v in f.values()))


if __name__ == "__main__":
    unittest.main()
