import io
import unittest

import numpy as np
from PIL import Image

from src import config
from src.preprocessing import (InvalidImageError, InvalidLabelError, load_image,
                               mobilenet_v2_reference_scaling, to_model_input, validate_label)
from tests.helpers import fake_fundus, png_bytes


class LoadImageTests(unittest.TestCase):
    def test_non_square_png_padded_to_224(self):
        arr = load_image(png_bytes((300, 200)))
        self.assertEqual(arr.shape, (224, 224, 3)); self.assertEqual(arr.dtype, np.uint8)
        self.assertEqual(int(arr[0, 0].sum()), 0)            # padding is black
        self.assertGreater(int(arr[112, 112].sum()), 0)

    def test_crop_mode_has_no_black_padding(self):
        arr = load_image(png_bytes((300, 200)), non_square="crop")
        self.assertGreater(int(arr[0, 0].sum()), 0)

    def test_exact_size_is_untouched(self):
        img = fake_fundus(1)[0]
        buf = io.BytesIO(); Image.fromarray(img).save(buf, format="PNG")
        np.testing.assert_array_equal(load_image(buf.getvalue()), img)   # PNG is lossless

    def test_grayscale_and_rgba_converted(self):
        self.assertEqual(load_image(png_bytes((64, 64), mode="L")).shape, (224, 224, 3))
        buf = io.BytesIO(); Image.new("RGBA", (64, 64), (1, 2, 3, 4)).save(buf, "PNG")
        self.assertEqual(load_image(buf.getvalue()).shape, (224, 224, 3))

    def test_jpeg_accepted(self):
        self.assertEqual(load_image(png_bytes((100, 100), fmt="JPEG")).shape, (224, 224, 3))

    def test_path_and_filelike_sources(self):
        import tempfile, pathlib
        p = pathlib.Path(tempfile.mkdtemp()) / "x.png"; p.write_bytes(png_bytes())
        self.assertEqual(load_image(p).shape, (224, 224, 3))
        self.assertEqual(load_image(io.BytesIO(png_bytes())).shape, (224, 224, 3))

    def test_rejections(self):
        bad = {
            "empty": b"",
            "text": b"hello world",
            "gif": png_bytes((64, 64), fmt="GIF"),
            "bmp": png_bytes((64, 64), fmt="BMP"),
            "tiny": png_bytes((8, 8)),
            "truncated": png_bytes((200, 200), fmt="PNG")[:-40],
        }
        for name, data in bad.items():
            with self.subTest(name), self.assertRaises(InvalidImageError):
                load_image(data)

    def test_missing_file_and_bad_type(self):
        with self.assertRaises(InvalidImageError):
            load_image("/no/such/file.png")
        with self.assertRaises(InvalidImageError):
            load_image(12345)

    def test_corrupt_jpeg_body(self):
        data = bytearray(png_bytes((120, 120), fmt="JPEG"))
        for i in range(200, min(len(data), 400)):
            data[i] = 0xFF
        # must either decode (JPEG is tolerant) or raise our error - never another exception type
        try:
            load_image(bytes(data))
        except InvalidImageError:
            pass

    def test_oversized_rejected(self):
        from unittest import mock
        with mock.patch.object(config, "MAX_UPLOAD_BYTES", 100):
            with self.assertRaises(InvalidImageError):
                load_image(png_bytes((200, 200)))


class LabelTests(unittest.TestCase):
    def test_valid(self):
        for raw, want in [(0, 0), (4, 4), ("3", 3), (" 2 ", 2), (1.0, 1), (np.int64(2), 2)]:
            self.assertEqual(validate_label(raw), want)

    def test_invalid(self):
        for raw in [-1, 5, 99, "5", "abc", "", None, True, 1.5, "1.5", "Mild", [1]]:
            with self.subTest(raw=raw), self.assertRaises(InvalidLabelError):
                validate_label(raw)


class ModelInputTests(unittest.TestCase):
    def test_shapes_and_dtype(self):
        one = to_model_input(fake_fundus(1)[0])
        self.assertEqual(one.shape, (1, 224, 224, 3)); self.assertEqual(one.dtype, np.float32)
        self.assertEqual(to_model_input(fake_fundus(3)).shape, (3, 224, 224, 3))
        self.assertEqual(to_model_input(list(fake_fundus(2))).shape, (2, 224, 224, 3))

    def test_values_stay_0_255(self):
        # the model rescales internally; feeding [0,1] or [-1,1] here would double-normalise
        x = to_model_input(fake_fundus(2))
        self.assertGreater(x.max(), 100.0)

    def test_wrong_shape_rejected(self):
        with self.assertRaises(InvalidImageError):
            to_model_input(np.zeros((1, 100, 100, 3), np.uint8))

    def test_reference_scaling_range(self):
        s = mobilenet_v2_reference_scaling(np.array([0, 127.5, 255]))
        np.testing.assert_allclose(s, [-1.0, 0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
