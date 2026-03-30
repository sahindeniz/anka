import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gui.histogram_editor import apply_camera_raw_adjustments, build_preview_proxy


class HistogramEditorPreviewTests(unittest.TestCase):
    def test_build_preview_proxy_downscales_large_image(self):
        img = np.zeros((2400, 3600, 3), dtype=np.float32)

        proxy, scale = build_preview_proxy(img, max_side=1440)

        self.assertEqual(proxy.dtype, np.float32)
        self.assertLess(scale, 1.0)
        self.assertEqual(proxy.shape[:2], (960, 1440))

    def test_build_preview_proxy_keeps_small_image(self):
        img = np.ones((320, 480, 3), dtype=np.float32)

        proxy, scale = build_preview_proxy(img, max_side=1440)

        self.assertEqual(scale, 1.0)
        self.assertEqual(proxy.shape, img.shape)
        np.testing.assert_allclose(proxy, img)


class HistogramEditorRawAdjustmentsTests(unittest.TestCase):
    def test_camera_raw_adjustments_lift_shadows_and_tame_highlights(self):
        img = np.array(
            [
                [[0.08, 0.10, 0.12], [0.92, 0.88, 0.84]],
                [[0.18, 0.16, 0.14], [0.65, 0.62, 0.60]],
            ],
            dtype=np.float32,
        )

        out = apply_camera_raw_adjustments(
            img,
            {
                "profile": "Renk",
                "exposure": 0.0,
                "contrast": 10.0,
                "highlights": -55.0,
                "shadows": 45.0,
                "whites": 0.0,
                "blacks": 0.0,
                "temp": 0.0,
                "tint": 0.0,
                "texture": 0.0,
                "sharpen": 0.0,
                "clarity": 0.0,
                "dehaze": 0.0,
                "vibrance": 0.0,
                "saturation": 0.0,
            },
        )

        self.assertEqual(out.shape, img.shape)
        self.assertEqual(out.dtype, np.float32)
        self.assertGreater(float(out[0, 0].mean()), float(img[0, 0].mean()))
        self.assertLess(float(out[0, 1].mean()), float(img[0, 1].mean()))

    def test_camera_raw_adjustments_bw_profile_equalizes_channels(self):
        img = np.array(
            [[[0.20, 0.40, 0.80], [0.90, 0.30, 0.10]]],
            dtype=np.float32,
        )

        out = apply_camera_raw_adjustments(
            img,
            {
                "profile": "Siyah Beyaz",
                "exposure": 0.0,
                "contrast": 0.0,
                "highlights": 0.0,
                "shadows": 0.0,
                "whites": 0.0,
                "blacks": 0.0,
                "temp": 40.0,
                "tint": -20.0,
                "texture": 0.0,
                "sharpen": 0.0,
                "clarity": 0.0,
                "dehaze": 0.0,
                "vibrance": 30.0,
                "saturation": 30.0,
            },
        )

        np.testing.assert_allclose(out[:, :, 0], out[:, :, 1], atol=1e-6)
        np.testing.assert_allclose(out[:, :, 1], out[:, :, 2], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
