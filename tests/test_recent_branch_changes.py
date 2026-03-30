import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gui.bg_composer import generate_welcome_overlay
from processing.stacking import _normalize_frames, _stack_weighted_mean, stack_aligned


class StackingRecentChangesTests(unittest.TestCase):
    def test_normalize_frames_matches_reference_per_channel(self):
        base = np.linspace(0.1, 2.0, 16 * 16, dtype=np.float32).reshape(16, 16)
        ref = np.stack(
            [
                base,
                base * 1.5 + 0.05,
                base * 0.8 + 0.02,
            ],
            axis=2,
        ).astype(np.float32)
        shifted = np.stack(
            [
                ref[:, :, 0] * 1.8 + 0.30,
                ref[:, :, 1] * 0.6 + 0.40,
                ref[:, :, 2] * 1.2 + 0.10,
            ],
            axis=2,
        ).astype(np.float32)
        masks = [np.ones((16, 16), dtype=np.float32), np.ones((16, 16), dtype=np.float32)]

        normalized = _normalize_frames([ref.copy(), shifted.copy()], masks, mode="additive_scaling")

        np.testing.assert_allclose(normalized[0], ref, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(normalized[1], ref, rtol=1e-5, atol=1e-5)

    def test_stack_weighted_mean_uses_valid_mask_and_progress(self):
        frames = [
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
        ]
        masks = [np.ones((2, 2), dtype=np.float32), np.ones((2, 2), dtype=np.float32)]
        valid_mask = np.array(
            [
                [[True, True], [True, True]],
                [[False, True], [True, False]],
            ],
            dtype=bool,
        )
        progress = []

        result = _stack_weighted_mean(
            frames,
            masks,
            valid_mask,
            np.array([1.0, 3.0], dtype=np.float32),
            progress_cb=lambda step, msg: progress.append((step, msg)),
            progress_label="Weighted mean",
        )

        expected = np.array([[1.0, 15.5], [23.25, 4.0]], dtype=np.float32)
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)
        self.assertEqual(progress, [(8, "Weighted mean (1/2)"), (8, "Weighted mean (2/2)")])

    def test_stack_aligned_auto_rejects_outlier_and_emits_progress(self):
        frame_a = np.ones((16, 16), dtype=np.float32)
        frame_b = np.ones((16, 16), dtype=np.float32)
        frame_c = np.ones((16, 16), dtype=np.float32)
        frame_c[4, 7] = 50.0
        progress = []

        result = stack_aligned(
            aligned_frames=[frame_a, frame_b, frame_c],
            method="auto",
            weight_mode="equal",
            normalization="none",
            progress_cb=lambda step, msg: progress.append((step, msg)),
        )

        self.assertEqual(result["method"], "percentile")
        self.assertGreater(result["n_rejected"], 0)
        self.assertAlmostEqual(float(result["result"][4, 7]), 1.0, places=4)
        self.assertTrue(any("Auto rejection:" in msg for _step, msg in progress))
        self.assertIn((8, "Percentile rejection (1/1)"), progress)
        self.assertTrue(any(step == 8 and msg.endswith("(3/3)") for step, msg in progress))
        self.assertTrue(any(msg.startswith("✅ Stacking tamamlandı") for _step, msg in progress))


class WelcomeOverlayTests(unittest.TestCase):
    def test_generate_welcome_overlay_returns_clipped_float32_image(self):
        bg = np.full((180, 320, 3), 0.15, dtype=np.float32)

        out = generate_welcome_overlay(bg)

        self.assertEqual(out.shape, bg.shape)
        self.assertEqual(out.dtype, np.float32)
        self.assertGreater(np.max(out), np.max(bg))
        self.assertFalse(np.allclose(out, bg))
        self.assertGreaterEqual(float(np.min(out)), 0.0)
        self.assertLessEqual(float(np.max(out)), 1.0)


if __name__ == "__main__":
    unittest.main()
