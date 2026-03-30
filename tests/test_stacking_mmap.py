import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from processing import stacking


class StackingMmapTests(unittest.TestCase):
    def test_build_master_forced_mmap_matches_reference_median(self):
        frames = {
            "a.fit": np.array([[0.1, 0.5], [0.2, 0.8]], dtype=np.float32),
            "b.fit": np.array([[0.2, 0.4], [0.3, 0.7]], dtype=np.float32),
            "c.fit": np.array([[0.15, 0.45], [0.25, 0.75]], dtype=np.float32),
        }
        expected = np.median(np.stack(list(frames.values()), axis=0), axis=0).astype(np.float32)

        with mock.patch.object(stacking, "_MMAP_THRESHOLD_GB", 0.0), mock.patch(
            "core.loader.load_image",
            side_effect=lambda path: frames[path].copy(),
        ):
            master = stacking._build_master(list(frames.keys()), method="median")

        np.testing.assert_allclose(master, expected, atol=1e-6)

    def test_align_frames_only_can_return_memmap_store(self):
        base = np.array(
            [
                [0.0, 0.2, 0.0],
                [0.2, 1.0, 0.2],
                [0.0, 0.2, 0.0],
            ],
            dtype=np.float32,
        )
        frames = {
            "ref.fit": base,
            "light2.fit": base * 0.95,
        }

        def fake_score(img):
            return {
                "score": float(np.mean(img)),
                "star_count": 12,
                "fwhm": 2.0,
                "snr": 5.0,
                "signal": float(np.mean(img)),
                "noise": 0.1,
            }

        with mock.patch.object(stacking, "_MMAP_THRESHOLD_GB", 0.0), mock.patch(
            "core.loader.load_image",
            side_effect=lambda path: frames[path].copy(),
        ), mock.patch.object(
            stacking,
            "score_frame",
            side_effect=fake_score,
        ), mock.patch.object(
            stacking,
            "_compute_homography",
            return_value=(np.eye(3, dtype=np.float32), {"rotation_deg": 0.0}),
        ), mock.patch.object(
            stacking,
            "_warp_image",
            side_effect=lambda img, h: img.copy(),
        ):
            aligned, frame_infos = stacking.align_frames_only(
                ["ref.fit", "light2.fit"],
                quality_threshold=-1.0,
            )
            try:
                self.assertIsInstance(aligned, stacking._MmapFrameStore)
                self.assertEqual(len(aligned), 2)
                np.testing.assert_allclose(aligned[0], base, atol=1e-6)
                np.testing.assert_allclose(aligned[1], base * 0.95, atol=1e-6)
                self.assertEqual(len(frame_infos), 2)
            finally:
                aligned.cleanup()

    def test_stack_aligned_sigma_clip_works_with_memmap_masks(self):
        frames = [
            np.full((4, 4), 0.20, dtype=np.float32),
            np.full((4, 4), 0.22, dtype=np.float32),
            np.full((4, 4), 0.21, dtype=np.float32),
        ]
        frames[2][1, 1] = 1.0

        store = stacking._MmapFrameStore(len(frames), frames[0].shape, dtype=np.float32)
        for i, frame in enumerate(frames):
            store[i] = frame

        with mock.patch.object(stacking, "_MMAP_THRESHOLD_GB", 0.0):
            try:
                result = stacking.stack_aligned(
                    store,
                    method="sigma_clip",
                    normalization="none",
                    weight_mode="equal",
                    iterations=1,
                    kappa_low=2.0,
                    kappa_high=2.0,
                )
            finally:
                store.cleanup()

        self.assertEqual(result["method"], "sigma_clip")
        self.assertLess(float(result["result"][1, 1]), 0.4)


if __name__ == "__main__":
    unittest.main()
