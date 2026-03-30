import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gui.bg_composer import generate_welcome_overlay
from processing.halo_reduction import reduce_halos
from processing.mastro_noise import process_denoise
from processing.mastro_starless import process_starless
from processing.noise_reduction import reduce_noise
from processing.stacking import _calibrate_frame, _normalize_frames, _stack_weighted_mean, stack_aligned
from processing.starsmaller import reduce_stars
from processing.noisexterminator import denoise as denoise_noisexterminator
from processing.veralux_silentium import denoise_silentium


class StackingRecentChangesTests(unittest.TestCase):
    @staticmethod
    def _frames_with_single_outlier(count):
        frames = [np.ones((16, 16), dtype=np.float32) for _ in range(count)]
        frames[-1][4, 7] = 50.0
        return frames

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
        progress = []

        result = stack_aligned(
            aligned_frames=self._frames_with_single_outlier(3),
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

    def test_stack_aligned_linear_fit_rejects_outlier(self):
        result = stack_aligned(
            aligned_frames=self._frames_with_single_outlier(5),
            method="linear_fit",
            weight_mode="equal",
            normalization="none",
        )

        self.assertEqual(result["method"], "linear_fit")
        self.assertGreater(result["n_rejected"], 0)
        self.assertAlmostEqual(float(result["result"][4, 7]), 1.0, places=4)

    def test_stack_aligned_sigma_clip_rejects_outlier(self):
        result = stack_aligned(
            aligned_frames=self._frames_with_single_outlier(8),
            method="sigma_clip",
            weight_mode="equal",
            normalization="none",
        )

        self.assertEqual(result["method"], "sigma_clip")
        self.assertGreater(result["n_rejected"], 0)
        self.assertAlmostEqual(float(result["result"][4, 7]), 1.0, places=4)

    def test_stack_aligned_winsorized_sigma_rejects_outlier(self):
        result = stack_aligned(
            aligned_frames=self._frames_with_single_outlier(8),
            method="winsorized_sigma",
            weight_mode="equal",
            normalization="none",
        )

        self.assertEqual(result["method"], "winsorized_sigma")
        self.assertGreater(result["n_rejected"], 0)
        self.assertAlmostEqual(float(result["result"][4, 7]), 1.0, places=4)

    def test_calibrate_frame_broadcasts_mono_masters_to_color_light(self):
        img = np.array(
            [
                [[0.9, 0.8, 0.7], [0.9, 0.8, 0.7]],
                [[0.9, 0.8, 0.7], [0.9, 0.8, 0.7]],
            ],
            dtype=np.float32,
        )
        master_bias = np.full((2, 2), 0.1, dtype=np.float32)
        master_dark = np.full((2, 2), 0.2, dtype=np.float32)
        master_flat = np.array([[0.5, 0.9], [0.5, 0.9]], dtype=np.float32)

        calibrated = _calibrate_frame(img, master_dark, master_flat, master_bias)

        flat_after_bias = np.clip(master_flat - master_bias, 0.0, None)
        flat_norm = flat_after_bias / np.mean(flat_after_bias)
        expected = np.clip(img - 0.3, 0.0, None) / flat_norm[:, :, None]

        self.assertEqual(calibrated.shape, img.shape)
        self.assertEqual(calibrated.dtype, np.float32)
        np.testing.assert_allclose(calibrated, expected.astype(np.float32), rtol=1e-6, atol=1e-6)


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


class MastroSelfContainedTests(unittest.TestCase):
    def test_mastro_noise_runs_without_external_model_files(self):
        base = np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32).reshape(64, 64)
        noisy = np.stack([base, np.flipud(base), base], axis=2)
        rng = np.random.default_rng(7)
        noisy = np.clip(noisy + rng.normal(0.0, 0.04, noisy.shape).astype(np.float32), 0, 1)
        progress = []

        result = process_denoise(
            noisy,
            modulation=0.7,
            progress_callback=lambda v: progress.append(v),
        )

        self.assertEqual(result.shape, noisy.shape)
        self.assertEqual(result.dtype, np.float32)
        self.assertGreaterEqual(min(progress), 0)
        self.assertEqual(progress[-1], 100)
        self.assertFalse(np.allclose(result, noisy))

    def test_mastro_starless_runs_without_siril(self):
        img = np.zeros((96, 96, 3), dtype=np.float32)
        img[48, 48] = 1.0
        img[24, 70] = 0.8
        img[70, 20] = 0.7
        progress = []

        starless, mask = process_starless(
            img,
            progress_callback=lambda v: progress.append(v),
        )

        self.assertEqual(starless.shape, img.shape)
        self.assertEqual(starless.dtype, np.float32)
        self.assertIsNotNone(mask)
        self.assertEqual(mask.shape, img.shape[:2])
        self.assertEqual(mask.dtype, np.float32)
        self.assertEqual(progress[-1], 100)
        self.assertLess(float(starless[48, 48, 0]), float(img[48, 48, 0]))


class HaloReductionTests(unittest.TestCase):
    @staticmethod
    def _make_star_field():
        h = w = 128
        yy, xx = np.mgrid[:h, :w]
        img = np.full((h, w, 3), 0.015, dtype=np.float32)

        def add_star(cx, cy, core_amp, halo_amp, core_sigma, halo_sigma, color):
            r2 = (xx - cx) ** 2 + (yy - cy) ** 2
            core = core_amp * np.exp(-r2 / (2.0 * core_sigma ** 2))
            halo = halo_amp * np.exp(-r2 / (2.0 * halo_sigma ** 2))
            profile = core + halo
            for ch, gain in enumerate(color):
                img[:, :, ch] += profile * gain

        add_star(64, 64, 0.95, 0.35, 1.2, 4.6, (1.0, 0.98, 0.96))
        add_star(34, 92, 0.72, 0.18, 1.0, 3.2, (0.95, 1.0, 1.05))
        add_star(94, 28, 0.58, 0.12, 0.9, 2.7, (1.05, 0.98, 0.95))
        return np.clip(img, 0.0, 1.0).astype(np.float32)

    def test_halo_reduction_lowers_outer_glow_but_keeps_core(self):
        img = self._make_star_field()
        yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]
        rr = np.sqrt((xx - 64) ** 2 + (yy - 64) ** 2)
        annulus = (rr >= 4.0) & (rr <= 8.0)
        core = rr <= 1.5

        result = reduce_halos(
            img,
            denoise_strength=0.0,
            halo_strength=0.35,
            chroma_cleanup=0.0,
            core_protect=0.85,
            recompose_opacity=1.0,
        )

        self.assertEqual(result["result"].shape, img.shape)
        self.assertEqual(result["starless"].shape, img.shape)
        self.assertEqual(result["stars"].shape, img.shape)
        self.assertEqual(result["result"].dtype, np.float32)
        self.assertLess(
            float(result["result"][annulus].mean()),
            float(img[annulus].mean()),
        )
        self.assertGreater(
            float(result["result"][core].mean()),
            float(img[core].mean()) * 0.60,
        )


class StarSmallerTests(unittest.TestCase):
    def test_reduce_stars_fills_halo_zone_from_surrounding_background(self):
        h = w = 96
        yy, xx = np.mgrid[:h, :w]
        bg = np.stack(
            [
                0.12 + xx.astype(np.float32) * 0.0007,
                0.18 + yy.astype(np.float32) * 0.0005,
                0.22 + (xx + yy).astype(np.float32) * 0.0003,
            ],
            axis=2,
        ).astype(np.float32)

        cx = cy = 48
        r2 = (xx - cx) ** 2 + (yy - cy) ** 2
        star = (
            0.95 * np.exp(-r2 / (2.0 * 1.3 ** 2))
            + 0.32 * np.exp(-r2 / (2.0 * 4.8 ** 2))
        ).astype(np.float32)
        img = np.clip(bg + star[:, :, None], 0.0, 1.0).astype(np.float32)

        result, mask = reduce_stars(
            img,
            strength=0.85,
            sensitivity=0.5,
            feather=3,
            max_sigma=8,
            min_sigma=1,
            threshold=0.02,
        )

        rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        halo_zone = (rr >= 4.0) & (rr <= 7.0)
        core_zone = rr <= 1.5

        before_bg_error = float(np.abs(img[halo_zone] - bg[halo_zone]).mean())
        after_bg_error = float(np.abs(result[halo_zone] - bg[halo_zone]).mean())

        self.assertEqual(result.shape, img.shape)
        self.assertEqual(mask.shape, img.shape[:2])
        self.assertLess(after_bg_error, before_bg_error * 0.35)
        self.assertGreater(float(result[core_zone].mean()), float(bg[core_zone].mean()) + 0.10)


class NoiseDispatchTests(unittest.TestCase):
    def test_noisexterminator_dispatch_uses_real_engine(self):
        img = np.random.default_rng(3).random((32, 32, 3), dtype=np.float32)
        direct, _meta = denoise_noisexterminator(img, strength=0.4, detail_preserve=0.6)
        buf = StringIO()
        with redirect_stdout(buf):
            dispatched = reduce_noise(img, method="noisexterminator", strength=0.4, detail=0.6)

        self.assertEqual(direct.shape, dispatched.shape)
        self.assertNotIn("noisexterminator failed", buf.getvalue())
        np.testing.assert_allclose(dispatched, direct, rtol=1e-6, atol=1e-6)

    def test_silentium_dispatch_uses_real_engine(self):
        img = np.random.default_rng(4).random((24, 24, 3), dtype=np.float32)
        direct = denoise_silentium(img, strength=0.35, detail=0.55)
        buf = StringIO()
        with redirect_stdout(buf):
            dispatched = reduce_noise(img, method="silentium", strength=0.35, detail=0.55)

        self.assertEqual(direct.shape, dispatched.shape)
        self.assertNotIn("silentium failed", buf.getvalue())
        np.testing.assert_allclose(dispatched, direct, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
