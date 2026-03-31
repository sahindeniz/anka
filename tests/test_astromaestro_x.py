import unittest

import numpy as np

from processing.astromaestro_x import SUITE
from processing.background import gradient_terminator
from processing.deconvolution import astro_blur_x
from processing.mastro_starless import astro_star_x
from processing.noise_reduction import reduce_noise
from processing.noisexterminator import astro_noise_x
from processing.star_shrink import astro_star_shrink


class AstroMaestroXTests(unittest.TestCase):
    def test_suite_exports_expected_profiles(self):
        self.assertIn("astro_blur_x", SUITE)
        self.assertIn("astro_noise_x", SUITE)
        self.assertIn("astro_star_x", SUITE)
        self.assertIn("astro_gradient_x", SUITE)
        self.assertIn("astro_star_shrink", SUITE)

    def test_astro_noise_x_returns_meta(self):
        rng = np.random.default_rng(11)
        img = rng.random((48, 48, 3), dtype=np.float32) * 0.12
        result, meta = astro_noise_x(img, strength=0.6, detail=0.7, iterations=3)
        self.assertEqual(result.shape, img.shape)
        self.assertEqual(meta["iterations"], 3)
        self.assertTrue(meta["used_color_separation"])

    def test_reduce_noise_dispatch_supports_astro_noise_x(self):
        rng = np.random.default_rng(12)
        img = rng.random((32, 32, 3), dtype=np.float32) * 0.15
        direct, _meta = astro_noise_x(img, strength=0.5, detail=0.6, iterations=2)
        dispatched = reduce_noise(img, method="astro_noise_x", strength=0.5, detail=0.6, iterations=2)
        np.testing.assert_allclose(dispatched, direct, rtol=1e-6, atol=1e-6)

    def test_astro_blur_x_correct_only_is_more_conservative(self):
        yy, xx = np.mgrid[:96, :96].astype(np.float32)
        star = np.exp(-(((xx - 32.0) ** 2 + (yy - 44.0) ** 2) / (2 * 2.8 ** 2)))
        galaxy = 0.35 * np.exp(-(((xx - 64.0) / 14.0) ** 2 + ((yy - 50.0) / 9.0) ** 2))
        img = np.clip(star + galaxy + 0.04, 0, 1).astype(np.float32)
        img = np.dstack([img, img * 0.97, img * 0.94]).astype(np.float32)

        correct_only = astro_blur_x(img, strength=0.7, correct_only=True)
        full = astro_blur_x(img, strength=0.7, correct_only=False)

        diff_correct = float(np.abs(correct_only - img).mean())
        diff_full = float(np.abs(full - img).mean())
        self.assertLess(diff_correct, diff_full)

    def test_gradient_terminator_neutralizes_background(self):
        yy, xx = np.mgrid[:120, :160].astype(np.float32)
        base = 0.12 + 0.08 * (xx / 159.0) + 0.05 * (yy / 119.0)
        img = np.dstack([
            base + 0.04,
            base - 0.01,
            base + 0.08,
        ]).astype(np.float32)
        galaxy = np.exp(-(((xx - 80.0) / 18.0) ** 2 + ((yy - 60.0) / 12.0) ** 2)) * 0.25
        img += galaxy[:, :, None].astype(np.float32)
        img = np.clip(img, 0, 1)

        result = gradient_terminator(
            img,
            detail="medium",
            strength="medium",
            balance_background_color=True,
        )

        bg_mask = (((xx - 80.0) / 26.0) ** 2 + ((yy - 60.0) / 18.0) ** 2) > 1.0
        before_spread = float(np.std(np.median(img[bg_mask], axis=0)))
        after_spread = float(np.std(np.median(result[bg_mask], axis=0)))
        self.assertLess(after_spread, before_spread)

    def test_astro_star_shrink_multi_pass_is_stronger(self):
        yy, xx = np.mgrid[:96, :96].astype(np.float32)
        star = np.exp(-(((xx - 48.0) ** 2 + (yy - 48.0) ** 2) / (2 * 3.2 ** 2)))
        img = np.dstack([star, star, star]).astype(np.float32) * 0.9 + 0.04
        img = np.clip(img, 0, 1)

        one_pass = astro_star_shrink(img, amount=0.7, passes=1, noise_level=0.0)
        two_pass = astro_star_shrink(img, amount=0.7, passes=2, noise_level=0.0)

        core = ((xx - 48.0) ** 2 + (yy - 48.0) ** 2) <= 4.0
        self.assertLess(float(two_pass[core].mean()), float(one_pass[core].mean()))

    def test_astro_star_x_returns_starless_layers(self):
        yy, xx = np.mgrid[:96, :96].astype(np.float32)
        background = 0.06 + 0.12 * np.exp(-(((xx - 56.0) / 18.0) ** 2 + ((yy - 54.0) / 12.0) ** 2))
        star1 = 0.8 * np.exp(-(((xx - 30.0) ** 2 + (yy - 28.0) ** 2) / (2 * 1.5 ** 2)))
        star2 = 0.6 * np.exp(-(((xx - 72.0) ** 2 + (yy - 62.0) ** 2) / (2 * 2.0 ** 2)))
        mono = np.clip(background + star1 + star2, 0, 1).astype(np.float32)
        img = np.dstack([mono, mono * 0.98, mono * 0.95]).astype(np.float32)

        result = astro_star_x(img, retain_structures=0.85, noise_match=True)

        self.assertIn("starless", result)
        self.assertIn("stars_only", result)
        self.assertEqual(result["starless"].shape, img.shape)
        self.assertEqual(result["stars_only"].shape, img.shape)
        self.assertLess(float(result["starless"][28, 30, 0]), float(img[28, 30, 0]))
        self.assertGreater(float(result["stars_only"][28, 30, 0]), 0.05)


if __name__ == "__main__":
    unittest.main()
