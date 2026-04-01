import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from processing.color_calibration import (
    _apply_catalog_calibration,
    _match_stars,
    _pcc_platesolve,
)


class ColorCalibrationTests(unittest.TestCase):
    def test_match_stars_reports_exclusion_stats(self):
        stars_px = [
            (0.0, 0.0, 1.0, 0.8, 0.9, 1.0),
            (1.0, 1.0, 0.9, 0.7, 0.8, 0.9),
            (2.0, 2.0, 0.8, 0.6, 0.7, 0.8),
            (3.0, 3.0, 0.7, 0.5, 0.6, 0.7),
        ]
        stars_radec = [
            (10.0, 20.0),
            (10.0, 20.0),
            (11.0, 21.0),
            (13.0, 24.0),
        ]
        catalog_stars = [
            (10.0, 20.0, 13.5, (1.0, 1.0, 1.0)),
            (11.0, 21.0, 14.5, (0.9, 1.0, 1.1)),
        ]

        matched, stats = _match_stars(
            stars_px, stars_radec, catalog_stars, np.zeros((4, 4, 3), dtype=np.float32), 1.8
        )

        self.assertEqual(len(matched), 1)
        self.assertEqual(stats["matched_count"], 1)
        self.assertEqual(stats["excluded_duplicate"], 1)
        self.assertEqual(stats["excluded_mag"], 1)
        self.assertEqual(stats["excluded_no_catalog"], 1)

    def test_apply_catalog_calibration_returns_fit_gain_and_background_stats(self):
        img = np.ones((16, 16, 3), dtype=np.float32)
        img[..., 0] *= 0.20
        img[..., 1] *= 0.25
        img[..., 2] *= 0.18

        expected_gains = np.array([0.80, 1.00, 1.20], dtype=np.float64)
        catalog_rgbs = [
            np.array([1.00, 1.00, 1.10]),
            np.array([0.92, 1.00, 1.04]),
            np.array([1.08, 1.00, 0.96]),
            np.array([1.03, 1.00, 1.00]),
            np.array([0.97, 1.00, 1.14]),
            np.array([1.01, 1.00, 0.92]),
        ]
        matched = [
            ((cat / expected_gains).tolist(), cat.tolist())
            for cat in catalog_rgbs
        ]

        result, stats = _apply_catalog_calibration(img, matched)

        self.assertEqual(result.shape, img.shape)
        self.assertEqual(stats["used_star_count"], len(matched))
        np.testing.assert_allclose(stats["gains"], expected_gains, atol=0.02)
        self.assertIsNotNone(stats["fit_rg"])
        self.assertIsNotNone(stats["fit_bg"])
        self.assertEqual(stats["background"].shape, (3,))
        self.assertEqual(stats["quality_warning"], "")

    def test_pcc_platesolve_logs_detailed_solution_summary(self):
        image = np.ones((32, 32, 3), dtype=np.float32) * 0.1
        detected_stars = [(float(i), float(i), 1.0, 0.8, 0.9, 1.0) for i in range(12)]
        stars_radec = [(188.0 + i * 1e-3, 2.3 + i * 1e-3) for i in range(12)]
        matched = [((0.8, 0.9, 1.0), (1.0, 1.0, 1.1))] * 6
        match_stats = {
            "matched_count": 6,
            "excluded_no_catalog": 2,
            "excluded_mag": 1,
            "excluded_duplicate": 3,
            "candidate_count": 12,
            "match_radius_arcsec": 54.0,
        }
        cal_stats = {
            "background": np.array([0.05, 0.01, 0.001], dtype=np.float64),
            "gains": np.array([0.707, 0.922, 1.000], dtype=np.float64),
            "input_star_count": 6,
            "used_star_count": 6,
            "excluded_outliers": 1,
            "fit_rg": {"intercept": 0.12, "slope": 0.88, "sigma": 0.05},
            "fit_bg": {"intercept": -0.40, "slope": 1.82, "sigma": 0.06},
            "quality_warning": "Fotometrik çözüm biraz belirsiz görünüyor; önce gradient düzeltmeyi deneyin",
        }
        catalog_meta = {"source_name": "Gaia DR3", "radius_deg": 1.0, "limit_mag": 16.5, "error": ""}
        catalog_stars = [(188.0 + i * 1e-3, 2.3 + i * 1e-3, 13.0, (1.0, 1.0, 1.1)) for i in range(12)]

        logs = []
        with (
            patch("processing.color_calibration._detect_stars_for_pcc", return_value=detected_stars),
            patch("processing.color_calibration._pixel_to_radec", return_value=stars_radec),
            patch("processing.color_calibration._query_gaia_colors", return_value=(catalog_stars, catalog_meta)),
            patch("processing.color_calibration._match_stars", return_value=(matched, match_stats)),
            patch("processing.color_calibration._apply_catalog_calibration", return_value=(image, cal_stats)),
        ):
            result = _pcc_platesolve(
                image,
                progress_cb=logs.append,
                solve_ra=188.0,
                solve_dec=2.3,
                solve_scale=1.8,
                solve_rotation=0.0,
                catalog_limit_mag=16.5,
            )

        self.assertIs(result, image)
        joined = "\n".join(logs)
        self.assertIn("limit mag 16.50", joined)
        self.assertIn("PCC Linear Fits", joined)
        self.assertIn("White balance factors:", joined)
        self.assertIn("K0: 0.707", joined)
        self.assertIn("Background reference:", joined)
        self.assertIn("B0: +5.00000e-02", joined)
        self.assertIn("gradient düzeltmeyi deneyin", joined)


if __name__ == "__main__":
    unittest.main()
