import shutil
import sys
import unittest
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.astap_bridge import (
    _build_fov_attempts,
    _build_solve_attempts,
    _parse_ini,
    _parse_star_count_from_output,
    _recommended_hint_radius,
)


class AstapBridgeTests(unittest.TestCase):
    def test_parse_ini_supports_pltsolvd_flags(self):
        tmp_root = ROOT / "_tmp_test_astap_bridge"
        tmp_root.mkdir(exist_ok=True)
        tmpdir = tmp_root / f"case_{uuid.uuid4().hex}"
        tmpdir.mkdir(parents=True, exist_ok=True)
        ini_path = tmpdir / "solve_input.ini"
        try:
            ini_path.write_text(
                "\n".join(
                    [
                        "[astap]",
                        "PLTSOLVD=T",
                        "CRVAL1=188.1234",
                        "CRVAL2=12.3456",
                        "CDELT1=-0.0005",
                        "CROTA2=178.5",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            result = _parse_ini(str(ini_path))

            self.assertEqual(result.get("solution"), "1")
            self.assertAlmostEqual(result.get("ra"), 188.1234, places=6)
            self.assertAlmostEqual(result.get("dec"), 12.3456, places=6)
            self.assertAlmostEqual(result.get("scale_arcsec"), 1.8, places=6)
            self.assertAlmostEqual(result.get("rotation_deg"), 178.5, places=6)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_parse_star_count_from_output_uses_astap_stdout(self):
        stdout = "Start finding stars\n1078 stars found of the requested 200. Background value is 0."

        stars = _parse_star_count_from_output(stdout, "")

        self.assertEqual(stars, 1078)

    def test_build_fov_attempts_uses_height_then_single_broader_retry(self):
        attempts = _build_fov_attempts(1.414, 2.075)

        self.assertEqual(len(attempts), 2)
        self.assertAlmostEqual(attempts[0], 1.414, places=3)
        self.assertAlmostEqual(attempts[1], 2.075, places=3)

    def test_build_solve_attempts_prefers_near_solve_before_blind(self):
        attempts = _build_solve_attempts(
            search_radius=60.0,
            ra_hint=188.0,
            dec_hint=1.4,
            fov_hint=1.414,
            fov_width_hint=2.075,
        )

        self.assertEqual(attempts[0]["label"], "near")
        self.assertAlmostEqual(
            attempts[0]["radius"],
            _recommended_hint_radius(60.0, 1.414),
            places=3,
        )
        self.assertEqual(attempts[-1]["label"], "blind")
        self.assertIsNone(attempts[-1]["ra_hint"])

    def test_build_solve_attempts_can_disable_near_search(self):
        attempts = _build_solve_attempts(
            search_radius=60.0,
            ra_hint=188.0,
            dec_hint=1.4,
            fov_hint=1.414,
            fov_width_hint=2.075,
            disable_near_search=True,
        )

        self.assertEqual(attempts[0]["label"], "near")
        self.assertAlmostEqual(attempts[0]["radius"], 60.0, places=3)
        self.assertEqual(attempts[-1]["label"], "blind")


if __name__ == "__main__":
    unittest.main()
