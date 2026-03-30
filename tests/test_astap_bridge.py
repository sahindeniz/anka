import shutil
import sys
import unittest
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.astap_bridge import _parse_ini, _parse_star_count_from_output


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


if __name__ == "__main__":
    unittest.main()
