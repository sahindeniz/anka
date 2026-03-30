import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import launcher_shortcut


class LauncherShortcutTests(unittest.TestCase):
    @staticmethod
    def _workspace_tempdir() -> Path:
        base = ROOT / "_tmp_test_launcher_shortcut"
        base.mkdir(exist_ok=True)
        return Path(tempfile.mkdtemp(dir=base))

    def test_build_shortcut_command_sets_icon_and_description(self):
        command = launcher_shortcut.build_shortcut_command(
            "C:/Users/O'Brien/Desktop/AstroMestro.lnk",
            "C:/Apps/AstroMestro/setup_and_run.bat",
            "C:/Apps/AstroMestro",
            "C:/Apps/AstroMestro/gui/icons/astromestro_space.ico",
        )

        self.assertEqual(command[:4], ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass"])
        script = command[-1]
        self.assertIn("AstroMestro", script)
        self.assertIn("IconLocation", script)
        self.assertIn("O''Brien", script)

    def test_ensure_desktop_shortcut_removes_legacy_shortcut(self):
        root = self._workspace_tempdir()
        desktop = self._workspace_tempdir()
        try:
            (root / "gui" / "icons").mkdir(parents=True)
            (root / "setup_and_run.bat").write_text("@echo off\n", encoding="utf-8")
            (root / "gui" / "icons" / "astromestro_space.ico").write_bytes(b"ico")
            legacy = desktop / "Astro Maestro Pro.lnk"
            legacy.write_text("legacy", encoding="utf-8")

            with mock.patch("core.launcher_shortcut.subprocess.run") as run_mock:
                shortcut = launcher_shortcut.ensure_desktop_shortcut(root, desktop)

            self.assertEqual(shortcut, desktop / "AstroMestro.lnk")
            self.assertFalse(legacy.exists())
            run_mock.assert_called_once()
            command = run_mock.call_args.args[0]
            self.assertIn("AstroMestro.lnk", command[-1])
            self.assertIn("astromestro_space.ico,0", command[-1])
        finally:
            shutil.rmtree(root, ignore_errors=True)
            shutil.rmtree(desktop, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
