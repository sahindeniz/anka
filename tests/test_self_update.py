import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.self_update import extract_release_archive, merge_release_tree, write_update_helper


class SelfUpdateTests(unittest.TestCase):
    @staticmethod
    def _workspace_tempdir() -> Path:
        base = ROOT / "_tmp_test_self_update"
        base.mkdir(exist_ok=True)
        return Path(tempfile.mkdtemp(dir=base))

    def test_extract_release_archive_finds_prefixed_app_root(self):
        tmpdir = self._workspace_tempdir()
        try:
            zip_path = Path(tmpdir) / "release.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr("AstroMaestroPro/main.py", "print('ok')\n")
                archive.writestr("AstroMaestroPro/core/version.py", 'APP_VERSION = "2.1.2"\n')

            staging_dir, app_root = extract_release_archive(zip_path)
            try:
                self.assertEqual(app_root.name, "AstroMaestroPro")
                self.assertTrue((app_root / "main.py").is_file())
                self.assertTrue((app_root / "core" / "version.py").is_file())
            finally:
                shutil.rmtree(staging_dir, ignore_errors=True)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_merge_release_tree_overwrites_code_and_preserves_settings(self):
        src_root = self._workspace_tempdir()
        dst_root = self._workspace_tempdir()
        try:

            (src_root / "core").mkdir()
            (src_root / "main.py").write_text("print('new')\n", encoding="utf-8")
            (src_root / "core" / "version.py").write_text('APP_VERSION = "2.1.2"\n', encoding="utf-8")
            (dst_root / "main.py").write_text("print('old')\n", encoding="utf-8")
            (dst_root / "settings.json").write_text('{"keep": true}\n', encoding="utf-8")

            merge_release_tree(src_root, dst_root)

            self.assertEqual((dst_root / "main.py").read_text(encoding="utf-8"), "print('new')\n")
            self.assertEqual(
                (dst_root / "settings.json").read_text(encoding="utf-8"),
                '{"keep": true}\n',
            )
        finally:
            shutil.rmtree(src_root, ignore_errors=True)
            shutil.rmtree(dst_root, ignore_errors=True)

    def test_write_update_helper_embeds_requested_paths(self):
        helper_dir = self._workspace_tempdir()
        helper_path = write_update_helper(
            "C:/src/app",
            "C:/dst/app",
            4242,
            module_root=ROOT,
            helper_dir=helper_dir,
        )
        try:
            text = helper_path.read_text(encoding="utf-8")
            self.assertIn("apply_update_and_restart", text)
            self.assertIn("C:/src/app", text)
            self.assertIn("C:/dst/app", text)
            self.assertIn("4242", text)
        finally:
            shutil.rmtree(helper_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
