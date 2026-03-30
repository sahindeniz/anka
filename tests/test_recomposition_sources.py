import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gui.recomposition_sources import collect_recomposition_sources


class _FakeTabBar:
    def __init__(self, index: int):
        self._index = index

    def currentIndex(self) -> int:
        return self._index


class RecompositionSourceTests(unittest.TestCase):
    def test_collect_sources_includes_tabs_and_history(self):
        current = np.full((2, 2), 1.0, dtype=np.float32)
        original = np.full((2, 2), 2.0, dtype=np.float32)
        tab_a = np.full((2, 2), 3.0, dtype=np.float32)
        tab_b = np.full((2, 2), 4.0, dtype=np.float32)
        film = np.full((2, 2), 5.0, dtype=np.float32)
        hist0 = np.full((2, 2), 6.0, dtype=np.float32)
        hist1 = np.full((2, 2), 7.0, dtype=np.float32)

        app = SimpleNamespace(
            _current=current,
            _orig=original,
            _img_tabs=_FakeTabBar(1),
            _img_tab_data=[
                {"key": "starless", "title": "Yıldızsız (StarNet)", "image": tab_a},
                {"key": "starmask", "title": "Yıldız Maskesi (StarNet)", "image": tab_b},
            ],
            _filmstrip_data=[{"path": str(ROOT / "demo.fit"), "img": film}],
            _history=[
                ("Original", hist0),
                ("Stretch", hist1),
            ],
        )

        options = collect_recomposition_sources(app)
        labels = [label for label, _img, _source in options]

        self.assertIn("🖼  Mevcut görüntü (aktif)", labels)
        self.assertIn("🗂  Sekme: Yıldızsız (StarNet)", labels)
        self.assertIn("🗂  Sekme (aktif): Yıldız Maskesi (StarNet)", labels)
        self.assertIn("📌  Orijinal görüntü", labels)
        self.assertTrue(any(label.startswith("📂  demo.fit") for label in labels))
        self.assertIn("📜  Step 1: Stretch", labels)
        self.assertNotIn("📜  Step 0: Original", labels)

        # Returned images should be copies, not shared references.
        options[0][1][:] = 99.0
        self.assertEqual(float(current[0, 0]), 1.0)

    def test_collect_sources_keeps_history_zero_when_no_original(self):
        hist0 = np.full((2, 2), 0.25, dtype=np.float32)
        app = SimpleNamespace(
            _current=None,
            _orig=None,
            _img_tabs=_FakeTabBar(-1),
            _img_tab_data=[],
            _filmstrip_data=[],
            _history=[("Imported", hist0)],
        )

        options = collect_recomposition_sources(app)
        self.assertEqual(len(options), 1)
        self.assertEqual(options[0][0], "📜  Step 0: Imported")


if __name__ == "__main__":
    unittest.main()
