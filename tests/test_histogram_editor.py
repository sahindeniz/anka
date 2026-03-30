import os
import sys
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication

from gui.histogram_editor import (
    DEFAULT_PREVIEW_MAX_SIDE,
    HistogramEditorPanel,
    apply_camera_raw_adjustments,
    build_preview_proxy,
)


_APP = QApplication.instance() or QApplication([])


class HistogramEditorPreviewTests(unittest.TestCase):
    def test_build_preview_proxy_downscales_large_image(self):
        img = np.zeros((2400, 3600, 3), dtype=np.float32)

        proxy, scale = build_preview_proxy(img, max_side=1440)

        self.assertEqual(proxy.dtype, np.float32)
        self.assertLess(scale, 1.0)
        self.assertEqual(proxy.shape[:2], (960, 1440))

    def test_build_preview_proxy_keeps_small_image(self):
        img = np.ones((320, 480, 3), dtype=np.float32)

        proxy, scale = build_preview_proxy(img, max_side=1440)

        self.assertEqual(scale, 1.0)
        self.assertEqual(proxy.shape, img.shape)
        np.testing.assert_allclose(proxy, img)


class HistogramEditorRawAdjustmentsTests(unittest.TestCase):
    def test_camera_raw_adjustments_lift_shadows_and_tame_highlights(self):
        img = np.array(
            [
                [[0.08, 0.10, 0.12], [0.92, 0.88, 0.84]],
                [[0.18, 0.16, 0.14], [0.65, 0.62, 0.60]],
            ],
            dtype=np.float32,
        )

        out = apply_camera_raw_adjustments(
            img,
            {
                "profile": "Renk",
                "exposure": 0.0,
                "contrast": 10.0,
                "highlights": -55.0,
                "shadows": 45.0,
                "whites": 0.0,
                "blacks": 0.0,
                "temp": 0.0,
                "tint": 0.0,
                "texture": 0.0,
                "sharpen": 0.0,
                "clarity": 0.0,
                "dehaze": 0.0,
                "vibrance": 0.0,
                "saturation": 0.0,
            },
        )

        self.assertEqual(out.shape, img.shape)
        self.assertEqual(out.dtype, np.float32)
        self.assertGreater(float(out[0, 0].mean()), float(img[0, 0].mean()))
        self.assertLess(float(out[0, 1].mean()), float(img[0, 1].mean()))

    def test_camera_raw_adjustments_bw_profile_equalizes_channels(self):
        img = np.array(
            [[[0.20, 0.40, 0.80], [0.90, 0.30, 0.10]]],
            dtype=np.float32,
        )

        out = apply_camera_raw_adjustments(
            img,
            {
                "profile": "Siyah Beyaz",
                "exposure": 0.0,
                "contrast": 0.0,
                "highlights": 0.0,
                "shadows": 0.0,
                "whites": 0.0,
                "blacks": 0.0,
                "temp": 40.0,
                "tint": -20.0,
                "texture": 0.0,
                "sharpen": 0.0,
                "clarity": 0.0,
                "dehaze": 0.0,
                "vibrance": 30.0,
                "saturation": 30.0,
            },
        )

        np.testing.assert_allclose(out[:, :, 0], out[:, :, 1], atol=1e-6)
        np.testing.assert_allclose(out[:, :, 1], out[:, :, 2], atol=1e-6)

    def test_camera_raw_adjustments_make_color_balance_shift_visible(self):
        img = np.full((2, 2, 3), 0.12, dtype=np.float32)

        out = apply_camera_raw_adjustments(
            img,
            {
                "profile": "Renk",
                "exposure": 0.0,
                "contrast": 0.0,
                "highlights": 0.0,
                "shadows": 0.0,
                "whites": 0.0,
                "blacks": 0.0,
                "temp": 60.0,
                "tint": 35.0,
                "texture": 0.0,
                "sharpen": 0.0,
                "clarity": 0.0,
                "dehaze": 0.0,
                "vibrance": 45.0,
                "saturation": 30.0,
            },
        )

        self.assertGreater(float(out[:, :, 0].mean()), float(out[:, :, 1].mean()))
        self.assertGreater(float(out[:, :, 1].mean()), float(out[:, :, 2].mean()))
        self.assertGreater(float(out[:, :, 0].mean() - out[:, :, 2].mean()), 0.05)


class HistogramEditorPanelTests(unittest.TestCase):
    def setUp(self):
        self.panel = HistogramEditorPanel()
        self.img = np.zeros((4, 4, 3), dtype=np.float32)
        self.img[:, :, 0] = 0.25
        self.img[:, :, 1] = 0.50
        self.img[:, :, 2] = 0.75
        self.panel.set_image(self.img, reset=True)

    def test_curve_section_is_first_and_expanded(self):
        section_order = []
        for idx in range(self.panel._panel_lay.count()):
            item = self.panel._panel_lay.itemAt(idx)
            widget = item.widget() if item is not None else None
            if widget is None:
                continue
            name = widget.objectName()
            if name.startswith("hist_section_"):
                section_order.append(name)
        self.assertEqual(
            section_order,
            [
                "hist_section_curve",
                "hist_section_basic",
                "hist_section_detail",
                "hist_section_levels",
                "hist_section_channels",
            ],
        )
        curve_body = self.panel.findChild(type(self.panel._curves_wgt.parentWidget()), "hist_section_body_curve")
        self.assertIsNotNone(curve_body)
        self.assertFalse(curve_body.isHidden())
        self.assertGreaterEqual(self.panel._curves_wgt.minimumHeight(), 320)

    def test_curve_channel_edit_unlinked_affects_selected_channel_only(self):
        self.panel._chk_link.setChecked(False)
        self.panel._set_channel("G")
        self.panel._curves_wgt._pts["G"] = [(0.0, 0.0), (0.5, 1.0), (1.0, 1.0)]
        self.panel._curves_wgt._mark_dirty("G")

        out = self.panel._apply(emit=False, preview=False)

        self.assertAlmostEqual(float(out[0, 0, 0]), 0.25, places=4)
        self.assertGreater(float(out[0, 0, 1]), 0.90)
        self.assertAlmostEqual(float(out[0, 0, 2]), 0.75, places=4)

    def test_levels_with_rgb_link_affect_all_rgb_channels(self):
        self.panel._chk_link.setChecked(True)
        self.panel._set_channel("R")
        self.panel._hist_wgt.set_state("R", 0.0, 0.35, 1.0, 0.0, 1.0)
        self.panel._on_levels_changed("R", 0.0, 0.35, 1.0, 0.0, 1.0)

        for channel in ("R", "G", "B"):
            self.assertEqual(
                self.panel._hist_wgt.get_state(channel),
                [0.0, 0.35, 1.0, 0.0, 1.0],
            )

        out = self.panel._apply(emit=False, preview=False)
        self.assertGreater(float(out[0, 0, 0]), 0.35)
        self.assertGreater(float(out[0, 0, 1]), 0.60)
        self.assertGreater(float(out[0, 0, 2]), 0.80)

    def test_live_preview_emits_proxy_payload_instead_of_full_resolution_image(self):
        panel = HistogramEditorPanel()
        large = np.zeros((2400, 3600, 3), dtype=np.float32)
        large[:, :, 1] = 0.2
        panel.set_image(large, reset=True)
        panel._chk_link.setChecked(False)
        panel._set_channel("G")
        panel._curves_wgt._pts["G"] = [(0.0, 0.0), (0.5, 1.0), (1.0, 1.0)]
        panel._curves_wgt._mark_dirty("G")

        payloads = []
        panel.preview_changed.connect(payloads.append)
        panel._emit_preview()

        self.assertEqual(len(payloads), 1)
        payload = payloads[0]
        self.assertIsInstance(payload, dict)
        self.assertEqual(tuple(payload["display_shape"]), (2400, 3600))
        self.assertLessEqual(max(payload["image"].shape[:2]), DEFAULT_PREVIEW_MAX_SIDE)
        self.assertLess(payload["image"].shape[0], large.shape[0])
        self.assertLess(payload["image"].shape[1], large.shape[1])


if __name__ == "__main__":
    unittest.main()
