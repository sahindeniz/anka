"""
Astro Mastro Pro — Processing Panels
Toolbar'dan açılan işlem panelleri:
  BgPanel, StretchPanel, NoisePanel, SharpenPanel,
  ColorPanel, DeconvPanel, RecompPanel
"""
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QComboBox, QCheckBox, QPushButton, QDoubleSpinBox,
    QSpinBox, QGroupBox, QScrollArea, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal


# ── Temel panel sınıfı ───────────────────────────────────────────────────────

class BasePanel(QWidget):
    """
    apply_requested(params: dict) → işlemi uygula
    preview_requested(params: dict) → önizleme iste
    """
    apply_requested = pyqtSignal(dict)
    preview_requested = pyqtSignal(dict)

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setMaximumWidth(430)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(6)

        # Başlık
        lbl = QLabel(title)
        lbl.setStyleSheet("font-size: 12pt; font-weight: bold; color: #88aaff; padding: 2px;")
        outer.addWidget(lbl)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #444;")
        outer.addWidget(sep)

        # Scroll area içerik
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._content = QWidget()
        self._layout = QVBoxLayout(self._content)
        self._layout.setContentsMargins(2, 2, 2, 2)
        self._layout.setSpacing(8)
        scroll.setWidget(self._content)
        outer.addWidget(scroll, 1)

        # Alt düğmeler
        btn_row = QHBoxLayout()
        self._btn_preview = QPushButton("Önizle")
        self._btn_preview.clicked.connect(self._on_preview)
        btn_row.addWidget(self._btn_preview)

        self._btn_apply = QPushButton("Uygula")
        self._btn_apply.setStyleSheet(
            "QPushButton { background-color: #285299; color: #fff; font-weight: bold; }"
            "QPushButton:hover { background-color: #355ea1; }"
        )
        self._btn_apply.clicked.connect(self._on_apply)
        btn_row.addWidget(self._btn_apply)
        outer.addLayout(btn_row)

    def get_params(self) -> dict:
        return {}

    def _on_apply(self):
        self.apply_requested.emit(self.get_params())

    def _on_preview(self):
        self.preview_requested.emit(self.get_params())

    # Yardımcılar
    def _add_combo(self, label: str, items: list, key: str) -> QComboBox:
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        cb = QComboBox()
        cb.addItems(items)
        cb.setProperty("param_key", key)
        row.addWidget(cb)
        self._layout.addLayout(row)
        return cb

    def _add_slider(self, label: str, mn: int, mx: int, default: int,
                    key: str, scale: float = 1.0) -> QSlider:
        grp = QGroupBox(label)
        v = QVBoxLayout(grp)
        row = QHBoxLayout()
        sl = QSlider(Qt.Orientation.Horizontal)
        sl.setRange(mn, mx)
        sl.setValue(default)
        sl.setProperty("param_key", key)
        sl.setProperty("scale", scale)
        lbl_val = QLabel(str(default * scale if scale != 1.0 else default))
        lbl_val.setFixedWidth(40)
        sl.valueChanged.connect(
            lambda v, lv=lbl_val, sc=scale: lv.setText(
                f"{v * sc:.2f}" if sc != 1.0 else str(v)
            )
        )
        row.addWidget(sl)
        row.addWidget(lbl_val)
        v.addLayout(row)
        self._layout.addWidget(grp)
        return sl

    def _add_spinbox(self, label: str, mn: float, mx: float,
                     default: float, step: float, key: str,
                     decimals: int = 1) -> QDoubleSpinBox:
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        sp = QDoubleSpinBox()
        sp.setRange(mn, mx)
        sp.setValue(default)
        sp.setSingleStep(step)
        sp.setDecimals(decimals)
        sp.setProperty("param_key", key)
        row.addWidget(sp)
        self._layout.addLayout(row)
        return sp

    def _add_checkbox(self, label: str, checked: bool, key: str) -> QCheckBox:
        cb = QCheckBox(label)
        cb.setChecked(checked)
        cb.setProperty("param_key", key)
        self._layout.addWidget(cb)
        return cb


# ── Arka Plan Çıkarma ────────────────────────────────────────────────────────

class BgPanel(BasePanel):
    def __init__(self, parent=None):
        super().__init__("🌌 Arka Plan Çıkarma", parent)
        self._method = self._add_combo(
            "Yöntem:", ["dbe_spline", "polynomial", "median_grid", "ai_gradient", "gaussian", "graxpert"],
            "method"
        )
        self._grid = self._add_slider("Grid Boyutu", 4, 64, 16, "grid_size")
        self._degree = self._add_slider("Polinom Derecesi", 1, 8, 4, "poly_degree")
        self._clip = self._add_spinbox("Alçak Klip (%)", 0, 10, 0, 0.5, "clip_low", 1)
        self._layout.addStretch()

    def get_params(self):
        return {
            "method": self._method.currentText(),
            "grid_size": self._grid.value(),
            "poly_degree": self._degree.value(),
            "clip_low": self._clip.value(),
        }


# ── Histogram Gerdirme ───────────────────────────────────────────────────────

class StretchPanel(BasePanel):
    def __init__(self, parent=None):
        super().__init__("📊 Histogram Gerdirme", parent)
        self._method = self._add_combo(
            "Yöntem:", ["auto_stf", "linear", "hyperbolic", "asinh", "log", "midtone", "statistical", "power"],
            "method"
        )
        self._low = self._add_spinbox("Düşük Klip (%)", 0, 10, 2, 0.5, "low", 1)
        self._high = self._add_spinbox("Yüksek Klip (%)", 90, 100, 98, 0.5, "high", 1)
        self._gamma = self._add_spinbox("Gamma", 0.1, 5.0, 1.0, 0.1, "gamma", 2)
        self._stf_target = self._add_spinbox("STF Hedef", 0.05, 0.5, 0.25, 0.01, "stf_target", 2)
        self._layout.addStretch()

    def get_params(self):
        return {
            "method": self._method.currentText(),
            "low": self._low.value(),
            "high": self._high.value(),
            "gamma": self._gamma.value(),
            "stf_target": self._stf_target.value(),
        }


# ── Gürültü Azaltma ─────────────────────────────────────────────────────────

class NoisePanel(BasePanel):
    def __init__(self, parent=None):
        super().__init__("🔊 Gürültü Azaltma", parent)
        self._method = self._add_combo(
            "Yöntem:", ["bilateral", "gaussian", "median", "nlm", "wavelet"],
            "method"
        )
        self._strength = self._add_slider("Güç", 0, 100, 70, "strength", 0.01)
        self._detail = self._add_slider("Detay Koruma", 0, 100, 50, "detail", 0.01)
        self._iters = self._add_slider("İterasyon", 1, 5, 1, "iterations")
        self._layout.addStretch()

    def get_params(self):
        return {
            "method": self._method.currentText(),
            "strength": self._strength.value() / 100.0,
            "detail": self._detail.value() / 100.0,
            "iterations": self._iters.value(),
        }


# ── Keskinleştirme ───────────────────────────────────────────────────────────

class SharpenPanel(BasePanel):
    def __init__(self, parent=None):
        super().__init__("🔬 Keskinleştirme", parent)
        self._method = self._add_combo(
            "Yöntem:", ["multiscale_vlc", "unsharp_mask", "laplacian_ai", "high_pass"],
            "method"
        )
        self._amount = self._add_slider("Miktar", 0, 300, 100, "amount", 0.01)
        self._radius = self._add_spinbox("Yarıçap", 0.5, 10.0, 2.0, 0.5, "radius", 1)
        self._threshold = self._add_spinbox("Eşik", 0.0, 0.1, 0.0, 0.005, "threshold", 3)
        self._levels = self._add_slider("Ölçek Seviyeleri", 1, 6, 4, "scale_levels")
        self._iters = self._add_slider("İterasyon", 1, 5, 1, "iterations")
        self._layout.addStretch()

    def get_params(self):
        return {
            "method": self._method.currentText(),
            "amount": self._amount.value() / 100.0,
            "radius": self._radius.value(),
            "threshold": self._threshold.value(),
            "scale_levels": self._levels.value(),
            "iterations": self._iters.value(),
        }


# ── Renk Kalibrasyonu ────────────────────────────────────────────────────────

class ColorPanel(BasePanel):
    def __init__(self, parent=None):
        super().__init__("🎨 Renk Kalibrasyonu", parent)
        self._method = self._add_combo(
            "Yöntem:", ["spcc_g2v", "avg_spiral", "pcc", "photometric", "white_balance", "ai_neutral"],
            "method"
        )
        self._neutral_pct = self._add_spinbox("Nötr Yüzdelik", 10, 90, 50, 5, "neutral_percentile", 0)
        self._layout.addStretch()

    def get_params(self):
        return {
            "method": self._method.currentText(),
            "neutral_percentile": self._neutral_pct.value(),
        }


# ── Dekonvolüsyon ────────────────────────────────────────────────────────────

class DeconvPanel(BasePanel):
    def __init__(self, parent=None):
        super().__init__("🌀 Dekonvolüsyon", parent)
        self._method = self._add_combo(
            "Yöntem:", ["richardson_lucy", "wiener", "blind", "total_variation", "blur_exterminator"],
            "method"
        )
        self._psf_type = self._add_combo(
            "PSF Tipi:", ["moffat", "gaussian", "airy", "lorentzian"],
            "psf_type"
        )
        self._psf_size = self._add_slider("PSF Boyutu", 3, 25, 5, "psf_size")
        self._iters = self._add_slider("İterasyon", 5, 50, 15, "iterations")
        self._clip = self._add_spinbox("Klip Üst Sınırı", 0.5, 2.0, 1.0, 0.1, "clip", 1)
        self._layout.addStretch()

    def get_params(self):
        return {
            "method": self._method.currentText(),
            "psf_type": self._psf_type.currentText(),
            "psf_size": self._psf_size.value(),
            "iterations": self._iters.value(),
            "clip": self._clip.value(),
        }


# ── Yıldız Küçültme (Star Shrink) ────────────────────────────────────────────

class StarShrinkPanel(BasePanel):
    """Yıldız küçültme — çekirdek/halo ayrımı ile."""
    def __init__(self, parent=None):
        super().__init__("✦ Yıldız Küçültme", parent)

        info = QLabel(
            "Yıldızların çekirdeklerini koruyarak\n"
            "halolarını zayıflatır ve küçültür."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #aaa; font-size: 9pt;")
        self._layout.addWidget(info)

        self._shrink = self._add_spinbox("Küçültme Faktörü", 0.1, 3.0, 1.0, 0.1, "shrink_factor", 2)
        self._halo = self._add_spinbox("Halo Dolgu Oranı", 0.0, 1.0, 0.3, 0.05, "halo_fill_ratio", 2)
        self._noise = self._add_spinbox("Gürültü Seviyesi", 0.0, 50.0, 5.0, 1.0, "noise_level", 1)
        self._threshold = self._add_spinbox("Yoğunluk Eşiği (σ)", 0.5, 5.0, 2.0, 0.1, "star_density_threshold", 1)

        self._layout.addStretch()

    def get_params(self):
        return {
            "shrink_factor": self._shrink.value(),
            "halo_fill_ratio": self._halo.value(),
            "noise_level": self._noise.value(),
            "star_density_threshold": self._threshold.value(),
        }


# ── Yıldız Yeniden Birleştirme ───────────────────────────────────────────────

class RecompPanel(BasePanel):
    """StarNet / StarXTerminator ile ayrıştırılmış yıldızları geri ekler."""
    def __init__(self, parent=None):
        super().__init__("✦+ Yıldız Yeniden Birleştirme", parent)

        info = QLabel(
            "Yıldızsız görüntü + Yıldız maskesini\n"
            "birleştirerek orijinal görüntüyü elde eder."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #aaa; font-size: 9pt;")
        self._layout.addWidget(info)

        self._mode = self._add_combo(
            "Birleştirme Modu:", ["add", "screen", "lighten"],
            "mode"
        )
        self._opacity = self._add_slider("Yıldız Opaklığı", 0, 100, 100, "opacity", 0.01)
        self._layout.addStretch()

        # Önizle düğmesini gizle (tek adım işlem)
        self._btn_preview.setVisible(False)

    def get_params(self):
        return {
            "mode": self._mode.currentText(),
            "opacity": self._opacity.value() / 100.0,
        }
