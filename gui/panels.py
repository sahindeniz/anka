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
from PyQt6.QtCore import Qt, pyqtSignal, QTimer


# ── Temel panel sınıfı ───────────────────────────────────────────────────────

class BasePanel(QWidget):
    """
    apply_requested(params: dict) → işlemi uygula
    preview_requested(params: dict) → önizleme iste
    """
    apply_requested = pyqtSignal(dict)
    preview_requested = pyqtSignal(dict)
    live_preview_requested = pyqtSignal(dict)

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setMaximumWidth(430)

        # Live preview debounce timer (300ms)
        self._live_timer = QTimer(self)
        self._live_timer.setSingleShot(True)
        self._live_timer.setInterval(300)
        self._live_timer.timeout.connect(self._emit_live_preview)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(6)

        # Başlık + Live checkbox satırı
        title_row = QHBoxLayout()
        lbl = QLabel(title)
        lbl.setStyleSheet("font-size: 12pt; font-weight: bold; color: #88aaff; padding: 2px;")
        title_row.addWidget(lbl)
        title_row.addStretch()
        self._chk_live = QCheckBox("Live")
        self._chk_live.setChecked(False)
        self._chk_live.setStyleSheet("color: #ff8844; font-weight: bold; font-size: 10pt;")
        title_row.addWidget(self._chk_live)
        outer.addLayout(title_row)

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

    def _schedule_live(self):
        """Parametre değiştiğinde debounce ile live preview tetikle."""
        if self._chk_live.isChecked():
            self._live_timer.start()

    def _emit_live_preview(self):
        self.live_preview_requested.emit(self.get_params())

    # Yardımcılar
    def _add_combo(self, label: str, items: list, key: str) -> QComboBox:
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        cb = QComboBox()
        cb.addItems(items)
        cb.setProperty("param_key", key)
        cb.currentIndexChanged.connect(lambda: self._schedule_live())
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
        sl.valueChanged.connect(lambda: self._schedule_live())
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
        sp.valueChanged.connect(lambda: self._schedule_live())
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

        # Her metot için bağımsız grup oluştur
        self._method_groups = {}

        # ── DBE Spline ──
        grp1 = QGroupBox("DBE Spline (RBF)")
        grp1.setStyleSheet("QGroupBox { font-weight: bold; color: #88ccff; border: 1px solid #444; border-radius: 4px; margin-top: 6px; padding-top: 14px; }")
        v1 = QVBoxLayout(grp1)
        self._dbe_grid = self._make_slider(v1, "Grid Boyutu", 4, 64, 16)
        self._dbe_clip = self._make_spinbox(v1, "Alçak Klip (%)", 0, 10, 0, 0.5, 1)
        self._make_buttons(v1, "dbe_spline")
        self._layout.addWidget(grp1)
        self._method_groups["dbe_spline"] = grp1

        # ── Polynomial ──
        grp2 = QGroupBox("Polinom Yüzey")
        grp2.setStyleSheet("QGroupBox { font-weight: bold; color: #88ccff; border: 1px solid #444; border-radius: 4px; margin-top: 6px; padding-top: 14px; }")
        v2 = QVBoxLayout(grp2)
        self._poly_degree = self._make_slider(v2, "Polinom Derecesi", 1, 8, 4)
        self._poly_clip = self._make_spinbox(v2, "Alçak Klip (%)", 0, 10, 0, 0.5, 1)
        self._make_buttons(v2, "polynomial")
        self._layout.addWidget(grp2)
        self._method_groups["polynomial"] = grp2

        # ── Median Grid ──
        grp3 = QGroupBox("Medyan Grid")
        grp3.setStyleSheet("QGroupBox { font-weight: bold; color: #88ccff; border: 1px solid #444; border-radius: 4px; margin-top: 6px; padding-top: 14px; }")
        v3 = QVBoxLayout(grp3)
        self._med_clip = self._make_spinbox(v3, "Alçak Klip (%)", 0, 10, 0, 0.5, 1)
        self._make_buttons(v3, "median_grid")
        self._layout.addWidget(grp3)
        self._method_groups["median_grid"] = grp3

        # ── AI Gradient ──
        grp4 = QGroupBox("AI Gradient")
        grp4.setStyleSheet("QGroupBox { font-weight: bold; color: #88ccff; border: 1px solid #444; border-radius: 4px; margin-top: 6px; padding-top: 14px; }")
        v4 = QVBoxLayout(grp4)
        self._ai_degree = self._make_slider(v4, "Polinom Derecesi", 1, 8, 3)
        self._ai_clip = self._make_spinbox(v4, "Alçak Klip (%)", 0, 10, 0, 0.5, 1)
        self._make_buttons(v4, "ai_gradient")
        self._layout.addWidget(grp4)
        self._method_groups["ai_gradient"] = grp4

        # ── Gaussian ──
        grp5 = QGroupBox("Gaussian Bulanıklaştırma")
        grp5.setStyleSheet("QGroupBox { font-weight: bold; color: #88ccff; border: 1px solid #444; border-radius: 4px; margin-top: 6px; padding-top: 14px; }")
        v5 = QVBoxLayout(grp5)
        self._gauss_clip = self._make_spinbox(v5, "Alçak Klip (%)", 0, 10, 0, 0.5, 1)
        self._make_buttons(v5, "gaussian")
        self._layout.addWidget(grp5)
        self._method_groups["gaussian"] = grp5

        # ── GraXpert ──
        grp6 = QGroupBox("GraXpert (DBE)")
        grp6.setStyleSheet("QGroupBox { font-weight: bold; color: #88ccff; border: 1px solid #444; border-radius: 4px; margin-top: 6px; padding-top: 14px; }")
        v6 = QVBoxLayout(grp6)
        self._grax_grid = self._make_slider(v6, "Grid Boyutu", 4, 64, 16)
        self._grax_clip = self._make_spinbox(v6, "Alçak Klip (%)", 0, 10, 0, 0.5, 1)
        self._make_buttons(v6, "graxpert")
        self._layout.addWidget(grp6)
        self._method_groups["graxpert"] = grp6

        self._layout.addStretch()

        # Alt butonları gizle — her grubun kendi butonları var
        self._btn_preview.setVisible(False)
        self._btn_apply.setVisible(False)

        # Son tıklanan metot bilgisi
        self._last_method = "dbe_spline"

    def _make_slider(self, layout, label, mn, mx, default):
        """Grup içi slider oluşturur (BasePanel._add_slider yerine)."""
        grp = QGroupBox(label)
        v = QVBoxLayout(grp)
        row = QHBoxLayout()
        sl = QSlider(Qt.Orientation.Horizontal)
        sl.setRange(mn, mx)
        sl.setValue(default)
        lbl_val = QLabel(str(default))
        lbl_val.setFixedWidth(40)
        sl.valueChanged.connect(lambda val, lv=lbl_val: lv.setText(str(val)))
        sl.valueChanged.connect(lambda: self._schedule_live())
        row.addWidget(sl)
        row.addWidget(lbl_val)
        v.addLayout(row)
        layout.addWidget(grp)
        return sl

    def _make_spinbox(self, layout, label, mn, mx, default, step, decimals):
        """Grup içi spinbox oluşturur."""
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        sp = QDoubleSpinBox()
        sp.setRange(mn, mx)
        sp.setValue(default)
        sp.setSingleStep(step)
        sp.setDecimals(decimals)
        sp.valueChanged.connect(lambda: self._schedule_live())
        row.addWidget(sp)
        layout.addLayout(row)
        return sp

    def _make_buttons(self, layout, method_name):
        """Her metot grubu için bağımsız Önizle/Uygula butonları."""
        row = QHBoxLayout()
        btn_prev = QPushButton("Önizle")
        btn_prev.clicked.connect(lambda _, m=method_name: self._emit_method(m, preview=True))
        row.addWidget(btn_prev)

        btn_apply = QPushButton("Uygula")
        btn_apply.setStyleSheet(
            "QPushButton { background-color: #285299; color: #fff; font-weight: bold; }"
            "QPushButton:hover { background-color: #355ea1; }"
        )
        btn_apply.clicked.connect(lambda _, m=method_name: self._emit_method(m, preview=False))
        row.addWidget(btn_apply)
        layout.addLayout(row)

    def _emit_method(self, method, preview=False):
        """Belirli metodun parametrelerini toplayıp sinyal gönderir."""
        self._last_method = method
        params = self._get_method_params(method)
        if preview:
            self.preview_requested.emit(params)
        else:
            self.apply_requested.emit(params)

    def _get_method_params(self, method):
        """Her metot için sadece kendi parametrelerini döndürür."""
        if method == "dbe_spline":
            return {
                "method": "dbe_spline",
                "grid_size": self._dbe_grid.value(),
                "clip_low": self._dbe_clip.value(),
            }
        elif method == "polynomial":
            return {
                "method": "polynomial",
                "poly_degree": self._poly_degree.value(),
                "clip_low": self._poly_clip.value(),
            }
        elif method == "median_grid":
            return {
                "method": "median_grid",
                "clip_low": self._med_clip.value(),
            }
        elif method == "ai_gradient":
            return {
                "method": "ai_gradient",
                "poly_degree": self._ai_degree.value(),
                "clip_low": self._ai_clip.value(),
            }
        elif method == "gaussian":
            return {
                "method": "gaussian",
                "clip_low": self._gauss_clip.value(),
            }
        elif method == "graxpert":
            return {
                "method": "graxpert",
                "grid_size": self._grax_grid.value(),
                "clip_low": self._grax_clip.value(),
            }
        return {"method": method}

    def get_params(self):
        return self._get_method_params(self._last_method)


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
