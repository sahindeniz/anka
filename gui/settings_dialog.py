"""
Astro Mastro Pro — Settings Dialog
GraXpert, StarNet++, ASTAP, tema, çıktı formatı ayarları
"""
from PyQt6.QtWidgets import (
    QDialog, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QDoubleSpinBox,
    QSpinBox, QCheckBox, QFileDialog, QDialogButtonBox, QGroupBox
)
from PyQt6.QtCore import Qt
from gui import settings_manager as SM


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ayarlar")
        self.setMinimumWidth(500)
        self.setModal(True)

        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        layout.addWidget(tabs)

        # Sekmeler
        tabs.addTab(self._make_general_tab(), "⚙ Genel")
        tabs.addTab(self._make_starnet_tab(), "✦ StarNet++")
        tabs.addTab(self._make_graxpert_tab(), "🌌 GraXpert")
        tabs.addTab(self._make_astap_tab(), "🔭 ASTAP")

        # Tamam / İptal
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self._save_and_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    # ── Sekmeler ────────────────────────────────────────────────────────────

    def _make_general_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)

        # Tema
        g1 = QGroupBox("Görünüm")
        g1l = QVBoxLayout(g1)
        row = QHBoxLayout()
        row.addWidget(QLabel("Tema:"))
        self._theme = QComboBox()
        self._theme.addItems(["dark", "light"])
        self._theme.setCurrentText(SM.get("theme", "dark"))
        row.addWidget(self._theme)
        g1l.addLayout(row)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Font Boyutu:"))
        self._font_size = QSpinBox()
        self._font_size.setRange(8, 16)
        self._font_size.setValue(SM.get("font_size", 10))
        row2.addWidget(self._font_size)
        g1l.addLayout(row2)
        v.addWidget(g1)

        # Çıktı
        g2 = QGroupBox("Kaydetme")
        g2l = QVBoxLayout(g2)
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Çıktı Formatı:"))
        self._out_fmt = QComboBox()
        self._out_fmt.addItems(["FITS", "TIFF", "PNG", "JPG"])
        self._out_fmt.setCurrentText(SM.get("output_format", "FITS"))
        row3.addWidget(self._out_fmt)
        g2l.addLayout(row3)
        v.addWidget(g2)

        # Panel
        g3 = QGroupBox("Arayüz")
        g3l = QVBoxLayout(g3)
        row4 = QHBoxLayout()
        row4.addWidget(QLabel("Panel Genişliği:"))
        self._panel_w = QSpinBox()
        self._panel_w.setRange(300, 600)
        self._panel_w.setValue(SM.get("panel_width", 415))
        row4.addWidget(self._panel_w)
        g3l.addLayout(row4)

        self._show_hist = QCheckBox("Geçmişi göster")
        self._show_hist.setChecked(SM.get("show_history", True))
        g3l.addWidget(self._show_hist)

        row5 = QHBoxLayout()
        row5.addWidget(QLabel("Canvas İnterpolasyon:"))
        self._interp = QComboBox()
        self._interp.addItems(["nearest", "smooth"])
        self._interp.setCurrentText(SM.get("canvas_interp", "nearest"))
        row5.addWidget(self._interp)
        g3l.addLayout(row5)
        v.addWidget(g3)

        v.addStretch()
        return w

    def _make_starnet_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)

        g = QGroupBox("StarNet++ v2 CLI")
        gl = QVBoxLayout(g)

        row = QHBoxLayout()
        row.addWidget(QLabel("Çalıştırılabilir:"))
        self._sn_exe = QLineEdit(SM.get("starnet_exe", ""))
        self._sn_exe.setPlaceholderText("starnet++.exe yolunu seçin...")
        row.addWidget(self._sn_exe)
        btn = QPushButton("Gözat")
        btn.clicked.connect(lambda: self._browse(self._sn_exe, "exe"))
        row.addWidget(btn)
        gl.addLayout(row)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Stride:"))
        self._sn_stride = QSpinBox()
        self._sn_stride.setRange(64, 1024)
        self._sn_stride.setSingleStep(64)
        self._sn_stride.setValue(SM.get("starnet_stride", 256))
        row2.addWidget(self._sn_stride)
        gl.addLayout(row2)

        self._sn_gpu = QCheckBox("GPU Kullan")
        self._sn_gpu.setChecked(SM.get("starnet_use_gpu", False))
        gl.addWidget(self._sn_gpu)

        v.addWidget(g)
        v.addStretch()
        return w

    def _make_graxpert_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)

        g = QGroupBox("GraXpert AI")
        gl = QVBoxLayout(g)

        row = QHBoxLayout()
        row.addWidget(QLabel("Çalıştırılabilir:"))
        self._gx_exe = QLineEdit(SM.get("graxpert_exe", ""))
        self._gx_exe.setPlaceholderText("GraXpert.exe yolunu seçin...")
        row.addWidget(self._gx_exe)
        btn = QPushButton("Gözat")
        btn.clicked.connect(lambda: self._browse(self._gx_exe, "exe"))
        row.addWidget(btn)
        gl.addLayout(row)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Smoothing:"))
        self._gx_smooth = QDoubleSpinBox()
        self._gx_smooth.setRange(0.0, 1.0)
        self._gx_smooth.setSingleStep(0.05)
        self._gx_smooth.setDecimals(2)
        self._gx_smooth.setValue(SM.get("graxpert_smoothing", 0.5))
        row2.addWidget(self._gx_smooth)
        gl.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Düzeltme:"))
        self._gx_correction = QComboBox()
        self._gx_correction.addItems(["Subtraction", "Division"])
        self._gx_correction.setCurrentText(SM.get("graxpert_correction", "Subtraction"))
        row3.addWidget(self._gx_correction)
        gl.addLayout(row3)

        row4 = QHBoxLayout()
        row4.addWidget(QLabel("Denoise Gücü:"))
        self._gx_denoise = QDoubleSpinBox()
        self._gx_denoise.setRange(0.0, 1.0)
        self._gx_denoise.setSingleStep(0.05)
        self._gx_denoise.setDecimals(2)
        self._gx_denoise.setValue(SM.get("graxpert_denoise_strength", 0.8))
        row4.addWidget(self._gx_denoise)
        gl.addLayout(row4)

        v.addWidget(g)
        v.addStretch()
        return w

    def _make_astap_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)

        g = QGroupBox("ASTAP Plate Solving")
        gl = QVBoxLayout(g)

        row = QHBoxLayout()
        row.addWidget(QLabel("ASTAP Exe:"))
        self._astap_exe = QLineEdit(SM.get("astap_exe", ""))
        self._astap_exe.setPlaceholderText("astap.exe yolunu seçin...")
        row.addWidget(self._astap_exe)
        btn = QPushButton("Gözat")
        btn.clicked.connect(lambda: self._browse(self._astap_exe, "exe"))
        row.addWidget(btn)
        gl.addLayout(row)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Yıldız Kataloğu (DB):"))
        self._astap_db = QLineEdit(SM.get("astap_db", ""))
        self._astap_db.setPlaceholderText("Opsiyonel — kataloğun bulunduğu klasör")
        row2.addWidget(self._astap_db)
        btn2 = QPushButton("Gözat")
        btn2.clicked.connect(lambda: self._browse_dir(self._astap_db))
        row2.addWidget(btn2)
        gl.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Arama Yarıçapı (°):"))
        self._astap_radius = QDoubleSpinBox()
        self._astap_radius.setRange(0.5, 180.0)
        self._astap_radius.setSingleStep(5.0)
        self._astap_radius.setValue(SM.get("astap_radius", 30.0))
        row3.addWidget(self._astap_radius)
        gl.addLayout(row3)

        row4 = QHBoxLayout()
        row4.addWidget(QLabel("Min Yıldız:"))
        self._astap_stars = QSpinBox()
        self._astap_stars.setRange(5, 500)
        self._astap_stars.setValue(SM.get("astap_min_stars", 10))
        row4.addWidget(self._astap_stars)
        gl.addLayout(row4)

        row5 = QHBoxLayout()
        row5.addWidget(QLabel("Zaman Aşımı (sn):"))
        self._astap_timeout = QSpinBox()
        self._astap_timeout.setRange(10, 600)
        self._astap_timeout.setSingleStep(10)
        self._astap_timeout.setValue(SM.get("astap_timeout", 120))
        row5.addWidget(self._astap_timeout)
        gl.addLayout(row5)

        v.addWidget(g)
        v.addStretch()
        return w

    # ── Kaydet ──────────────────────────────────────────────────────────────

    def _save_and_accept(self):
        SM.set("theme", self._theme.currentText())
        SM.set("font_size", self._font_size.value())
        SM.set("output_format", self._out_fmt.currentText())
        SM.set("panel_width", self._panel_w.value())
        SM.set("show_history", self._show_hist.isChecked())
        SM.set("canvas_interp", self._interp.currentText())

        SM.set("starnet_exe", self._sn_exe.text())
        SM.set("starnet_stride", self._sn_stride.value())
        SM.set("starnet_use_gpu", self._sn_gpu.isChecked())

        SM.set("graxpert_exe", self._gx_exe.text())
        SM.set("graxpert_smoothing", self._gx_smooth.value())
        SM.set("graxpert_correction", self._gx_correction.currentText())
        SM.set("graxpert_denoise_strength", self._gx_denoise.value())

        SM.set("astap_exe", self._astap_exe.text())
        SM.set("astap_db", self._astap_db.text())
        SM.set("astap_radius", self._astap_radius.value())
        SM.set("astap_min_stars", self._astap_stars.value())
        SM.set("astap_timeout", self._astap_timeout.value())

        SM.save()
        self.accept()

    # ── Gözat yardımcıları ───────────────────────────────────────────────────

    def _browse(self, line_edit: QLineEdit, ftype: str = "exe"):
        if ftype == "exe":
            path, _ = QFileDialog.getOpenFileName(
                self, "Seç", "",
                "Çalıştırılabilir (*.exe);;Tümü (*)"
            )
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Seç", "", "Tümü (*)")
        if path:
            line_edit.setText(path)

    def _browse_dir(self, line_edit: QLineEdit):
        path = QFileDialog.getExistingDirectory(self, "Klasör Seç", "")
        if path:
            line_edit.setText(path)
