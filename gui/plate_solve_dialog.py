"""
Astro Mastro Pro — Plate Solve Dialog  (Siril tarzı, v3)
=========================================================
Siril'deki "Image Plate Solver" dialogunun tam karşılığı:
  • RA / Dec ipucu (h m s + d m s ayrı kutucuklar)
  • SIMBAD objesi arama
  • Focal length + pixel size → FOV hesaplama
  • Siril Solver / ASTAP seçimi
  • Flip, Downsample, Auto-crop, Save distortion
  • Search radius
  • Katalog: ASTAP exe'nin yanındaki katalog klasörü (G17/H17/W08/D05)
  • Catalogue limit magnitude
  • Star Detection parametreleri
  • Solution order (Linear/Quadratic/Cubic SIP)
  • Progress log + sonuç kartı
"""

import os, re, sys
import numpy as np

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QDoubleSpinBox, QSpinBox,
    QCheckBox, QComboBox, QGroupBox, QTextEdit,
    QProgressBar, QWidget, QFrame, QLineEdit,
    QSizePolicy, QMessageBox, QFileDialog, QSplitter
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QColor

# ── Renk paleti ──────────────────────────────────────────────────────────────
BG      = "#050e1a"
BG2     = "#081526"
BG3     = "#0c1e35"
BG4     = "#102644"
BORDER  = "#1a3a5c"
BORDER2 = "#2a5a8a"
ACCENT  = "#3d9bd4"
ACCENT2 = "#5bb8f0"
GOLD    = "#d4a44f"
GREEN   = "#3dbd6e"
RED     = "#d45555"
TEXT    = "#ddeeff"
MUTED   = "#7aa0c0"
HEAD    = "#a8d4f0"
SUBTEXT = "#4a7a9a"

_BTN = (
    f"QPushButton{{background:{BG3};color:{TEXT};border:1px solid {BORDER};"
    f"border-radius:4px;padding:3px 10px;font-size:10px;}}"
    f"QPushButton:hover{{background:{BG4};border:1px solid {ACCENT};}}"
    f"QPushButton:pressed{{background:{BG};}}"
    f"QPushButton:disabled{{color:{SUBTEXT};}}"
)
_RUN = (
    f"QPushButton{{background:{ACCENT};color:#000;border:none;"
    f"border-radius:4px;padding:5px 18px;font-size:11px;font-weight:700;}}"
    f"QPushButton:hover{{background:{ACCENT2};}}"
    f"QPushButton:pressed{{background:{BG4};color:{TEXT};}}"
    f"QPushButton:disabled{{background:{BG3};color:{SUBTEXT};}}"
)
_SPIN = (
    f"QDoubleSpinBox,QSpinBox{{background:{BG};color:{TEXT};"
    f"border:1px solid {BORDER};border-radius:3px;padding:2px 4px;font-size:10px;}}"
    f"QDoubleSpinBox:focus,QSpinBox:focus{{border:1px solid {ACCENT};}}"
)
_EDIT = (
    f"QLineEdit{{background:{BG};color:{TEXT};border:1px solid {BORDER};"
    f"border-radius:3px;padding:3px 6px;font-size:10px;}}"
    f"QLineEdit:focus{{border:1px solid {ACCENT};}}"
)
_COMBO = (
    f"QComboBox{{background:{BG};color:{TEXT};border:1px solid {BORDER};"
    f"border-radius:3px;padding:2px 6px;font-size:10px;}}"
    f"QComboBox QAbstractItemView{{background:{BG2};color:{TEXT};"
    f"selection-background-color:{BG4};border:1px solid {BORDER};}}"
    f"QComboBox::drop-down{{border:none;width:16px;}}"
)
_CHK = (
    f"QCheckBox{{color:{HEAD};font-size:10px;spacing:5px;}}"
    f"QCheckBox::indicator{{width:13px;height:13px;border-radius:2px;"
    f"border:1px solid {BORDER};background:{BG};}}"
    f"QCheckBox::indicator:checked{{background:{ACCENT};border:1px solid {ACCENT2};}}"
)
_GRP = (
    f"QGroupBox{{background:{BG3};border:1px solid {BORDER};"
    f"border-radius:6px;margin-top:14px;padding:6px;}}"
    f"QGroupBox::title{{color:{HEAD};font-size:10px;font-weight:700;"
    f"subcontrol-origin:margin;left:8px;padding:0 4px;}}"
)
_LBL = f"color:{MUTED};font-size:10px;"


# ─────────────────────────────────────────────────────────────────────────────
#  Worker thread
# ─────────────────────────────────────────────────────────────────────────────
class _SolveWorker(QThread):
    finished = pyqtSignal(dict)
    progress = pyqtSignal(str)
    error    = pyqtSignal(str)

    def __init__(self, image, params: dict):
        super().__init__()
        self.image  = image
        self.params = params
        self.setObjectName("PlateSolveWorker")

    def run(self):
        try:
            from ai.astap_bridge import solve_image
            p = self.params
            result = solve_image(
                image         = self.image,
                astap_exe     = p["astap_exe"],
                db_path       = p["db_path"],
                search_radius = p["search_radius"],
                downsample    = p["downsample"],
                min_stars     = p["min_stars"],
                timeout       = p["timeout"],
                ra_hint       = p.get("ra_hint"),
                dec_hint      = p.get("dec_hint"),
                fov_hint      = p.get("fov_hint"),
                progress_cb   = lambda m: self.progress.emit(str(m)),
            )
            self.finished.emit(result)
        except Exception:
            import traceback
            self.error.emit(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
#  H M S  /  D M S  satır widget'ı
# ─────────────────────────────────────────────────────────────────────────────
class _HMSWidget(QWidget):
    """Siril tarzı H — M — S  veya  D — M — S giriş satırı."""
    def __init__(self, is_dms=False, parent=None):
        super().__init__(parent)
        self._dms = is_dms
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0,0,0,0); lay.setSpacing(3)

        def _sp(mn, mx, val, w=52):
            sp = QSpinBox()
            sp.setRange(mn, mx); sp.setValue(val)
            sp.setFixedWidth(w); sp.setStyleSheet(_SPIN)
            sp.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
            return sp

        def _dsep():
            l = QPushButton("—"); l.setFixedSize(18,22)
            l.setStyleSheet(f"QPushButton{{background:{BG4};color:{SUBTEXT};"
                            f"border:1px solid {BORDER};border-radius:3px;font-size:9px;}}")
            l.setEnabled(False)
            return l

        def _plus(sp):
            b = QPushButton("+"); b.setFixedSize(18,22)
            b.setStyleSheet(f"QPushButton{{background:{BG4};color:{ACCENT2};"
                            f"border:1px solid {BORDER};border-radius:3px;font-size:10px;font-weight:700;}}"
                            f"QPushButton:hover{{background:{BG3};}}")
            b.clicked.connect(lambda: sp.setValue(sp.value()+1))
            return b

        def _minus(sp):
            b = QPushButton("−"); b.setFixedSize(18,22)
            b.setStyleSheet(f"QPushButton{{background:{BG4};color:{MUTED};"
                            f"border:1px solid {BORDER};border-radius:3px;font-size:10px;font-weight:700;}}"
                            f"QPushButton:hover{{background:{BG3};}}")
            b.clicked.connect(lambda: sp.setValue(sp.value()-1))
            return b

        if is_dms:
            self._d = _sp(-90, 90, 0, 52)
            lay.addWidget(self._d); lay.addWidget(_minus(self._d)); lay.addWidget(_plus(self._d))
            lay.addWidget(_dsep())
            self._m = _sp(0, 59, 0, 44)
            lay.addWidget(self._m); lay.addWidget(_minus(self._m)); lay.addWidget(_plus(self._m))
            lay.addWidget(_dsep())
        else:
            self._h = _sp(0, 23, 0, 44)
            lay.addWidget(self._h); lay.addWidget(_minus(self._h)); lay.addWidget(_plus(self._h))
            lay.addWidget(_dsep())
            self._m = _sp(0, 59, 0, 44)
            lay.addWidget(self._m); lay.addWidget(_minus(self._m)); lay.addWidget(_plus(self._m))
            lay.addWidget(_dsep())

        self._sec = QDoubleSpinBox()
        self._sec.setRange(0, 59.9999); self._sec.setDecimals(4)
        self._sec.setSingleStep(1.0); self._sec.setFixedWidth(78)
        self._sec.setStyleSheet(_SPIN)
        self._sec.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        lay.addWidget(self._sec)
        lay.addStretch()

    def value_deg(self) -> float:
        """Derece cinsinden değer döner."""
        if self._dms:
            d = self._d.value()
            m = self._m.value()
            s = self._sec.value()
            sign = -1 if d < 0 else 1
            return sign * (abs(d) + m/60.0 + s/3600.0)
        else:
            h = self._h.value()
            m = self._m.value()
            s = self._sec.value()
            return (h + m/60.0 + s/3600.0) * 15.0

    def set_from_deg(self, deg: float):
        """Derece değerinden widget'ı doldur."""
        if self._dms:
            sign = -1 if deg < 0 else 1
            deg  = abs(deg)
            d    = int(deg)
            rem  = (deg - d) * 60
            m    = int(rem)
            s    = (rem - m) * 60
            self._d.setValue(sign * d)
            self._m.setValue(m)
            self._sec.setValue(round(s, 4))
        else:
            deg  = (float(deg) % 360)
            tot  = deg / 15.0 * 3600
            h    = int(tot // 3600)
            m    = int((tot % 3600) // 60)
            s    = tot % 60
            self._h.setValue(h)
            self._m.setValue(m)
            self._sec.setValue(round(s, 4))


# ─────────────────────────────────────────────────────────────────────────────
#  Ana Dialog
# ─────────────────────────────────────────────────────────────────────────────
class PlateSolveDialog(QDialog):
    """
    Siril tarzı Image Plate Solver dialog.
    on_annotate_cb(annotated_image) → ana pencerede overlay göster
    """

    CATALOGS = [
        "G17  (Gaia DR3, ~1.7B yıldız)",
        "H17  (Tycho-2 + UCAC, ~100M)",
        "W08  (UCAC4, ~100M)",
        "D05  (Dwarfs 5, ~5M)",
        "T01  (Tycho-2)",
        "S01  (Stars, ~1M)",
    ]
    CATALOG_CODES = [
        "D80",  # ASTAP varsayılanı — Gaia EDR3, ~1.5B yıldız
        "G17",  # Gaia DR3, ~1.7B
        "H17",  # Tycho-2 + UCAC, ~100M
        "W08",  # UCAC4, ~113M
        "D05",  # Dwarfs 5, ~5M
        "T01",  # Tycho-2
        "S01",  # Stars, ~1M
        "V01",  # USNO-B, ~1B
    ]

    SOL_ORDERS = ["Linear", "Quadratic", "Cubic (SIP)"]

    def __init__(self, image, settings: dict, on_annotate_cb=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🔭  Image Plate Solver")
        self.setMinimumSize(620, 820)
        self.resize(640, 880)
        self.setStyleSheet(f"background:{BG};color:{TEXT};font-size:10px;")
        self._image       = image
        self._settings    = settings
        self._on_annotate = on_annotate_cb
        self._worker      = None
        self._last_result = None
        self._build()
        self._refresh_catalog_path()

    # ─────────────────────────────────────────────────────────────────────────
    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(6)

        # Scroll için sarıcı
        scroll_wrap = QWidget()
        sv = QVBoxLayout(scroll_wrap)
        sv.setContentsMargins(0, 0, 0, 0); sv.setSpacing(6)

        # ── 1. Image Parameters ───────────────────────────────────────────
        sv.addWidget(self._sec_image_params())

        # ── 2. Solver / Options ───────────────────────────────────────────
        sv.addWidget(self._sec_solver_options())

        # ── 3. Catalogue Parameters ───────────────────────────────────────
        sv.addWidget(self._sec_catalogue())

        # ── 4. Star Detection ─────────────────────────────────────────────
        sv.addWidget(self._sec_star_detection())

        # ── 5. Progress log ───────────────────────────────────────────────
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFixedHeight(110)
        self._log.setStyleSheet(
            f"QTextEdit{{background:#020810;color:{GREEN};"
            f"border:1px solid {BORDER};border-radius:4px;"
            f"font-family:'Consolas','Courier New',monospace;"
            f"font-size:10px;padding:4px;}}")
        sv.addWidget(self._log)

        # ── 6. Progress bar ───────────────────────────────────────────────
        self._pbar = QProgressBar()
        self._pbar.setRange(0, 0); self._pbar.setFixedHeight(5)
        self._pbar.setTextVisible(False)
        self._pbar.setStyleSheet(
            f"QProgressBar{{background:{BG2};border:none;border-radius:3px;}}"
            f"QProgressBar::chunk{{background:qlineargradient("
            f"x1:0,y1:0,x2:1,y2:0,stop:0 {BORDER2},stop:1 {ACCENT});"
            f"border-radius:3px;}}")
        self._pbar.setVisible(False)
        sv.addWidget(self._pbar)

        # ── 7. Sonuç kartı ────────────────────────────────────────────────
        self._result_grp = QGroupBox("Sonuç")
        self._result_grp.setStyleSheet(_GRP)
        self._result_grp.setVisible(False)
        rgl = QVBoxLayout(self._result_grp)
        self._lbl_result = QLabel()
        self._lbl_result.setStyleSheet(
            f"color:{GREEN};font-size:10px;font-family:monospace;")
        self._lbl_result.setWordWrap(True)
        self._lbl_result.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        rgl.addWidget(self._lbl_result)
        sv.addWidget(self._result_grp)

        sv.addStretch()
        root.addWidget(scroll_wrap, 1)

        # ── Bottom buttons ────────────────────────────────────────────────
        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"color:{BORDER};")
        root.addWidget(sep)

        btn_row = QHBoxLayout(); btn_row.setSpacing(8)

        b_metadata = QPushButton("Get Metadata From Image")
        b_metadata.setStyleSheet(_BTN); b_metadata.setFixedHeight(26)
        b_metadata.setToolTip("FITS header'dan fokal uzunluk ve piksel boyutu oku")
        b_metadata.clicked.connect(self._get_metadata)
        btn_row.addWidget(b_metadata)

        btn_row.addStretch()

        self._btn_overlay = QPushButton("🗺 Overlay")
        self._btn_overlay.setStyleSheet(_BTN); self._btn_overlay.setFixedHeight(28)
        self._btn_overlay.setEnabled(False)
        self._btn_overlay.clicked.connect(self._show_annotation)
        btn_row.addWidget(self._btn_overlay)

        b_close = QPushButton("Close")
        b_close.setStyleSheet(_BTN); b_close.setFixedHeight(30)
        b_close.clicked.connect(self.reject)
        btn_row.addWidget(b_close)

        self._btn_ok = QPushButton("OK")
        self._btn_ok.setStyleSheet(_RUN); self._btn_ok.setFixedHeight(32)
        self._btn_ok.setMinimumWidth(80)
        self._btn_ok.clicked.connect(self._start_solve)
        btn_row.addWidget(self._btn_ok)

        root.addLayout(btn_row)

    # ─────────────────────────────────────────────────────────────────────────
    #  Section builders
    # ─────────────────────────────────────────────────────────────────────────
    def _sec_image_params(self):
        """Image Parameters bölümü — RA/Dec + SIMBAD arama + Focal/Pixel."""
        grp = _CollapsibleGroup("Image Parameters", expanded=True)
        lay = grp.content_layout()
        lay.setSpacing(5)

        # ── SIMBAD arama satırı ───────────────────────────────────────────
        simbad_row = QHBoxLayout(); simbad_row.setSpacing(4)
        self._edit_object = QLineEdit()
        self._edit_object.setPlaceholderText("Object name…")
        self._edit_object.setStyleSheet(_EDIT)
        self._edit_object.returnPressed.connect(self._simbad_search)
        simbad_row.addWidget(self._edit_object, 1)

        b_find = QPushButton("🔍 Find")
        b_find.setStyleSheet(_BTN); b_find.setFixedHeight(24)
        b_find.clicked.connect(self._simbad_search)
        simbad_row.addWidget(b_find)

        srv_lbl = QLabel("Server:"); srv_lbl.setStyleSheet(_LBL)
        self._combo_server = QComboBox()
        self._combo_server.addItems(["SIMBAD", "NED", "VizieR"])
        self._combo_server.setStyleSheet(_COMBO); self._combo_server.setFixedWidth(90)
        simbad_row.addWidget(srv_lbl); simbad_row.addWidget(self._combo_server)
        lay.addLayout(simbad_row)

        # ── RA satırı ─────────────────────────────────────────────────────
        ra_row = QHBoxLayout(); ra_row.setSpacing(6)
        ra_lbl = QLabel("Right Ascension:"); ra_lbl.setStyleSheet(_LBL); ra_lbl.setFixedWidth(115)
        self._ra_widget = _HMSWidget(is_dms=False)
        ra_row.addWidget(ra_lbl); ra_row.addWidget(self._ra_widget); ra_row.addStretch()
        lay.addLayout(ra_row)

        # Önceki RA ile doldur
        last_ra = self._settings.get("last_platesolve_ra", 0.0)
        if last_ra:
            self._ra_widget.set_from_deg(float(last_ra))

        # ── Dec satırı ────────────────────────────────────────────────────
        dec_row = QHBoxLayout(); dec_row.setSpacing(6)
        dec_lbl = QLabel("Declination:"); dec_lbl.setStyleSheet(_LBL); dec_lbl.setFixedWidth(115)
        self._dec_widget = _HMSWidget(is_dms=True)
        dec_row.addWidget(dec_lbl); dec_row.addWidget(self._dec_widget); dec_row.addStretch()
        lay.addLayout(dec_row)

        last_dec = self._settings.get("last_platesolve_dec", 0.0)
        if last_dec:
            self._dec_widget.set_from_deg(float(last_dec))

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"color:{BORDER};margin:3px 0;")
        lay.addWidget(sep)

        # ── Aperture + Reducer → Focal length hesaplama ─────────────────
        fp_row = QGridLayout(); fp_row.setSpacing(6)

        fp_row.addWidget(self._lbl("Aperture (mm):"), 0, 0)
        self._sp_aperture = QDoubleSpinBox()
        self._sp_aperture.setRange(1, 50000); self._sp_aperture.setDecimals(1)
        self._sp_aperture.setValue(float(self._settings.get("aperture_mm", 200.0)))
        self._sp_aperture.setFixedWidth(90); self._sp_aperture.setStyleSheet(_SPIN)
        self._sp_aperture.valueChanged.connect(self._calc_focal_and_fov)
        fp_row.addWidget(self._sp_aperture, 0, 1)

        fp_row.addWidget(self._lbl("Focal ratio (f/):"), 0, 2)
        self._sp_fratio = QDoubleSpinBox()
        self._sp_fratio.setRange(0.5, 100); self._sp_fratio.setDecimals(1)
        self._sp_fratio.setValue(float(self._settings.get("focal_ratio", 5.0)))
        self._sp_fratio.setFixedWidth(90); self._sp_fratio.setStyleSheet(_SPIN)
        self._sp_fratio.valueChanged.connect(self._calc_focal_and_fov)
        fp_row.addWidget(self._sp_fratio, 0, 3)

        fp_row.addWidget(self._lbl("Reducer/Barlow:"), 1, 0)
        self._sp_reducer = QDoubleSpinBox()
        self._sp_reducer.setRange(0.1, 10.0); self._sp_reducer.setDecimals(2)
        self._sp_reducer.setValue(float(self._settings.get("reducer", 1.0)))
        self._sp_reducer.setFixedWidth(90); self._sp_reducer.setStyleSheet(_SPIN)
        self._sp_reducer.setToolTip("Reducer < 1 (orn: 0.73), Barlow > 1 (orn: 2.0), yok = 1.0")
        self._sp_reducer.valueChanged.connect(self._calc_focal_and_fov)
        fp_row.addWidget(self._sp_reducer, 1, 1)

        fp_row.addWidget(self._lbl("Focal length (mm):"), 1, 2)
        self._lbl_focal = QLabel("—")
        self._lbl_focal.setStyleSheet(f"color:{GOLD};font-size:10px;font-weight:bold;")
        fp_row.addWidget(self._lbl_focal, 1, 3)

        fp_row.addWidget(self._lbl("Pixel size (µm):"), 2, 0)
        self._sp_pixel = QDoubleSpinBox()
        self._sp_pixel.setRange(0.1, 100); self._sp_pixel.setDecimals(2)
        self._sp_pixel.setValue(float(self._settings.get("pixel_size_um", 4.63)))
        self._sp_pixel.setFixedWidth(90); self._sp_pixel.setStyleSheet(_SPIN)
        self._sp_pixel.valueChanged.connect(self._calc_focal_and_fov)
        fp_row.addWidget(self._sp_pixel, 2, 1)

        self._lbl_resolution = QLabel("Resolution: —")
        self._lbl_resolution.setStyleSheet(f"color:{ACCENT2};font-size:10px;")
        fp_row.addWidget(self._lbl_resolution, 2, 2, 1, 2)

        lay.addLayout(fp_row)
        self._calc_focal_and_fov()  # İlk hesaplama
        return grp

    def _sec_solver_options(self):
        """Solver / Options bölümü."""
        grp = _CollapsibleGroup("Solver Options", expanded=True)
        lay = grp.content_layout()
        lay.setSpacing(5)

        # Solver seçimi + Solution order
        r1 = QHBoxLayout(); r1.setSpacing(8)
        r1.addWidget(self._lbl("Solver:"))
        self._combo_solver = QComboBox()
        self._combo_solver.addItems(["ASTAP solver", "Siril solver (local)"])
        self._combo_solver.setStyleSheet(_COMBO); self._combo_solver.setFixedWidth(160)
        r1.addWidget(self._combo_solver)
        r1.addSpacing(20)
        r1.addWidget(self._lbl("Solution order:"))
        self._combo_sol_order = QComboBox()
        self._combo_sol_order.addItems(self.SOL_ORDERS)
        self._combo_sol_order.setCurrentText("Cubic (SIP)")
        self._combo_sol_order.setStyleSheet(_COMBO); self._combo_sol_order.setFixedWidth(120)
        r1.addWidget(self._combo_sol_order)
        r1.addStretch()
        lay.addLayout(r1)

        # Checkboxlar + Search radius
        r2 = QHBoxLayout(); r2.setSpacing(16)
        self._chk_flip       = QCheckBox("Flip image if needed")
        self._chk_downsample = QCheckBox("Downsample image")
        self._chk_autocrop   = QCheckBox("Auto-crop (for wide field)")
        self._chk_savedist   = QCheckBox("Save distortion")
        self._chk_flip.setChecked(True)
        self._chk_autocrop.setChecked(True)
        for chk in (self._chk_flip, self._chk_downsample,
                    self._chk_autocrop, self._chk_savedist):
            chk.setStyleSheet(_CHK)

        # 2x2 grid
        chk_grid = QGridLayout(); chk_grid.setSpacing(4)
        chk_grid.addWidget(self._chk_flip,       0, 0)
        chk_grid.addWidget(self._chk_downsample,  1, 0)
        chk_grid.addWidget(self._chk_autocrop,    0, 1)
        chk_grid.addWidget(self._chk_savedist,    1, 1)
        lay.addLayout(chk_grid)

        # Search radius + disable near search
        r3 = QHBoxLayout(); r3.setSpacing(8)
        r3.addWidget(self._lbl("Search radius:"))
        self._sp_radius = QDoubleSpinBox()
        self._sp_radius.setRange(0.1, 180); self._sp_radius.setDecimals(1)
        self._sp_radius.setValue(float(self._settings.get("astap_radius", 10.0)))
        self._sp_radius.setFixedWidth(70); self._sp_radius.setStyleSheet(_SPIN)
        # +/- butonları
        b_rm = QPushButton("—"); b_rm.setFixedSize(20,22)
        b_rm.setStyleSheet(_BTN)
        b_rm.clicked.connect(lambda: self._sp_radius.setValue(
            max(0.1, self._sp_radius.value() - 1)))
        b_rp = QPushButton("+"); b_rp.setFixedSize(20,22)
        b_rp.setStyleSheet(_BTN)
        b_rp.clicked.connect(lambda: self._sp_radius.setValue(
            min(180, self._sp_radius.value() + 1)))
        r3.addWidget(self._sp_radius)
        r3.addWidget(b_rm); r3.addWidget(b_rp)
        r3.addSpacing(20)
        self._chk_no_near = QCheckBox("disable near search")
        self._chk_no_near.setStyleSheet(_CHK)
        r3.addWidget(self._chk_no_near)
        r3.addStretch()
        lay.addLayout(r3)

        # ASTAP exe durumu göstergesi
        self._lbl_exe_status = QLabel()
        self._lbl_exe_status.setWordWrap(True)
        self._lbl_exe_status.setStyleSheet(
            f"color:{MUTED};font-size:9px;padding:3px 4px;")
        lay.addWidget(self._lbl_exe_status)
        self._update_exe_status()

        return grp

    def _sec_catalogue(self):
        """Catalogue Parameters bölümü."""
        grp = _CollapsibleGroup("Catalogue Parameters", expanded=True)
        lay = grp.content_layout()
        lay.setSpacing(5)

        # Star Catalogue
        r1 = QHBoxLayout(); r1.setSpacing(8)
        r1.addWidget(self._lbl("Star Catalogue:"))
        self._combo_catalog = QComboBox()
        self._combo_catalog.addItems(self.CATALOG_CODES)
        self._combo_catalog.setCurrentText(
            self._settings.get("astap_catalog", "G17"))
        self._combo_catalog.setStyleSheet(_COMBO)
        self._combo_catalog.setFixedWidth(80)
        self._combo_catalog.currentTextChanged.connect(self._refresh_catalog_path)
        r1.addWidget(self._combo_catalog)

        self._chk_cat_auto = QCheckBox("Auto")
        self._chk_cat_auto.setChecked(True)
        self._chk_cat_auto.setStyleSheet(_CHK)
        r1.addWidget(self._chk_cat_auto)

        self._lbl_cat_type = QLabel("(local catalogue)")
        self._lbl_cat_type.setStyleSheet(f"color:{SUBTEXT};font-size:9px;")
        r1.addWidget(self._lbl_cat_type)
        r1.addStretch()
        lay.addLayout(r1)

        # Katalog yolu göstergesi
        self._lbl_cat_path = QLabel()
        self._lbl_cat_path.setStyleSheet(
            f"color:{MUTED};font-size:9px;padding:2px 4px;")
        self._lbl_cat_path.setWordWrap(True)
        lay.addWidget(self._lbl_cat_path)

        # Catalogue Limit Magnitude
        r2 = QHBoxLayout(); r2.setSpacing(8)
        r2.addWidget(self._lbl("Catalogue Limit Mag:"))
        self._sp_lim_mag = QSpinBox()
        self._sp_lim_mag.setRange(8, 21)
        self._sp_lim_mag.setValue(int(self._settings.get("cat_limit_mag", 12)))
        self._sp_lim_mag.setFixedWidth(60); self._sp_lim_mag.setStyleSheet(_SPIN)
        b_mm = QPushButton("—"); b_mm.setFixedSize(20,22); b_mm.setStyleSheet(_BTN)
        b_mm.clicked.connect(lambda: self._sp_lim_mag.setValue(
            max(8, self._sp_lim_mag.value()-1)))
        b_mp = QPushButton("+"); b_mp.setFixedSize(20,22); b_mp.setStyleSheet(_BTN)
        b_mp.clicked.connect(lambda: self._sp_lim_mag.setValue(
            min(21, self._sp_lim_mag.value()+1)))
        self._chk_mag_auto = QCheckBox("Auto")
        self._chk_mag_auto.setChecked(True); self._chk_mag_auto.setStyleSheet(_CHK)
        r2.addWidget(self._sp_lim_mag)
        r2.addWidget(b_mm); r2.addWidget(b_mp)
        r2.addWidget(self._chk_mag_auto)
        r2.addStretch()
        lay.addLayout(r2)

        return grp

    def _sec_star_detection(self):
        """Star Detection bölümü."""
        grp = _CollapsibleGroup("Star Detection", expanded=False)
        lay = grp.content_layout()
        lay.setSpacing(5)

        def _row(label, widget, hint=""):
            r = QHBoxLayout(); r.setSpacing(6)
            l = QLabel(label); l.setStyleSheet(_LBL); l.setFixedWidth(160)
            r.addWidget(l); r.addWidget(widget)
            if hint:
                hl = QLabel(hint); hl.setStyleSheet(
                    f"color:{SUBTEXT};font-size:9px;")
                r.addWidget(hl)
            r.addStretch(); lay.addLayout(r)

        self._sp_min_stars = QSpinBox()
        self._sp_min_stars.setRange(5, 500)
        self._sp_min_stars.setValue(int(self._settings.get("astap_min_stars", 10)))
        self._sp_min_stars.setFixedWidth(70); self._sp_min_stars.setStyleSheet(_SPIN)
        _row("Min stars for solve:", self._sp_min_stars, "5–500")

        self._sp_max_stars = QSpinBox()
        self._sp_max_stars.setRange(10, 1000)
        self._sp_max_stars.setValue(int(self._settings.get("astap_max_stars", 500)))
        self._sp_max_stars.setFixedWidth(70); self._sp_max_stars.setStyleSheet(_SPIN)
        _row("Max stars for solve:", self._sp_max_stars)

        self._sp_timeout = QSpinBox()
        self._sp_timeout.setRange(10, 600); self._sp_timeout.setSingleStep(10)
        self._sp_timeout.setValue(int(self._settings.get("astap_timeout", 120)))
        self._sp_timeout.setFixedWidth(70); self._sp_timeout.setStyleSheet(_SPIN)
        _row("Timeout (s):", self._sp_timeout)

        self._sp_downsample = QSpinBox()
        self._sp_downsample.setRange(0, 4)
        self._sp_downsample.setValue(int(self._settings.get("astap_downsample", 0)))
        self._sp_downsample.setFixedWidth(60); self._sp_downsample.setStyleSheet(_SPIN)
        _row("Downsample factor:", self._sp_downsample, "0=auto")

        return grp

    # ─────────────────────────────────────────────────────────────────────────
    #  Katalog yolu — ASTAP exe'nin yanındaki klasörden otomatik bul
    # ─────────────────────────────────────────────────────────────────────────
    def _refresh_catalog_path(self, _=None):
        """ASTAP exe klasöründe katalog ara (.pkg, .dat, .bin dahil)."""
        exe    = self._settings.get("astap_exe", "").strip()
        cat_id = self._combo_catalog.currentText() if hasattr(self, "_combo_catalog") else "D80"

        if not exe or not os.path.isfile(exe):
            if hasattr(self, "_lbl_cat_path"):
                self._lbl_cat_path.setText("⚠ ASTAP yolu ayarlı değil — Settings → ASTAP")
                self._lbl_cat_path.setStyleSheet(
                    f"color:{GOLD};font-size:9px;padding:2px 4px;")
            return

        exe_dir    = os.path.dirname(exe)
        found_path = _find_catalog(exe_dir, cat_id)

        if not hasattr(self, "_lbl_cat_path"):
            return

        if found_path:
            # Bulunan katalog dosyalarını listele
            cat_lower = cat_id.lower()
            cat_files = []
            try:
                for f in sorted(os.listdir(found_path)):
                    fl = f.lower()
                    if fl.startswith(cat_lower) and any(
                            fl.endswith(e) for e in (".pkg", ".dat", ".bin", ".cat")):
                        size_mb = os.path.getsize(
                            os.path.join(found_path, f)) / 1024 / 1024
                        cat_files.append(f"{f}  ({size_mb:.0f} MB)")
            except Exception:
                pass
            extra = ("\n  " + "\n  ".join(cat_files[:4])) if cat_files else ""
            self._lbl_cat_path.setText(f"✅  {found_path}{extra}")
            self._lbl_cat_path.setStyleSheet(
                f"color:{GREEN};font-size:9px;padding:2px 4px;")
        else:
            # Mevcut katalogları tara ve göster
            available = _scan_available_catalogs(exe_dir)
            if available:
                avail_str = ", ".join(available.keys())
                hint = f"\n  💡 Mevcut kataloglar: {avail_str}"
            else:
                hint = (f"\n  Örn: d80_star_database.pkg dosyasını"
                        f"\n  {exe_dir} klasörüne koyun.")
            self._lbl_cat_path.setText(f"⚠ {cat_id} bulunamadı.{hint}")
            self._lbl_cat_path.setStyleSheet(
                f"color:{GOLD};font-size:9px;padding:2px 4px;")


    def _get_catalog_db_path(self) -> str:
        """Kullanılacak katalog klasörünü döner."""
        exe    = self._settings.get("astap_exe", "").strip()
        cat_id = self._combo_catalog.currentText()

        if exe and os.path.isfile(exe):
            exe_dir = os.path.dirname(exe)
            found   = _find_catalog(exe_dir, cat_id)
            if found:
                return found
            # Seçili katalog yoksa mevcut olanı kullan
            auto_cat = _auto_detect_catalog(exe_dir)
            if auto_cat:
                found2 = _find_catalog(exe_dir, auto_cat)
                if found2:
                    return found2

        # Son çare: settings'teki astap_db
        return self._settings.get("astap_db", "")

    def _update_exe_status(self):
        exe = self._settings.get("astap_exe", "").strip()
        if not hasattr(self, "_lbl_exe_status"):
            return
        if exe and os.path.isfile(exe):
            size_mb = os.path.getsize(exe) / 1024 / 1024
            self._lbl_exe_status.setText(
                f"✅  {os.path.basename(exe)}  ({size_mb:.0f} MB)  —  {exe}")
            self._lbl_exe_status.setStyleSheet(
                f"color:{GREEN};font-size:9px;padding:3px 4px;")

            # Otomatik katalog seçimi — exe klasöründe ne var?
            exe_dir = os.path.dirname(exe)
            auto_cat = _auto_detect_catalog(exe_dir)
            if hasattr(self, "_combo_catalog") and auto_cat:
                if auto_cat in self.CATALOG_CODES:
                    self._combo_catalog.setCurrentText(auto_cat)
                    self._settings["astap_catalog"] = auto_cat
        else:
            self._lbl_exe_status.setText(
                "⚠  ASTAP bulunamadı — Settings → ASTAP sekmesini açın")
            self._lbl_exe_status.setStyleSheet(
                f"color:{GOLD};font-size:9px;padding:3px 4px;")

    # ─────────────────────────────────────────────────────────────────────────
    #  FOV hesaplama
    # ─────────────────────────────────────────────────────────────────────────
    def _get_focal_mm(self):
        """Aperture x focal ratio x reducer → focal length (mm)."""
        aperture = self._sp_aperture.value()
        fratio   = self._sp_fratio.value()
        reducer  = self._sp_reducer.value()
        return aperture * fratio * reducer

    def _calc_focal_and_fov(self):
        """Aperture/ratio/reducer'dan focal length hesapla, sonra FOV."""
        try:
            focal_mm = self._get_focal_mm()
            self._lbl_focal.setText(f"{focal_mm:.1f} mm")
            pixel_um = self._sp_pixel.value()
            if focal_mm > 0 and pixel_um > 0:
                arcsec_per_px = (pixel_um / focal_mm) * 206.265
                self._lbl_resolution.setText(
                    f"Resolution: {arcsec_per_px:.4f}\"")
        except Exception:
            pass

    def _fov_deg(self) -> float:
        """Görüntünün tahmini FOV genişliğini derece döner."""
        try:
            if self._image is None:
                return 0.0
            w = self._image.shape[1]
            focal_mm  = self._get_focal_mm()
            pixel_um  = self._sp_pixel.value()
            if focal_mm > 0 and pixel_um > 0:
                arcsec_per_px = (pixel_um / focal_mm) * 206.265
                return (w * arcsec_per_px) / 3600.0
        except Exception:
            pass
        return 0.0

    # ─────────────────────────────────────────────────────────────────────────
    #  SIMBAD arama
    # ─────────────────────────────────────────────────────────────────────────
    def _simbad_search(self):
        name = self._edit_object.text().strip()
        if not name:
            return
        server = self._combo_server.currentText() if hasattr(self, "_combo_server") else "SIMBAD"
        self._log_line(f"🔍 {server}'da aranıyor: {name}…")

        ra_deg, dec_deg = None, None

        # ── SIMBAD ────────────────────────────────────────────────────────
        if server in ("SIMBAD", "VizieR"):
            try:
                from astroquery.simbad import Simbad
                import astropy.units as u
                from astropy.coordinates import SkyCoord

                simbad = Simbad()
                result = simbad.query_object(name)
                if result is None or len(result) == 0:
                    self._log_line(f"❌  '{name}' SIMBAD'da bulunamadı.")
                else:
                    # Kolon adları: astroquery versiyonuna göre "RA"/"DEC" veya "ra"/"dec"
                    cols_upper = {c.upper(): c for c in result.colnames}
                    ra_col  = cols_upper.get("RA")
                    dec_col = cols_upper.get("DEC")

                    if ra_col and dec_col:
                        ra_str  = str(result[ra_col][0]).strip()
                        dec_str = str(result[dec_col][0]).strip()
                        # "12 33 45.67"  "+08 50 06.2" formatı
                        try:
                            coord   = SkyCoord(ra_str, dec_str,
                                               unit=(u.hourangle, u.deg),
                                               frame="icrs")
                            ra_deg  = float(coord.ra.deg)
                            dec_deg = float(coord.dec.deg)
                        except Exception:
                            # Bazen direkt derece olarak gelir
                            try:
                                ra_deg  = float(ra_str)
                                dec_deg = float(dec_str)
                            except Exception as e2:
                                self._log_line(f"❌  Koordinat parse hatası: {e2}  (RA='{ra_str}' Dec='{dec_str}')")
                    else:
                        self._log_line(f"⚠  Beklenen kolonlar yok. Mevcut: {result.colnames}")
            except ImportError:
                self._log_line("⚠  astroquery yüklü değil. pip install astroquery")
            except Exception as e:
                self._log_line(f"⚠  SIMBAD: {type(e).__name__}: {e}")
                # Bağlantı hatası ise manuel URL ile dene
                if "connection" in str(e).lower() or "service" in str(e).lower():
                    ra_deg, dec_deg = self._simbad_manual(name)

        # ── NED ───────────────────────────────────────────────────────────
        elif server == "NED":
            try:
                try:
                    from astroquery.ipac.ned import Ned
                except ImportError:
                    from astroquery.ned import Ned  # eski versiyon
                import astropy.units as u
                result = Ned.query_object(name)
                if result is not None and len(result) > 0:
                    cols_upper = {c.upper(): c for c in result.colnames}
                    ra_col  = cols_upper.get("RA") or cols_upper.get("RA(DEG)")
                    dec_col = cols_upper.get("DEC") or cols_upper.get("DEC(DEG)")
                    if ra_col and dec_col:
                        ra_deg  = float(result[ra_col][0])
                        dec_deg = float(result[dec_col][0])
                    else:
                        self._log_line(f"⚠  NED kolon bulunamadı: {result.colnames}")
                else:
                    self._log_line(f"❌  '{name}' NED'de bulunamadı.")
            except ImportError:
                self._log_line("⚠  astroquery yüklü değil.")
            except Exception as e:
                self._log_line(f"⚠  NED: {type(e).__name__}: {e}")

        # ── Sonuç ─────────────────────────────────────────────────────────
        if ra_deg is not None and dec_deg is not None:
            self._ra_widget.set_from_deg(ra_deg)
            self._dec_widget.set_from_deg(dec_deg)
            self._log_line(
                f"✅  {name}  →  RA={ra_deg:.4f}°  Dec={dec_deg:+.4f}°")
        elif ra_deg is None:
            self._log_line(
                f"❌  '{name}' bulunamadı.\n"
                f"   RA/Dec'i manuel girebilirsiniz.")

    def _simbad_manual(self, name: str):
        """SIMBAD script arayüzü üzerinden basit HTTP sorgusu."""
        try:
            import urllib.request, urllib.parse
            encoded = urllib.parse.quote(name)
            url = (f"https://simbad.cds.unistra.fr/simbad/sim-id"
                   f"?output.format=ASCII&obj.coo1=on&Ident={encoded}")
            req = urllib.request.Request(url, headers={"User-Agent": "AstroMastroPro/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                text = resp.read().decode("utf-8", errors="replace")
            # RA/DEC satırlarını parse et
            # Örnek: "Coordinates(ICRS,ep=J2000,eq=2000): 12 33 45.68 +08 50 06.2"
            for line in text.splitlines():
                if "Coordinates" in line and (":" in line):
                    parts = line.split(":")[-1].strip().split()
                    if len(parts) >= 6:
                        import astropy.units as u
                        from astropy.coordinates import SkyCoord
                        ra_str  = " ".join(parts[:3])
                        dec_str = " ".join(parts[3:6])
                        coord   = SkyCoord(ra_str, dec_str,
                                           unit=(u.hourangle, u.deg))
                        self._log_line(f"✅  (manual) RA={coord.ra.deg:.4f}°  Dec={coord.dec.deg:+.4f}°")
                        return float(coord.ra.deg), float(coord.dec.deg)
        except Exception as e:
            self._log_line(f"⚠  Manuel SIMBAD: {e}")
        return None, None

    # ─────────────────────────────────────────────────────────────────────────
    #  FITS metadata oku
    # ─────────────────────────────────────────────────────────────────────────
    def _get_metadata(self):
        """FITS header'dan fokal uzunluk ve piksel boyutu oku."""
        last_file = self._settings.get("last_open_file", "")
        if not last_file or not os.path.isfile(last_file):
            path, _ = QFileDialog.getOpenFileName(
                self, "FITS Dosyası Seç", "",
                "FITS (*.fits *.fit *.fts);;Tümü (*)")
            if not path: return
            last_file = path

        try:
            from astropy.io import fits as _fits
            with _fits.open(last_file, memmap=False) as hdul:
                hdr = hdul[0].header
                focal = (hdr.get("FOCALLEN") or hdr.get("FOCAL") or
                         hdr.get("FLENGTH") or hdr.get("TELESCOP_FOCALLEN"))
                pixel = (hdr.get("XPIXSZ") or hdr.get("PIXSCALE") or
                         hdr.get("PIXSIZE1") or hdr.get("PIXELSIZE"))
                ra    = hdr.get("RA") or hdr.get("CRVAL1")
                dec   = hdr.get("DEC") or hdr.get("CRVAL2")

            found = []
            if focal:
                # Focal length'i aperture ve ratio'dan ters hesapla
                focal_val = float(focal)
                aperture  = self._sp_aperture.value()
                reducer   = self._sp_reducer.value()
                if aperture > 0 and reducer > 0:
                    new_ratio = focal_val / (aperture * reducer)
                    self._sp_fratio.setValue(round(new_ratio, 1))
                found.append(f"FL={focal_val:.1f}mm")
            if pixel:
                self._sp_pixel.setValue(float(pixel))
                found.append(f"px={float(pixel):.2f}µm")
            if ra is not None:
                self._ra_widget.set_from_deg(float(ra))
                found.append(f"RA={float(ra):.4f}°")
            if dec is not None:
                self._dec_widget.set_from_deg(float(dec))
                found.append(f"Dec={float(dec):+.4f}°")

            if found:
                self._log_line("📋 FITS header: " + "  ".join(found))
            else:
                self._log_line("⚠  FITS header'da uygun bilgi bulunamadı.")
        except Exception as e:
            self._log_line(f"❌  FITS okuma hatası: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    #  Solve başlat
    # ─────────────────────────────────────────────────────────────────────────
    def _start_solve(self):
        if self._image is None:
            QMessageBox.warning(self, "Uyarı", "Önce bir görüntü açın.")
            return
        if self._worker and self._worker.isRunning():
            return

        exe = self._settings.get("astap_exe", "").strip()
        if not exe or not os.path.isfile(exe):
            QMessageBox.warning(self, "ASTAP",
                "ASTAP yolu ayarlı değil.\n"
                "Settings → ASTAP sekmesinden yolu ayarlayın.")
            return

        # Parametreleri topla
        ra_deg  = self._ra_widget.value_deg()
        dec_deg = self._dec_widget.value_deg()
        fov_deg = self._fov_deg()

        # Sonucu settings'e kaydet (bir sonraki açılış için)
        self._settings["last_platesolve_ra"]  = ra_deg
        self._settings["last_platesolve_dec"] = dec_deg
        self._settings["aperture_mm"]          = self._sp_aperture.value()
        self._settings["focal_ratio"]          = self._sp_fratio.value()
        self._settings["reducer"]              = self._sp_reducer.value()
        self._settings["focal_length_mm"]      = self._get_focal_mm()
        self._settings["pixel_size_um"]        = self._sp_pixel.value()
        self._settings["astap_catalog"]        = self._combo_catalog.currentText()
        self._settings["astap_radius"]         = self._sp_radius.value()
        self._settings["astap_min_stars"]      = self._sp_min_stars.value()
        self._settings["astap_timeout"]        = self._sp_timeout.value()
        self._settings["astap_downsample"]     = self._sp_downsample.value()

        params = {
            "astap_exe":     exe,
            "db_path":       self._get_catalog_db_path(),
            "search_radius": self._sp_radius.value(),
            "downsample":    (2 if self._chk_downsample.isChecked()
                              else self._sp_downsample.value()),
            "min_stars":     self._sp_min_stars.value(),
            "timeout":       self._sp_timeout.value(),
            "ra_hint":       ra_deg  if ra_deg  != 0.0 else None,
            "dec_hint":      dec_deg if dec_deg != 0.0 else None,
            "fov_hint":      fov_deg if fov_deg > 0.01 else None,
        }

        self._log.clear()
        self._result_grp.setVisible(False)
        self._btn_overlay.setEnabled(False)
        self._btn_ok.setEnabled(False)
        self._btn_ok.setText("⏳ Çözüyor…")
        self._pbar.setVisible(True)

        self._log_line(
            f"🔭 Plate solve başlıyor…\n"
            f"   Katalog : {self._combo_catalog.currentText()}\n"
            f"   DB yolu : {params['db_path'] or '(ASTAP varsayılanı)'}\n"
            f"   Radius  : {params['search_radius']}°\n"
            f"   Min★    : {params['min_stars']}\n"
            f"   FOV tahmini: {fov_deg:.3f}°" if fov_deg > 0.01 else
            f"🔭 Plate solve başlıyor…")

        self._worker = _SolveWorker(self._image, params)
        self._worker.progress.connect(self._log_line)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    # ─────────────────────────────────────────────────────────────────────────
    #  Sonuç
    # ─────────────────────────────────────────────────────────────────────────
    def _on_done(self, result):
        self._pbar.setVisible(False)
        self._btn_ok.setEnabled(True)
        self._btn_ok.setText("OK")
        self._last_result = result

        from ai.astap_bridge import format_result
        text = format_result(result)

        if result.get("ra") is not None:
            self._lbl_result.setStyleSheet(
                f"color:{GREEN};font-size:10px;font-family:monospace;")
            self._btn_overlay.setEnabled(True)
            # RA/Dec kutularını sonuçla güncelle
            self._ra_widget.set_from_deg(result["ra"])
            self._dec_widget.set_from_deg(result["dec"])
            self._log_line(
                f"✅  Çözüldü!\n"
                f"   RA  = {result['ra']:.6f}°\n"
                f"   Dec = {result['dec']:+.6f}°\n"
                f"   Ölçek = {result.get('scale_arcsec','?'):.3f}\"/px"
                if isinstance(result.get('scale_arcsec'), float) else
                f"✅  Çözüldü!")
        else:
            self._lbl_result.setStyleSheet(
                f"color:{RED};font-size:10px;font-family:monospace;")
            header = result.get("fits_header", {})
            if header:
                self._log_line("─── INI içeriği ───")
                for k, v in list(header.items())[:15]:
                    self._log_line(f"  {k}={v}")
            stars = result.get("star_count", 0)
            if stars:
                self._log_line(f"  Bulunan yıldız: {stars}")
            if result.get("warning"):
                self._log_line(f"  Uyarı: {result['warning']}")
            self._log_line(
                f"❌  Çözüm bulunamadı (exit={result.get('astap_exit_code','?')})")

        self._lbl_result.setText(text)
        self._result_grp.setVisible(True)

        # Settings kaydet
        try:
            from gui.settings import save as _save
            _save(self._settings)
        except Exception:
            pass

    def _on_error(self, msg):
        self._pbar.setVisible(False)
        self._btn_ok.setEnabled(True)
        self._btn_ok.setText("OK")
        self._log_line(f"❌  HATA:\n{msg[:500]}")

    def _show_annotation(self):
        if not self._last_result or self._last_result.get("ra") is None:
            return
        try:
            from astrometry.wcs_annotator import annotate_image
            annotated = annotate_image(
                self._image, self._last_result,
                show_grid    = True,
                show_catalog = False,
            )
            if self._on_annotate:
                self._on_annotate(annotated)
            self._log_line("✅  Overlay ana ekranda gösteriliyor.")
            self.accept()
        except Exception as e:
            self._log_line(f"⚠  Overlay hatası: {e}")

    def _log_line(self, msg: str):
        self._log.append(str(msg))
        self._log.verticalScrollBar().setValue(
            self._log.verticalScrollBar().maximum())

    def _lbl(self, text):
        l = QLabel(text)
        l.setStyleSheet(_LBL)
        l.setFixedWidth(160)
        return l

    def closeEvent(self, event):
        if self._worker and self._worker.isRunning():
            self._worker.requestInterruption()
            self._worker.wait(1000)
        super().closeEvent(event)


# ─────────────────────────────────────────────────────────────────────────────
#  Collapsible Group widget  (Siril'deki ▼ / ▶ açılır kapanır grup)
# ─────────────────────────────────────────────────────────────────────────────
class _CollapsibleGroup(QWidget):
    def __init__(self, title: str, expanded: bool = True, parent=None):
        super().__init__(parent)
        self._expanded = expanded
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(0)

        # Header
        self._hdr = QWidget()
        self._hdr.setFixedHeight(26)
        self._hdr.setStyleSheet(
            f"background:{BG4};border-radius:4px 4px 0 0;"
            f"border:1px solid {BORDER};")
        self._hdr.setCursor(Qt.CursorShape.PointingHandCursor)
        hl = QHBoxLayout(self._hdr)
        hl.setContentsMargins(8, 0, 8, 0); hl.setSpacing(4)
        self._arrow = QLabel("∨" if expanded else "›")
        self._arrow.setStyleSheet(
            f"color:{SUBTEXT};font-size:11px;min-width:12px;")
        self._title_lbl = QLabel(title)
        self._title_lbl.setStyleSheet(
            f"color:{HEAD};font-size:10px;font-weight:700;")
        hl.addWidget(self._arrow)
        hl.addWidget(self._title_lbl, 1)
        outer.addWidget(self._hdr)

        # Body
        self._body = QWidget()
        self._body.setStyleSheet(
            f"background:{BG3};border:1px solid {BORDER};"
            f"border-top:none;border-radius:0 0 4px 4px;")
        self._body_lay = QVBoxLayout(self._body)
        self._body_lay.setContentsMargins(10, 8, 10, 8)
        self._body_lay.setSpacing(4)
        self._body.setVisible(expanded)
        outer.addWidget(self._body)

        self._hdr.mousePressEvent = lambda _: self._toggle()

    def _toggle(self):
        self._expanded = not self._expanded
        self._body.setVisible(self._expanded)
        self._arrow.setText("∨" if self._expanded else "›")
        self._hdr.setStyleSheet(
            f"background:{BG4 if self._expanded else BG3};"
            f"border-radius:{'4px 4px 0 0' if self._expanded else '4px'};"
            f"border:1px solid {BORDER};"
            + ("border-bottom:none;" if self._expanded else ""))

    def content_layout(self):
        return self._body_lay


# ─────────────────────────────────────────────────────────────────────────────
#  Katalog yolu bulucu — ASTAP exe'nin yanında katalog klasörlerini ara
# ─────────────────────────────────────────────────────────────────────────────
def _find_catalog(exe_dir: str, cat_id: str) -> str:
    """
    ASTAP katalog dosyasının yolunu döner.

    .pkg dosyası için TAM DOSYA YOLUNU döner (ASTAP -d ile kullanmak için).
    .dat klasörü için KLASÖR YOLUNU döner.

    Dönüş değeri:
      - "C:/.../d80_star_database.pkg"  →  .pkg tam yolu (astap -d bu yolu ister)
      - "C:/.../G17"                    →  .dat klasörü
      - ""                               →  bulunamadı
    """
    if not exe_dir or not os.path.isdir(exe_dir):
        return ""

    cat_lower  = cat_id.lower()
    EXTENSIONS = (".pkg", ".dat", ".bin", ".cat")

    def _find_pkg_full_path(directory: str, prefix: str) -> str:
        """Klasörde prefix.pkg dosyasının TAM YOLUNU döner."""
        if not os.path.isdir(directory):
            return ""
        try:
            for f in sorted(os.listdir(directory)):
                fl = f.lower()
                if fl.startswith(prefix) and fl.endswith(".pkg"):
                    return os.path.join(directory, f)  # TAM DOSYA YOLU
        except Exception:
            pass
        return ""

    def _has_dat_files(directory: str, prefix: str) -> bool:
        """Klasörde prefix ile başlayan .dat/.bin dosyası var mı?"""
        if not os.path.isdir(directory):
            return False
        try:
            for f in os.listdir(directory):
                fl = f.lower()
                if fl.startswith(prefix) and any(
                        fl.endswith(e) for e in (".dat", ".bin", ".cat")):
                    return True
        except Exception:
            pass
        return False

    search_dirs = [exe_dir, os.path.dirname(exe_dir)]

    for base in search_dirs:
        if not base or not os.path.isdir(base):
            continue

        # 1. Direkt klasörde .pkg dosyası var mı? → TAM DOSYA YOLU döndür
        pkg_path = _find_pkg_full_path(base, cat_lower)
        if pkg_path:
            return pkg_path  # "C:/Program Files\astap/d80_star_database.pkg"

        # 2. Alt klasörde .pkg var mı? (D80/d80_star_database.pkg)
        for sub_name in [cat_id, cat_id.lower(), cat_id.upper()]:
            sub = os.path.join(base, sub_name)
            if os.path.isdir(sub):
                pkg_path2 = _find_pkg_full_path(sub, cat_lower)
                if pkg_path2:
                    return pkg_path2  # TAM DOSYA YOLU

                # .dat formatı — klasör yolunu döndür
                if _has_dat_files(sub, cat_lower):
                    return sub

                # Klasör var, içinde dosya var (format bilinmiyor)
                try:
                    if os.listdir(sub):
                        return sub
                except Exception:
                    pass

        # 3. Direkt klasörde .dat dosyaları var mı?
        if _has_dat_files(base, cat_lower):
            return base

        # 4. Büyük/küçük harf farklı alt klasörler
        try:
            for entry in os.listdir(base):
                if entry.upper() == cat_id.upper():
                    full = os.path.join(base, entry)
                    if os.path.isdir(full):
                        pkg3 = _find_pkg_full_path(full, cat_lower)
                        if pkg3:
                            return pkg3
                        if _has_dat_files(full, cat_lower):
                            return full
        except Exception:
            pass

    return ""

def _scan_available_catalogs(exe_dir: str) -> dict:
    """
    ASTAP exe klasöründe bulunan tüm katalogları tarar.
    Dönüş: {"D80": "/path/to/D80", "G17": "", ...}
    """
    all_cats = ["D80", "G17", "H17", "W08", "D05", "T01", "S01", "V01"]
    found = {}
    for cat in all_cats:
        path = _find_catalog(exe_dir, cat)
        if path:
            found[cat] = path
    return found


def _auto_detect_catalog(exe_dir: str) -> str:
    """
    ASTAP exe klasöründe hangi katalog var, onu döner.
    Öncelik: D80 > G17 > H17 > W08 > D05
    .pkg formatı dahil tüm katalog türlerini tanır.
    """
    priority = ["D80", "G17", "H17", "W08", "D05", "T01", "S01", "V01"]
    for cat in priority:
        if _find_catalog(exe_dir, cat):
            return cat
    return ""
