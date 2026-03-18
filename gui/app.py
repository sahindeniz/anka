"""\nAstro Maestro Pro  —  PixInsight-style GUI  (v4.0)\n\nDüzeltmeler:\n  • Worker thread GC crash → _workers listesi ile referans tutulur\n  • Thread bitmeden yeni işlem → isRunning() koruması\n  • Exception → sys.excepthook + QMessageBox (program kapanmaz)\n  • matplotlib backend thread güvenliği → canvas.draw() yerine flush_events\n  • StackingDialog timer crash → QTimer doğru import\n  • lambda closure capture → default arg ile sabitlendi\n"""

import sys
import os
import re
import traceback
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QSlider, QCheckBox,
    QScrollArea, QStatusBar, QProgressBar, QSpinBox, QDoubleSpinBox,
    QComboBox, QTabWidget, QFrame, QMessageBox, QToolTip,
    QSplitter, QSizePolicy, QGroupBox, QListWidget, QListWidgetItem,
    QDialog, QTextEdit, QGridLayout, QLineEdit, QTabBar, QStyleFactory
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt6.QtGui import (QPalette, QColor, QFont, QCursor,
                          QLinearGradient, QBrush, QPainter,
                          QPixmap, QImage, QIcon)

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from gui.histogram_editor import HistogramEditorPanel
from gui.script_editor import ScriptEditorDialog
from matplotlib.widgets import RectangleSelector

# ═══════════════════════════════ THEME ══════════════════════════════════════
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
PURPLE  = "#9b6bd4"
TEXT    = "#ddeeff"
MUTED   = "#7aa0c0"
HEAD    = "#a8d4f0"
SUBTEXT = "#4a7a9a"

SLIDER_CSS = (
    f"QSlider::groove:horizontal{{height:3px;background:{BORDER};border-radius:2px;}}"
    f"QSlider::handle:horizontal{{width:12px;height:12px;margin:-5px 0;"
    f"background:{ACCENT};border-radius:6px;border:1px solid {ACCENT2};}}"
    f"QSlider::sub-page:horizontal{{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
    f"stop:0 {BORDER2},stop:1 {ACCENT});}}"
)
SPIN_CSS = (
    f"QDoubleSpinBox,QSpinBox{{background:{BG};color:{TEXT};"
    f"border:1px solid {BORDER};border-radius:3px;padding:1px 3px;font-size:10px;}}"
    f"QDoubleSpinBox:focus,QSpinBox:focus{{border:1px solid {ACCENT};}}"
)
COMBO_CSS = (
    f"QComboBox{{background:{BG};color:{TEXT};border:1px solid {BORDER};"
    f"border-radius:3px;padding:2px 5px;font-size:10px;}}"
    f"QComboBox QAbstractItemView{{background:{BG2};color:{TEXT};"
    f"selection-background-color:{BG4};border:1px solid {BORDER};}}"
    f"QComboBox::drop-down{{border:none;width:16px;}}"
)
CHECK_CSS = (
    f"QCheckBox{{color:{HEAD};font-size:11px;spacing:5px;}}"
    f"QCheckBox::indicator{{width:14px;height:14px;border-radius:2px;"
    f"border:1px solid {BORDER};background:{BG};}}"
    f"QCheckBox::indicator:checked{{background:{ACCENT};border:1px solid {ACCENT2};}}"
)
GROUP_CSS = (
    f"QGroupBox{{background:{BG3};border:1px solid {BORDER};"
    f"border-radius:6px;margin-top:14px;padding:6px;}}"
    f"QGroupBox::title{{color:{HEAD};font-size:11px;font-weight:600;"
    f"subcontrol-origin:margin;subcontrol-position:top left;"
    f"left:8px;padding:0 4px;}}"
)
SEP_CSS = f"color:{BORDER};"
LBL_CSS = f"color:{MUTED};font-size:10px;"


def _btn(color=BG4, hover=None, h=28, bold=False):
    hc = hover or BORDER2
    w  = "700" if bold else "500"
    return (f"QPushButton{{background:{color};color:{TEXT};border:1px solid {BORDER};"
            f"border-radius:4px;padding:3px 10px;font-size:11px;font-weight:{w};"
            f"min-height:{h}px;}}"
            f"QPushButton:hover{{background:{hc};border:1px solid {ACCENT};}}"
            f"QPushButton:pressed{{background:{BG};}}"
            f"QPushButton:disabled{{color:{SUBTEXT};background:{BG2};}}")


def _run_btn(color=GREEN):
    return (f"QPushButton{{background:{color};color:#fff;border:none;"
            f"border-radius:4px;padding:4px 14px;font-size:11px;font-weight:700;"
            f"min-height:26px;}}"
            f"QPushButton:hover{{background:#5ad48a;}}"
            f"QPushButton:pressed{{background:#2a9a50;}}"
            f"QPushButton:disabled{{background:{BG3};color:{SUBTEXT};}}")


def make_icon_btn(emoji, label, color=BG4, hover=None):
    hc = hover or BORDER2
    b  = QPushButton(f"{emoji}\n{label}")
    b.setStyleSheet(
        f"QPushButton{{background:{color};color:{TEXT};border:1px solid {BORDER};"
        f"border-radius:6px;font-size:10px;font-weight:500;"
        f"min-width:72px;min-height:52px;max-width:72px;max-height:52px;}}"
        f"QPushButton:hover{{background:{hc};border:1px solid {ACCENT2};color:{ACCENT2};}}"
        f"QPushButton:pressed{{background:{BG};}}"
        f"QPushButton:disabled{{color:{SUBTEXT};background:{BG2};}}")
    b.setFixedSize(72, 52)
    return b


# ═══════════════════════════════ GLOBAL EXCEPTION HOOK ══════════════════════
def _global_exception_handler(exc_type, exc_value, exc_tb):
    """Yakalanmayan tüm exceptionları dialog ile göster, programı kapatma."""
    msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    print(msg, file=sys.stderr)
    try:
        dlg = QMessageBox()
        dlg.setWindowTitle("Unexpected Error")
        dlg.setIcon(QMessageBox.Icon.Critical)
        dlg.setText("An unexpected error occurred. The program will continue.")
        dlg.setDetailedText(msg)
        dlg.exec()
    except Exception:
        pass  # GUI yoksa sessizce geç


# ═══════════════════════════════ WORKERS ════════════════════════════════════
class Worker(QThread):
    """\n    Process worker thread with progress reporting.\n    progress signal: (step:int, total:int, msg:str)\n    """
    finished = pyqtSignal(object)           # np.ndarray or dict
    progress = pyqtSignal(int, int, str)    # step, total, message
    preview  = pyqtSignal(object)           # ara preview np.ndarray
    error    = pyqtSignal(str)

    def __init__(self, fn, image, params):
        super().__init__()
        self.fn     = fn
        self.image  = image
        self.params = params
        self.setObjectName("ProcessWorker")
        self._step  = 0
        self._total = 1

    def _progress_cb(self, msg: str, step: int = -1, total: int = -1,
                     preview: object = None):
        """Called from processing functions to report progress."""
        if step >= 0:
            self._step = step
        else:
            self._step = min(self._step + 1, self._total)
        if total > 0:
            self._total = total
        if not self.isInterruptionRequested():
            self.progress.emit(self._step, self._total, str(msg))
            if preview is not None:
                self.preview.emit(preview)

    def run(self):
        try:
            # Inject progress callback into params if function supports it
            params_with_cb = dict(self.params)
            params_with_cb["_progress_cb"] = self._progress_cb
            try:
                result = self.fn(self.image, **params_with_cb)
            except TypeError:
                # Function doesn't accept _progress_cb — call without it
                result = self.fn(self.image, **self.params)
            if not self.isInterruptionRequested():
                self.finished.emit(result)
        except Exception:
            if not self.isInterruptionRequested():
                self.error.emit(traceback.format_exc())


class StackWorker(QThread):
    """
    2-aşamalı stacking worker:
      Faz 1 — Hizalama  (align_only=True)  → aligned kareler diske/memory
      Faz 2 — Stacking  (aligned kareler)  → final sonuç

    progress sinyali: str mesaj
    phase_done sinyali: (phase:int, data)
      phase=1 → data = aligned frame listesi (np.ndarray list)
      phase=2 → data = stacking result dict
    """
    finished   = pyqtSignal(object)   # dict — final stacking result
    progress   = pyqtSignal(str)
    phase_done = pyqtSignal(int, object)   # 1=align done, 2=stack done
    error      = pyqtSignal(str)

    def __init__(self, params, mode="full"):
        super().__init__()
        self.params = params
        self.mode   = mode    # "align" | "stack" | "full"
        self.setObjectName("StackWorker")

    def run(self):
        try:
            from processing.stacking import (stack_lights, align_frames_only,
                                              stack_aligned)
            def cb(step, msg):
                if not self.isInterruptionRequested():
                    self.progress.emit(f"[{step}] {msg}")

            if self.mode == "align":
                # Sadece hizalama — aligned kareleri döndür
                aligned = align_frames_only(**self.params, progress_cb=cb)
                if not self.isInterruptionRequested():
                    self.phase_done.emit(1, aligned)
                    self.finished.emit({"aligned": aligned, "phase": "align"})

            elif self.mode == "stack":
                # Sadece stacking — önceden hizalanmış kareler geldi
                result = stack_aligned(**self.params, progress_cb=cb)
                if not self.isInterruptionRequested():
                    self.finished.emit(result)

            else:
                # Tam pipeline — hizalama + stacking
                result = stack_lights(**self.params, progress_cb=cb)
                if not self.isInterruptionRequested():
                    self.finished.emit(result)

        except Exception:
            if not self.isInterruptionRequested():
                self.error.emit(traceback.format_exc())


# ═══════════════════════════════ PARAM WIDGETS ══════════════════════════════
class PS(QWidget):
    """ParamSlider"""
    def __init__(self, lbl, mn, mx, val, dec=0, tip="", parent=None):
        super().__init__(parent)
        self._sc = 10 ** dec
        r = QHBoxLayout(self); r.setContentsMargins(0,1,0,1); r.setSpacing(4)
        l = QLabel(lbl); l.setFixedWidth(140); l.setStyleSheet(LBL_CSS)
        if tip:
            l.setToolTip(tip)
            l.setCursor(QCursor(Qt.CursorShape.WhatsThisCursor))
        r.addWidget(l)
        self.sl = QSlider(Qt.Orientation.Horizontal)
        self.sl.setRange(int(mn*self._sc), int(mx*self._sc))
        self.sl.setValue(int(val*self._sc))
        self.sl.setStyleSheet(SLIDER_CSS)
        r.addWidget(self.sl, 1)
        if dec:
            self.sp = QDoubleSpinBox()
            self.sp.setDecimals(dec)
            self.sp.setSingleStep(1/self._sc)
        else:
            self.sp = QSpinBox()
        self.sp.setRange(mn, mx); self.sp.setValue(val)
        self.sp.setFixedWidth(62); self.sp.setStyleSheet(SPIN_CSS)
        r.addWidget(self.sp)
        # Döngüsel sinyali kır: blockSignals ile
        self.sl.valueChanged.connect(self._sl_to_sp)
        self.sp.valueChanged.connect(self._sp_to_sl)

    def _sl_to_sp(self, v):
        self.sp.blockSignals(True)
        val = v / self._sc
        # QSpinBox requires int, QDoubleSpinBox requires float
        self.sp.setValue(int(round(val)) if self._sc == 1 else float(val))
        self.sp.blockSignals(False)

    def _sp_to_sl(self, v):
        self.sl.blockSignals(True)
        self.sl.setValue(int(v * self._sc))
        self.sl.blockSignals(False)

    def v(self):   return float(self.sp.value())
    def set(self, val): self.sp.setValue(val)


class PC(QWidget):
    """ParamCombo"""
    def __init__(self, lbl, items, default="", tip="", parent=None):
        super().__init__(parent)
        r = QHBoxLayout(self); r.setContentsMargins(0,1,0,1); r.setSpacing(4)
        l = QLabel(lbl); l.setFixedWidth(140); l.setStyleSheet(LBL_CSS)
        if tip: l.setToolTip(tip)
        r.addWidget(l)
        self.cb = QComboBox(); self.cb.addItems(items)
        self.cb.setStyleSheet(COMBO_CSS)
        if default in items: self.cb.setCurrentText(default)
        r.addWidget(self.cb, 1)

    def v(self):      return self.cb.currentText()
    def set(self, v): self.cb.setCurrentText(v)


# ═══════════════════════════════ PROCESS PANEL ══════════════════════════════
class ProcessPanel(QWidget):
    """\n    Collapsible process panel.\n    Header: icon + title (click to expand/collapse) + ▶ Run button\n    Body: parameters (hidden when collapsed)\n    """
    run_requested = pyqtSignal(dict)

    def __init__(self, icon, title, key="", parent=None):
        super().__init__(parent)
        self._params   = {}
        self._key      = key
        self._icon     = icon
        self._title    = title
        self._expanded = True
        self.setStyleSheet(f"background:{BG3};border:1px solid {BORDER};border-radius:6px;")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0,0,0,0); outer.setSpacing(0)

        # ── Header ────────────────────────────────────────────────────────
        self._hdr = QWidget()
        self._hdr.setFixedHeight(34)
        self._hdr.setStyleSheet(
            f"background:{BG4};border-radius:6px 6px 0 0;"
            f"border-bottom:1px solid {BORDER};")
        self._hdr.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        hlay = QHBoxLayout(self._hdr)
        hlay.setContentsMargins(8,0,6,0); hlay.setSpacing(4)

        self._arrow = QLabel("▼")
        self._arrow.setStyleSheet(f"color:{SUBTEXT};font-size:9px;min-width:12px;")
        self._title_lbl = QLabel(f"{icon}  {title}")
        self._title_lbl.setStyleSheet(
            f"color:{HEAD};font-size:11px;font-weight:600;")

        self.btn_run = QPushButton("▶")
        self.btn_run.setFixedSize(28, 22)
        self.btn_run.setStyleSheet(
            f"QPushButton{{background:{GREEN};color:#fff;border:none;border-radius:4px;"
            f"font-size:11px;font-weight:700;}}"
            f"QPushButton:hover{{background:#5ad48a;}}"
            f"QPushButton:pressed{{background:#2a9a50;}}"
            f"QPushButton:disabled{{background:{BG};color:{SUBTEXT};}}")
        self.btn_run.setToolTip("Run this process")
        self.btn_run.clicked.connect(self._emit)

        hlay.addWidget(self._arrow)
        hlay.addWidget(self._title_lbl, 1)
        hlay.addWidget(self.btn_run)
        outer.addWidget(self._hdr)

        # ── Body ──────────────────────────────────────────────────────────
        self._body = QWidget()
        self._body.setStyleSheet("background:transparent;")
        self._bl = QVBoxLayout(self._body)
        self._bl.setContentsMargins(8,6,8,6); self._bl.setSpacing(3)
        outer.addWidget(self._body)

        # ── Progress bar ──────────────────────────────────────────────────
        prog_row = QHBoxLayout(); prog_row.setContentsMargins(8,0,8,4); prog_row.setSpacing(4)
        self._prog_bar = QProgressBar()
        self._prog_bar.setFixedHeight(5)
        self._prog_bar.setTextVisible(False)
        self._prog_bar.setStyleSheet(
            f"QProgressBar{{background:{BG};border:none;border-radius:2px;}}"
            f"QProgressBar::chunk{{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            f"stop:0 {BORDER2},stop:1 {ACCENT});border-radius:2px;}}")
        self._prog_lbl = QLabel("")
        self._prog_lbl.setStyleSheet(f"color:{MUTED};font-size:9px;")
        prog_row.addWidget(self._prog_bar, 1); prog_row.addWidget(self._prog_lbl)
        outer.addLayout(prog_row)
        self._prog_bar.setVisible(False)

        # Click to collapse
        self._hdr.mousePressEvent = lambda e: self.toggle()

    def toggle(self):
        self._expanded = not self._expanded
        self._body.setVisible(self._expanded)
        self._arrow.setText("▼" if self._expanded else "▶")
        self._hdr.setStyleSheet(
            f"background:{BG4 if self._expanded else BG3};"
            f"border-radius:{'6px 6px 0 0' if self._expanded else '6px'};"
            f"border-bottom:{'1px solid '+BORDER if self._expanded else 'none'};")

    def expand(self):
        if not self._expanded: self.toggle()

    def collapse(self):
        if self._expanded: self.toggle()

    def add(self, key, widget, getter=None):
        self._bl.addWidget(widget)
        self._params[key] = (widget, getter or (lambda w: w.v()))
        return widget

    def add_sep(self):
        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(SEP_CSS); self._bl.addWidget(sep)

    def add_combo(self, key, lbl, items, default="", tip=""):
        return self.add(key, PC(lbl, items, default, tip))

    def add_slider(self, key, lbl, mn, mx, val, dec=0, tip=""):
        return self.add(key, PS(lbl, mn, mx, val, dec, tip))

    def add_check(self, key, lbl, default=True):
        w = QCheckBox(lbl); w.setChecked(default); w.setStyleSheet(CHECK_CSS)
        return self.add(key, w, lambda ww: ww.isChecked())

    def collect(self):
        return {k: g(w) for k,(w,g) in self._params.items()}

    def _emit(self):
        self.run_requested.emit(self.collect())

    def set_running(self, v, msg=""):
        self.btn_run.setEnabled(not v)
        if v:
            self.btn_run.setText("⏳ …")
        else:
            self.btn_run.setText("▶  Run")
        if hasattr(self, "_prog_bar"):
            self._prog_bar.setVisible(v)
            if not v:
                self._prog_bar.setValue(0)
                self._prog_lbl.setText("")

    def set_progress(self, step: int, total: int, msg: str):
        if not hasattr(self, "_prog_bar"): return
        self._prog_bar.setVisible(True)
        if total > 0:
            self._prog_bar.setRange(0, total)
            self._prog_bar.setValue(step)
        else:
            self._prog_bar.setRange(0, 0)
        # truncate long messages
        short = msg if len(msg) <= 48 else "…" + msg[-46:]
        self._prog_lbl.setText(short)


class _StarNetWorker(QThread):
    """\n    Background thread that runs StarNet++ (real exe or AI fallback)\n    and saves starless + stars_only TIFF files.\n    Emits finished(dict) with keys: starless, stars_only, saved, error\n    """
    finished = pyqtSignal(object)       # dict
    progress = pyqtSignal(int, int, str)  # step, total, msg

    def __init__(self, image, exe, stride, use_gpu,
                 starless_path, stars_only_path):
        super().__init__()
        self.image           = image
        self.exe             = exe
        self.stride          = stride
        self.use_gpu         = use_gpu
        self.starless_path   = starless_path
        self.stars_only_path = stars_only_path
        self.setObjectName("StarNetWorker")

    def _cb(self, msg):
        """Parse [N/T] prefix and emit progress signal."""
        if self.isInterruptionRequested():
            return
        m = re.match(r"\[(\d+)/(\d+)\]", msg)
        if m:
            self.progress.emit(int(m.group(1)), int(m.group(2)), msg)
        else:
            self.progress.emit(0, 0, msg)

    def run(self):
        import re
        try:
            # ── Run StarNet++ or AI fallback ──────────────────────────────
            if self.exe and os.path.isfile(self.exe):
                self.progress.emit(1, 5, "[1/5] Starting StarNet++ executable…")
                from ai.starnet_bridge import run_starnet
                result = run_starnet(
                    self.image,
                    exe_path    = self.exe,
                    stride      = self.stride,
                    use_gpu     = self.use_gpu,
                    progress_cb = self._cb,
                )
            else:
                self.progress.emit(1, 4, "[1/4] AI fallback — detecting stars…")
                from ai.star_net import separate_stars
                r = separate_stars(self.image)
                result = {
                    "starless":   r["starless"],
                    "stars_only": r["stars_only"],
                    "star_mask":  r.get("star_mask"),
                    "exe_used":   "AI fallback",
                }

            if self.isInterruptionRequested():
                return

            starless   = result["starless"]
            stars_only = result["stars_only"]
            saved      = []

            # ── Save files ────────────────────────────────────────────────
            step = result.get("_step", 4)
            self.progress.emit(step, 5, "[4/5] Saving starless image…")
            _save_tif(starless,   self.starless_path)
            saved.append(self.starless_path)

            self.progress.emit(step+1, 5, "[5/5] Saving stars-only image…")
            _save_tif(stars_only, self.stars_only_path)
            saved.append(self.stars_only_path)

            self.progress.emit(5, 5, "Done ✅")
            self.finished.emit({
                "starless":   starless,
                "stars_only": stars_only,
                "saved":      saved,
                "error":      None,
            })

        except Exception:
            err = traceback.format_exc()
            self.finished.emit({
                "starless":   None,
                "stars_only": None,
                "saved":      [],
                "error":      err,
            })


def _save_tif(img: np.ndarray, path: str):
    """Save float32 [0,1] image as 16-bit TIFF."""
    import cv2
    img16 = (np.clip(img, 0, 1) * 65535).astype(np.uint16)
    if img16.ndim == 3:
        img16 = cv2.cvtColor(img16, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(path, img16)
    if not ok:
        raise RuntimeError(f"Failed to save: {path}")



# ═══════════════════════════════ SETTINGS DIALOG ════════════════════════════
class SettingsDialog(QDialog):
    """\n    Application Settings — StarNet++, Display, Processing defaults.\n    """
    def __init__(self, current_settings: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("⚙  Settings — Astro Maestro Pro")
        self.setMinimumSize(620, 560)
        self.setStyleSheet(f"background:{BG};color:{TEXT};font-size:11px;")
        self._s = dict(current_settings)
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 12)
        root.setSpacing(0)

        # Title bar
        title = QLabel("  ⚙  Settings")
        title.setStyleSheet(
            f"background:{BG4};color:{ACCENT2};font-size:14px;font-weight:700;"
            f"letter-spacing:1px;padding:10px 0;border-bottom:1px solid {BORDER};")
        root.addWidget(title)

        # Tab widget
        tabs = QTabWidget()
        tabs.setStyleSheet(
            f"QTabWidget::pane{{border:none;background:{BG};}}"
            f"QTabBar::tab{{background:{BG2};color:{MUTED};padding:8px 18px;"
            f"border-bottom:2px solid transparent;font-size:11px;}}"
            f"QTabBar::tab:selected{{color:{ACCENT2};border-bottom:2px solid {ACCENT2};}}"
            f"QTabBar::tab:hover{{color:{TEXT};}}")
        root.addWidget(tabs, 1)

        tabs.addTab(self._tab_starnet(),  "✦  StarNet++")
        tabs.addTab(self._tab_graxpert(), "🌌  GraXpert")
        tabs.addTab(self._tab_astap(),    "🔭  ASTAP")
        tabs.addTab(self._tab_display(),  "🖥  Display")
        tabs.addTab(self._tab_processing(),"⚙  Processing")
        tabs.addTab(self._tab_update(),   "🔄  Güncelleme")
        tabs.addTab(self._tab_paths(),    "📁  Paths")

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(12, 8, 12, 0)
        btn_row.addStretch()
        b_cancel = QPushButton("Cancel")
        b_cancel.setStyleSheet(_btn()); b_cancel.setFixedHeight(32)
        b_save = QPushButton("💾  Save Settings")
        b_save.setStyleSheet(_run_btn(ACCENT)); b_save.setFixedHeight(34)
        btn_row.addWidget(b_cancel); btn_row.addWidget(b_save)
        root.addLayout(btn_row)

        b_cancel.clicked.connect(self.reject)
        b_save.clicked.connect(self.accept)

    # ── StarNet++ Tab ─────────────────────────────────────────────────────
    def _tab_starnet(self):
        w = QWidget(); w.setStyleSheet(f"background:{BG};")
        lay = QVBoxLayout(w); lay.setContentsMargins(16,16,16,16); lay.setSpacing(12)

        # Info box
        info = QLabel(
            "StarNet++ removes stars from astrophotos using deep learning.\n\n"
            "Supported versions:\n"
            "  \u2022 StarNet++ v1  (starnet++.exe)\n"
            "  \u2022 StarNet2      (StarNet2.exe)\n\n"
            "Download: www.starnetastro.com\n\n"
            "After downloading, click Browse and select the .exe file.\n"
            "If no exe is set, an AI fallback method is used automatically.")
        info.setStyleSheet(
            f"background:{BG3};color:{MUTED};font-size:10px;"
            f"border:1px solid {BORDER};border-radius:6px;padding:10px;")
        info.setWordWrap(True)
        lay.addWidget(info)

        # Exe path row
        grp = QGroupBox("StarNet++ Executable"); grp.setStyleSheet(GROUP_CSS)
        glay = QVBoxLayout(grp); glay.setSpacing(6)

        path_row = QHBoxLayout()
        self.edit_starnet_exe = QLineEdit(self._s.get("starnet_exe",""))
        self.edit_starnet_exe.setPlaceholderText("e.g.  C:\\StarNet2\\StarNet2.exe")
        self.edit_starnet_exe.setStyleSheet(
            f"QLineEdit{{background:{BG};color:{TEXT};border:1px solid {BORDER};"
            f"border-radius:3px;padding:4px 8px;font-size:10px;}}"
            f"QLineEdit:focus{{border:1px solid {ACCENT};}}")
        b_browse = QPushButton("📂 Browse")
        b_browse.setStyleSheet(_btn()); b_browse.setFixedHeight(28)
        b_auto = QPushButton("🔍 Auto-Detect")
        b_auto.setStyleSheet(_btn(color=f"#0a1a2a",hover=ACCENT)); b_auto.setFixedHeight(28)
        path_row.addWidget(self.edit_starnet_exe, 1)
        path_row.addWidget(b_browse); path_row.addWidget(b_auto)
        glay.addLayout(path_row)

        b_browse.clicked.connect(self._browse_starnet)
        b_auto.clicked.connect(self._auto_detect_starnet)

        # Verify button
        b_verify = QPushButton("✅  Verify Executable")
        b_verify.setStyleSheet(_btn(color=f"#0a2a0a",hover=GREEN)); b_verify.setFixedHeight(26)
        b_verify.clicked.connect(self._verify_starnet)
        glay.addWidget(b_verify)

        lay.addWidget(grp)

        # Options
        opt_grp = QGroupBox("StarNet++ Options"); opt_grp.setStyleSheet(GROUP_CSS)
        olay = QVBoxLayout(opt_grp); olay.setSpacing(6)

        stride_row = QHBoxLayout()
        stride_lbl = QLabel("Tile Stride:"); stride_lbl.setStyleSheet(LBL_CSS)
        stride_lbl.setFixedWidth(100)
        self.combo_stride = QComboBox()
        self.combo_stride.addItems(["64","128","256"])
        self.combo_stride.setCurrentText(str(self._s.get("starnet_stride",256)))
        self.combo_stride.setStyleSheet(COMBO_CSS); self.combo_stride.setFixedWidth(80)
        stride_hint = QLabel("smaller = better quality, slower")
        stride_hint.setStyleSheet(f"color:{SUBTEXT};font-size:9px;")
        stride_row.addWidget(stride_lbl); stride_row.addWidget(self.combo_stride)
        stride_row.addWidget(stride_hint); stride_row.addStretch()
        olay.addLayout(stride_row)

        self.chk_gpu = QCheckBox("Use GPU  (StarNet2 only — requires CUDA)")
        self.chk_gpu.setChecked(bool(self._s.get("starnet_use_gpu", False)))
        self.chk_gpu.setStyleSheet(CHECK_CSS)
        olay.addWidget(self.chk_gpu)

        lay.addWidget(opt_grp)
        lay.addStretch()
        return w

    def _browse_starnet(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select StarNet++ Executable", "",
            "Executable (*.exe);;All Files (*)" if sys.platform=="win32"
            else "All Files (*)")
        if path: self.edit_starnet_exe.setText(path)

    def _auto_detect_starnet(self):
        from ai.starnet_bridge import find_starnet_exe
        found = find_starnet_exe()
        if found:
            self.edit_starnet_exe.setText(found)
            QMessageBox.information(self,"Auto-Detect",f"Found:\n{found}")
        else:
            QMessageBox.information(self,"Auto-Detect","StarNet++ not found in common locations.\nPlease use Browse to locate the executable manually.")

    def _verify_starnet(self):
        exe = self.edit_starnet_exe.text().strip()
        if not exe:
            QMessageBox.warning(self,"Verify","No path entered."); return
        if not os.path.isfile(exe):
            QMessageBox.critical(self,"Verify",f"File not found:\n{exe}"); return
        size_mb = os.path.getsize(exe) / 1024 / 1024
        name = os.path.basename(exe).lower()
        if any(x in name for x in ["starnet","star_net","starnet2"]):
            QMessageBox.information(self,"Verify",
                f"\u2705 Executable found\n\nName: {os.path.basename(exe)}\nSize: {size_mb:.1f} MB\nPath: {exe}\n\nLooks like a valid StarNet++ executable.")
        else:
            QMessageBox.warning(self,"Verify",
                f"File found but name does not look like StarNet++:\n{exe}\n\nMake sure you selected the correct executable.")

    # ── GraXpert Tab ──────────────────────────────────────────────────────
    def _tab_graxpert(self):
        w = QWidget(); w.setStyleSheet(f"background:{BG};")
        lay = QVBoxLayout(w); lay.setContentsMargins(16,16,16,16); lay.setSpacing(12)

        # Info
        info = QLabel(
            "GraXpert removes background gradients using deep learning.\n\n"
            "Supported versions:\n"
            "  \u2022 GraXpert 2.x  (GraXpert-cli.exe)\n"
            "  \u2022 GraXpert 3.x  (GraXpert.exe --cli)\n\n"
            "Download: https://graxpert.com\n\n"
            "After downloading, click Browse to select the executable.\n"
            "GraXpert will be used automatically when 'graxpert' is\n"
            "selected as the BG Extract method.")
        info.setStyleSheet(
            f"background:{BG3};color:{MUTED};font-size:10px;"
            f"border:1px solid {BORDER};border-radius:6px;padding:10px;")
        info.setWordWrap(True)
        lay.addWidget(info)

        # Exe path
        grp = QGroupBox("GraXpert Executable"); grp.setStyleSheet(GROUP_CSS)
        glay = QVBoxLayout(grp); glay.setSpacing(6)

        path_row = QHBoxLayout()
        self.edit_graxpert_exe = QLineEdit(self._s.get("graxpert_exe",""))
        self.edit_graxpert_exe.setPlaceholderText(
            "e.g.  C:\\GraXpert\\GraXpert-cli.exe")
        self.edit_graxpert_exe.setStyleSheet(
            f"QLineEdit{{background:{BG};color:{TEXT};border:1px solid {BORDER};"
            f"border-radius:3px;padding:4px 8px;font-size:10px;}}"
            f"QLineEdit:focus{{border:1px solid {ACCENT};}}")
        b_browse = QPushButton("📂 Browse"); b_browse.setStyleSheet(_btn()); b_browse.setFixedHeight(28)
        b_auto   = QPushButton("🔍 Auto-Detect"); b_auto.setStyleSheet(_btn(color="0a1a2a",hover=ACCENT)); b_auto.setFixedHeight(28)
        path_row.addWidget(self.edit_graxpert_exe, 1)
        path_row.addWidget(b_browse); path_row.addWidget(b_auto)
        glay.addLayout(path_row)

        b_verify = QPushButton("✅  Verify Executable")
        b_verify.setStyleSheet(_btn(color="0a2a0a",hover=GREEN)); b_verify.setFixedHeight(26)
        glay.addWidget(b_verify)
        lay.addWidget(grp)

        b_browse.clicked.connect(self._browse_graxpert)
        b_auto.clicked.connect(self._auto_detect_graxpert)
        b_verify.clicked.connect(self._verify_graxpert)

        # Options
        opt_grp = QGroupBox("GraXpert Options"); opt_grp.setStyleSheet(GROUP_CSS)
        olay = QVBoxLayout(opt_grp); olay.setSpacing(6)

        # Correction type
        corr_row = QHBoxLayout()
        corr_lbl = QLabel("Correction:"); corr_lbl.setStyleSheet(LBL_CSS); corr_lbl.setFixedWidth(90)
        self.combo_graxpert_corr = QComboBox()
        self.combo_graxpert_corr.addItems(["Subtraction","Division"])
        self.combo_graxpert_corr.setCurrentText(self._s.get("graxpert_correction","Subtraction"))
        self.combo_graxpert_corr.setStyleSheet(COMBO_CSS); self.combo_graxpert_corr.setFixedWidth(120)
        corr_hint = QLabel("Subtraction=additive bg  Division=multiplicative bg")
        corr_hint.setStyleSheet(f"color:{SUBTEXT};font-size:9px;")
        corr_row.addWidget(corr_lbl); corr_row.addWidget(self.combo_graxpert_corr)
        corr_row.addWidget(corr_hint); corr_row.addStretch()
        olay.addLayout(corr_row)

        # Smoothing
        sm_row = QHBoxLayout()
        sm_lbl = QLabel("Smoothing:"); sm_lbl.setStyleSheet(LBL_CSS); sm_lbl.setFixedWidth(90)
        self.spin_graxpert_smooth = QDoubleSpinBox()
        self.spin_graxpert_smooth.setRange(0.0, 1.0)
        self.spin_graxpert_smooth.setDecimals(2); self.spin_graxpert_smooth.setSingleStep(0.05)
        self.spin_graxpert_smooth.setValue(float(self._s.get("graxpert_smoothing", 0.5)))
        self.spin_graxpert_smooth.setFixedWidth(70); self.spin_graxpert_smooth.setStyleSheet(SPIN_CSS)
        sm_hint = QLabel("0.0=sharp edges  1.0=smooth gradient")
        sm_hint.setStyleSheet(f"color:{SUBTEXT};font-size:9px;")
        sm_row.addWidget(sm_lbl); sm_row.addWidget(self.spin_graxpert_smooth)
        sm_row.addWidget(sm_hint); sm_row.addStretch()
        olay.addLayout(sm_row)

        ai_row = QHBoxLayout()
        ai_lbl = QLabel("AI Version:"); ai_lbl.setStyleSheet(LBL_CSS); ai_lbl.setFixedWidth(90)
        self.edit_graxpert_ai = QLineEdit(self._s.get("graxpert_ai_version","latest"))
        self.edit_graxpert_ai.setStyleSheet(
            f"QLineEdit{{background:{BG};color:{TEXT};border:1px solid {BORDER};"
            f"border-radius:3px;padding:3px 6px;font-size:10px;}}")
        self.edit_graxpert_ai.setFixedWidth(100)
        ai_hint = QLabel("leave 'latest' for auto")
        ai_hint.setStyleSheet(f"color:{SUBTEXT};font-size:9px;")
        ai_row.addWidget(ai_lbl); ai_row.addWidget(self.edit_graxpert_ai)
        ai_row.addWidget(ai_hint); ai_row.addStretch()
        olay.addLayout(ai_row)

        # Denoise strength
        dn_row = QHBoxLayout()
        dn_lbl = QLabel("Denoise Str:"); dn_lbl.setStyleSheet(LBL_CSS); dn_lbl.setFixedWidth(90)
        self.spin_graxpert_denoise = QDoubleSpinBox()
        self.spin_graxpert_denoise.setRange(0.0, 1.0)
        self.spin_graxpert_denoise.setDecimals(2); self.spin_graxpert_denoise.setSingleStep(0.05)
        self.spin_graxpert_denoise.setValue(float(self._s.get("graxpert_denoise_strength", 0.8)))
        self.spin_graxpert_denoise.setFixedWidth(70); self.spin_graxpert_denoise.setStyleSheet(SPIN_CSS)
        dn_hint = QLabel("GraXpert Denoising strength")
        dn_hint.setStyleSheet(f"color:{SUBTEXT};font-size:9px;")
        dn_row.addWidget(dn_lbl); dn_row.addWidget(self.spin_graxpert_denoise)
        dn_row.addWidget(dn_hint); dn_row.addStretch()
        olay.addLayout(dn_row)
        lay.addWidget(opt_grp)

        lay.addStretch()
        return w

    def _browse_graxpert(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select GraXpert Executable", "",
            "Executable (*.exe);;All Files (*)" if sys.platform=="win32"
            else "All Files (*)")
        if path: self.edit_graxpert_exe.setText(path)

    def _auto_detect_graxpert(self):
        from ai.graxpert_bridge import find_graxpert_exe
        found = find_graxpert_exe()
        if found:
            self.edit_graxpert_exe.setText(found)
            QMessageBox.information(self,"Auto-Detect",f"Found:\n{found}")
        else:
            QMessageBox.information(self,"Auto-Detect",
                "GraXpert not found in common locations.\n"
                "Please use Browse to locate the executable manually.")

    def _verify_graxpert(self):
        exe = self.edit_graxpert_exe.text().strip()
        if not exe:
            QMessageBox.warning(self,"Verify","No path entered."); return
        if not os.path.isfile(exe):
            QMessageBox.critical(self,"Verify",f"File not found:\n{exe}"); return
        size_mb = os.path.getsize(exe)/1024/1024
        name = os.path.basename(exe).lower()
        if "graxpert" in name:
            QMessageBox.information(self,"Verify",
                f"\u2705 Executable found\n\n"
                f"Name: {os.path.basename(exe)}\n"
                f"Size: {size_mb:.1f} MB\n"
                f"Path: {exe}\n\n"
                "Looks like a valid GraXpert executable.")
        else:
            QMessageBox.warning(self,"Verify",
                f"File found but name does not look like GraXpert:\n{exe}")



    # ── ASTAP Tab ─────────────────────────────────────────────────────────
    def _tab_astap(self):
        w = QWidget(); w.setStyleSheet(f"background:{BG};")
        lay = QVBoxLayout(w); lay.setContentsMargins(16,16,16,16); lay.setSpacing(12)

        # Bilgi kutusu
        info = QLabel(
            "ASTAP (Astrometric STAcking Program) ile Plate Solving.\n\n"
            "ASTAP, görüntüdeki yıldızları yıldız kataloglarıyla eşleştirerek\n"
            "görüntünün gökyüzündeki konumunu (RA/Dec) ve ölçeğini bulur.\n\n"
            "İndirmek için:  https://www.hnsky.org/astap.htm\n\n"
            "G17 veya H17 kataloğunu da indirip aynı klasöre koyun.")
        info.setStyleSheet(
            f"background:{BG3};color:{MUTED};font-size:10px;"
            f"border:1px solid {BORDER};border-radius:6px;padding:10px;")
        info.setWordWrap(True)
        lay.addWidget(info)

        # Exe path
        grp = QGroupBox("ASTAP Executable"); grp.setStyleSheet(GROUP_CSS)
        glay = QVBoxLayout(grp); glay.setSpacing(6)

        path_row = QHBoxLayout()
        self.edit_astap_exe = QLineEdit(self._s.get("astap_exe", ""))
        self.edit_astap_exe.setPlaceholderText("e.g.  C:\\astap\\astap.exe")
        self.edit_astap_exe.setStyleSheet(
            f"QLineEdit{{background:{BG};color:{TEXT};border:1px solid {BORDER};"
            f"border-radius:3px;padding:4px 8px;font-size:10px;}}"
            f"QLineEdit:focus{{border:1px solid {ACCENT};}}")
        b_browse = QPushButton("📂 Browse"); b_browse.setStyleSheet(_btn()); b_browse.setFixedHeight(28)
        b_auto   = QPushButton("🔍 Auto-Detect"); b_auto.setStyleSheet(_btn(color="#0a1a2a", hover=ACCENT)); b_auto.setFixedHeight(28)
        path_row.addWidget(self.edit_astap_exe, 1)
        path_row.addWidget(b_browse); path_row.addWidget(b_auto)
        glay.addLayout(path_row)

        b_verify = QPushButton("✅  Verify")
        b_verify.setStyleSheet(_btn(color="#0a2a0a", hover=GREEN)); b_verify.setFixedHeight(26)
        glay.addWidget(b_verify)
        lay.addWidget(grp)

        b_browse.clicked.connect(self._browse_astap)
        b_auto.clicked.connect(self._auto_detect_astap)
        b_verify.clicked.connect(self._verify_astap)

        # Katalog path
        cat_grp = QGroupBox("Yıldız Kataloğu (Opsiyonel)"); cat_grp.setStyleSheet(GROUP_CSS)
        catl = QVBoxLayout(cat_grp); catl.setSpacing(6)

        db_row = QHBoxLayout()
        self.edit_astap_db = QLineEdit(self._s.get("astap_db", ""))
        self.edit_astap_db.setPlaceholderText("Katalog klasörü (boş = ASTAP varsayılanı)")
        self.edit_astap_db.setStyleSheet(
            f"QLineEdit{{background:{BG};color:{TEXT};border:1px solid {BORDER};"
            f"border-radius:3px;padding:4px 8px;font-size:10px;}}")
        b_db = QPushButton("📂"); b_db.setFixedSize(26, 26); b_db.setStyleSheet(_btn())
        b_db.clicked.connect(lambda: self._browse_dir(self.edit_astap_db))
        db_row.addWidget(self.edit_astap_db, 1); db_row.addWidget(b_db)
        catl.addLayout(db_row)
        lay.addWidget(cat_grp)

        # Arama parametreleri
        opt_grp = QGroupBox("Arama Parametreleri"); opt_grp.setStyleSheet(GROUP_CSS)
        olay = QVBoxLayout(opt_grp); olay.setSpacing(6)

        def _opt_row(lbl_text, widget, hint=""):
            r = QHBoxLayout(); r.setSpacing(8)
            l = QLabel(lbl_text); l.setStyleSheet(LBL_CSS); l.setFixedWidth(160)
            r.addWidget(l); r.addWidget(widget)
            if hint:
                hl = QLabel(hint); hl.setStyleSheet(f"color:{SUBTEXT};font-size:9px;")
                r.addWidget(hl)
            r.addStretch(); olay.addLayout(r)

        self.spin_astap_radius = QDoubleSpinBox()
        self.spin_astap_radius.setRange(1.0, 180.0)
        self.spin_astap_radius.setSingleStep(5.0); self.spin_astap_radius.setDecimals(1)
        self.spin_astap_radius.setValue(float(self._s.get("astap_radius", 30.0)))
        self.spin_astap_radius.setFixedWidth(80); self.spin_astap_radius.setStyleSheet(SPIN_CSS)
        _opt_row("Arama Yarıçapı (°):", self.spin_astap_radius, "kör aramada daha büyük = daha yavaş")

        self.spin_astap_minstars = QSpinBox()
        self.spin_astap_minstars.setRange(5, 500)
        self.spin_astap_minstars.setValue(int(self._s.get("astap_min_stars", 10)))
        self.spin_astap_minstars.setFixedWidth(70); self.spin_astap_minstars.setStyleSheet(SPIN_CSS)
        _opt_row("Min Yıldız:", self.spin_astap_minstars, "yüksek = daha güvenilir ama yavaş")

        self.spin_astap_timeout = QSpinBox()
        self.spin_astap_timeout.setRange(10, 600); self.spin_astap_timeout.setSingleStep(10)
        self.spin_astap_timeout.setValue(int(self._s.get("astap_timeout", 120)))
        self.spin_astap_timeout.setFixedWidth(70); self.spin_astap_timeout.setStyleSheet(SPIN_CSS)
        _opt_row("Zaman Aşımı (s):", self.spin_astap_timeout)

        self.spin_astap_downsample = QSpinBox()
        self.spin_astap_downsample.setRange(0, 4)
        self.spin_astap_downsample.setValue(int(self._s.get("astap_downsample", 0)))
        self.spin_astap_downsample.setFixedWidth(60); self.spin_astap_downsample.setStyleSheet(SPIN_CSS)
        _opt_row("Downsample:", self.spin_astap_downsample, "0=auto, 1-4")

        lay.addWidget(opt_grp)
        lay.addStretch()
        return w

    def _browse_astap(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select ASTAP Executable", "",
            "Executable (*.exe);;All Files (*)" if sys.platform == "win32"
            else "All Files (*)")
        if path: self.edit_astap_exe.setText(path)

    def _auto_detect_astap(self):
        from ai.astap_bridge import find_astap_exe
        found = find_astap_exe()
        if found:
            self.edit_astap_exe.setText(found)
            QMessageBox.information(self, "Auto-Detect", f"ASTAP bulundu:\n{found}")
        else:
            QMessageBox.information(self, "Auto-Detect",
                "ASTAP sistemde bulunamadı.\n"
                "Lütfen Manuel yolu belirleyin.")

    def _verify_astap(self):
        exe = self.edit_astap_exe.text().strip()
        if not exe:
            QMessageBox.warning(self, "Verify", "Yol girilmedi."); return
        if not os.path.isfile(exe):
            QMessageBox.critical(self, "Verify", f"Dosya bulunamadı:\n{exe}"); return
        size_mb = os.path.getsize(exe) / 1024 / 1024
        name = os.path.basename(exe).lower()
        if "astap" in name:
            QMessageBox.information(self, "Verify",
                f"✅ ASTAP bulundu\n\nDosya: {os.path.basename(exe)}\n"
                f"Boyut: {size_mb:.1f} MB\nYol: {exe}")
        else:
            QMessageBox.warning(self, "Verify",
                f"Dosya bulundu ama adı ASTAP'a benzemiyor:\n{exe}")

    # ── Display Tab ───────────────────────────────────────────────────────
    def _tab_display(self):
        w = QWidget(); w.setStyleSheet(f"background:{BG};")
        lay = QVBoxLayout(w); lay.setContentsMargins(16,16,16,16); lay.setSpacing(10)

        grp = QGroupBox("Display Settings"); grp.setStyleSheet(GROUP_CSS)
        glayout = QVBoxLayout(grp); glayout.setSpacing(8)

        def row(lbl_text, widget):
            r = QHBoxLayout()
            l = QLabel(lbl_text); l.setFixedWidth(160); l.setStyleSheet(LBL_CSS)
            r.addWidget(l); r.addWidget(widget); r.addStretch()
            return r

        self.combo_interp = QComboBox()
        self.combo_interp.addItems(["nearest","bilinear","bicubic"])
        self.combo_interp.setCurrentText(self._s.get("canvas_interp","nearest"))
        self.combo_interp.setStyleSheet(COMBO_CSS); self.combo_interp.setFixedWidth(120)
        glayout.addLayout(row("Image Interpolation:", self.combo_interp))
        hint1 = QLabel("  nearest = crisp pixels (recommended for astronomy)")
        hint1.setStyleSheet(f"color:{SUBTEXT};font-size:9px;"); glayout.addWidget(hint1)

        self.spin_font = QSpinBox()
        self.spin_font.setRange(8,16); self.spin_font.setValue(int(self._s.get("font_size",10)))
        self.spin_font.setFixedWidth(60); self.spin_font.setStyleSheet(SPIN_CSS)
        glayout.addLayout(row("Font Size (restart):", self.spin_font))

        self.chk_history = QCheckBox("Show history panel")
        self.chk_history.setChecked(bool(self._s.get("show_history",True)))
        self.chk_history.setStyleSheet(CHECK_CSS)
        glayout.addWidget(self.chk_history)

        lay.addWidget(grp)
        lay.addStretch()
        return w

    # ── Processing Tab ────────────────────────────────────────────────────
    def _tab_processing(self):
        w = QWidget(); w.setStyleSheet(f"background:{BG};")
        lay = QVBoxLayout(w); lay.setContentsMargins(16,16,16,16); lay.setSpacing(10)

        grp = QGroupBox("Processing Defaults"); grp.setStyleSheet(GROUP_CSS)
        glayout = QVBoxLayout(grp); glayout.setSpacing(8)

        def row(lbl_text, widget):
            r = QHBoxLayout()
            l = QLabel(lbl_text); l.setFixedWidth(180); l.setStyleSheet(LBL_CSS)
            r.addWidget(l); r.addWidget(widget); r.addStretch()
            return r

        self.combo_def_stretch = QComboBox()
        self.combo_def_stretch.addItems(["auto_stf","linear","hyperbolic","asinh","statistical"])
        self.combo_def_stretch.setCurrentText(self._s.get("default_stretch","auto_stf"))
        self.combo_def_stretch.setStyleSheet(COMBO_CSS); self.combo_def_stretch.setFixedWidth(130)
        glayout.addLayout(row("Default Stretch Method:", self.combo_def_stretch))

        self.combo_def_bg = QComboBox()
        self.combo_def_bg.addItems(["dbe_spline","polynomial","ai_gradient","gaussian_sub"])
        self.combo_def_bg.setCurrentText(self._s.get("default_bg","dbe_spline"))
        self.combo_def_bg.setStyleSheet(COMBO_CSS); self.combo_def_bg.setFixedWidth(130)
        glayout.addLayout(row("Default BG Method:", self.combo_def_bg))

        self.combo_outfmt = QComboBox()
        self.combo_outfmt.addItems(["FITS","PNG","TIFF","JPEG"])
        self.combo_outfmt.setCurrentText(self._s.get("output_format","FITS"))
        self.combo_outfmt.setStyleSheet(COMBO_CSS); self.combo_outfmt.setFixedWidth(80)
        glayout.addLayout(row("Default Save Format:", self.combo_outfmt))

        lay.addWidget(grp)
        lay.addStretch()
        return w

    # ── Update Tab ─────────────────────────────────────────────────────────
    def _tab_update(self):
        w = QWidget(); w.setStyleSheet(f"background:{BG};")
        lay = QVBoxLayout(w); lay.setContentsMargins(16,16,16,16); lay.setSpacing(12)

        from core.version import APP_VERSION, APP_NAME, APP_AUTHOR, APP_BUILD_DATE

        # Sürüm bilgisi kartı
        ver_grp = QGroupBox("Uygulama Bilgisi"); ver_grp.setStyleSheet(GROUP_CSS)
        vgl = QVBoxLayout(ver_grp); vgl.setSpacing(6)

        def _info_row(label, value, color=None):
            r = QHBoxLayout(); r.setSpacing(8)
            l = QLabel(label); l.setStyleSheet(LBL_CSS); l.setFixedWidth(140)
            v = QLabel(value)
            v.setStyleSheet(f"color:{color or TEXT};font-size:10px;font-weight:bold;")
            r.addWidget(l); r.addWidget(v); r.addStretch()
            vgl.addLayout(r)

        _info_row("Program:",     APP_NAME,       ACCENT2)
        _info_row("Sürüm:",       f"v{APP_VERSION}", GREEN)
        _info_row("Geliştirici:", APP_AUTHOR,     GOLD)
        _info_row("Yapı Tarihi:", APP_BUILD_DATE, MUTED)
        lay.addWidget(ver_grp)

        # Güncelleme ayarları
        upd_grp = QGroupBox("Güncelleme Ayarları"); upd_grp.setStyleSheet(GROUP_CSS)
        ugl = QVBoxLayout(upd_grp); ugl.setSpacing(8)

        self.chk_update_startup = QCheckBox(
            "Program başlarken güncelleme kontrolü yap")
        self.chk_update_startup.setChecked(
            bool(self._s.get("check_updates_on_startup", False)))
        self.chk_update_startup.setStyleSheet(CHECK_CSS)
        ugl.addWidget(self.chk_update_startup)

        hint = QLabel(
            "Güncelleme kontrolü için internet bağlantısı gerekir.\n"
            "GitHub deposu ayarlanmamışsa bu seçeneğin etkisi olmaz.")
        hint.setStyleSheet(f"color:{SUBTEXT};font-size:9px;")
        hint.setWordWrap(True)
        ugl.addWidget(hint)

        # Manuel kontrol butonu
        b_check = QPushButton("🔍  Şimdi Güncelleme Kontrol Et")
        b_check.setStyleSheet(_btn()); b_check.setFixedHeight(30)
        b_check.clicked.connect(self._open_update_from_settings)
        ugl.addWidget(b_check)
        lay.addWidget(upd_grp)

        # Changelog
        cl_grp = QGroupBox("Sürüm Geçmişi"); cl_grp.setStyleSheet(GROUP_CSS)
        cgl = QVBoxLayout(cl_grp)
        from core.version import CHANGELOG
        cl_edit = QTextEdit()
        cl_edit.setReadOnly(True)
        cl_edit.setFixedHeight(160)
        cl_edit.setStyleSheet(
            f"QTextEdit{{background:{BG};color:{TEXT};"
            f"border:1px solid {BORDER};border-radius:3px;"
            f"font-family:Consolas,monospace;font-size:9px;padding:4px;}}")
        cl_edit.setPlainText(CHANGELOG.strip())
        cgl.addWidget(cl_edit)
        lay.addWidget(cl_grp)

        lay.addStretch()
        return w

    def _open_update_from_settings(self):
        """Settings içinden update dialogunu aç."""
        from gui.update_dialog import UpdateDialog
        dlg = UpdateDialog(parent=self, auto_check=True)
        dlg.exec()

    # ── Paths Tab ─────────────────────────────────────────────────────────
    def _tab_paths(self):
        w = QWidget(); w.setStyleSheet(f"background:{BG};")
        lay = QVBoxLayout(w); lay.setContentsMargins(16,16,16,16); lay.setSpacing(10)

        grp = QGroupBox("Default Directories"); grp.setStyleSheet(GROUP_CSS)
        glayout = QVBoxLayout(grp); glayout.setSpacing(8)

        def path_row(lbl_text, attr, default=""):
            r = QHBoxLayout()
            l = QLabel(lbl_text); l.setFixedWidth(140); l.setStyleSheet(LBL_CSS)
            edit = QLineEdit(self._s.get(default,""))
            edit.setStyleSheet(
                f"QLineEdit{{background:{BG};color:{TEXT};border:1px solid {BORDER};"
                f"border-radius:3px;padding:3px 6px;font-size:10px;}}")
            setattr(self, attr, edit)
            b = QPushButton("📂"); b.setFixedSize(26,26)
            b.setStyleSheet(_btn())
            b.clicked.connect(lambda _, e=edit: self._browse_dir(e))
            r.addWidget(l); r.addWidget(edit,1); r.addWidget(b)
            return r

        glayout.addLayout(path_row("Open Directory:",  "edit_open_dir",  "last_open_dir"))
        glayout.addLayout(path_row("Save Directory:",  "edit_save_dir",  "last_save_dir"))
        lay.addWidget(grp)
        lay.addStretch()
        return w

    def _browse_dir(self, edit: QLineEdit):
        d = QFileDialog.getExistingDirectory(self, "Select Directory",
                                              edit.text() or os.path.expanduser("~"))
        if d: edit.setText(d)

    # ── Collect settings ──────────────────────────────────────────────────
    def get_settings(self) -> dict:
        s = dict(self._s)
        s["starnet_exe"]       = self.edit_starnet_exe.text().strip()
        s["starnet_stride"]    = int(self.combo_stride.currentText())
        s["starnet_use_gpu"]   = self.chk_gpu.isChecked()
        if hasattr(self,"edit_graxpert_exe"):
            s["graxpert_exe"]             = self.edit_graxpert_exe.text().strip()
            s["graxpert_smoothing"]        = self.spin_graxpert_smooth.value()
            s["graxpert_ai_version"]       = self.edit_graxpert_ai.text().strip() or "latest"
            s["graxpert_correction"]       = self.combo_graxpert_corr.currentText()
            s["graxpert_denoise_strength"] = self.spin_graxpert_denoise.value()
        if hasattr(self, "edit_astap_exe"):
            s["astap_exe"]        = self.edit_astap_exe.text().strip()
            s["astap_db"]         = self.edit_astap_db.text().strip()
            s["astap_radius"]     = self.spin_astap_radius.value()
            s["astap_min_stars"]  = self.spin_astap_minstars.value()
            s["astap_timeout"]    = self.spin_astap_timeout.value()
            s["astap_downsample"] = self.spin_astap_downsample.value()
        s["canvas_interp"]     = self.combo_interp.currentText()
        s["font_size"]         = self.spin_font.value()
        s["show_history"]      = self.chk_history.isChecked()
        s["default_stretch"]   = self.combo_def_stretch.currentText()
        s["default_bg"]        = self.combo_def_bg.currentText()
        s["output_format"]     = self.combo_outfmt.currentText()
        s["last_open_dir"]     = self.edit_open_dir.text().strip()
        s["last_save_dir"]     = self.edit_save_dir.text().strip()
        if hasattr(self, "chk_update_startup"):
            s["check_updates_on_startup"] = self.chk_update_startup.isChecked()
        return s


# ═══════════════════════════════ RECOMPOSITION DIALOG ═══════════════════════
class RecompositionDialog(QDialog):
    """\n    Star Recomposition — blend starless + stars-only with full layer control.\n    • Live preview (downsampled for speed)\n    • Blend mode, opacity, star size, hue/sat/brightness, luminosity mask\n    • Apply → adds result to main viewer history\n    """

    MODES = ["screen","add","lighten","soft_light","hard_light",
             "luminosity","overlay","multiply"]

    def __init__(self, starless, stars,
                 on_apply, parent=None):
        super().__init__(parent)
        self.setWindowTitle("✦+  Star Recomposition")
        self.setMinimumSize(960, 660)
        self.setStyleSheet(f"background:{BG};color:{TEXT};font-size:11px;")
        self._starless  = starless   # may be None initially
        self._stars     = stars      # may be None initially
        self._on_apply  = on_apply
        self._preview   = None
        self._dirty     = True
        self._build()
        if self._starless is not None and self._stars is not None:
            self._schedule_preview()
        else:
            self._update_load_labels()

    # ── UI ────────────────────────────────────────────────────────────────
    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0,0,0,0); root.setSpacing(0)

        # ── Left: controls ────────────────────────────────────────────────
        ctrl = QWidget(); ctrl.setFixedWidth(320)
        ctrl.setStyleSheet(f"background:{BG2};border-right:1px solid {BORDER};")
        cl = QVBoxLayout(ctrl); cl.setContentsMargins(0,0,0,0); cl.setSpacing(0)

        hdr = QLabel("  ✦+  Star Recomposition")
        hdr.setStyleSheet(
            f"background:{BG4};color:{GOLD};font-size:13px;font-weight:700;"
            f"padding:10px 0;border-bottom:1px solid {BORDER};letter-spacing:1px;")
        cl.addWidget(hdr)

        # ── File load section ─────────────────────────────────────────────
        file_grp = QGroupBox("📂  Source Files"); file_grp.setStyleSheet(GROUP_CSS)
        fgl = QVBoxLayout(file_grp); fgl.setSpacing(4); fgl.setContentsMargins(8,6,8,8)

        # Starless row
        sl_row = QHBoxLayout(); sl_row.setSpacing(4)
        self.lbl_sl_file = QLabel("No starless loaded")
        self.lbl_sl_file.setStyleSheet(f"color:{MUTED};font-size:9px;")
        b_load_sl = QPushButton("📂 Starless"); b_load_sl.setStyleSheet(_btn(h=22)); b_load_sl.setFixedHeight(22)
        sl_row.addWidget(b_load_sl); sl_row.addWidget(self.lbl_sl_file, 1)
        fgl.addLayout(sl_row)

        # Stars-only row
        st_row = QHBoxLayout(); st_row.setSpacing(4)
        self.lbl_st_file = QLabel("No stars-only loaded")
        self.lbl_st_file.setStyleSheet(f"color:{MUTED};font-size:9px;")
        b_load_st = QPushButton("📂 Stars Only"); b_load_st.setStyleSheet(_btn(h=22)); b_load_st.setFixedHeight(22)
        st_row.addWidget(b_load_st); st_row.addWidget(self.lbl_st_file, 1)
        fgl.addLayout(st_row)

        b_load_sl.clicked.connect(self._load_starless_file)
        b_load_st.clicked.connect(self._load_stars_file)
        cl.addWidget(file_grp)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            f"QScrollArea{{border:none;background:{BG2};}}"
            f"QScrollBar:vertical{{background:{BG};width:6px;border-radius:3px;}}"
            f"QScrollBar::handle:vertical{{background:{BORDER2};border-radius:3px;}}"
            f"QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{{height:0;}}")
        inner = QWidget(); inner.setStyleSheet(f"background:{BG2};")
        vb = QVBoxLayout(inner); vb.setContentsMargins(10,10,10,10); vb.setSpacing(8)

        # ── Layer group ───────────────────────────────────────────────────
        grp_layer = self._make_group("🎚  Layer")
        gl = QVBoxLayout(grp_layer); gl.setSpacing(4)

        self.combo_mode = self._make_combo("Blend Mode", self.MODES, "screen")
        gl.addWidget(self.combo_mode)
        self.sl_opacity = self._make_slider("Opacity", 0, 100, 100, suffix="%")
        gl.addWidget(self.sl_opacity)
        vb.addWidget(grp_layer)

        # ── Star Size group ───────────────────────────────────────────────
        grp_size = self._make_group("⭐  Star Size")
        gs = QVBoxLayout(grp_size); gs.setSpacing(4)
        self.sl_size = self._make_slider("Size Factor", 20, 200, 100,
                                          suffix="%", tip="100%=original  <100=smaller  >100=larger")
        gs.addWidget(self.sl_size)
        vb.addWidget(grp_size)

        # ── Colour group ──────────────────────────────────────────────────
        grp_col = self._make_group("🎨  Star Colour")
        gc = QVBoxLayout(grp_col); gc.setSpacing(4)
        self.sl_hue  = self._make_slider("Hue Shift",    -180, 180,   0, suffix="°",
                                          tip="Rotate star hue (-180–180°)")
        self.sl_sat  = self._make_slider("Saturation",      0, 300, 100, suffix="%",
                                          tip="100%=original  0=grey  200=vivid")
        self.sl_bri  = self._make_slider("Brightness",      0, 300, 100, suffix="%",
                                          tip="100%=original")
        gc.addWidget(self.sl_hue)
        gc.addWidget(self.sl_sat)
        gc.addWidget(self.sl_bri)
        b_reset_col = QPushButton("↺  Reset Colour")
        b_reset_col.setStyleSheet(_btn(h=22)); b_reset_col.setFixedHeight(24)
        b_reset_col.clicked.connect(self._reset_colour)
        gc.addWidget(b_reset_col)
        vb.addWidget(grp_col)

        # ── Luminosity Mask group ─────────────────────────────────────────
        grp_lum = self._make_group("🌟  Luminosity Mask")
        glm = QVBoxLayout(grp_lum); glm.setSpacing(4)
        self.chk_lum = QCheckBox("Enable luminosity mask")
        self.chk_lum.setStyleSheet(CHECK_CSS)
        lum_info = QLabel("Protects bright nebula regions from star overlay")
        lum_info.setStyleSheet(f"color:{SUBTEXT};font-size:9px;"); lum_info.setWordWrap(True)
        self.sl_lum_thr = self._make_slider("Threshold", 0, 100, 50, suffix="%",
                                             tip="Pixels brighter than threshold are protected")
        glm.addWidget(self.chk_lum)
        glm.addWidget(lum_info)
        glm.addWidget(self.sl_lum_thr)
        vb.addWidget(grp_lum)

        vb.addStretch()
        scroll.setWidget(inner)
        cl.addWidget(scroll, 1)

        # ── Buttons ───────────────────────────────────────────────────────
        btn_w = QWidget()
        btn_w.setStyleSheet(f"background:{BG3};border-top:1px solid {BORDER};")
        btn_lay = QVBoxLayout(btn_w); btn_lay.setContentsMargins(10,8,10,8); btn_lay.setSpacing(6)
        self.btn_preview = QPushButton("🔄  Refresh Preview")
        self.btn_preview.setStyleSheet(_btn(color=BG4, h=28)); self.btn_preview.setFixedHeight(28)
        self.btn_preview.clicked.connect(self._do_preview)
        self.pbar = QProgressBar(); self.pbar.setFixedHeight(5)
        self.pbar.setTextVisible(False); self.pbar.setRange(0,0)
        self.pbar.setStyleSheet(
            f"QProgressBar{{background:{BG};border:none;border-radius:2px;}}"
            f"QProgressBar::chunk{{background:{GOLD};border-radius:2px;}}")
        self.pbar.hide()
        row_btn = QHBoxLayout(); row_btn.setSpacing(6)
        b_cancel = QPushButton("Cancel")
        b_cancel.setStyleSheet(_btn(h=32)); b_cancel.setFixedHeight(32)
        self.b_apply = QPushButton("✅  Apply to Image")
        self.b_apply.setStyleSheet(_run_btn(GREEN)); self.b_apply.setFixedHeight(34)
        row_btn.addWidget(b_cancel, 1); row_btn.addWidget(self.b_apply, 2)
        btn_lay.addWidget(self.btn_preview)
        btn_lay.addWidget(self.pbar)
        btn_lay.addLayout(row_btn)
        cl.addWidget(btn_w)

        b_cancel.clicked.connect(self.reject)
        self.b_apply.clicked.connect(self._apply)

        # ── Right: single result preview with zoom ────────────────────────
        right = QWidget(); right.setStyleSheet(f"background:{BG};")
        rl = QVBoxLayout(right); rl.setContentsMargins(0,0,0,0); rl.setSpacing(0)

        # Header with zoom controls
        prev_hdr = QWidget(); prev_hdr.setFixedHeight(32)
        prev_hdr.setStyleSheet(f"background:{BG2};border-bottom:1px solid {BORDER};")
        ph = QHBoxLayout(prev_hdr); ph.setContentsMargins(8,2,8,2); ph.setSpacing(6)
        self.lbl_mode = QLabel("Result Preview")
        self.lbl_mode.setStyleSheet(f"color:{GREEN};font-size:11px;font-weight:600;")
        ph.addWidget(self.lbl_mode, 1)

        # Zoom buttons
        for icon, tip, fn in [
            ("🔍+", "Zoom In",  lambda: self._preview_zoom(1.3)),
            ("🔍−", "Zoom Out", lambda: self._preview_zoom(0.77)),
            ("⛶",  "Fit",      lambda: self._preview_fit()),
        ]:
            b = QPushButton(icon); b.setFixedSize(30, 24)
            b.setStyleSheet(_btn(color=BG3, h=24))
            b.setToolTip(tip); b.clicked.connect(fn)
            ph.addWidget(b)

        self.lbl_zoom = QLabel("100%")
        self.lbl_zoom.setStyleSheet(f"color:{SUBTEXT};font-size:9px;min-width:36px;")
        ph.addWidget(self.lbl_zoom)
        rl.addWidget(prev_hdr)

        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

        # Single full-size result canvas
        self._fig_preview = Figure(facecolor=BG)
        self._ax = self._fig_preview.add_subplot(111)
        self._ax.set_facecolor(BG); self._ax.set_axis_off()
        self._ax.text(0.5, 0.5, "Load files and adjust settings",
                      ha="center", va="center", color=SUBTEXT,
                      fontsize=12, transform=self._ax.transAxes)
        self._fig_preview.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Aliases
        self._fig       = self._fig_preview
        self._fig_left  = self._fig_preview
        self._fig_right = self._fig_preview
        # Dummy axes for _show_slot (not displayed, but avoid AttributeError)
        self._ax_left  = self._ax
        self._ax_right = self._ax

        self._canvas = FigureCanvas(self._fig_preview)
        self._canvas.setStyleSheet(f"background:{BG};")
        # Scroll zoom
        self._canvas.mpl_connect("scroll_event", self._preview_scroll)
        rl.addWidget(self._canvas, 1)

        root.addWidget(ctrl)
        root.addWidget(right, 1)

        # Zoom state
        self._prev_zoom = 1.0

        # Connect all controls to preview update
        for sl in [self.sl_opacity, self.sl_size, self.sl_hue,
                   self.sl_sat, self.sl_bri, self.sl_lum_thr]:
            sl.sl.valueChanged.connect(self._on_change)
        self.combo_mode.cb.currentTextChanged.connect(self._on_change)
        self.chk_lum.stateChanged.connect(self._on_change)

    # ── Widget helpers ────────────────────────────────────────────────────
    def _make_group(self, title):
        g = QGroupBox(title); g.setStyleSheet(GROUP_CSS); return g

    def _make_combo(self, lbl, items, default=""):
        return PC(lbl, items, default)

    def _make_slider(self, lbl, mn, mx, val, suffix="", tip=""):
        w = PS(lbl, mn, mx, val, dec=0, tip=tip)
        return w

    def _reset_colour(self):
        self.sl_hue.set(0)
        self.sl_sat.set(100)
        self.sl_bri.set(100)

    # ── Preview logic ─────────────────────────────────────────────────────
    def _on_change(self):
        self._dirty = True
        self._schedule_preview()

    def _schedule_preview(self):
        if not hasattr(self, "_prev_timer"):
            self._prev_timer = QTimer(self)
            self._prev_timer.setSingleShot(True)
            self._prev_timer.timeout.connect(self._do_preview)
        self._prev_timer.start(300)   # 300 ms debounce

    def _do_preview(self):
        self._dirty = False
        if self._starless is None or self._stars is None:
            self.lbl_mode.setText("Load starless + stars-only files to preview")
            return
        self.pbar.show()
        try:
            from gui.recomposition import recompose
            params = self._collect()

            # Downsample for speed
            h, w = self._starless.shape[:2]
            scale = min(1.0, 800.0 / max(h, w, 1))
            nh, nw = max(1, int(h*scale)), max(1, int(w*scale))

            import cv2 as _cv2
            def ds(img):
                if scale >= 0.99: return img
                return _cv2.resize(img, (nw, nh), interpolation=_cv2.INTER_AREA)

            sl_s = ds(self._starless)
            st_s = ds(self._stars)

            result = recompose(sl_s, st_s, **params)
            self._preview = result

            self._ax.clear()
            self._ax.set_facecolor(BG)
            self._ax.imshow(result,
                cmap=("gray" if result.ndim==2 else None),
                origin="upper", aspect="equal", interpolation="nearest")
            self._ax.set_title(
                f"{params['blend_mode']}  {int(params['opacity']*100)}%  "
                f"size:{int(params['star_size']*100)}%",
                color=GREEN, fontsize=9, pad=3)
            self._ax.set_axis_off()
            self._fig_preview.subplots_adjust(left=0, right=1, top=0.95, bottom=0)
            try: self._fig_preview.canvas.draw()
            except Exception: pass

            mode = params["blend_mode"]
            op   = int(params["opacity"]*100)
            sz   = int(params["star_size"]*100)
            self.lbl_zoom.setText(f"{int(self._prev_zoom*100)}%")
        except Exception as e:
            self.lbl_mode.setText(f"Preview error: {e}")
        finally:
            self.pbar.hide()

    def _update_load_labels(self):
        if self._starless is not None:
            h, w = self._starless.shape[:2]
            self.lbl_sl_file.setText(f"✅  {w}×{h} loaded")
            self.lbl_sl_file.setStyleSheet(f"color:{GREEN};font-size:9px;")
        else:
            self.lbl_sl_file.setText("No starless loaded")
            self.lbl_sl_file.setStyleSheet(f"color:{MUTED};font-size:9px;")
        if self._stars is not None:
            h, w = self._stars.shape[:2]
            self.lbl_st_file.setText(f"✅  {w}×{h} loaded")
            self.lbl_st_file.setStyleSheet(f"color:{GREEN};font-size:9px;")
        else:
            self.lbl_st_file.setText("No stars-only loaded")
            self.lbl_st_file.setStyleSheet(f"color:{MUTED};font-size:9px;")

    def _load_starless_file(self):
        from core.loader import load_image
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Starless Image", "",
            "Images (*.fits *.fit *.tif *.tiff *.png *.jpg *.jpeg);;All (*)")
        if not path: return
        try:
            self._starless = load_image(path)
            h, w = self._starless.shape[:2]
            self.lbl_sl_file.setText(f"✅  {os.path.basename(path)}  ({w}×{h})")
            self.lbl_sl_file.setStyleSheet(f"color:{GREEN};font-size:9px;")
            self._show_slot(self._starless, "Starless", 0)
            self._schedule_preview()
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))

    def _load_stars_file(self):
        from core.loader import load_image
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Stars-Only Image", "",
            "Images (*.fits *.fit *.tif *.tiff *.png *.jpg *.jpeg);;All (*)")
        if not path: return
        try:
            self._stars = load_image(path)
            h, w = self._stars.shape[:2]
            self.lbl_st_file.setText(f"✅  {os.path.basename(path)}  ({w}×{h})")
            self.lbl_st_file.setStyleSheet(f"color:{GREEN};font-size:9px;")
            self._show_slot(self._stars, "Stars Only", 1)
            self._schedule_preview()
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))

    def _show_slot(self, img: np.ndarray, title: str, slot: int):
        """\n        Slot 0=starless, 1=stars — stored for blending.\n        Preview shows result when both are loaded, otherwise shows the loaded image.\n        """
        # Show the loaded image immediately in result panel
        self._ax.clear()
        self._ax.set_facecolor(BG)
        self._ax.imshow(img, cmap=("gray" if img.ndim==2 else None),
                        origin="upper", aspect="equal", interpolation="nearest")
        label = "Starless" if slot==0 else "Stars Only"
        self._ax.set_title(f"Loaded: {label}", color=GOLD if slot==0 else ACCENT2,
                           fontsize=10, pad=3)
        self._ax.set_axis_off()
        self._fig_preview.subplots_adjust(left=0, right=1, top=0.95, bottom=0)
        try: self._fig_preview.canvas.draw()
        except Exception: pass

    def _preview_zoom(self, factor: float):
        ax = self._ax
        xl = ax.get_xlim(); yl = ax.get_ylim()
        cx = (xl[0]+xl[1])/2; cy = (yl[0]+yl[1])/2
        hw = (xl[1]-xl[0])/2/factor
        hh = abs(yl[1]-yl[0])/2/factor
        ax.set_xlim(cx-hw, cx+hw); ax.set_ylim(cy+hh, cy-hh)
        self._prev_zoom *= factor
        self.lbl_zoom.setText(f"{int(self._prev_zoom*100)}%")
        try: self._fig_preview.canvas.draw()
        except Exception: pass

    def _preview_fit(self):
        if self._preview is not None:
            h, w = self._preview.shape[:2]
        elif self._starless is not None:
            h, w = self._starless.shape[:2]
        else:
            return
        self._ax.set_xlim(0, w); self._ax.set_ylim(h, 0)
        self._prev_zoom = 1.0; self.lbl_zoom.setText("Fit")
        try: self._fig_preview.canvas.draw()
        except Exception: pass

    def _preview_scroll(self, event):
        factor = 1.2 if event.button == "up" else (1/1.2)
        xd = event.xdata; yd = event.ydata
        ax = self._ax
        xl = ax.get_xlim(); yl = ax.get_ylim()
        if xd is None: xd = (xl[0]+xl[1])/2
        if yd is None: yd = (yl[0]+yl[1])/2
        ax.set_xlim(xd+(xl[0]-xd)/factor, xd+(xl[1]-xd)/factor)
        ax.set_ylim(yd+(yl[0]-yd)/factor, yd+(yl[1]-yd)/factor)
        self._prev_zoom *= factor
        self.lbl_zoom.setText(f"{int(self._prev_zoom*100)}%")
        try: self._fig_preview.canvas.draw()
        except Exception: pass

    def _collect(self) -> dict:
        return {
            "blend_mode":   self.combo_mode.v(),
            "opacity":      self.sl_opacity.v() / 100.0,
            "star_size":    self.sl_size.v() / 100.0,
            "hue_shift":    self.sl_hue.v(),
            "saturation":   self.sl_sat.v() / 100.0,
            "brightness":   self.sl_bri.v() / 100.0,
            "use_lum_mask": self.chk_lum.isChecked(),
            "lum_threshold":self.sl_lum_thr.v() / 100.0,
        }

    def _apply(self):
        """Apply full-resolution recomposition."""
        if self._starless is None or self._stars is None:
            QMessageBox.warning(self,"Missing Images",
                "Please load both starless and stars-only images first.")
            return
        self.b_apply.setEnabled(False)
        self.b_apply.setText("⏳  Processing…")
        self.pbar.show()
        try:
            from gui.recomposition import recompose
            params  = self._collect()
            result  = recompose(self._starless, self._stars, **params)
            self._on_apply(result)
            self.accept()
        except Exception as e:
            import traceback
            QMessageBox.critical(self, "Recomposition Error",
                                 f"{e}\n\n{traceback.format_exc()[:500]}")
        finally:
            self.b_apply.setEnabled(True)
            self.b_apply.setText("✅  Apply to Image")
            self.pbar.hide()


# ═══════════════════════════════ PROCESS FLYOUT ══════════════════════════════
class WorkflowPanel(QFrame):
    """
    Astro Fotoğrafçılık Workflow Rehberi.
    Her adıma tıklayınca ilgili işlem paneli açılır.
    """
    step_clicked = pyqtSignal(str)   # panel key
    closed       = pyqtSignal()

    STEPS = [
        # (key, icon, başlık, faz_rengi, faz_etiket, açıklama)
        ("stack",   "🗂",  "Stacking",          "#2255aa", "HAM VERİ",
         "Bias→Dark→Flat kalibrasyon, hizalama ve yığınlama. Tüm ham kareleri tek temiz görüntüye dönüştür."),
        ("bg",      "🌌", "BG Extract",          "#22aa55", "LİNEER",
         "Gradyan ve ışık kirliliğini çıkar. GraXpert AI veya Nox kullan. Stretch öncesi zorunlu!"),
        ("noise",   "✨", "Noise Reduction",     "#22aa55", "LİNEER",
         "Lineer gürültüyü bastır. Silentium burada en iyi çalışır — fiziksel gürültü modeli."),
        ("deconv",  "🔭", "Deconvolution",       "#22aa55", "LİNEER",
         "Yıldız PSF'ini düzelt, optik bulanıklığı gider. Blur Exterminator önerilen."),
        ("color",   "🎨", "Color Calibration",   "#22aa55", "LİNEER",
         "Renk dengesi — SPCC G2V (Güneş tipi) veya Avg Spiral Galaxy referansı. Stretch rengi bozar!"),
        ("stretch", "📊", "Stretch",             "#aa5522", "GEÇİŞ ★",
         "Lineer→Non-lineer. Veralux HMS önerilen. Bu noktadan sonra lineer işlem yapılmaz!"),
        ("stars",   "⭐", "Star Removal",        "#884488", "NON-LİNEER",
         "Yıldızları ayır (StarNet++). Yıldızsız katmanda kontrast işlemleri çok daha etkili."),
        ("sharp",   "🔪", "Sharpen",             "#884488", "NON-LİNEER",
         "Yapı ve doku detayları. Revela (Veralux) yıldızsız katmanda çok daha temiz çalışır."),
        ("color",   "🎨", "Color Grading",       "#884488", "NON-LİNEER",
         "Vectra ile LCH renk cerrahi. Ha/OIII/SHO vektörlerini ince ayarla."),
        ("recomp",  "✦+","Star Recomposition",   "#884488", "NON-LİNEER",
         "Keskinleştirilmiş starless + orijinal yıldız maskesini blend ile birleştir."),
        ("hist",    "📈","Histogram / Curves",   "#555577", "FİNAL",
         "Levels, Curves, Adjustments ile tonlama. Highlights/Shadows/Contrast ince ayarı."),
    ]

    def __init__(self, anchor_btn, parent=None):
        super().__init__(parent, Qt.WindowType.Popup)
        self._anchor = anchor_btn
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            f"WorkflowPanel{{background:{BG2};border:2px solid {ACCENT2};"
            f"border-radius:8px;}}")
        self.setMinimumWidth(420)
        self.setMaximumWidth(460)
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0,0,0,0); lay.setSpacing(0)

        # Header
        hdr = QWidget()
        hdr.setFixedHeight(40)
        hdr.setStyleSheet(
            f"background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            f"stop:0 #0a1a2a,stop:1 #0a2a1a);"
            f"border-radius:6px 6px 0 0;"
            f"border-bottom:1px solid {BORDER};")
        hl = QHBoxLayout(hdr); hl.setContentsMargins(12,0,8,0)
        title = QLabel("🌌  Astro Workflow Rehberi")
        title.setStyleSheet(
            f"color:{ACCENT2};font-size:12px;font-weight:700;letter-spacing:0.5px;")
        hl.addWidget(title, 1)
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(22,22)
        close_btn.setStyleSheet(
            f"QPushButton{{background:transparent;color:{SUBTEXT};border:none;font-size:12px;}}"
            f"QPushButton:hover{{color:{RED};}}")
        close_btn.clicked.connect(self.close)
        hl.addWidget(close_btn)
        lay.addWidget(hdr)

        # Sub-header
        sub = QLabel("  Tıkla → ilgili panel açılır  |  Faz sırası kritik!")
        sub.setStyleSheet(
            f"color:{MUTED};font-size:8px;background:{BG3};"
            f"padding:3px 12px;border-bottom:1px solid {BORDER};")
        lay.addWidget(sub)

        # Scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            f"QScrollArea{{border:none;background:{BG2};}}"
            f"QScrollBar:vertical{{background:{BG};width:5px;border-radius:2px;}}"
            f"QScrollBar::handle:vertical{{background:{BORDER2};border-radius:2px;}}"
            f"QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{{height:0;}}")
        scroll.setMaximumHeight(520)

        content = QWidget(); content.setStyleSheet(f"background:{BG2};")
        cl = QVBoxLayout(content)
        cl.setContentsMargins(0,4,0,4); cl.setSpacing(2)

        prev_phase = None
        for i, (key, icon, title_t, phase_color, phase_lbl, desc) in enumerate(self.STEPS):
            # Faz ayırıcı
            if phase_lbl != prev_phase:
                sep_w = QWidget(); sep_w.setStyleSheet(
                    f"background:{phase_color}20;border-top:1px solid {phase_color}40;")
                sep_w.setFixedHeight(18)
                sl = QHBoxLayout(sep_w); sl.setContentsMargins(12,0,8,0)
                sl.addWidget(QLabel(f"── {phase_lbl} ──").__class__(
                    f"<span style='color:{phase_color};font-size:8px;"
                    f"font-weight:700;letter-spacing:1px;'>"
                    f"── {phase_lbl} ──</span>"))
                sl.addStretch()
                cl.addWidget(sep_w)
                prev_phase = phase_lbl

            # Adım satırı
            step_w = QWidget()
            step_w.setStyleSheet(
                f"QWidget{{background:transparent;}}"
                f"QWidget:hover{{background:{phase_color}15;}}")
            step_w.setCursor(Qt.CursorShape.PointingHandCursor)
            step_w.setFixedHeight(52)
            sl2 = QHBoxLayout(step_w); sl2.setContentsMargins(8,4,8,4); sl2.setSpacing(8)

            # Numara balonu
            num_lbl = QLabel(f"{i+1}")
            num_lbl.setFixedSize(22,22)
            num_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            num_lbl.setStyleSheet(
                f"background:{phase_color};color:white;"
                f"border-radius:11px;font-size:9px;font-weight:700;")
            sl2.addWidget(num_lbl)

            # İkon + başlık + açıklama
            txt_w = QWidget(); txt_w.setStyleSheet("background:transparent;")
            tl = QVBoxLayout(txt_w); tl.setContentsMargins(0,0,0,0); tl.setSpacing(1)
            title_lbl = QLabel(f"{icon}  {title_t}")
            title_lbl.setStyleSheet(
                f"color:{TEXT};font-size:10px;font-weight:700;background:transparent;")
            desc_lbl = QLabel(desc)
            desc_lbl.setStyleSheet(
                f"color:{MUTED};font-size:8px;background:transparent;")
            desc_lbl.setWordWrap(True)
            tl.addWidget(title_lbl); tl.addWidget(desc_lbl)
            sl2.addWidget(txt_w, 1)

            # Ok
            arr = QLabel("›")
            arr.setStyleSheet(
                f"color:{phase_color};font-size:16px;font-weight:700;background:transparent;")
            arr.setFixedWidth(14)
            sl2.addWidget(arr)

            # Tıklama
            def _click(event, k=key):
                if event.button() == Qt.MouseButton.LeftButton:
                    self.step_clicked.emit(k)
                    self.close()
            step_w.mousePressEvent = _click
            cl.addWidget(step_w)

        scroll.setWidget(content)
        lay.addWidget(scroll)

    def showEvent(self, event):
        super().showEvent(event)
        self._position()

    def _position(self):
        if not self._anchor: return
        try:
            pos = self._anchor.mapToGlobal(self._anchor.rect().bottomLeft())
            sg  = self.screen().availableGeometry() if self.screen() else None
            self.adjustSize()
            x = pos.x()
            y = pos.y() + 4
            if sg:
                if x + self.width() > sg.right():
                    x = sg.right() - self.width() - 4
                if y + self.height() > sg.bottom():
                    y = pos.y() - self.height() - 4
            self.move(x, y)
        except Exception:
            pass

class ProcessFlyout(QFrame):
    """\n    Popup panel that appears below a toolbar button.\n    Contains all parameters for one process + Run button.\n    Closes when user clicks outside.\n    """
    closed = pyqtSignal()

    def __init__(self, panel: "ProcessPanel", anchor_btn: QPushButton,
                 parent=None, extra_panels=None):
        super().__init__(parent, Qt.WindowType.Popup)
        self._panels_list = [panel] + (extra_panels or [])
        self._panel = self._panels_list[0]
        self._anchor = anchor_btn
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            f"ProcessFlyout{{background:{BG2};border:2px solid {ACCENT};"
            f"border-radius:8px;}}")
        self.setMinimumWidth(340)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0,0,0,0); lay.setSpacing(0)

        # Header
        hdr = QWidget()
        hdr.setFixedHeight(36)
        hdr.setStyleSheet(
            f"background:{BG4};border-radius:6px 6px 0 0;"
            f"border-bottom:1px solid {BORDER};")
        hl = QHBoxLayout(hdr)
        hl.setContentsMargins(10,0,8,0); hl.setSpacing(6)
        icon_title = panel._title_lbl.text()
        title_lbl = QLabel(icon_title)
        title_lbl.setStyleSheet(
            f"color:{HEAD};font-size:12px;font-weight:700;")
        hl.addWidget(title_lbl, 1)

        close_btn = QPushButton("✕")
        close_btn.setFixedSize(22, 22)
        close_btn.setStyleSheet(
            f"QPushButton{{background:transparent;color:{SUBTEXT};"
            f"border:none;font-size:12px;}}"
            f"QPushButton:hover{{color:{RED};}}")
        close_btn.clicked.connect(self.close)
        hl.addWidget(close_btn)
        lay.addWidget(hdr)

        # Body — the panel's _body widget (parameters)
        body_scroll = QScrollArea()
        body_scroll.setWidgetResizable(True)
        body_scroll.setMaximumHeight(420)
        body_scroll.setStyleSheet(
            f"QScrollArea{{border:none;background:{BG2};}}"
            f"QScrollBar:vertical{{background:{BG};width:5px;border-radius:2px;}}"
            f"QScrollBar::handle:vertical{{background:{BORDER2};border-radius:2px;}}"
            f"QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{{height:0;}}")

        # Body — one section per panel
        body_wrap = QWidget()
        body_wrap.setStyleSheet(f"background:{BG2};")
        bwl = QVBoxLayout(body_wrap)
        bwl.setContentsMargins(0,0,0,0); bwl.setSpacing(0)

        for idx, pnl in enumerate(self._panels_list):
            # Separator between panels
            if idx > 0:
                sep = QWidget()
                sep.setFixedHeight(2)
                sep.setStyleSheet(f"background:{BORDER};")
                bwl.addWidget(sep)

                # Sub-header for additional panels
                sub_hdr = QWidget()
                sub_hdr.setFixedHeight(30)
                sub_hdr.setStyleSheet(f"background:{BG3};")
                sh_lay = QHBoxLayout(sub_hdr)
                sh_lay.setContentsMargins(10,0,10,0)
                sh_lbl = QLabel(pnl._title_lbl.text())
                sh_lbl.setStyleSheet(f"color:{GOLD};font-size:11px;font-weight:700;")
                sh_lay.addWidget(sh_lbl, 1)
                # Run butonu sub-header'a
                bwl.addWidget(sub_hdr)

            # Panel body
            pw = QWidget()
            pw.setStyleSheet("background:transparent;")
            pwl = QVBoxLayout(pw)
            pwl.setContentsMargins(10,8,10,4); pwl.setSpacing(4)
            pnl._body.setParent(pw)
            pnl._body.setStyleSheet("background:transparent;")
            pwl.addWidget(pnl._body)

            # Per-panel progress + run button
            pnl._prog_bar.setParent(pw)
            pnl._prog_lbl.setParent(pw)
            prog_row = QHBoxLayout(); prog_row.setSpacing(4)
            prog_row.addWidget(pnl._prog_bar, 1)
            prog_row.addWidget(pnl._prog_lbl)
            pwl.addLayout(prog_row)

            run_row = QHBoxLayout()
            run_row.addWidget(pnl.btn_run)
            pnl.btn_run.setFixedHeight(30)
            pnl.btn_run.setStyleSheet(
                f"QPushButton{{background:{GREEN};color:#fff;border:none;"
                f"border-radius:5px;font-size:11px;font-weight:700;min-height:30px;}}"
                f"QPushButton:hover{{background:#5ad48a;}}"
                f"QPushButton:pressed{{background:#2a9a50;}}"
                f"QPushButton:disabled{{background:{BG};color:{SUBTEXT};}}")
            pwl.addLayout(run_row)
            bwl.addWidget(pw)

        body_scroll.setWidget(body_wrap)
        lay.addWidget(body_scroll)
        self.adjustSize()

    def _position(self):
        """Position flyout below the anchor button."""
        if not self._anchor: return
        btn_global = self._anchor.mapToGlobal(
            self._anchor.rect().bottomLeft())
        self.move(btn_global.x(), btn_global.y() + 2)

    def showEvent(self, e):
        super().showEvent(e)
        self._position()

    def closeEvent(self, e):
        # Re-parent body back to panel
        panel = self._panel
        panel._body.setParent(panel)
        panel._body.setStyleSheet("background:transparent;")
        # Re-parent progress
        panel._prog_bar.setParent(panel)
        panel._prog_lbl.setParent(panel)
        self.closed.emit()
        super().closeEvent(e)


# ═══════════════════════════════ PANEL CUSTOMIZE DIALOG ═════════════════════
class PanelCustomizeDialog(QDialog):
    """\n    Reorder and show/hide process panels.\n    Drag-like Up/Down buttons to reorder, checkboxes to show/hide.\n    """
    def __init__(self, panels: dict, order: list, visible: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🔧  Customize Panels")
        self.setMinimumSize(400, 480)
        self.setStyleSheet(f"background:{BG};color:{TEXT};font-size:11px;")
        self._panels  = panels
        self._order   = list(order)
        self._visible = dict(visible)
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12,12,12,12); root.setSpacing(8)

        title = QLabel("🔧  Customize Process Panels")
        title.setStyleSheet(f"color:{ACCENT2};font-size:13px;font-weight:700;padding:4px 0 8px 0;")
        root.addWidget(title)

        info = QLabel("Check/uncheck to show or hide panels.  Use ▲▼ to reorder.")
        info.setStyleSheet(f"color:{MUTED};font-size:10px;")
        root.addWidget(info)

        # Panel list
        self._list = QListWidget()
        self._list.setStyleSheet(
            f"QListWidget{{background:{BG2};border:1px solid {BORDER};"
            f"border-radius:4px;color:{TEXT};font-size:11px;}}"
            f"QListWidget::item{{padding:6px 8px;border-bottom:1px solid {BORDER};}}"
            f"QListWidget::item:selected{{background:{BG4};color:{ACCENT2};}}")
        self._list.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self._refresh_list()
        root.addWidget(self._list, 1)

        # Up/Down buttons
        btn_row = QHBoxLayout()
        b_up   = QPushButton("▲  Move Up");   b_up.setStyleSheet(_btn()); b_up.setFixedHeight(28)
        b_down = QPushButton("▼  Move Down"); b_down.setStyleSheet(_btn()); b_down.setFixedHeight(28)
        b_all  = QPushButton("☑  Show All");  b_all.setStyleSheet(_btn(color="#0a2a0a",hover=GREEN))
        b_none = QPushButton("☐  Hide All");  b_none.setStyleSheet(_btn(color="#2a0a0a",hover=RED))
        for b in (b_up,b_down,b_all,b_none): b.setFixedHeight(28); btn_row.addWidget(b)
        root.addLayout(btn_row)

        b_up.clicked.connect(self._move_up)
        b_down.clicked.connect(self._move_down)
        b_all.clicked.connect(lambda: self._set_all(True))
        b_none.clicked.connect(lambda: self._set_all(False))

        # OK/Cancel
        ok_row = QHBoxLayout(); ok_row.addStretch()
        b_cancel = QPushButton("Cancel"); b_cancel.setStyleSheet(_btn()); b_cancel.setFixedHeight(30)
        b_ok = QPushButton("✅  Apply"); b_ok.setStyleSheet(_run_btn(ACCENT)); b_ok.setFixedHeight(32)
        ok_row.addWidget(b_cancel); ok_row.addWidget(b_ok)
        root.addLayout(ok_row)
        b_cancel.clicked.connect(self.reject)
        b_ok.clicked.connect(self._apply)

    _LABELS = {
        "bg":      "🌌  Background Extraction",
        "noise":   "✨  Noise Reduction",
        "stars":   "⭐  Star Removal",
        "deconv":  "🔭  Deconvolution",
        "sharp":   "🔪  Sharpening",
        "nebula":  "🌠  Nebula Enhancement",
        "color":   "🎨  Color Calibration",
        "morph":   "🔮  Morphology",
        "galaxy":  "🌀  Galaxy Detection",
        "stretch": "📊  Histogram Stretch",
    }

    def _refresh_list(self):
        self._list.clear()
        for key in self._order:
            lbl = self._LABELS.get(key, key)
            item = QListWidgetItem(lbl)
            item.setData(Qt.ItemDataRole.UserRole, key)
            item.setCheckState(
                Qt.CheckState.Checked if self._visible.get(key, True)
                else Qt.CheckState.Unchecked)
            self._list.addItem(item)

    def _move_up(self):
        row = self._list.currentRow()
        if row <= 0: return
        item = self._list.takeItem(row)
        self._list.insertItem(row-1, item)
        self._list.setCurrentRow(row-1)

    def _move_down(self):
        row = self._list.currentRow()
        if row < 0 or row >= self._list.count()-1: return
        item = self._list.takeItem(row)
        self._list.insertItem(row+1, item)
        self._list.setCurrentRow(row+1)

    def _set_all(self, checked: bool):
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for i in range(self._list.count()):
            self._list.item(i).setCheckState(state)

    def _apply(self):
        self._order   = []
        self._visible = {}
        for i in range(self._list.count()):
            item = self._list.item(i)
            key  = item.data(Qt.ItemDataRole.UserRole)
            self._order.append(key)
            self._visible[key] = (item.checkState() == Qt.CheckState.Checked)
        self.accept()

    def get_order(self):   return list(self._order)
    def get_visible(self): return dict(self._visible)


# ═══════════════════════════════ STACKING DIALOG ════════════════════════════
class FramePreviewDialog(QDialog):
    """Tek kare önizleme penceresi."""
    def __init__(self, path, info, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"👁  {os.path.basename(path)}")
        self.setMinimumSize(560, 520)
        self.setStyleSheet(f"background:{BG};color:{TEXT};")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8,8,8,8); lay.setSpacing(6)

        # Bilgi satırı
        info_lbl = QLabel(info)
        info_lbl.setStyleSheet(
            f"color:{MUTED};font-size:9px;background:{BG3};"
            f"border:1px solid {BORDER};border-radius:3px;padding:4px 8px;")
        info_lbl.setWordWrap(True)
        lay.addWidget(info_lbl)

        # Görüntü
        self._lbl = QLabel("Yükleniyor...")
        self._lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl.setStyleSheet(f"background:{BG2};border:1px solid {BORDER};")
        self._lbl.setMinimumSize(540, 420)
        lay.addWidget(self._lbl, 1)

        btn = QPushButton("Kapat")
        btn.setStyleSheet(_btn()); btn.setFixedHeight(28)
        btn.clicked.connect(self.close)
        lay.addWidget(btn)

        self._path = path
        QTimer.singleShot(50, self._load)

    def _load(self):
        try:
            from core.loader import load_image
            import numpy as _np
            img = load_image(self._path)
            # Auto-stretch için
            med  = float(_np.median(img))
            mad  = float(_np.median(_np.abs(img - med))) * 1.4826
            c0   = max(0.0, med - 2.8 * mad)
            if img.max() > c0 + 1e-9:
                disp = _np.clip((img - c0) / (img.max() - c0), 0, 1)
            else:
                disp = img.copy()

            # Küçük ekrana sığdır
            h, w = disp.shape[:2]
            max_w, max_h = 530, 410
            scale = min(max_w / max(w,1), max_h / max(h,1), 1.0)
            if scale < 1.0:
                import cv2 as _cv2
                disp = _cv2.resize(disp, (int(w*scale), int(h*scale)),
                                   interpolation=_cv2.INTER_AREA)

            # numpy → QPixmap
            disp8 = (_np.clip(disp, 0, 1) * 255).astype(_np.uint8)
            if disp8.ndim == 2:
                disp8 = _np.stack([disp8]*3, 2)
            h2, w2 = disp8.shape[:2]
            img_data = disp8.astype(_np.uint8).tobytes()
            qimg = QImage(img_data, w2, h2, w2*3, QImage.Format.Format_RGB888)
            pix  = QPixmap.fromImage(qimg)
            self._lbl.setPixmap(pix)
        except Exception as e:
            self._lbl.setText(f"Yüklenemedi:\n{e}")


class FrameTableWidget(QWidget):
    """
    Light frame listesi — tablo görünümü.
    Her kare: küçük thumbnail | dosya adı | pozlama | yıldız | skor | red/green durum
    Tıklanınca FramePreviewDialog açılır.
    """
    frames_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frames = []   # list of dict: {path, name, info, score, rejected}
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0,0,0,0); lay.setSpacing(3)

        # Toolbar
        tb = QHBoxLayout(); tb.setSpacing(4)
        self.btn_add  = QPushButton("+ Ekle");   self.btn_add.setStyleSheet(_btn(h=22));  self.btn_add.setFixedHeight(22)
        self.btn_del  = QPushButton("✕ Sil");    self.btn_del.setStyleSheet(_btn(color="#2a0a0a",h=22)); self.btn_del.setFixedHeight(22)
        self.btn_scan = QPushButton("📊 Analiz"); self.btn_scan.setStyleSheet(_btn(h=22));  self.btn_scan.setFixedHeight(22)
        self.btn_clr  = QPushButton("Temizle");  self.btn_clr.setStyleSheet(_btn(color="#2a0a0a",h=22)); self.btn_clr.setFixedHeight(22)
        self.lbl_cnt  = QLabel("0 kare")
        self.lbl_cnt.setStyleSheet(f"color:{MUTED};font-size:9px;")
        tb.addWidget(self.btn_add); tb.addWidget(self.btn_del)
        tb.addWidget(self.btn_scan); tb.addWidget(self.btn_clr)
        tb.addStretch(); tb.addWidget(self.lbl_cnt)
        lay.addLayout(tb)

        # Sütun başlıkları
        hdr = QWidget(); hdr.setStyleSheet(f"background:{BG3};border-bottom:1px solid {BORDER};")
        hdr_lay = QHBoxLayout(hdr); hdr_lay.setContentsMargins(4,2,4,2); hdr_lay.setSpacing(0)
        for lbl, w in [("Durum",44),("Dosya",0),("Poz.",64),("Yıldız",52),("Skor",52),("Aksiyon",52)]:
            l = QLabel(lbl); l.setStyleSheet(f"color:{SUBTEXT};font-size:8px;font-weight:700;")
            if w: l.setFixedWidth(w)
            else: l.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            hdr_lay.addWidget(l)
        lay.addWidget(hdr)

        # Kaydırılabilir liste
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            f"QScrollArea{{border:1px solid {BORDER};background:{BG2};}}"
            f"QScrollBar:vertical{{background:{BG};width:6px;border-radius:3px;}}"
            f"QScrollBar::handle:vertical{{background:{BORDER2};border-radius:3px;}}"
            f"QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{{height:0;}}")
        self._list_w = QWidget()
        self._list_w.setStyleSheet(f"background:{BG2};")
        self._list_lay = QVBoxLayout(self._list_w)
        self._list_lay.setContentsMargins(0,0,0,0); self._list_lay.setSpacing(0)
        self._list_lay.addStretch()
        scroll.setWidget(self._list_w)
        lay.addWidget(scroll, 1)

        self.btn_add.clicked.connect(self._add_files)
        self.btn_del.clicked.connect(self._del_rejected)
        self.btn_scan.clicked.connect(self._scan_all)
        self.btn_clr.clicked.connect(self._clear)

    def _add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Light karelerini seç", "",
            "Görseller (*.fits *.fit *.fts *.tif *.tiff *.png *.jpg *.jpeg *.bmp);;Tümü (*)")
        for p in paths:
            if not any(f["path"] == p for f in self._frames):
                info = self._read_meta(p)
                self._frames.append({"path":p, "name":os.path.basename(p),
                                      "info":info, "score":-1, "rejected":False})
        self._rebuild_rows()
        self.frames_changed.emit()

    def _read_meta(self, path):
        """FITS header veya dosya özelliklerinden metadata oku."""
        info = {"exposure": "?", "stars": "?", "iso": "", "date": ""}
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext in (".fits", ".fit", ".fts"):
                try:
                    from astropy.io import fits as _fits
                    with _fits.open(path, memmap=False) as hdul:
                        hdr = hdul[0].header
                        exp = hdr.get("EXPTIME") or hdr.get("EXPOSURE") or hdr.get("EXP_TIME")
                        iso = hdr.get("ISO") or hdr.get("GAIN") or ""
                        dat = str(hdr.get("DATE-OBS",""))[:10]
                        info["exposure"] = f"{float(exp):.1f}s" if exp else "?"
                        info["iso"]  = str(iso)
                        info["date"] = dat
                except Exception:
                    pass
            else:
                # PNG/TIFF/JPG — dosya boyutundan tahmin
                size_kb = os.path.getsize(path) // 1024
                info["exposure"] = f"~{size_kb}KB"
        except Exception:
            pass
        return info

    def _rebuild_rows(self):
        """Tüm satırları yeniden oluştur."""
        # Mevcut widget'ları kaldır
        while self._list_lay.count() > 1:
            item = self._list_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for i, fr in enumerate(self._frames):
            row = self._make_row(i, fr)
            self._list_lay.insertWidget(i, row)

        self.lbl_cnt.setText(f"{len(self._frames)} kare")

    def _make_row(self, idx, fr):
        """Tek kare satırı widget'ı."""
        rejected = fr.get("rejected", False)
        bg = "#1a0808" if rejected else (BG3 if idx % 2 == 0 else BG2)
        row = QWidget()
        row.setStyleSheet(
            f"QWidget#frame_row{{background:{bg};}}"
            f"QWidget#frame_row:hover{{background:#1a2a1a;}}")
        row.setObjectName("frame_row")
        row.setFixedHeight(32)
        rl = QHBoxLayout(row); rl.setContentsMargins(4,0,4,0); rl.setSpacing(4)

        # Durum göstergesi
        score = fr.get("score", -1)
        if score < 0:
            status_clr = SUBTEXT; status_txt = "●"
        elif rejected:
            status_clr = RED;    status_txt = "✕"
        elif score > 0.7:
            status_clr = GREEN;  status_txt = "✓"
        elif score > 0.4:
            status_clr = GOLD;   status_txt = "~"
        else:
            status_clr = RED;    status_txt = "✕"

        status_lbl = QLabel(status_txt)
        status_lbl.setStyleSheet(f"color:{status_clr};font-size:13px;font-weight:700;")
        status_lbl.setFixedWidth(22); status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Dosya adı
        name_lbl = QLabel(fr["name"])
        name_lbl.setStyleSheet(
            f"color:{'#666' if rejected else TEXT};"
            f"font-size:9px;{'text-decoration:line-through;' if rejected else ''}")
        name_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        name_lbl.setToolTip(fr["path"])

        # Meta bilgiler
        meta = fr["info"]
        exp_lbl   = QLabel(meta.get("exposure","?"))
        stars_lbl = QLabel(str(fr.get("stars","?")))
        score_lbl = QLabel(f"{score:.2f}" if score >= 0 else "—")
        for lbl, w in [(exp_lbl,64),(stars_lbl,52),(score_lbl,52)]:
            lbl.setStyleSheet(f"color:{MUTED};font-size:9px;")
            lbl.setFixedWidth(w); lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Reddet/Kabul butonu
        tog_btn = QPushButton("✕" if not rejected else "↩")
        tog_btn.setFixedSize(32, 22)
        tog_btn.setStyleSheet(
            f"QPushButton{{background:{'#2a0a0a' if not rejected else '#0a2a0a'};"
            f"color:{'#ff6666' if not rejected else '#66ff66'};"
            f"border:1px solid {BORDER};border-radius:3px;font-size:10px;}}"
            f"QPushButton:hover{{background:{'#3a1010' if not rejected else '#1a3a1a'}}}")
        tog_btn.setToolTip("Reddet / Geri Al")

        rl.addWidget(status_lbl)
        rl.addWidget(name_lbl, 1)
        rl.addWidget(exp_lbl)
        rl.addWidget(stars_lbl)
        rl.addWidget(score_lbl)
        rl.addWidget(tog_btn)

        # Tıklama — önizleme
        def _open_preview(event, i=idx):
            if event.button() == Qt.MouseButton.LeftButton:
                fr = self._frames[i]
                info_str = (f"Dosya: {fr['name']}  |  "
                            f"Pozlama: {fr['info'].get('exposure','?')}  |  "
                            f"ISO/Gain: {fr['info'].get('iso','')}  |  "
                            f"Yıldız: {fr.get('stars','?')}  |  "
                            f"Skor: {fr.get('score',-1):.3f}"
                            if fr.get('score',-1) >= 0
                            else f"Dosya: {fr['name']}  |  Pozlama: {fr['info'].get('exposure','?')}")
                dlg = FramePreviewDialog(fr["path"], info_str, self.window())
                dlg.show()

        def _toggle(i=idx):
            self._frames[i]["rejected"] = not self._frames[i].get("rejected", False)
            self._rebuild_rows()
            self.frames_changed.emit()

        row.mousePressEvent  = _open_preview
        name_lbl.mousePressEvent = _open_preview
        tog_btn.clicked.connect(_toggle)

        return row

    def _scan_all(self):
        """Tüm light karelerini analiz et (arka planda)."""
        from processing.stacking import score_frame
        from core.loader import load_image
        self.btn_scan.setText("⏳ Analiz...")
        self.btn_scan.setEnabled(False)
        total = len(self._frames)

        for i, fr in enumerate(self._frames):
            try:
                img = load_image(fr["path"])
                sc  = score_frame(img)
                self._frames[i]["score"]   = sc["score"]
                self._frames[i]["stars"]   = sc["star_count"]
                self._frames[i]["fwhm"]    = sc["fwhm"]
                self._frames[i]["snr"]     = sc["snr"]
                # Düşük skor → auto-reject
                if sc["score"] < 0.15:
                    self._frames[i]["rejected"] = True
            except Exception:
                pass

        self._rebuild_rows()
        self.frames_changed.emit()
        self.btn_scan.setText("📊 Analiz")
        self.btn_scan.setEnabled(True)

    def _del_rejected(self):
        """Reddedilmiş kareleri listeden kaldır."""
        self._frames = [f for f in self._frames if not f.get("rejected", False)]
        self._rebuild_rows()
        self.frames_changed.emit()

    def _clear(self):
        self._frames.clear()
        self._rebuild_rows()
        self.frames_changed.emit()

    def get_accepted_paths(self):
        return [f["path"] for f in self._frames if not f.get("rejected", False)]

    def get_all_paths(self):
        return [f["path"] for f in self._frames]

    def count(self):
        return len(self._frames)


class StackingDialog(QDialog):
    """
    DeepSkyStacker-style stacking dialog.
    Light kareleri tablo görünümü + metadata + tıklayınca preview.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🗂  Image Stacking — DeepSkyStacker Style")
        self.setMinimumSize(1100, 820)
        self.setStyleSheet(f"background:{BG};color:{TEXT};font-size:11px;")
        self._lists   = {}
        self._worker  = None
        self._result  = None
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10,10,10,10); root.setSpacing(6)

        # Başlık
        hdr = QHBoxLayout()
        title = QLabel("🗂  IMAGE STACKING")
        title.setStyleSheet(f"color:{ACCENT2};font-size:14px;font-weight:700;letter-spacing:1px;")
        hdr.addWidget(title); hdr.addStretch()
        hdr.addWidget(QLabel("DeepSkyStacker-style | kareye tıkla = önizleme")
                      .__class__(f"<span style='color:{SUBTEXT};font-size:9px;'>"
                                 f"DeepSkyStacker-style | kareye tıkla = önizleme</span>"))
        root.addLayout(hdr)

        # Ana splitter
        main_split = QSplitter(Qt.Orientation.Horizontal)
        main_split.setStyleSheet(f"QSplitter::handle{{background:{BORDER};width:3px;}}")

        # ── Sol: Light tablo + kalibrasyon kareleri ──────────────────────
        left = QWidget()
        ll = QVBoxLayout(left); ll.setContentsMargins(0,0,4,0); ll.setSpacing(6)

        # Light frame tablo (büyük, genişleyen)
        light_grp = QGroupBox("💡 Light Frames")
        light_grp.setStyleSheet(
            f"QGroupBox{{background:{BG3};border:1px solid {ACCENT2};"
            f"border-radius:5px;margin-top:14px;padding:4px;}}"
            f"QGroupBox::title{{color:{ACCENT2};font-size:10px;font-weight:700;"
            f"subcontrol-origin:margin;left:8px;padding:0 4px;}}")
        lg_lay = QVBoxLayout(light_grp); lg_lay.setContentsMargins(4,4,4,4)
        self._frame_table = FrameTableWidget()
        self._frame_table.setMinimumHeight(280)
        self._frame_table.frames_changed.connect(self._update_summary)
        lg_lay.addWidget(self._frame_table)
        ll.addWidget(light_grp, 1)

        # Kalibrasyon kareleri — kompakt
        calib_lbl = QLabel("KALİBRASYON KARELERİ")
        calib_lbl.setStyleSheet(f"color:{SUBTEXT};font-size:8px;font-weight:700;letter-spacing:1px;")
        ll.addWidget(calib_lbl)

        calib_grid = QWidget()
        cg_lay = QGridLayout(calib_grid); cg_lay.setSpacing(4)
        for (key,lbl_text,color), (row,col) in zip([
            ("dark",     "🌑 Dark",      "#888899"),
            ("flat",     "⬜ Flat",      GOLD),
            ("dark_flat","🔲 Dark Flat", SUBTEXT),
            ("bias",     "⬛ Bias",      MUTED),
        ], [(0,0),(0,1),(1,0),(1,1)]):
            cg_lay.addWidget(self._make_calib_group(key, lbl_text, color), row, col)
        ll.addWidget(calib_grid)

        # Özet
        self._lbl_summary = QLabel("0 light kare seçildi")
        self._lbl_summary.setStyleSheet(
            f"color:{MUTED};font-size:9px;background:{BG3};"
            f"border:1px solid {BORDER};border-radius:3px;padding:4px 8px;")
        ll.addWidget(self._lbl_summary)
        main_split.addWidget(left)

        # ── Sağ: Ayarlar paneli ───────────────────────────────────────────
        right = QWidget()
        rl = QVBoxLayout(right); rl.setContentsMargins(4,0,0,0); rl.setSpacing(6)

        settings_lbl = QLabel("STACKING AYARLARI")
        settings_lbl.setStyleSheet(f"color:{SUBTEXT};font-size:8px;font-weight:700;letter-spacing:1px;")
        rl.addWidget(settings_lbl)

        tabs = QTabWidget()
        tabs.setStyleSheet(
            f"QTabWidget::pane{{background:{BG2};border:1px solid {BORDER};}}"
            f"QTabBar::tab{{background:{BG3};color:{MUTED};padding:5px 10px;font-size:10px;}}"
            f"QTabBar::tab:selected{{background:{BG2};color:{ACCENT2};}}")
        tabs.addTab(self._tab_stacking(), "⚙ Stacking")
        tabs.addTab(self._tab_alignment(), "🎯 Hizalama")
        tabs.addTab(self._tab_quality(), "📊 Kalite")
        tabs.addTab(self._tab_advanced(), "🔬 İleri")
        rl.addWidget(tabs, 1)
        main_split.addWidget(right)
        main_split.setSizes([680, 380])
        root.addWidget(main_split, 1)

        # Log
        self.log = QTextEdit()
        self.log.setReadOnly(True); self.log.setFixedHeight(90)
        self.log.setStyleSheet(
            f"QTextEdit{{background:{BG2};color:{GREEN};border:1px solid {BORDER};"
            f"font-size:10px;font-family:Consolas,monospace;padding:4px;}}")
        root.addWidget(self.log)

        # Progress bar (stacking için)
        pbar_row = QHBoxLayout(); pbar_row.setSpacing(8)
        self._stack_pbar = QProgressBar()
        self._stack_pbar.setRange(0, 100); self._stack_pbar.setValue(0)
        self._stack_pbar.setTextVisible(True)
        self._stack_pbar.setFormat("Hazır")
        self._stack_pbar.setFixedHeight(18)
        self._stack_pbar.setStyleSheet(
            f"QProgressBar{{background:{BG3};border:1px solid {BORDER};"
            f"border-radius:4px;color:{TEXT};font-size:9px;text-align:center;}}"
            f"QProgressBar::chunk{{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            f"stop:0 {PURPLE},stop:1 {ACCENT2});border-radius:3px;}}")
        pbar_row.addWidget(self._stack_pbar, 1)
        self._pbar_lbl = QLabel(""); 
        self._pbar_lbl.setStyleSheet(f"color:{MUTED};font-size:9px;min-width:60px;")
        pbar_row.addWidget(self._pbar_lbl)
        root.addLayout(pbar_row)

        # Butonlar
        btn_row = QHBoxLayout(); btn_row.setSpacing(8)
        self._lbl_progress = QLabel("")
        self._lbl_progress.setStyleSheet(f"color:{MUTED};font-size:9px;")
        btn_row.addWidget(self._lbl_progress, 1)
        self.btn_cancel = QPushButton("Kapat")
        self.btn_cancel.setStyleSheet(_btn()); self.btn_cancel.setFixedHeight(32)
        self.btn_stack  = QPushButton("🚀  Stacking Başlat")
        self.btn_stack.setStyleSheet(_run_btn(PURPLE))
        self.btn_stack.setFixedHeight(34); self.btn_stack.setMinimumWidth(180)
        self.btn_stack.setEnabled(False)
        btn_row.addWidget(self.btn_cancel); btn_row.addWidget(self.btn_stack)
        root.addLayout(btn_row)

        self.btn_stack.clicked.connect(self._start_stack)
        self.btn_cancel.clicked.connect(self.reject)

    # ── Kalibrasyon grubu (kompakt) ──────────────────────────────────────────
    def _make_calib_group(self, key, lbl_text, color):
        g = QGroupBox(); g.setTitle(lbl_text)
        g.setStyleSheet(
            f"QGroupBox{{background:{BG2};border:1px solid {BORDER};"
            f"border-radius:4px;margin-top:12px;padding:2px;}}"
            f"QGroupBox::title{{color:{color};font-size:9px;font-weight:700;"
            f"subcontrol-origin:margin;left:6px;padding:0 3px;}}")
        lay = QVBoxLayout(g); lay.setContentsMargins(3,3,3,3); lay.setSpacing(2)
        lst = QListWidget()
        lst.setStyleSheet(
            f"QListWidget{{background:{BG};border:none;color:{TEXT};font-size:8px;}}"
            f"QListWidget::item{{padding:1px 3px;border-bottom:1px solid {BORDER};}}"
            f"QListWidget::item:selected{{background:{BG4};}}")
        lst.setFixedHeight(55)
        lst.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self._lists[key] = lst; lay.addWidget(lst)
        row = QHBoxLayout(); row.setSpacing(3)
        b_add = QPushButton("+"); b_add.setStyleSheet(_btn(h=18)); b_add.setFixedSize(28,18)
        b_clr = QPushButton("✕"); b_clr.setStyleSheet(_btn(color="#2a0a0a",h=18)); b_clr.setFixedSize(28,18)
        cnt = QLabel("0"); cnt.setStyleSheet(f"color:{MUTED};font-size:8px;")
        self._lists[key+"_cnt"] = cnt
        b_add.clicked.connect(lambda _, k=key: self._add_calib(k))
        b_clr.clicked.connect(lambda _, k=key: (self._lists[k].clear(), self._update_cnt(k)))
        row.addWidget(b_add); row.addWidget(b_clr); row.addStretch(); row.addWidget(cnt)
        lay.addLayout(row)
        return g

    def _add_calib(self, key):
        paths, _ = QFileDialog.getOpenFileNames(
            self, f"{key} kareleri seç", "",
            "Görseller (*.fits *.fit *.fts *.tif *.tiff *.png *.jpg *.jpeg *.bmp);;Tümü (*)")
        lst = self._lists[key]
        for p in paths:
            item = QListWidgetItem(os.path.basename(p))
            item.setData(Qt.ItemDataRole.UserRole, p)
            item.setToolTip(p); lst.addItem(item)
        self._update_cnt(key)

    def _update_cnt(self, key):
        c = self._lists.get(key+"_cnt")
        if c: c.setText(str(self._lists[key].count()))

    def _update_summary(self):
        n = self._frame_table.count()
        acc = len(self._frame_table.get_accepted_paths())
        rej = n - acc
        n_dark = self._lists["dark"].count()
        n_flat = self._lists["flat"].count()
        n_bias = self._lists["bias"].count()
        self._lbl_summary.setText(
            f"{acc} light (✓)  |  {rej} reddedildi  |  "
            f"{n_dark} dark  |  {n_flat} flat  |  {n_bias} bias")
        self.btn_stack.setEnabled(acc > 0)

    def _get_calib_paths(self, key):
        lst = self._lists[key]
        return [lst.item(i).data(Qt.ItemDataRole.UserRole) for i in range(lst.count())] or None

    # ── Tab: Stacking ─────────────────────────────────────────────────────────
    def _tab_stacking(self):
        w = QWidget(); w.setStyleSheet(f"background:{BG2};")
        lay = QVBoxLayout(w); lay.setContentsMargins(12,12,12,12); lay.setSpacing(8)
        def _row(label, widget, tip=""):
            r = QHBoxLayout(); r.setSpacing(8)
            l = QLabel(label); l.setStyleSheet(LBL_CSS); l.setFixedWidth(140)
            r.addWidget(l); r.addWidget(widget)
            if tip:
                ht = QLabel(tip); ht.setStyleSheet(f"color:{SUBTEXT};font-size:9px;")
                r.addWidget(ht)
            r.addStretch(); lay.addLayout(r)
        self.combo_method = QComboBox()
        self.combo_method.addItems(["kappa_sigma","winsorized","median","mean",
                                    "linear_fit","entropy_weighted","maximum","minimum","sum"])
        self.combo_method.setStyleSheet(COMBO_CSS); self.combo_method.setFixedWidth(160)
        _row("Stacking Metodu:", self.combo_method)
        self.spin_kappa = QDoubleSpinBox()
        self.spin_kappa.setRange(1.0,5.0); self.spin_kappa.setValue(2.0)
        self.spin_kappa.setDecimals(1); self.spin_kappa.setSingleStep(0.5)
        self.spin_kappa.setFixedWidth(70); self.spin_kappa.setStyleSheet(SPIN_CSS)
        _row("Kappa:", self.spin_kappa, "DSS: 2.0")
        kw_row = QHBoxLayout(); kw_row.setSpacing(6)
        l = QLabel("Kappa Low/High:"); l.setStyleSheet(LBL_CSS); l.setFixedWidth(140)
        self.spin_kappa_low  = QDoubleSpinBox(); self.spin_kappa_high = QDoubleSpinBox()
        for sp in (self.spin_kappa_low, self.spin_kappa_high):
            sp.setRange(1.0,5.0); sp.setValue(2.0); sp.setDecimals(1)
            sp.setSingleStep(0.5); sp.setFixedWidth(65); sp.setStyleSheet(SPIN_CSS)
        kw_row.addWidget(l); kw_row.addWidget(self.spin_kappa_low)
        kw_row.addWidget(QLabel("/")); kw_row.addWidget(self.spin_kappa_high)
        kw_row.addStretch(); lay.addLayout(kw_row)
        self.spin_iter = QSpinBox()
        self.spin_iter.setRange(1,10); self.spin_iter.setValue(5)
        self.spin_iter.setFixedWidth(60); self.spin_iter.setStyleSheet(SPIN_CSS)
        _row("İterasyon:", self.spin_iter, "DSS: 5")
        lay.addWidget(self._sep())
        self.chk_normalize = QCheckBox("Kareleri normalize et (median eşitleme)")
        self.chk_normalize.setChecked(True); self.chk_normalize.setStyleSheet(CHECK_CSS)
        lay.addWidget(self.chk_normalize)
        self.combo_weight = QComboBox()
        self.combo_weight.addItems(["none","snr","fwhm","stars"])
        self.combo_weight.setStyleSheet(COMBO_CSS); self.combo_weight.setFixedWidth(120)
        _row("Kare Ağırlıklandırma:", self.combo_weight)
        lay.addStretch(); return w

    # ── Tab: Hizalama ─────────────────────────────────────────────────────────
    def _tab_alignment(self):
        w = QWidget(); w.setStyleSheet(f"background:{BG2};")
        lay = QVBoxLayout(w); lay.setContentsMargins(12,12,12,12); lay.setSpacing(8)
        def _row(label, widget, tip=""):
            r = QHBoxLayout(); r.setSpacing(8)
            l = QLabel(label); l.setStyleSheet(LBL_CSS); l.setFixedWidth(140)
            r.addWidget(l); r.addWidget(widget)
            if tip:
                ht = QLabel(tip); ht.setStyleSheet(f"color:{SUBTEXT};font-size:9px;")
                r.addWidget(ht)
            r.addStretch(); lay.addLayout(r)

        # Hizalama metodu
        self.combo_align = QComboBox()
        self.combo_align.addItems(["ecc_euclidean","ecc_translation","ecc_affine",
                                   "orb_homography","sift_homography","star_match","none"])
        self.combo_align.setStyleSheet(COMBO_CSS); self.combo_align.setFixedWidth(160)
        self.combo_align.setToolTip(
            "ecc_euclidean  — ECC öteleme+dönme (önerilen)\n"
            "ecc_translation— ECC sadece öteleme (hızlı)\n"
            "ecc_affine     — ECC tam affine\n"
            "orb_homography — ORB özellik eşleme\n"
            "sift_homography— SIFT özellik eşleme\n"
            "star_match     — Yıldız koordinat eşleme\n"
            "none           — Hizalama yok")
        _row("Hizalama Metodu:", self.combo_align)

        lay.addWidget(self._sep())

        # Referans kare
        self.combo_ref_mode = QComboBox()
        self.combo_ref_mode.addItems(["best_quality","median_quality","manual"])
        self.combo_ref_mode.setStyleSheet(COMBO_CSS); self.combo_ref_mode.setFixedWidth(160)
        self.combo_ref_mode.currentTextChanged.connect(self._on_ref_mode_change)
        _row("Referans Mod:", self.combo_ref_mode)
        self.spin_ref = QSpinBox()
        self.spin_ref.setRange(1,999); self.spin_ref.setValue(1)
        self.spin_ref.setFixedWidth(70); self.spin_ref.setStyleSheet(SPIN_CSS)
        self.spin_ref.setEnabled(False)
        _row("Referans Kare #:", self.spin_ref, "(manuel mod)")

        lay.addWidget(self._sep())

        # Durum göstergesi
        self._align_status = QLabel("⬜  Henüz hizalanmadı")
        self._align_status.setStyleSheet(
            f"color:{MUTED};font-size:10px;background:{BG3};"
            f"border:1px solid {BORDER};border-radius:4px;padding:6px 10px;")
        self._align_status.setWordWrap(True)
        lay.addWidget(self._align_status)

        # Hizalama butonu
        self.btn_align = QPushButton("🎯  Hizalamayı Çalıştır")
        self.btn_align.setStyleSheet(_run_btn(ACCENT))
        self.btn_align.setFixedHeight(36)
        self.btn_align.setToolTip("Önce hizalama yap, sonra Stacking Başlat ile stack'le")
        self.btn_align.clicked.connect(self._run_alignment)
        lay.addWidget(self.btn_align)

        # Hizalanmış kare önizleme listesi
        self._aligned_list_lbl = QLabel("Hizalanmış Kareler:")
        self._aligned_list_lbl.setStyleSheet(f"color:{SUBTEXT};font-size:9px;font-weight:700;")
        self._aligned_list_lbl.setVisible(False)
        lay.addWidget(self._aligned_list_lbl)

        self._aligned_list = QListWidget()
        self._aligned_list.setStyleSheet(
            f"QListWidget{{background:{BG};color:{TEXT};border:1px solid {BORDER};"
            f"font-size:9px;}}"
            f"QListWidget::item{{padding:2px 4px;border-bottom:1px solid {BORDER};}}"
            f"QListWidget::item:selected{{background:{BG4};}}")
        self._aligned_list.setFixedHeight(120)
        self._aligned_list.setVisible(False)
        lay.addWidget(self._aligned_list)

        # Hizalanmış kareleri sakla
        self._aligned_frames = None   # list of np.ndarray

        lay.addStretch(); return w

    # ── Tab: Kalite ───────────────────────────────────────────────────────────
    def _tab_quality(self):
        w = QWidget(); w.setStyleSheet(f"background:{BG2};")
        lay = QVBoxLayout(w); lay.setContentsMargins(12,12,12,12); lay.setSpacing(8)
        info = QLabel("Kalite Skorlama: Her kare FWHM, SNR ve yıldız sayısına göre puanlanır.\n"
                      "Kırmızı durum = düşük kalite, ✕ butonu ile manuel reddet.")
        info.setStyleSheet(
            f"color:{MUTED};font-size:9px;background:{BG3};"
            f"border:1px solid {BORDER};border-radius:4px;padding:8px;")
        info.setWordWrap(True); lay.addWidget(info)
        self.chk_quality_reject = QCheckBox("Kötü kareleri otomatik reddet")
        self.chk_quality_reject.setChecked(True)
        self.chk_quality_reject.setStyleSheet(CHECK_CSS); lay.addWidget(self.chk_quality_reject)
        thr_row = QHBoxLayout(); thr_row.setSpacing(8)
        tl = QLabel("Kalite Eşiği:"); tl.setStyleSheet(LBL_CSS); tl.setFixedWidth(140)
        self.spin_quality_thr = QDoubleSpinBox()
        self.spin_quality_thr.setRange(0.0,0.9); self.spin_quality_thr.setValue(0.2)
        self.spin_quality_thr.setDecimals(2); self.spin_quality_thr.setSingleStep(0.05)
        self.spin_quality_thr.setFixedWidth(70); self.spin_quality_thr.setStyleSheet(SPIN_CSS)
        thr_row.addWidget(tl); thr_row.addWidget(self.spin_quality_thr)
        thr_row.addWidget(QLabel("(0=hepsi  0.9=sadece en iyiler)"))
        thr_row.addStretch(); lay.addLayout(thr_row)
        lay.addWidget(self._sep())
        self._score_table = QTextEdit(); self._score_table.setReadOnly(True)
        self._score_table.setStyleSheet(
            f"background:{BG};color:{TEXT};border:1px solid {BORDER};"
            f"font-family:Consolas,monospace;font-size:9px;")
        lay.addWidget(self._score_table, 1); return w

    # ── Tab: İleri ────────────────────────────────────────────────────────────
    def _tab_advanced(self):
        w = QWidget(); w.setStyleSheet(f"background:{BG2};")
        lay = QVBoxLayout(w); lay.setContentsMargins(12,12,12,12); lay.setSpacing(8)
        self.chk_save_masters = QCheckBox("Master kareleri kaydet (dark/flat/bias)")
        self.chk_save_masters.setChecked(False); self.chk_save_masters.setStyleSheet(CHECK_CSS)
        lay.addWidget(self.chk_save_masters)
        save_row = QHBoxLayout()
        self.edit_save_dir = QLineEdit(); self.edit_save_dir.setPlaceholderText("Master kayıt klasörü...")
        self.edit_save_dir.setStyleSheet(
            f"QLineEdit{{background:{BG};color:{TEXT};border:1px solid {BORDER};"
            f"border-radius:3px;padding:3px 6px;font-size:10px;}}")
        b_browse = QPushButton("📂"); b_browse.setFixedSize(28,24); b_browse.setStyleSheet(_btn(h=24))
        b_browse.clicked.connect(lambda: self.edit_save_dir.setText(
            QFileDialog.getExistingDirectory(self,"Master kayıt klasörü seç") or self.edit_save_dir.text()))
        save_row.addWidget(self.edit_save_dir); save_row.addWidget(b_browse)
        lay.addLayout(save_row); lay.addStretch(); return w

    def _sep(self):
        line = QFrame(); line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet(f"color:{BORDER};background:{BORDER};max-height:1px;"); return line

    def _on_ref_mode_change(self, mode):
        self.spin_ref.setEnabled(mode == "manual")

    def _log(self, msg):
        self.log.append(msg)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())
        self._lbl_progress.setText(msg.split("\\n")[0][:80])

    def _run_alignment(self):
        """Faz 1: Sadece hizalama — kareleri hizala, hafızada tut."""
        lights = self._frame_table.get_accepted_paths()
        if not lights:
            QMessageBox.warning(self,"Uyarı","Önce Light karelerini ekleyin."); return
        if self.combo_align.currentText() == "none":
            QMessageBox.information(self,"Bilgi",
                "Hizalama metodu 'none' seçili.\n"
                "Hizalama atlanıyor — direkt stacking yapabilirsiniz."); return

        self.btn_align.setEnabled(False)
        self.btn_align.setText("⏳ Hizalanıyor…")
        self.btn_stack.setEnabled(False)
        self._align_status.setText("⏳  Hizalama çalışıyor…")
        self._align_status.setStyleSheet(
            f"color:{GOLD};font-size:10px;background:{BG3};"
            f"border:1px solid {GOLD};border-radius:4px;padding:6px 10px;")
        self._aligned_frames = None
        self.log.clear()
        self._log(f"🎯 Hizalama başlıyor — {len(lights)} kare, "
                  f"metot: {self.combo_align.currentText()}")

        params = {
            "light_paths":   lights,
            "dark_paths":    self._get_calib_paths("dark"),
            "flat_paths":    self._get_calib_paths("flat"),
            "dark_flat_paths": self._get_calib_paths("dark_flat"),
            "bias_paths":    self._get_calib_paths("bias"),
            "align_method":  self.combo_align.currentText(),
            "ref_index":     int(self.spin_ref.value()) - 1,
            "ref_mode":      self.combo_ref_mode.currentText(),
            "normalize":     self.chk_normalize.isChecked(),
            "quality_reject":self.chk_quality_reject.isChecked(),
            "quality_threshold": float(self.spin_quality_thr.value()),
        }
        self._worker = StackWorker(params, mode="align")
        self._worker.progress.connect(self._log)
        self._worker.finished.connect(self._on_align_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_align_done(self, result):
        """Faz 1 tamamlandı — hizalanmış kareler hazır."""
        aligned = result.get("aligned", [])
        self._aligned_frames = aligned
        n = len(aligned)

        self.btn_align.setEnabled(True)
        self.btn_align.setText("🔄 Tekrar Hizala")
        self.btn_stack.setEnabled(n > 0)
        self.btn_stack.setText("🚀  Stacking Başlat")
        # Pbar'ı hizalama tamamlandı olarak göster
        self._stack_pbar.setValue(100)
        self._stack_pbar.setFormat(f"✅  {n} kare hizalandı — stacking için hazır")
        self._pbar_lbl.setText("✓")

        self._align_status.setText(
            f"✅  {n} kare hizalandı  —  stacking için hazır")
        self._align_status.setStyleSheet(
            f"color:{GREEN};font-size:10px;background:{BG3};"
            f"border:1px solid {GREEN};border-radius:4px;padding:6px 10px;")

        # Hizalanan kareleri listele
        self._aligned_list.clear()
        for i, fr_arr in enumerate(aligned):
            lbl = f"#{i+1}  {fr_arr.shape[1]}×{fr_arr.shape[0]}  ✓ hizalandı"
            self._aligned_list.addItem(lbl)
        self._aligned_list_lbl.setVisible(True)
        self._aligned_list.setVisible(True)

        self._log(f"✅ Hizalama tamamlandı — {n} kare hazır. "
                  f"'Stacking Başlat' butonuna basın.")

    def _start_stack(self):
        """Faz 2: Stacking — hizalanmış kareler varsa onları kullan."""
        lights = self._frame_table.get_accepted_paths()
        if not lights:
            QMessageBox.warning(self,"Uyarı","En az 1 kabul edilmiş Light karesi gerekli."); return
        self.spin_ref.setMaximum(len(lights))

        self.btn_stack.setEnabled(False); self.btn_stack.setText("⏳ Stack yapılıyor…")
        self._stack_pbar.setValue(0); self._stack_pbar.setFormat("Başlatılıyor…")
        self._pbar_lbl.setText("0%")
        self.log.clear()

        # Hizalanmış kareler varsa onları kullan (faz 2)
        if self._aligned_frames and len(self._aligned_frames) > 0:
            self._log(f"▶ Stacking (hizalanmış {len(self._aligned_frames)} kare) "
                      f"— {self.combo_method.currentText()}")
            params = {
                "aligned_frames":    self._aligned_frames,
                "method":            self.combo_method.currentText(),
                "kappa":             float(self.spin_kappa.value()),
                "kappa_low":         float(self.spin_kappa_low.value()),
                "kappa_high":        float(self.spin_kappa_high.value()),
                "iterations":        int(self.spin_iter.value()),
                "weight_mode":       self.combo_weight.currentText(),
                "quality_reject":    self.chk_quality_reject.isChecked(),
                "quality_threshold": float(self.spin_quality_thr.value()),
            }
            self._worker = StackWorker(params, mode="stack")
        else:
            # Hizalama yapılmamış — tam pipeline
            self._log(f"▶ Tam pipeline (hizalama + stacking) — "
                      f"{len(lights)} light, {self.combo_method.currentText()}")
            self._align_status.setText(
                "⚠  Önce hizalama yapılması önerilir. Tam pipeline çalışıyor…")
            self._align_status.setStyleSheet(
                f"color:{GOLD};font-size:10px;background:{BG3};"
                f"border:1px solid {GOLD};border-radius:4px;padding:6px 10px;")
            params = {
                "light_paths":       lights,
                "dark_paths":        self._get_calib_paths("dark"),
                "flat_paths":        self._get_calib_paths("flat"),
                "dark_flat_paths":   self._get_calib_paths("dark_flat"),
                "bias_paths":        self._get_calib_paths("bias"),
                "method":            self.combo_method.currentText(),
                "align_method":      self.combo_align.currentText(),
                "ref_index":         int(self.spin_ref.value()) - 1,
                "ref_mode":          self.combo_ref_mode.currentText(),
                "kappa":             float(self.spin_kappa.value()),
                "kappa_low":         float(self.spin_kappa_low.value()),
                "kappa_high":        float(self.spin_kappa_high.value()),
                "iterations":        int(self.spin_iter.value()),
                "quality_reject":    self.chk_quality_reject.isChecked(),
                "quality_threshold": float(self.spin_quality_thr.value()),
                "normalize":         self.chk_normalize.isChecked(),
                "weight_mode":       self.combo_weight.currentText(),
            }
            self._worker = StackWorker(params, mode="full")

        self._score_table.clear()
        self._worker.progress.connect(self._log)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_done(self, result):
        self._result = result
        n  = result.get("n_lights", 0)
        nr = result.get("n_rejected", 0)

        # Progress bar tamamlandı
        self._stack_pbar.setValue(100)
        self._stack_pbar.setFormat("✅  Tamamlandı!")
        self._pbar_lbl.setText("100%")

        self._log(f"✅ Tamamlandı — {n} kare birleştirildi, {nr} reddedildi")
        self.btn_stack.setEnabled(True); self.btn_stack.setText("🔄 Tekrar Çalıştır")

        # Kalite tablosunu doldur
        scores = result.get("frame_scores", [])
        if scores:
            lines = ["Kare  Skor   FWHM   SNR    Yıldız  Durum", "-"*50]
            rejected = set(result.get("rejected_frames", []))
            for i, sc in enumerate(scores):
                durum = "REDDEDİLDİ" if i in rejected else "OK"
                lines.append(f"#{i+1:<4} {sc['score']:.3f}  {sc['fwhm']:5.1f}px  "
                             f"{sc['snr']:5.1f}   {sc['star_count']:<6}  {durum}")
            self._score_table.setPlainText("\n".join(lines))

        # Sonucu light klasörüne kaydet
        img = result.get("result")
        if img is not None:
            self._save_result(img, n, result.get("method", "stack"))

        self._close_timer = QTimer(self)
        self._close_timer.setSingleShot(True)
        self._close_timer.timeout.connect(self.accept)
        self._close_timer.start(2500)

    def _save_result(self, img, n_frames, method):
        """Stacking sonucunu light karelerin klasörüne TIFF olarak kaydet."""
        import numpy as _np, datetime as _dt
        lights = self._frame_table.get_accepted_paths()
        if not lights:
            return
        save_dir = os.path.dirname(os.path.abspath(lights[0]))
        timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"stack_{n_frames}fr_{method}_{timestamp}.tif"
        save_path = os.path.join(save_dir, fname)
        try:
            img16 = (_np.clip(img, 0, 1) * 65535).astype(_np.uint16)
            try:
                import tifffile as _tf
                _tf.imwrite(save_path, img16,
                            photometric="rgb" if img.ndim==3 else "minisblack",
                            compression=None)
            except ImportError:
                import cv2 as _cv2
                if img.ndim == 3:
                    img16 = _cv2.cvtColor(img16, _cv2.COLOR_RGB2BGR)
                _cv2.imwrite(save_path, img16, [_cv2.IMWRITE_TIFF_COMPRESSION, 1])
            self._log(f"💾 Kaydedildi: {save_path}")
            self._lbl_progress.setText(f"💾 {fname}")
        except Exception as e:
            self._log(f"⚠ Kayıt hatası: {e}")

    def _on_error(self, msg):
        self._log(f"❌ HATA:\n{msg}")
        self._stack_pbar.setValue(0)
        self._stack_pbar.setFormat("❌  Hata oluştu")
        self._pbar_lbl.setText("0%")
        QMessageBox.critical(self,"Stacking Hatası",msg[:1000])
        self.btn_stack.setEnabled(True); self.btn_stack.setText("🚀  Stacking Başlat")

    def get_result(self):
        return self._result

class ImageViewer(QWidget):
    """\n    Multi-panel image viewer with:\n      • 1/2/4 panel layout (side by side comparison)\n      • Scroll wheel zoom + pan on each panel\n      • Draggable histogram black/white point lines → live image update\n      • Crop, Stats tabs\n    """
    stretch_changed = pyqtSignal(float, float)   # black_pt, white_pt

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{BG};")

        # Stored images per slot [0..3]
        self._imgs   = [None, None, None, None]
        self._titles = ["", "", "", ""]
        self._active = 0      # which slot is "current"
        self._layout_n = 1    # 1, 2 or 4 panels

        # Histogram per-channel state
        # Each channel: [black, midtone, white]
        # ch: "L" (luminance/linked), "R", "G", "B"
        self._hist_pts = {
            "L": [0.0, 0.5, 1.0],
            "R": [0.0, 0.5, 1.0],
            "G": [0.0, 0.5, 1.0],
            "B": [0.0, 0.5, 1.0],
        }
        self._hist_ch   = "L"      # active channel
        self._hist_drag = None     # None|"black"|"mid"|"white"
        self._hist_black = 0.0     # compat alias → L black
        self._hist_white = 1.0     # compat alias → L white

        self._sel = None
        self._crop_coords = None
        # Inline crop state (resim üzerinde sürükle-seç)
        self._crop_mode   = False   # crop modu aktif mi
        self._crop_rect   = None    # (x0,y0,x1,y1) piksel koordinatları
        self._crop_drag   = False   # sol tık basılı
        self._crop_start_pt = None  # sürükleme başlangıcı (data coords)
        self._crop_patch    = None   # overlay patch
        self._crop_apply_cb  = None  # AstroApp tarafından set edilir
        self._crop_cancel_cb = None  # AstroApp tarafından set edilir

        lay = QVBoxLayout(self); lay.setContentsMargins(0,0,0,0); lay.setSpacing(0)

        # ── Top bar ───────────────────────────────────────────────────────
        tb = QWidget(); tb.setFixedHeight(30)
        tb.setStyleSheet(f"background:{BG2};border-bottom:1px solid {BORDER};")
        tbl = QHBoxLayout(tb); tbl.setContentsMargins(6,2,6,2); tbl.setSpacing(6)

        self.lbl_info = QLabel("—")
        self.lbl_info.setStyleSheet(f"color:{MUTED};font-size:10px;")
        tbl.addWidget(self.lbl_info, 1)

        # Layout buttons
        for n, icon in [(1,"▣"),(2,"▣▣"),(4,"⊞")]:
            b = QPushButton(icon); b.setFixedSize(30, 22)
            b.setCheckable(True)
            b.setStyleSheet(
                f"QPushButton{{background:{BG3};color:{MUTED};border:1px solid {BORDER};"
                f"border-radius:3px;font-size:11px;}}"
                f"QPushButton:checked{{background:{BG4};color:{ACCENT2};"
                f"border:1px solid {ACCENT};}}"
                f"QPushButton:hover{{color:{TEXT};}}")
            b.setToolTip(f"{n} panel")
            b.clicked.connect(lambda _, nn=n: self._set_layout(nn))
            tbl.addWidget(b)
        self._layout_btns = [tbl.itemAt(tbl.count()-3).widget(),
                              tbl.itemAt(tbl.count()-2).widget(),
                              tbl.itemAt(tbl.count()-1).widget()]
        self._layout_btns[0].setChecked(True)

        # Zoom buttons
        b_zi = QPushButton("🔍+"); b_zi.setFixedSize(32,22)
        b_zo = QPushButton("🔍−"); b_zo.setFixedSize(32,22)
        b_fit= QPushButton("⛶");   b_fit.setFixedSize(28,22)
        for b in (b_zi, b_zo, b_fit):
            b.setStyleSheet(_btn(color=BG3, h=22))
        b_zi.clicked.connect(lambda: self._zoom_step(1.25))
        b_zo.clicked.connect(lambda: self._zoom_step(0.80))
        b_fit.clicked.connect(self._zoom_fit)
        tbl.addWidget(b_zi); tbl.addWidget(b_zo); tbl.addWidget(b_fit)

        cmap_lbl = QLabel("Colormap:"); cmap_lbl.setStyleSheet(LBL_CSS)
        self.cmap_cb = QComboBox()
        self.cmap_cb.addItems(["gray","inferno","plasma","viridis","hot","coolwarm","nipy_spectral"])
        self.cmap_cb.setStyleSheet(COMBO_CSS); self.cmap_cb.setFixedWidth(100)
        tbl.addWidget(cmap_lbl); tbl.addWidget(self.cmap_cb)
        lay.addWidget(tb)

        # ── Tabs ──────────────────────────────────────────────────────────
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(
            f"QTabWidget::pane{{border:none;background:{BG};}}"
            f"QTabBar::tab{{background:{BG2};color:{MUTED};padding:5px 14px;"
            f"border-bottom:2px solid transparent;font-size:10px;}}"
            f"QTabBar::tab:selected{{color:{ACCENT2};border-bottom:2px solid {ACCENT2};}}"
            f"QTabBar::tab:hover{{color:{TEXT};}}")
        lay.addWidget(self.tabs, 1)

        # ── Image tab (main) ─────────────────────────────────────────────
        # Ana sekme: SOL=resim grid, SAĞ=histogram editör (sürüklenebilir splitter)
        self._img_tab = QWidget()
        self._img_tab.setStyleSheet(f"background:{BG};")
        _img_tab_lay = QHBoxLayout(self._img_tab)
        _img_tab_lay.setContentsMargins(0,0,0,0)
        _img_tab_lay.setSpacing(0)

        # QSplitter: sol=canvas grid, sağ=histogram panel
        self._main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._main_splitter.setStyleSheet(
            f"QSplitter::handle{{background:{BORDER};width:4px;}}"
            f"QSplitter::handle:hover{{background:{ACCENT};}}")
        _img_tab_lay.addWidget(self._main_splitter)

        # ── Sol: image grid ───────────────────────────────────────────────
        self._img_canvas_widget = QWidget()
        self._img_canvas_widget.setStyleSheet(f"background:{BG};")
        self._grid_lay = QGridLayout(self._img_canvas_widget)
        self._grid_lay.setContentsMargins(0,0,0,0); self._grid_lay.setSpacing(2)

        self._figs_view = []
        self._axes_view = []
        self._canvases  = []
        for i in range(4):
            fig = Figure(facecolor=BG); ax = fig.add_subplot(111)
            ax.set_facecolor(BG); ax.set_axis_off()
            fig.tight_layout(pad=0)
            canvas = FigureCanvas(fig)
            canvas.setStyleSheet(f"background:{BG};")
            canvas.setMouseTracking(True)
            # Store slot index on canvas for Qt events
            canvas._slot = i
            canvas._viewer = self
            # Override Qt mouse events for crop
            canvas.mousePressEvent   = lambda ev, c=canvas: self._qt_press(ev, c._slot)
            canvas.mouseMoveEvent    = lambda ev, c=canvas: self._qt_move(ev, c._slot)
            canvas.mouseReleaseEvent = lambda ev, c=canvas: self._qt_release(ev, c._slot)
            canvas.contextMenuEvent  = lambda ev, c=canvas: None  # block default
            fig.canvas.mpl_connect("scroll_event",
                lambda e,i=i: self._on_scroll(e,i))
            fig.canvas.mpl_connect("motion_notify_event",
                lambda e,i=i: self._on_mouse_move(e,i))
            self._figs_view.append(fig)
            self._axes_view.append(ax)
            self._canvases.append(canvas)

        self._main_splitter.addWidget(self._img_canvas_widget)

        # ── Sağ: histogram editör paneli ──────────────────────────────────
        self._hist_panel_wrap = QWidget()
        self._hist_panel_wrap.setStyleSheet(
            f"background:{BG2};border-left:2px solid {BORDER};")
        self._hist_panel_wrap.setMinimumWidth(280)
        self._hist_panel_wrap.setMaximumWidth(480)
        _hpw_lay = QVBoxLayout(self._hist_panel_wrap)
        _hpw_lay.setContentsMargins(0,0,0,0); _hpw_lay.setSpacing(0)

        # Panel başlık çubuğu + toggle butonu
        _hdr = QWidget(); _hdr.setFixedHeight(28)
        _hdr.setStyleSheet(f"background:{BG3};border-bottom:1px solid {BORDER};")
        _hdr_lay = QHBoxLayout(_hdr)
        _hdr_lay.setContentsMargins(8,2,4,2); _hdr_lay.setSpacing(4)
        _hdr_lay.addWidget(QLabel("📊 Histogram Editor"))
        _hdr_lay.itemAt(0).widget().setStyleSheet(
            f"color:{HEAD};font-size:10px;font-weight:700;")
        _hdr_lay.addStretch()
        self._btn_hist_toggle = QPushButton("◀")
        self._btn_hist_toggle.setFixedSize(24, 20)
        self._btn_hist_toggle.setStyleSheet(_btn(color=BG4, h=20))
        self._btn_hist_toggle.setToolTip("Hide/Show histogram panel")
        self._btn_hist_toggle.clicked.connect(self._toggle_hist_panel)
        _hdr_lay.addWidget(self._btn_hist_toggle)
        _hpw_lay.addWidget(_hdr)

        self._hist_editor = HistogramEditorPanel()
        self._hist_editor.preview_changed.connect(self._on_hist_preview)
        self._hist_editor.apply_requested.connect(self._on_hist_apply)
        _hpw_lay.addWidget(self._hist_editor, 1)

        self._main_splitter.addWidget(self._hist_panel_wrap)
        # Başlangıç boyutları: resim geniş, histogram dar
        self._main_splitter.setSizes([10000, 320])
        self._main_splitter.setCollapsible(0, False)
        self._main_splitter.setCollapsible(1, True)

        self.tabs.addTab(self._img_tab, "🖼  Image")

        # ── Histogram compat stubs ────────────────────────────────────────
        self._hist_apply_cb = None
        self._hist_pts = {"L":[0.0,0.5,1.0],"R":[0.0,0.5,1.0],"G":[0.0,0.5,1.0],"B":[0.0,0.5,1.0]}
        self._hist_ch  = "L"
        self.chk_hist_linked = QCheckBox(); self.chk_hist_linked.setChecked(True)
        self.spin_hist_black = QDoubleSpinBox(); self.spin_hist_black.setValue(0.0)
        self.spin_hist_mid   = QDoubleSpinBox(); self.spin_hist_mid.setValue(0.5)
        self.spin_hist_white = QDoubleSpinBox(); self.spin_hist_white.setValue(1.0)
        self.spin_hist_gamma = QDoubleSpinBox(); self.spin_hist_gamma.setValue(1.0)
        self.chk_hist_live   = QCheckBox();      self.chk_hist_live.setChecked(True)
        self.lbl_hist_bp = QLabel(); self.lbl_hist_wp = QLabel()
        self._hist_ch_btns = {}

        # Crop tab kaldirildi — inline crop kullaniliyor

        # ── Stats tab ─────────────────────────────────────────────────────
        self._fig_stat = Figure(facecolor=BG)
        self._ax_stat  = self._fig_stat.add_subplot(111)
        self._ax_stat.set_facecolor(BG); self._ax_stat.set_axis_off()
        self._canvas_stat = FigureCanvas(self._fig_stat)
        self._canvas_stat.setStyleSheet(f"background:{BG};")
        self.tabs.addTab(self._canvas_stat, "📈  Stats")

        self.cmap_cb.currentTextChanged.connect(lambda _: self._redraw_all())
        self._set_layout(1)

        # Pan state
        self._pan_start = {}   # slot → (x, y, xlim, ylim)

    # ── Helper ───────────────────────────────────────────────────────────
    @staticmethod
    def _sax_style(ax):
        ax.set_facecolor(BG); ax.tick_params(colors=ACCENT, labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)

    # ── Layout ───────────────────────────────────────────────────────────
    def _set_layout(self, n: int):
        self._layout_n = n
        while self._grid_lay.count():
            item = self._grid_lay.takeAt(0)
            if item.widget():
                item.widget().hide()
        positions = {1:[(0,0)], 2:[(0,0),(0,1)], 4:[(0,0),(0,1),(1,0),(1,1)]}
        for idx, (r, c) in enumerate(positions[n]):
            self._canvases[idx].show()
            self._grid_lay.addWidget(self._canvases[idx], r, c)
        for idx in range(n, 4):
            self._canvases[idx].hide()
        for i, b in enumerate(self._layout_btns):
            b.setChecked([1,2,4][i] == n)
        # Ensure we're on image tab
        self.tabs.setCurrentIndex(0)
        self._redraw_all()

    # ── Show image ────────────────────────────────────────────────────────
    def show_image(self, img: np.ndarray, title: str = "", slot: int = -1):
        """\n        Display image. slot=-1 → auto-select next free or slot 0.\n        Always updates slot 0 (current) for backward compat.\n        """
        if slot < 0:
            slot = 0
        slot = max(0, min(3, slot))
        self._imgs[slot]   = img.copy()
        self._titles[slot] = title
        self._active = slot
        self._hist_black = 0.0
        self._hist_white = 1.0
        self._draw_slot(slot)
        self._hist_editor.set_image(img)
        self._draw_stats(img)
        h, w = img.shape[:2]; ch = "RGB" if img.ndim==3 else "Gray"
        self.lbl_info.setText(f"Slot {slot+1}  {w}×{h}  {ch}  min={img.min():.3f}  max={img.max():.3f}")
        self.tabs.setCurrentIndex(0)

    def show_comparison(self, imgs: list, titles: list = None):
        """Load multiple images into slots 0..n-1 and set layout."""
        n = min(len(imgs), 4)
        self._set_layout(n if n in (1,2,4) else (2 if n==3 else 1))
        for i in range(n):
            self._imgs[i]   = np.clip(imgs[i], 0, 1).astype(np.float32)
            self._titles[i] = (titles[i] if titles and i < len(titles) else f"Image {i+1}")
        for i in range(n, 4):
            self._imgs[i] = None
        self._redraw_all()

    def _draw_slot(self, slot: int):
        img = self._imgs[slot]
        if img is None: return
        ax  = self._axes_view[slot]
        fig = self._figs_view[slot]

        # Remember zoom state
        try:
            xlim = ax.get_xlim(); ylim = ax.get_ylim()
            had_limits = (xlim != (0,1))
        except Exception:
            had_limits = False

        ax.clear(); self._sax_style(ax)
        display = self._apply_hist_points(img)
        cmap = self.cmap_cb.currentText() if img.ndim==2 else None
        ax.imshow(display, cmap=cmap, origin="upper",
                  aspect="equal", interpolation="nearest")
        if self._titles[slot]:
            ax.set_title(self._titles[slot], color=HEAD, fontsize=9, pad=2)
        ax.set_axis_off()
        fig.tight_layout(pad=0.1)

        if had_limits:
            ax.set_xlim(xlim); ax.set_ylim(ylim)

        try: fig.canvas.draw()
        except Exception: pass

    def _apply_hist_points(self, img: np.ndarray) -> np.ndarray:
        """Apply black/white point adjustment to image for display."""
        bp, wp = self._hist_black, self._hist_white
        rng = max(wp - bp, 1e-9)
        return np.clip((img.astype(np.float32) - bp) / rng, 0, 1)

    def _redraw_all(self):
        for i in range(self._layout_n):
            if self._imgs[i] is not None:
                self._draw_slot(i)

    # ── Zoom & Pan ────────────────────────────────────────────────────────
    def _zoom_step(self, factor: float):
        """Zoom all visible panels."""
        for i in range(self._layout_n):
            ax = self._axes_view[i]
            if self._imgs[i] is None: continue
            xl = ax.get_xlim(); yl = ax.get_ylim()
            cx = (xl[0]+xl[1])/2; cy = (yl[0]+yl[1])/2
            hw = (xl[1]-xl[0])/2/factor; hh = abs(yl[1]-yl[0])/2/factor
            ax.set_xlim(cx-hw, cx+hw); ax.set_ylim(cy+hh, cy-hh)
            try: self._figs_view[i].canvas.draw()
            except Exception: pass

    def _zoom_fit(self):
        for i in range(self._layout_n):
            img = self._imgs[i]
            if img is None: continue
            h, w = img.shape[:2]
            self._axes_view[i].set_xlim(0, w)
            self._axes_view[i].set_ylim(h, 0)
            try: self._figs_view[i].canvas.draw()
            except Exception: pass

    def _on_scroll(self, event, slot: int):
        if self._imgs[slot] is None: return
        ax = self._axes_view[slot]
        factor = 1.2 if event.button == "up" else (1/1.2)
        xdata = event.xdata or (ax.get_xlim()[0]+ax.get_xlim()[1])/2
        ydata = event.ydata or (ax.get_ylim()[0]+ax.get_ylim()[1])/2
        xl = ax.get_xlim(); yl = ax.get_ylim()
        ax.set_xlim(xdata+(xl[0]-xdata)/factor, xdata+(xl[1]-xdata)/factor)
        ax.set_ylim(ydata+(yl[0]-ydata)/factor, ydata+(yl[1]-ydata)/factor)
        try: self._figs_view[slot].canvas.draw()
        except Exception: pass

    def _on_mouse_move(self, event, slot: int):
        if event.inaxes and self._imgs[slot] is not None:
            img = self._imgs[slot]
            x, y = int(event.xdata or 0), int(event.ydata or 0)
            h, w = img.shape[:2]
            if 0<=x<w and 0<=y<h:
                if img.ndim == 2:
                    pix = f"{img[y,x]:.4f}"
                else:
                    pix = f"R={img[y,x,0]:.3f} G={img[y,x,1]:.3f} B={img[y,x,2]:.3f}"
                self.lbl_info.setText(
                    f"X:{x} Y:{y}  {pix}")

    def _on_click(self, event, slot: int):
        """Matplotlib click — sadece orta tık pan için."""
        if event.button == 2 and event.inaxes:
            ax = self._axes_view[slot]
            self._pan_start[slot] = (event.xdata, event.ydata,
                                      ax.get_xlim(), ax.get_ylim())

    # _show_crop_context replaced by _show_crop_menu

    def _draw_hist(self, img):
        pass  # handled by HistogramEditorPanel

    def _hist_set_channel(self, ch: str):
        """Compat stub."""
        pass

    def _hist_spin_changed(self):
        pass

    def _set_hist_pts(self, ch, b, m, w):
        pass

    def _hist_press(self,e): pass
    def _hist_motion(self,e): pass
    def _hist_release(self,e): pass
    def _reset_hist_points(self): pass

    def _apply_hist_points(self, img: np.ndarray) -> np.ndarray:
        """Compat stub — actual processing now done in HistogramEditorPanel."""
        return img  # Editor handles its own apply

    def _hist_bake(self):
        """Compat stub — Apply button is inside HistogramEditorPanel."""
        pass


    def _toggle_hist_panel(self):
        """Hide/show histogram panel by collapsing splitter."""
        sizes = self._main_splitter.sizes()
        if sizes[1] > 10:
            # Collapse → sakla
            self._hist_panel_wrap.setMinimumWidth(0)
            self._main_splitter.setSizes([10000, 0])
            self._btn_hist_toggle.setText("▶")
            self._btn_hist_toggle.setToolTip("Show histogram panel")
        else:
            # Expand → göster
            self._hist_panel_wrap.setMinimumWidth(280)
            self._main_splitter.setSizes([10000, 320])
            self._btn_hist_toggle.setText("◀")
            self._btn_hist_toggle.setToolTip("Hide histogram panel")

    def _on_hist_preview(self, result):
        """Live preview — update display only, do NOT touch hist_editor._orig_img."""
        slot = self._active
        # Store preview separately — draw without overwriting stored img for editor
        self._preview_img = np.clip(result, 0, 1).astype(np.float32)
        self._draw_slot_direct(slot, self._preview_img)

    def _preview_slot(self, img: "np.ndarray", title: str = "⏳ işleniyor…"):
        """İşlem sırasında ara sonucu viewer'da göster (history'e gitmez)."""
        slot = self._active
        self._draw_slot_direct(slot, img)
        # Title güncelle
        try:
            ax = self._axes_view[slot]
            ax.set_title(title, color="#aaffcc", fontsize=8, pad=2)
            self._figs_view[slot].canvas.draw_idle()
        except Exception:
            pass

    def _draw_slot_direct(self, slot, img):
        """Draw img to slot without updating self._imgs (used for live preview)."""
        ax  = self._axes_view[slot]
        fig = self._figs_view[slot]
        try:
            xlim = ax.get_xlim(); ylim = ax.get_ylim()
            had_limits = (xlim != (0.0, 1.0))
        except Exception:
            had_limits = False
        ax.clear(); self._sax_style(ax)
        ax.set_axis_off()
        cmap = None if img.ndim == 3 else self.cmap_cb.currentText()
        ax.imshow(np.clip(img,0,1), aspect="equal", interpolation="bilinear",
                  origin="upper", cmap=cmap)
        if had_limits:
            ax.set_xlim(xlim); ax.set_ylim(ylim)
        fig.tight_layout(pad=0)
        try: fig.canvas.draw_idle()
        except Exception: pass

    def _on_hist_apply(self, result):
        """User clicked Apply — bake to history and reset editor to new baseline."""
        result = np.clip(result, 0, 1).astype(np.float32)
        slot = self._active
        self._imgs[slot] = result
        # Update editor's original so next edit starts from here
        self._hist_editor.set_image(result)
        if self._hist_apply_cb is not None:
            self._hist_apply_cb(result)
        self._draw_slot(slot)

    # ── Inline Crop (resim üzerinde sürükle + sağ tık menüsü) ───────────────────
    # ── Qt Native Crop Handlers ──────────────────────────────────────────────
    def _data_coords(self, slot, qx, qy):
        """Qt widget piksel → matplotlib data koordinatı."""
        canvas = self._canvases[slot]
        ax     = self._axes_view[slot]
        # Matplotlib canvas yüksekliğini al (Y ekseni ters)
        h_canvas = canvas.height()
        # display coords (matplotlib uses bottom-left origin)
        disp_x = qx
        disp_y = h_canvas - qy
        try:
            inv = ax.transData.inverted()
            dx, dy = inv.transform((disp_x, disp_y))
            return float(dx), float(dy)
        except Exception:
            return None, None

    def _qt_press(self, ev, slot):
        """Qt sol/sağ tık."""
        from PyQt6.QtCore import Qt as _Qt
        from PyQt6.QtGui import QCursor

        btn = ev.button()
        if btn == _Qt.MouseButton.LeftButton and self._crop_mode:
            dx, dy = self._data_coords(slot, ev.pos().x(), ev.pos().y())
            if dx is not None:
                self._crop_start_xy = (dx, dy)
                self._crop_rect     = None
            ev.accept(); return

        if btn == _Qt.MouseButton.RightButton and self._crop_mode:
            self._show_crop_menu(slot)
            ev.accept(); return

        # Pass through to matplotlib for pan etc.
        FigureCanvas.mousePressEvent(self._canvases[slot], ev)

    def _qt_move(self, ev, slot):
        """Qt mouse hareket — crop overlay çiz."""
        from PyQt6.QtCore import Qt as _Qt
        if (self._crop_mode
                and self._crop_start_xy is not None
                and ev.buttons() & _Qt.MouseButton.LeftButton):
            dx, dy = self._data_coords(slot, ev.pos().x(), ev.pos().y())
            if dx is not None:
                x0, y0 = self._crop_start_xy
                self._draw_crop_overlay(slot, x0, y0, dx, dy)
            ev.accept(); return
        FigureCanvas.mouseMoveEvent(self._canvases[slot], ev)

    def _qt_release(self, ev, slot):
        """Qt sol tık bırakıldı — seçim koordinatlarını kaydet."""
        from PyQt6.QtCore import Qt as _Qt
        if (ev.button() == _Qt.MouseButton.LeftButton
                and self._crop_mode
                and self._crop_start_xy is not None):
            dx, dy = self._data_coords(slot, ev.pos().x(), ev.pos().y())
            if dx is not None:
                img = self._imgs[slot]
                if img is not None:
                    h, w = img.shape[:2]
                    x0d, y0d = self._crop_start_xy
                    x0 = int(np.clip(min(x0d, dx),   0, w-1))
                    x1 = int(np.clip(max(x0d, dx)+1, 1, w))
                    y0 = int(np.clip(min(y0d, dy),   0, h-1))
                    y1 = int(np.clip(max(y0d, dy)+1, 1, h))
                    if x1-x0 > 4 and y1-y0 > 4:
                        self._crop_rect = (x0, y0, x1, y1)
                        self.lbl_info.setText(
                            f"Secim: {x1-x0}x{y1-y0} px  "
                            f"[Sag tik = menu]")
                    else:
                        self._crop_rect = None
            ev.accept(); return
        FigureCanvas.mouseReleaseEvent(self._canvases[slot], ev)

    def _draw_crop_overlay(self, slot, x0d, y0d, x1d, y1d):
        """Sarı kesik çizgi dikdörtgen çiz."""
        import matplotlib.patches as mpatches
        ax  = self._axes_view[slot]
        fig = self._figs_view[slot]
        if self._crop_patch and self._crop_patch in ax.patches:
            try: self._crop_patch.remove()
            except Exception: pass
        rx = min(x0d, x1d); ry = min(y0d, y1d)
        rw = abs(x1d - x0d); rh = abs(y1d - y0d)
        patch = mpatches.Rectangle(
            (rx, ry), rw, rh,
            linewidth=1.5, edgecolor=GOLD, facecolor=(1,1,0,0.07),
            linestyle="--", zorder=10)
        ax.add_patch(patch)
        self._crop_patch = patch
        try: fig.canvas.draw_idle()
        except Exception: pass

    def _show_crop_menu(self, slot):
        """Sağ tık context menüsü."""
        from PyQt6.QtWidgets import QMenu
        from PyQt6.QtGui import QCursor
        menu = QMenu(self)
        menu.setStyleSheet(
            f"QMenu{{background:{BG2};color:{TEXT};border:1px solid {BORDER};"
            f"border-radius:6px;padding:4px;}}"
            f"QMenu::item{{padding:7px 22px;border-radius:3px;font-size:11px;}}"
            f"QMenu::item:selected{{background:{BG4};color:{ACCENT2};}}"
            f"QMenu::separator{{height:1px;background:{BORDER};margin:3px 8px;}}")

        rect = self._crop_rect
        if rect:
            w = rect[2]-rect[0]; h = rect[3]-rect[1]
            a_apply = menu.addAction(f"✂  Crop  ({w} x {h} px)")
            a_apply.setFont(a_apply.font())
            menu.addSeparator()
        a_all    = menu.addAction("⬜  Tamami sec")
        menu.addSeparator()
        a_cancel = menu.addAction("✕  Crop modundan cik")

        chosen = menu.exec(QCursor.pos())
        if chosen is None: return

        txt = chosen.text()
        if txt.startswith("✂") and rect:
            if self._crop_apply_cb: self._crop_apply_cb()
        elif txt.startswith("⬜"):
            img = self._imgs[slot]
            if img is not None:
                h2, w2 = img.shape[:2]
                self._crop_rect = (0, 0, w2, h2)
                self._draw_crop_overlay(slot, 0, 0, w2, h2)
                self.lbl_info.setText(f"Tamami: {w2}x{h2}  [Sag tik = menu]")
        elif txt.startswith("✕"):
            if self._crop_cancel_cb: self._crop_cancel_cb()
            else: self.exit_crop_mode()

    def enter_crop_mode(self):
        self._crop_mode     = True
        self._crop_start_xy = None
        self._crop_patch    = None
        for c in self._canvases:
            c.setCursor(Qt.CursorShape.CrossCursor)
        # Baslangicta tam resmi sec
        slot = self._active
        img  = self._imgs[slot]
        if img is not None:
            h, w = img.shape[:2]
            self._crop_rect = (0, 0, w, h)
            self._draw_crop_overlay(slot, 0, 0, w, h)
            self.lbl_info.setText(
                f"✂  Tam resim secili ({w}x{h})  |  "
                f"Sol tik surukle yeni alan sec  |  Sag tik = menu")
        else:
            self._crop_rect = None
            self.lbl_info.setText("✂  Sol tik surukle sec  |  Sag tik = menu")

    def exit_crop_mode(self):
        self._crop_mode     = False
        self._crop_rect     = None
        self._crop_start_xy = None
        if self._crop_patch:
            try: self._crop_patch.remove()
            except Exception: pass
            self._crop_patch = None
        for c in self._canvases:
            c.setCursor(Qt.CursorShape.ArrowCursor)
        self._redraw_all()

    # _crop_overlay replaced by _draw_crop_overlay

    def get_crop_rect(self):
        return self._crop_rect

    def apply_inline_crop(self):
        if self._crop_rect is None: return None
        slot = self._active
        img  = self._imgs[slot]
        if img is None: return None
        x0, y0, x1, y1 = self._crop_rect
        h, w = img.shape[:2]
        x0 = max(0, min(x0, w-1))
        x1 = max(1, min(x1, w))
        y0 = max(0, min(y0, h-1))
        y1 = max(1, min(y1, h))
        return np.clip(img[y0:y1, x0:x1], 0, 1).astype(np.float32)

    def _draw_stats(self, img: np.ndarray):
        ax = self._ax_stat; ax.clear(); ax.set_facecolor(BG); ax.set_axis_off()
        flat = img.ravel()
        rows = [("Mean",f"{np.mean(flat):.5f}"),("Median",f"{np.median(flat):.5f}"),
                ("Std Dev",f"{np.std(flat):.5f}"),("Min",f"{flat.min():.5f}"),
                ("Max",f"{flat.max():.5f}"),("SNR",f"{np.mean(flat)/(np.std(flat)+1e-9):.2f}"),
                ("Size",f"{img.shape[1]}×{img.shape[0]}"),
                ("Channel","RGB" if img.ndim==3 else "Gray")]
        y = 0.92
        for lbl, val in rows:
            ax.text(0.05,y,lbl,transform=ax.transAxes,color=MUTED,fontsize=10)
            ax.text(0.60,y,val,transform=ax.transAxes,color=TEXT,fontsize=10,fontweight="bold")
            y -= 0.10
        ax.set_title("Image Statistics", color=HEAD, fontsize=11, pad=6)
        self._fig_stat.tight_layout()
        try: self._canvas_stat.draw()
        except Exception: pass

    # ── Crop (unchanged interface) ────────────────────────────────────────
    def _ph_crop(self, msg):
        pass

    def start_crop_mode(self, img):
        self.enter_crop_mode()  # redirect

    def _on_sel_done(self, ec, er):
        img = self._imgs[self._active]
        if img is None: return
        h, w = img.shape[:2]
        x0 = int(np.clip(min(ec.xdata,er.xdata),  0,w-1))
        x1 = int(np.clip(max(ec.xdata,er.xdata)+1,1,w))
        y0 = int(np.clip(min(ec.ydata,er.ydata),  0,h-1))
        y1 = int(np.clip(max(ec.ydata,er.ydata)+1,1,h))
        if x1>x0 and y1>y0: self._crop_coords = (x0,y0,x1,y1)

    def apply_crop(self):
        return self.apply_inline_crop()

    def reset_crop(self):
        self.exit_crop_mode()

    def _img(self): return self._imgs[self._active]


# ── QLabel helper ─────────────────────────────────────────────────────────────
def _qlabel_also_style(lbl, style):
    lbl.setStyleSheet(style)
    return lbl
QLabel.also_style = _qlabel_also_style
# ═══════════════════════════════ GALAXY BG ══════════════════════════════════
class StarfieldBg(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        rng=np.random.default_rng(42); self._stars=rng.random((350,3))

    def paintEvent(self,_):
        p=QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W,H=self.width(),self.height()
        grad=QLinearGradient(0,0,0,H)
        grad.setColorAt(0.0,QColor("#020810")); grad.setColorAt(0.4,QColor("#04101e"))
        grad.setColorAt(1.0,QColor("#020810")); p.fillRect(0,0,W,H,QBrush(grad))
        for cx,cy,r,cr,cg,cb in [(0.15,0.3,0.18,20,40,80),(0.75,0.65,0.22,10,25,55),(0.5,0.15,0.12,30,15,60)]:
            ng=QLinearGradient(int(cx*W-r*W),int(cy*H),int(cx*W+r*W),int(cy*H))
            ng.setColorAt(0.0,QColor(0,0,0,0)); ng.setColorAt(0.5,QColor(cr,cg,cb,35))
            ng.setColorAt(1.0,QColor(0,0,0,0)); p.fillRect(0,0,W,H,QBrush(ng))
        for sx,sy,sb in self._stars:
            bri=int(sb*255); sz=0.6+sb*1.2
            p.setPen(QColor(bri,bri,min(255,int(bri*1.1)),int(sb*220)))
            p.drawEllipse(int(sx*W)-1,int(sy*H)-1,int(sz*2),int(sz*2))
        p.end()


# ═══════════════════════════════ HISTORY PANEL ══════════════════════════════
class HistoryPanel(QWidget):
    jump_requested = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{BG2};"); self.setFixedWidth(170)
        lay=QVBoxLayout(self); lay.setContentsMargins(0,0,0,0); lay.setSpacing(0)
        hdr=QLabel("  HISTORY")
        hdr.setStyleSheet(f"background:{BG4};color:{HEAD};font-size:10px;font-weight:700;"
                          f"letter-spacing:2px;padding:6px 0;border-bottom:1px solid {BORDER};")
        lay.addWidget(hdr)
        self.lst=QListWidget()
        self.lst.setStyleSheet(
            f"QListWidget{{background:{BG2};border:none;color:{TEXT};font-size:10px;}}"
            f"QListWidget::item{{padding:5px 8px;border-bottom:1px solid {BORDER};}}"
            f"QListWidget::item:selected{{background:{BG4};color:{ACCENT2};}}"
            f"QListWidget::item:hover{{background:{BG3};}}")
        self.lst.itemClicked.connect(
            lambda item: self.jump_requested.emit(item.data(Qt.ItemDataRole.UserRole)))
        lay.addWidget(self.lst,1)
        clr=QPushButton("🗑  Clear History")
        clr.setStyleSheet(_btn(color="#200a0a",hover=RED)); clr.setFixedHeight(26)
        clr.clicked.connect(self.lst.clear); lay.addWidget(clr)

    def push(self, label, index):
        item=QListWidgetItem(f"#{index}  {label}")
        item.setData(Qt.ItemDataRole.UserRole,index)
        self.lst.addItem(item); self.lst.scrollToBottom(); self.lst.setCurrentItem(item)

    def select(self, index):
        for i in range(self.lst.count()):
            item=self.lst.item(i)
            if item.data(Qt.ItemDataRole.UserRole)==index:
                self.lst.setCurrentItem(item); break

    def truncate_to(self, index):
        to_del=[i for i in range(self.lst.count())
                if self.lst.item(i).data(Qt.ItemDataRole.UserRole)>index]
        for i in reversed(to_del): self.lst.takeItem(i)


# ═══════════════════════════════ MAIN WINDOW ════════════════════════════════
class AstroApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🔭  Astro Maestro Pro")
        self.resize(1540,900)
        self._orig        = None
        self._current        = None
        self._history        = []
        self._before_process = None   # son işlem öncesi snapshot
        self._workers     = []   # GC crash guard
        self._crop_src    = None
        self._starnet_fn  = None  # holds last StarNet++ result accessor
        self._settings    = {}    # loaded at startup
        self._apply_theme(); self._build_ui()
        self._load_settings()

    def _apply_theme(self):
        pal=QPalette()
        for role,color in [
            (QPalette.ColorRole.Window,BG),(QPalette.ColorRole.WindowText,TEXT),
            (QPalette.ColorRole.Base,BG2),(QPalette.ColorRole.AlternateBase,BG),
            (QPalette.ColorRole.Text,TEXT),(QPalette.ColorRole.ButtonText,TEXT),
            (QPalette.ColorRole.Button,BG4),(QPalette.ColorRole.ToolTipBase,BG3),
            (QPalette.ColorRole.ToolTipText,TEXT),
            (QPalette.ColorRole.Highlight,QColor(ACCENT)),
            (QPalette.ColorRole.HighlightedText,QColor("#fff")),
        ]:
            pal.setColor(role,QColor(color) if isinstance(color,str) else color)
        self.setPalette(pal)

    def _build_ui(self):
        central=QWidget(); self.setCentralWidget(central)
        root=QVBoxLayout(central); root.setContentsMargins(0,0,0,0); root.setSpacing(0)
        root.addWidget(self._make_toolbar())
        content=QWidget(); content.setStyleSheet(f"background:{BG};")
        cl=QVBoxLayout(content); cl.setContentsMargins(0,0,0,0)
        self._bg=StarfieldBg(content); self._bg.setGeometry(0,0,1540,900); self._bg.lower()
        splitter=QSplitter(Qt.Orientation.Horizontal)
        splitter.setStyleSheet(f"QSplitter::handle{{background:{BORDER};width:2px;}}")
        splitter.addWidget(self._build_viewer())
        splitter.addWidget(self._build_history_panel())
        splitter.setSizes([1200,170]); splitter.setCollapsible(1,True)
        cl.addWidget(splitter,1); root.addWidget(content,1)
        # Build panels (hidden, used by flyout popups)
        self._panels={}
        self._panel_order=[]
        self._panel_visible={}
        self._make_process_panels_headless()
        self.status=QStatusBar()
        self.status.setStyleSheet(
            f"QStatusBar{{background:{BG};border-top:1px solid {BORDER};"
            f"color:{ACCENT};font-size:10px;padding:2px 8px;}}")
        self.setStatusBar(self.status)
        self.pbar=QProgressBar(); self.pbar.setFixedWidth(260); self.pbar.setTextVisible(False)
        self.pbar.setStyleSheet(
            f"QProgressBar{{background:{BG2};border:1px solid {BORDER};"
            f"border-radius:3px;height:8px;}}"
            f"QProgressBar::chunk{{background:{ACCENT};border-radius:3px;}}")
        self.status.addPermanentWidget(self.pbar); self.pbar.hide()
        self.status.showMessage("Ready  —  Open a file to start")
        self._build_menubar()

    def resizeEvent(self,e):
        super().resizeEvent(e)
        if hasattr(self,'_bg'): self._bg.setGeometry(0,0,self.width(),self.height())

    # ── Menu Bar ─────────────────────────────────────────────────────────
    def _build_menubar(self):
        """Tam menü çubuğu: Dosya, Düzenle, Görünüm, İşlem, Araçlar, Yardım"""
        mb = self.menuBar()
        mb.setStyleSheet(
            f"QMenuBar{{background:{BG2};color:{TEXT};border-bottom:1px solid {BORDER};"
            f"font-size:11px;padding:2px 0;}}"
            f"QMenuBar::item{{padding:4px 10px;border-radius:3px;}}"
            f"QMenuBar::item:selected{{background:{BG4};color:{ACCENT2};}}"
            f"QMenu{{background:{BG2};color:{TEXT};border:1px solid {BORDER};"
            f"border-radius:6px;padding:4px;}}"
            f"QMenu::item{{padding:6px 20px 6px 28px;border-radius:3px;font-size:11px;}}"
            f"QMenu::item:selected{{background:{BG4};color:{ACCENT2};}}"
            f"QMenu::separator{{height:1px;background:{BORDER};margin:3px 8px;}}"
            f"QMenu::icon{{padding-left:8px;}}")

        # ── Dosya ──────────────────────────────────────────────────────────
        fm = mb.addMenu("📁  Dosya")
        self._act(fm, "📂  Aç…",            "Ctrl+O",  self._open_file)
        self._act(fm, "💾  Kaydet…",         "Ctrl+S",  self._save_file)
        fm.addSeparator()
        self._act(fm, "🌌  Yıldızsız Aç…",   None,      self._open_starless_file)
        self._act(fm, "⭐  Yıldız Maskesi Aç…",None,    self._open_stars_file)
        fm.addSeparator()
        # Son açılan dosyalar
        self._recent_menu = fm.addMenu("🕐  Son Açılanlar")
        self._update_recent_menu()
        fm.addSeparator()
        self._act(fm, "❌  Çıkış",           "Alt+F4",  self.close)

        # ── Düzenle ────────────────────────────────────────────────────────
        em = mb.addMenu("✏️  Düzenle")
        self._act(em, "↺  Geri Al",          "Ctrl+Z",  self._undo)
        self._act(em, "🔄  Tümünü Sıfırla",  None,      self._reset_all)
        em.addSeparator()
        self._act(em, "✂  Kırp",             "Ctrl+X",  lambda: self._toggle_crop_mode())
        em.addSeparator()
        self._act(em, "⚙  Ayarlar…",         "Ctrl+,",  self._open_settings)

        # ── Görünüm ────────────────────────────────────────────────────────
        vm = mb.addMenu("👁  Görünüm")
        self._act(vm, "🖼  Orijinali Göster", "Ctrl+H",  self._show_original)
        vm.addSeparator()
        self._act(vm, "⛶  Ekrana Sığdır",    "Ctrl+0",  lambda: self.viewer._zoom_fit())
        self._act(vm, "🔍+  Yakınlaştır",     "Ctrl+=",  lambda: self.viewer._zoom_step(1.25))
        self._act(vm, "🔍−  Uzaklaştır",      "Ctrl+-",  lambda: self.viewer._zoom_step(0.8))
        vm.addSeparator()
        self._act(vm, "▣  1 Panel",          "Ctrl+1",  lambda: self.viewer._set_layout(1))
        self._act(vm, "▣▣  2 Panel",         "Ctrl+2",  lambda: self.viewer._set_layout(2))
        self._act(vm, "⊞  4 Panel",          "Ctrl+4",  lambda: self.viewer._set_layout(4))
        vm.addSeparator()
        self._act(vm, "📊  Histogram Paneli", "Ctrl+H",
                  lambda: self._toggle_hist_panel())
        self._act(vm, "📋  Geçmiş Paneli",    "Ctrl+J",
                  lambda: self._toggle_history_panel())
        vm.addSeparator()
        self._act(vm, "🌑  Karanlık Tema",    None,
                  lambda: self._switch_theme("dark"))
        self._act(vm, "☀️  Açık Tema",        None,
                  lambda: self._switch_theme("light"))

        # ── İşlem ──────────────────────────────────────────────────────────
        pm = mb.addMenu("⚙  İşlem")
        procs = [
            ("🌌  BG Çıkarma",       "bg"),
            ("✨  Gürültü Azaltma",  "noise"),
            ("🔭  Dekonvolüsyon",    "deconv"),
            ("🔪  Keskinleştirme",   "sharp"),
            ("🌠  Nebula Güçlendir", "nebula"),
            ("🎨  Renk Kalibrasyonu","color"),
            ("🔮  Morfoloji",        "morph"),
            ("🌀  Galaksi Tespiti",  "galaxy"),
            ("📊  Histogram Stretch","stretch"),
        ]
        for label, key in procs:
            self._act(pm, label, None,
                      lambda _, k=key: self._show_process_flyout(
                          k, self._proc_btns.get(k, QPushButton())))
        pm.addSeparator()
        self._act(pm, "🗂  Stacking…",       "Ctrl+T",  self._open_stacking)
        self._act(pm, "⭐  StarNet++…",       None,      self._run_starnet_and_save)
        self._act(pm, "✦+  Yıldız Birleştir…",None,     self._open_recomposition)

        # ── Araçlar ────────────────────────────────────────────────────────
        tm = mb.addMenu("🛠  Araçlar")
        self._act(tm, "🔭  Plate Solve…",    "Ctrl+P",  self._open_plate_solve)
        tm.addSeparator()
        self._act(tm, "⚡  Script Editörü…", "Ctrl+E",  self._open_script_editor)
        tm.addSeparator()
        self._act(tm, "🔧  Panelleri Özelleştir…", None, self._open_panel_customize)

        # ── Yardım ─────────────────────────────────────────────────────────
        hm = mb.addMenu("❓  Yardım")
        self._act(hm, "🌌  Workflow Rehberi", None,     self._show_workflow)
        hm.addSeparator()
        self._act(hm, "🔄  Güncelleme Kontrolü…", None, self._open_update_dialog)
        hm.addSeparator()
        self._act(hm, "ℹ️  Hakkında",         None,     self._show_about)

    def _act(self, menu, label, shortcut, slot):
        """Menüye kısayollu bir aksiyon ekler."""
        from PyQt6.QtGui import QAction, QKeySequence
        act = QAction(label, self)
        if shortcut:
            act.setShortcut(QKeySequence(shortcut))
        act.triggered.connect(slot)
        menu.addAction(act)
        return act

    def _update_recent_menu(self):
        """Son açılan dosyalar menüsünü günceller."""
        if not hasattr(self, "_recent_menu"):
            return
        self._recent_menu.clear()
        recent = self._settings.get("recent_files", [])
        if not recent:
            act = self._recent_menu.addAction("(yok)")
            act.setEnabled(False)
            return
        for path in recent[:10]:
            if os.path.isfile(path):
                act = self._recent_menu.addAction(
                    f"  {os.path.basename(path)}")
                act.setToolTip(path)
                act.triggered.connect(lambda _, p=path: self._load_path(p))
        self._recent_menu.addSeparator()
        self._recent_menu.addAction("🗑 Temizle").triggered.connect(
            self._clear_recent)

    def _add_to_recent(self, path: str):
        recent = self._settings.get("recent_files", [])
        if path in recent:
            recent.remove(path)
        recent.insert(0, path)
        self._settings["recent_files"] = recent[:15]
        self._update_recent_menu()

    def _clear_recent(self):
        self._settings["recent_files"] = []
        from gui.settings import save as _save
        _save(self._settings)
        self._update_recent_menu()

    def _toggle_crop_mode(self):
        b = self._proc_btns.get("crop")
        if getattr(self, "_crop_active", False):
            self._inline_crop_cancel()
        else:
            if self._current is None: return
            self._crop_active = True
            if b: b.setChecked(True)
            self.viewer._crop_apply_cb  = self._inline_crop_apply
            self.viewer._crop_cancel_cb = self._inline_crop_cancel
            self.viewer.enter_crop_mode()
            self.status.showMessage("✂  Sol tık sürükle seç  |  Sağ tık → menü")

    def _toggle_hist_panel(self):
        """Histogram panelini göster/gizle."""
        if hasattr(self.viewer, "_toggle_hist_panel"):
            self.viewer._toggle_hist_panel()

    def _toggle_history_panel(self):
        """Geçmiş panelini göster/gizle."""
        if hasattr(self, "hist_panel"):
            vis = self.hist_panel.isVisible()
            self.hist_panel.setVisible(not vis)

    def _switch_theme(self, theme: str):
        self._settings["theme"] = theme
        from gui.settings import save as _save
        _save(self._settings)
        self.status.showMessage(
            f"Tema '{theme}' ayarlandı — tam etki için yeniden başlatın.")

    def _show_about(self):
        QMessageBox.information(
            self, "Astro Maestro Pro",
            "🔭  Astro Maestro Pro\n\n"
            "Astrofotoğraf işleme uygulaması\n"
            "by Deniz\n\n"
            "Özellikler:\n"
            "  • VeraLux HyperMetric Stretch\n"
            "  • StarNet++ / AI yıldız ayırma\n"
            "  • GraXpert AI arka plan\n"
            "  • ASTAP Plate Solving\n"
            "  • DSS-style stacking\n"
            "  • Photoshop-style histogram\n"
            "  • Python script editörü")

    # ── Plate Solve ──────────────────────────────────────────────────────
    def _open_plate_solve(self):
        """Plate Solve dialogunu açar."""
        if self._current is None:
            QMessageBox.information(self, "Plate Solve",
                "Önce bir görüntü açın.")
            return

        from gui.plate_solve_dialog import PlateSolveDialog

        def _on_annotate(annotated_img):
            self._set_image(annotated_img, "Plate Solve Overlay")
            self.status.showMessage("✅  Plate solve overlay uygulandı")

        dlg = PlateSolveDialog(
            image         = self._current.copy(),
            settings      = self._settings,
            on_annotate_cb= _on_annotate,
            parent        = self,
        )
        dlg.exec()
        # Güncellenmiş ASTAP ayarlarını kaydet
        try:
            from gui.settings import save as _save
            _save(self._settings)
        except Exception:
            pass

    def _open_starless_file(self):
        """Yıldızsız görüntü aç (recomposition için)."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Yıldızsız Görüntü Aç",
            self._settings.get("last_open_dir", ""),
            "Görüntüler (*.fits *.fit *.tif *.tiff *.png *.jpg);;Tümü (*)")
        if not path: return
        try:
            from core.loader import load_image
            img = load_image(path)
            # StarNet sonucu olarak sakla
            if not hasattr(self, "_starless_img"):
                self._starless_img = None
            self._starless_img = img
            self.status.showMessage(
                f"✅  Yıldızsız yüklendi: {os.path.basename(path)} "
                f"({img.shape[1]}×{img.shape[0]})")
        except Exception as e:
            QMessageBox.critical(self, "Yükleme Hatası", str(e))

    def _open_stars_file(self):
        """Yıldız maskesi / stars-only aç (recomposition için)."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Yıldız Maskesi Aç",
            self._settings.get("last_open_dir", ""),
            "Görüntüler (*.fits *.fit *.tif *.tiff *.png *.jpg);;Tümü (*)")
        if not path: return
        try:
            from core.loader import load_image
            img = load_image(path)
            if not hasattr(self, "_stars_img"):
                self._stars_img = None
            self._stars_img = img
            self.status.showMessage(
                f"✅  Yıldız maskesi yüklendi: {os.path.basename(path)} "
                f"({img.shape[1]}×{img.shape[0]})")
        except Exception as e:
            QMessageBox.critical(self, "Yükleme Hatası", str(e))

    # ── Toolbar ─────────────────────────────────────────────────────────
    def _make_toolbar(self):
        """\n        Two-row toolbar:\n          Row 1: Logo image + App name + File/Edit/Stack/StarNet/View/Tools buttons\n          Row 2: Process shortcuts (click → expand that panel in left sidebar)\n        """
        container = QWidget()
        container.setStyleSheet(
            f"background:qlineargradient(x1:0,y1:0,x2:0,y2:1,"
            f"stop:0 {BG4},stop:0.7 {BG3},stop:1 {BG2});"
            f"border-bottom:2px solid {BORDER};")
        vlay = QVBoxLayout(container)
        vlay.setContentsMargins(0,0,0,0); vlay.setSpacing(0)

        # ── Row 1: main toolbar ───────────────────────────────────────────
        row1 = QWidget(); row1.setFixedHeight(68)
        row1.setStyleSheet("background:transparent;border:none;")
        lay = QHBoxLayout(row1); lay.setContentsMargins(8,4,8,4); lay.setSpacing(4)

        # Logo image
        logo_lbl = QLabel()
        logo_path = os.path.join(os.path.dirname(__file__), "logo_thumb.jpg")
        if os.path.exists(logo_path):
            pix = QPixmap(logo_path).scaled(52, 52,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation)
            logo_lbl.setPixmap(pix)
            logo_lbl.setFixedSize(54, 54)
            logo_lbl.setStyleSheet("border:1px solid "+BORDER+";border-radius:4px;")
        else:
            logo_lbl.setText("🔭")
            logo_lbl.setStyleSheet("font-size:28px;")
        lay.addWidget(logo_lbl)

        # App name + author
        name_widget = QWidget()
        name_widget.setStyleSheet("background:transparent;border:none;")
        name_vlay = QVBoxLayout(name_widget)
        name_vlay.setContentsMargins(4,0,0,0); name_vlay.setSpacing(0)
        name_lbl = QLabel("ASTRO MAESTRO PRO")
        name_lbl.setStyleSheet(
            f"color:{ACCENT2};font-size:12px;font-weight:700;letter-spacing:2px;")
        by_lbl = QLabel("by Deniz")
        by_lbl.setStyleSheet(
            f"color:{GOLD};font-size:9px;font-weight:500;letter-spacing:1px;")
        name_vlay.addWidget(name_lbl)
        name_vlay.addWidget(by_lbl)
        lay.addWidget(name_widget)
        lay.addSpacing(4); lay.addWidget(self._vsep()); lay.addSpacing(4)

        # FILE
        lay.addWidget(self._grp_lbl("FILE"))
        self._tb_open  = make_icon_btn("📂","Open")
        self._tb_save  = make_icon_btn("💾","Save")
        lay.addWidget(self._tb_open); lay.addWidget(self._tb_save)
        self._tb_open.clicked.connect(self._open_file)
        self._tb_save.clicked.connect(self._save_file)

        lay.addWidget(self._vsep())
        lay.addWidget(self._grp_lbl("EDIT"))
        self._tb_undo  = make_icon_btn("↺","Undo")
        self._tb_reset = make_icon_btn("🔄","Reset","#2a1010",RED)
        lay.addWidget(self._tb_undo); lay.addWidget(self._tb_reset)
        self._tb_undo.clicked.connect(self._undo)
        self._tb_reset.clicked.connect(self._reset_all)

        lay.addWidget(self._vsep())
        lay.addWidget(self._grp_lbl("STACK"))
        self._tb_stack = make_icon_btn("🗂","Stack","#0a1a2a",ACCENT)
        lay.addWidget(self._tb_stack)
        self._tb_stack.clicked.connect(self._open_stacking)

        lay.addWidget(self._vsep())
        lay.addWidget(self._grp_lbl("STARNET"))
        self._tb_starless = make_icon_btn("🌌","Starless","#1a0a2a",PURPLE)
        self._tb_recomp   = make_icon_btn("✦+","Recompose","#0a1a2a",GOLD)
        lay.addWidget(self._tb_starless)
        lay.addWidget(self._tb_recomp)
        self._tb_starless.clicked.connect(self._run_starnet_and_save)
        self._tb_recomp.clicked.connect(self._open_recomposition)
        self._tb_starless.setToolTip(
            "Run StarNet++ on current image\n"
            "Saves starless + stars-only files\n"
            "Set StarNet++ path in ⚙ Settings first")
        self._tb_recomp.setToolTip(
            "Star Recomposition\n"
            "Blend starless + stars-only back together\n"
            "with layer controls, colour and star size options")

        lay.addWidget(self._vsep())
        lay.addWidget(self._grp_lbl("VIEW"))
        self._tb_orig = make_icon_btn("🖼","Original")
        self._tb_zfit = make_icon_btn("⛶","Fit")
        lay.addWidget(self._tb_orig); lay.addWidget(self._tb_zfit)
        self._tb_orig.clicked.connect(self._show_original)
        self._tb_zfit.clicked.connect(lambda: self.viewer._zoom_fit())

        lay.addWidget(self._vsep())
        lay.addWidget(self._grp_lbl("TOOLS"))
        self._tb_settings   = make_icon_btn("⚙","Settings","#0a0a1a",ACCENT2)
        self._tb_customize  = make_icon_btn("🔧","Panels","#0a1a0a",GREEN)
        self._tb_platesolve = make_icon_btn("🔭","PlateSolve","#0a1a2a","#88ddff")
        lay.addWidget(self._tb_settings); lay.addWidget(self._tb_customize)
        lay.addWidget(self._tb_platesolve)
        self._tb_settings.clicked.connect(self._open_settings)
        self._tb_customize.clicked.connect(self._open_panel_customize)
        self._tb_platesolve.clicked.connect(self._open_plate_solve)
        self._tb_platesolve.setToolTip(
            "Plate Solve — ASTAP\n"
            "Görüntünün gökyüzündeki konumunu bul\n"
            "Settings → ASTAP sekmesinden yolu ayarlayın")

        # Update butonu (sağ köşe)
        self._tb_update = make_icon_btn("🔄","Update","#0a1a0a","#44dd88")
        lay.addWidget(self._tb_update)
        self._tb_update.clicked.connect(self._open_update_dialog)
        self._tb_update.setToolTip(
            "Güncelleme Kontrolü\n"
            "Yeni sürüm var mı kontrol et")

        lay.addStretch()
        self.lbl_file = QLabel("No file")
        self.lbl_file.setStyleSheet(f"color:{SUBTEXT};font-size:9px;")
        self.lbl_file.setAlignment(
            Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter)
        lay.addWidget(self.lbl_file)
        vlay.addWidget(row1)

        # ── Row 2: process shortcut bar ───────────────────────────────────
        row2 = QWidget(); row2.setFixedHeight(32)
        row2.setStyleSheet(
            f"background:{BG2};border-top:1px solid {BORDER};border-bottom:none;")
        r2lay = QHBoxLayout(row2)
        r2lay.setContentsMargins(8,2,8,2); r2lay.setSpacing(2)

        self._proc_btns = {}
        proc_shortcuts = [
            ("bg",      "🌌","BG Extract"),
            ("noise",   "✨","Noise"),
            ("deconv",  "🔭","Deconv"),
            ("sharp",   "🔪","Sharpen"),
            ("nebula",  "🌠","Nebula"),
            ("color",   "🎨","Color"),
            ("morph",   "🔮","Morph"),
            ("galaxy",  "🌀","Galaxy"),
            ("stretch", "📊","Stretch"),
            ("crop",    "✂", "Crop"),
            ("script",  "⚡","Script"),
        ]
        for key, icon, label in proc_shortcuts:
            b = QPushButton(f"{icon}  {label}")
            b.setFixedHeight(26)
            b.setStyleSheet(
                f"QPushButton{{background:{BG3};color:{MUTED};border:1px solid {BORDER};"
                f"border-radius:4px;padding:0 10px;font-size:10px;font-weight:500;}}"
                f"QPushButton:hover{{background:{BG4};color:{ACCENT2};"
                f"border:1px solid {ACCENT};}}"
                f"QPushButton:checked{{background:{BG4};color:{GOLD};"
                f"border:1px solid {GOLD};}}")
            b.setCheckable(True)
            b.setToolTip(f"Click to open {label} settings")
            b.clicked.connect(lambda checked, k=key, btn=b: self._show_process_flyout(k, btn))
            self._proc_btns[key] = b
            r2lay.addWidget(b)

        r2lay.addStretch()

        # Workflow rehber butonu
        self._btn_workflow = QPushButton("🌌  Workflow")
        self._btn_workflow.setFixedHeight(26)
        self._btn_workflow.setCheckable(True)
        self._btn_workflow.setStyleSheet(
            f"QPushButton{{background:#0a1a2a;color:{ACCENT2};"
            f"border:1px solid {ACCENT2};border-radius:4px;"
            f"padding:0 10px;font-size:10px;font-weight:600;}}"
            f"QPushButton:hover{{background:#0a2a3a;border-color:{ACCENT2};}}"
            f"QPushButton:checked{{background:#0a2a3a;color:{GOLD};"
            f"border-color:{GOLD};}}")
        self._btn_workflow.setToolTip("Astro Fotoğrafçılık Workflow Rehberi\nHer adıma tıklayınca ilgili panel açılır")
        self._btn_workflow.clicked.connect(self._show_workflow)
        r2lay.addWidget(self._btn_workflow)

        # Customize panels button
        btn_cust = QPushButton("🔧")
        btn_cust.setFixedSize(28, 26)
        btn_cust.setStyleSheet(
            f"QPushButton{{background:{BG};color:{SUBTEXT};border:1px solid {BORDER};"
            f"border-radius:3px;font-size:13px;}}"
            f"QPushButton:hover{{color:{TEXT};border-color:{ACCENT};}}")
        btn_cust.setToolTip("Customize panel order/visibility")
        btn_cust.clicked.connect(self._open_panel_customize)
        r2lay.addWidget(btn_cust)
        vlay.addWidget(row2)

        return container

    def _grp_lbl(self, t):
        l=QLabel(t); l.setStyleSheet(f"color:{SUBTEXT};font-size:8px;font-weight:700;letter-spacing:1px;padding:0 2px;")
        l.setAlignment(Qt.AlignmentFlag.AlignBottom|Qt.AlignmentFlag.AlignHCenter)
        return l

    def _vsep(self):
        s=QFrame(); s.setFrameShape(QFrame.Shape.VLine)
        s.setStyleSheet(f"color:{BORDER};"); s.setFixedWidth(1); return s

    # ── Left panel ──────────────────────────────────────────────────────

    def _make_crop_panel(self):
        g=QGroupBox(); g.setTitle("✂  Crop"); g.setStyleSheet(GROUP_CSS)
        lay=QVBoxLayout(g); lay.setContentsMargins(6,4,6,6); lay.setSpacing(4)
        info=QLabel("'Start' → Draw selection → 'Apply'")
        info.setStyleSheet(f"color:{SUBTEXT};font-size:10px;"); lay.addWidget(info)
        row=QHBoxLayout()
        self.btn_crop_start=QPushButton("✂  Start")
        self.btn_crop_apply=QPushButton("✅  Apply")
        self.btn_crop_reset=QPushButton("↺  Reset")
        self.btn_crop_start.setStyleSheet(_btn(color="#0a2a1a",hover=GREEN))
        self.btn_crop_apply.setStyleSheet(_btn(color="#0a3a1a",hover=GREEN))
        self.btn_crop_reset.setStyleSheet(_btn(color="#2a1a00",hover="#8a5a00"))
        for b in (self.btn_crop_apply,self.btn_crop_reset): b.setEnabled(False)
        for b in (self.btn_crop_start,self.btn_crop_apply,self.btn_crop_reset): row.addWidget(b)
        lay.addLayout(row)
        self.btn_crop_start.clicked.connect(self._crop_start)
        self.btn_crop_apply.clicked.connect(self._crop_apply)
        self.btn_crop_reset.clicked.connect(self._crop_reset)
        return g


    def _update_stretch_vis(self):
        m=self._st_method.v()
        for w in self._ghs_widgets:  w.setVisible(m=="hyperbolic")
        for w in self._stat_widgets: w.setVisible(m=="statistical")
        for w in self._pow_widgets:  w.setVisible(m=="power")
        for w in self._stf_widgets:  w.setVisible(m=="auto_stf")

    def _build_viewer(self):
        w=QWidget(); w.setStyleSheet(f"background:{BG};")
        lay=QVBoxLayout(w); lay.setContentsMargins(0,0,0,0)

        hbar=QWidget(); hbar.setFixedHeight(30)
        hbar.setStyleSheet(f"background:{BG2};border-bottom:1px solid {BORDER};")
        hl=QHBoxLayout(hbar); hl.setContentsMargins(8,0,8,0); hl.setSpacing(4)
        self.lbl_step=QLabel("Step: —")
        self.lbl_step.setStyleSheet(f"color:{MUTED};font-size:10px;")
        hl.addWidget(self.lbl_step); hl.addStretch()

        # Slot selector for multi-view
        slot_lbl = QLabel("Load into slot:")
        slot_lbl.setStyleSheet(LBL_CSS)
        hl.addWidget(slot_lbl)
        self._slot_combo = QComboBox()
        self._slot_combo.addItems(["1","2","3","4"])
        self._slot_combo.setStyleSheet(COMBO_CSS)
        self._slot_combo.setFixedWidth(46)
        self._slot_combo.setToolTip("Which panel slot to load images into")
        hl.addWidget(self._slot_combo)

        lay.addWidget(hbar)
        self.viewer=ImageViewer(); lay.addWidget(self.viewer,1)
        # Wire histogram Apply button → create new history step
        self.viewer._hist_apply_cb = lambda img: self._hist_apply(img)
        return w

    def _build_history_panel(self):
        self.hist_panel=HistoryPanel()
        self.hist_panel.jump_requested.connect(self._jump_to)
        return self.hist_panel

    # ── File ops ────────────────────────────────────────────────────────
    def _open_file(self):
        # En son kullanılan dosyadan başla
        last_file = self._settings.get("last_open_file", "")
        start_dir = (os.path.dirname(last_file)
                     if last_file and os.path.isfile(last_file)
                     else self._settings.get("last_open_dir", ""))
        path,_=QFileDialog.getOpenFileName(self,"Open Image", start_dir,
            "All Supported (*.fits *.fit *.fts *.png *.tiff *.tif *.jpg *.jpeg *.bmp *.cr2 *.nef *.arw);;"
            "FITS (*.fits *.fit *.fts);;PNG (*.png);;TIFF (*.tiff *.tif);;"
            "JPEG (*.jpg *.jpeg);;RAW (*.cr2 *.nef *.arw);;All (*)")
        if not path: return
        self._load_path(path)

    def _load_path(self, path):
        """Verilen dosyayı yükle ve ayarlara kaydet."""
        try:
            from core.loader import load_image
            img=load_image(path); self._orig=img.copy()
            self._set_image(img,"Original",reset=True)
            self.lbl_file.setText(os.path.basename(path))
            h,w=img.shape[:2]; ch="RGB" if img.ndim==3 else "Gray"
            self.status.showMessage(f"✅  {os.path.basename(path)}  |  {w}×{h}  |  {ch}")
            # Son kullanılan dosyayı kaydet
            self._settings["last_open_file"] = path
            self._settings["last_open_dir"]  = os.path.dirname(path)
            self._add_to_recent(path)
            from gui.settings import save as _save_cfg
            _save_cfg(self._settings)
        except Exception as e:
            QMessageBox.critical(self,"Error",f"{e}\n\n{traceback.format_exc()}")

    def _save_file(self):
        if self._current is None: QMessageBox.information(self,"Info","No image loaded."); return
        path,_=QFileDialog.getSaveFileName(self,"Save","result.fits",
            "FITS (*.fits);;PNG (*.png);;TIFF (*.tiff);;JPEG (*.jpg);;All (*)")
        if not path: return
        try:
            from core.loader import save_image; save_image(path,self._current)
            self.status.showMessage(f"💾  Kaydedildi: {path}")
        except Exception as e:
            QMessageBox.critical(self,"Save Error",str(e))

    # ── Process flyout system ────────────────────────────────────────────
    def _show_process_flyout(self, key: str, btn: QPushButton):
        """Show/hide flyout popup for a process panel."""
        # Script: editoru ac
        if key == "script":
            b = self._proc_btns.get("script")
            if b: b.setChecked(False)
            self._open_script_editor()
            return

        # Crop: toggle crop mode directly — no flyout
        if key == "crop":
            b = self._proc_btns.get("crop")
            if getattr(self, "_crop_active", False):
                # Crop modu aktifse kapat
                self._crop_active = False
                if b: b.setChecked(False)
                self.viewer.exit_crop_mode()
                self.status.showMessage("Crop iptal edildi")
            else:
                # Crop modunu başlat
                if self._current is None:
                    QMessageBox.information(self, "Info", "Once bir resim acin.")
                    if b: b.setChecked(False)
                    return
                self._crop_active = True
                if b: b.setChecked(True)
                self.viewer._crop_apply_cb  = self._inline_crop_apply
                self.viewer._crop_cancel_cb = self._inline_crop_cancel
                self.viewer.enter_crop_mode()
                self.status.showMessage("✂  Sol tik surukle sec  |  Sag tik → Crop Uygula")
            return

        # Close any other open flyout first
        if hasattr(self, "_active_flyout") and self._active_flyout:
            try:
                self._active_flyout.close()
            except Exception:
                pass
            self._active_flyout = None
            # Deselect all buttons
            for b in self._proc_btns.values():
                b.setChecked(False)
            # If we just closed the same panel, don't reopen
            if not btn.isChecked():
                return

        panel = self._panels.get(key)
        if panel is None:
            btn.setChecked(False)
            return

        # Mark button as active
        for k, b in self._proc_btns.items():
            b.setChecked(k == key)

        # Bazi butonlar birden fazla panel gosterir
        extra_panels = []
        if key == "deconv":
            stars_panel = self._panels.get("stars")
            if stars_panel:
                extra_panels = [stars_panel]

        # Create flyout
        flyout = ProcessFlyout(panel, btn, parent=self, extra_panels=extra_panels)

        def _on_flyout_closed():
            btn.setChecked(False)
            for b in self._proc_btns.values():
                b.setChecked(False)
            if (hasattr(self,"_active_flyout") and
                    self._active_flyout is flyout):
                self._active_flyout = None

        flyout.closed.connect(_on_flyout_closed)
        self._active_flyout = flyout
        flyout.show()
        flyout._position()

        # Stretch paneli açılınca goruntu analiz et
        if key == "stretch" and self._current is not None:
            self._auto_analyse_stretch()

        # Veralux panel'in mode/sensor combo'larını run_requested'e inject et
        if key == "stretch":
            def _inject_vl_params(params):
                if hasattr(self, "_vl_mode"):
                    params["vl_mode"]   = self._vl_mode.currentText()
                    params["vl_sensor"] = self._vl_sensor.currentText()
                return self._run_key("stretch", params)
            panel.btn_run.clicked.disconnect()
            panel.btn_run.clicked.connect(lambda: _inject_vl_params(panel.collect()))

    def _make_process_panels_headless(self):
        """\n        Create all process panels without adding them to any layout.\n        They are shown on demand via ProcessFlyout popups.\n        """
        def _make(icon, title, key):
            p = ProcessPanel(icon, title, key=key)
            self._panels[key] = p
            return p

        # Background Extraction — GraXpert Siril tarzı
        p = _make("🌌","Background Extraction","bg")
        p.add_combo("method","Method",
                    ["graxpert","nox","dbe_spline","polynomial","ai_gradient","median_grid","gaussian_sub"],
                    "graxpert" if self._settings.get("graxpert_exe","") else "nox",
                    "graxpert   — GraXpert AI (en iyi kalite)\n"
                    "nox        — Veralux Nox (membran)\n"
                    "dbe_spline — RBF spline\n"
                    "polynomial — 2D polynomial\n"
                    "ai_gradient— Yapı maskelı polynomial")
        # ── GraXpert parametreleri (Siril tarzı) ──
        gx_hdr = QLabel("── GraXpert AI ──")
        gx_hdr.setStyleSheet(f"color:#7af0a0;font-size:10px;font-weight:700;")
        gx_hdr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        p._bl.addWidget(gx_hdr)
        self._gx_model  = p.add_combo("gx_model","Select Model",
                                       ["1.0.0","2.0.0","3.0.0","latest"],"1.0.0")
        self._gx_smooth = p.add_slider("gx_smoothing","Smoothing",0.0,1.0,0.5,2)
        self._gx_corr   = p.add_combo("gx_correction","Correction Type",
                                       ["Subtraction","Division"],"Subtraction")
        self._gx_keep   = p.add_check("gx_keep_bg","Keep background")
        self._gx_keep.setChecked(False)
        # Advanced grubu
        adv_hdr = QLabel("Advanced")
        adv_hdr.setStyleSheet(
            f"color:{SUBTEXT};font-size:9px;font-weight:600;"
            f"border:1px solid {BORDER};border-radius:3px;padding:2px 6px;"
            f"background:{BG3};margin-top:4px;")
        p._bl.addWidget(adv_hdr)
        self._gx_batch = p.add_slider("gx_batch","Batch Size",1,8,4,0)
        self._gx_gpu   = p.add_check("gx_gpu","Use GPU acceleration (if available)")
        self._gx_gpu.setChecked(True)
        # Klasik metodlar
        cls_hdr = QLabel("── Klasik Metodlar ──")
        cls_hdr.setStyleSheet(f"color:{SUBTEXT};font-size:9px;font-weight:600;margin-top:4px;")
        cls_hdr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        p._bl.addWidget(cls_hdr)
        p.add_slider("grid_size","Grid Size",8,32,16,0)
        p.add_slider("poly_degree","Poly Degree",2,6,4,0)
        p.add_slider("clip_low","Clip Low (%)",0,10,0,1)
        self._gx_widgets = [gx_hdr, self._gx_model, self._gx_smooth,
                            self._gx_corr, self._gx_keep, adv_hdr,
                            self._gx_batch, self._gx_gpu]
        # method degisince GraXpert kontrollerini goster/gizle
        def _bg_method_changed(method):
            is_gx = (method == "graxpert")
            for w in self._gx_widgets: w.setVisible(is_gx)
            cls_hdr.setVisible(not is_gx)
        # p._params["method"][0] = PC widget (combo)
        p._params["method"][0].cb.currentTextChanged.connect(_bg_method_changed)
        # Baslangicta goster/gizle
        cur_method = "graxpert" if self._settings.get("graxpert_exe","") else "nox"
        _bg_method_changed(cur_method)
        p.run_requested.connect(lambda s,k="bg": self._run_key(k,s))

        # Noise Reduction
        p = _make("✨","Noise Reduction","noise")
        p.add_combo("method","Method",
                    ["silentium","bilateral","gaussian","median","nlm","noisexterminator","graxpert"],
                    "silentium",
                    "silentium        — Veralux Silentium (linear-phase, en iyi)\n"
                    "bilateral        — Kenar-koruyucu bilateral\n"
                    "gaussian         — Gaussian bulaniklik\n"
                    "median           — Medyan filtre\n"
                    "nlm              — Non-local means\n"
                    "noisexterminator — Wavelet astro gurultu\n"
                    "graxpert         — GraXpert AI (exe gerekli)")
        p.add_slider("strength","Strength",0,1,0.7,2)
        p.add_slider("detail","Detail Preserve",0,1,0.5,2,
                     "Yuksek = detay koru (noisexterminator)")
        p.add_slider("iterations","Iterations",1,5,1,0)
        p.run_requested.connect(lambda s,k="noise": self._run_key(k,s))

        # Deconvolution
        p = _make("🔭","Deconvolution (R-L)","deconv")
        p.add_combo("psf_type","PSF Type",["moffat","gaussian","airy","lorentzian","box"],"moffat")
        p.add_combo("method","Method",
                    ["richardson_lucy","blind","wiener","total_variation","blur_exterminator"],"richardson_lucy",
                    "richardson_lucy — Classic RL\nblind — AI PSF from stars\nwiener — Noise-robust\n"
                    "total_variation — TV regularised\nblur_exterminator — Blind+multi-scale+TV (best)")
        p.add_slider("psf_size","PSF Size (px)",3,21,5,0)
        p.add_slider("iterations","Iterations",5,100,20,0)
        p.add_slider("clip","Clip Limit",0,1,1.0,2)
        p.add_slider("wiener_snr","Wiener SNR",5,200,30,0)
        p.add_slider("tv_weight","TV Weight",0.01,1.0,0.1,2)
        p.add_slider("strength","BE Strength",0,1,1.0,2,"Blur Exterminator blend")
        p.add_slider("noise_level","BE Noise",0.001,0.1,0.01,3,"Blur Exterminator noise level")
        p.run_requested.connect(lambda s,k="deconv": self._run_key(k,s))

        # Star Removal (Deconv panelinin altinda)
        p = _make("⭐","Star Removal","stars")
        p.add_combo("method","Method",
                    ["starcomposer","starsmalerx","local_median","global_median","inpaint"],
                    "starcomposer",
                    "starcomposer — Veralux StarComposer (yildiz yeniden birlestir)\n"
                    "starsmalerx  — Wavelet yildiz kaldir\n"
                    "local_median — Yerel medyan dolgu\n"
                    "global_median— Genel medyan dolgu\n"
                    "inpaint      — OpenCV inpainting")
        p.add_slider("strength","Strength",0,1,0.9,2,
                     "Yildiz azaltma gucu")
        p.add_slider("sensitivity","Sensitivity",0,1,0.5,2,
                     "Yildiz tespiti hassasiyeti")
        p.add_slider("feather","Feather Radius",1,10,3,1,
                     "Kenar yumusatma piksel")
        p.add_sep()
        p.add_slider("max_sigma","Max Sigma (klasik)",1,25,6,0)
        p.add_slider("min_sigma","Min Sigma (klasik)",1,10,1,0)
        p.add_slider("threshold","Threshold (klasik)",0.001,0.3,0.03,3)
        p.run_requested.connect(lambda s,k="stars": self._run_key(k,s))

        # Sharpening
        p = _make("🔪","Sharpening","sharp")
        p.add_combo("method","Method",
                    ["revela","multiscale_vlc","unsharp_mask","laplacian_ai","high_pass"],
                    "revela",
                    "revela        — Veralux Revela (wavelet, en iyi)\n"
                    "multiscale_vlc— Multiscale VLC\n"
                    "unsharp_mask  — Unsharp Mask\n"
                    "laplacian_ai  — Laplacian AI\n"
                    "high_pass     — High Pass")
        p.add_slider("radius","Radius (px)",0.5,10,2.0,1)
        p.add_slider("amount","Amount",0,5,1.0,2)
        p.add_slider("threshold","Threshold",0,0.5,0.0,3)
        p.add_slider("scale_levels","Scale Levels",2,6,4,0,"Multiscale VLC levels")
        p.add_slider("iterations","Iterations",1,5,1,0)
        p.run_requested.connect(lambda s,k="sharp": self._run_key(k,s))

        # Nebula Enhancement
        p = _make("🌠","Nebula Enhancement","nebula")
        p.add_combo("method","Method",["multiscale_lce","hdrgc","structure_amp","clahe_astro","legacy"],"multiscale_lce",
                    "multiscale_lce — PixInsight LCE\nhdrgc — HDR Gamma\nstructure_amp — AI Structure\nclahe_astro — CLAHE")
        p.add_slider("strength","Strength (0–5)",0,5,2.0,1)
        p.add_slider("blur_kernel","Blur Kernel (px)",11,101,51,0)
        p.add_slider("blend","Blend (0–1)",0,1,1.0,2)
        p.add_slider("clahe_clip","CLAHE Clip",1.0,10.0,3.0,1)
        p.run_requested.connect(lambda s,k="nebula": self._run_key(k,s))

        # Color Calibration
        p = _make("🎨","Color Calibration","color")
        p.add_combo("method","Method",
                    ["spcc_g2v","avg_spiral","vectra","alchemy","pcc","ai_neutral","photometric","white_balance"],
                    "spcc_g2v",
                    "spcc_g2v   — G2V Yildiz ref. (Gunes, 5778K) — EN DOGRU RENK\n"
                    "avg_spiral — Average Spiral Galaxy (PixInsight PCC default)\n"
                    "vectra     — Veralux Vectra (LCH renk cerrahisi)\n"
                    "alchemy    — Veralux Alchemy (narrowband karisim)\n"
                    "pcc        — Yildiz tabanlı PCC\n"
                    "ai_neutral — AI arka plan notralizasyonu\n"
                    "photometric— Basit fotometri\n"
                    "white_balance — Yuzdelik beyaz nokta")
        p.add_slider("neutral_percentile","Neutral %ile",10,90,50,0,"AI neutral: background percentile")
        p.run_requested.connect(lambda s,k="color": self._run_key(k,s))

        # Morphology
        p = _make("🔮","Morphology","morph")
        p.add_combo("operation","Operation",["erosion","dilation","opening","closing"],"opening")
        p.add_slider("kernel_size","Kernel (px)",1,11,3,0)
        p.add_slider("iterations","Iterations",1,5,1,0)
        p.run_requested.connect(lambda s,k="morph": self._run_key(k,s))

        # Galaxy Detection
        p = _make("🌀","Galaxy Detection","galaxy")
        p.add_slider("percentile","Threshold (%)",50,99,95,0)
        p.add_slider("blur_size","Blur (px)",11,101,31,0)
        p.run_requested.connect(lambda s,k="galaxy": self._run_key(k,s))

        # Histogram Stretch
        p = _make("📊","Histogram Stretch","stretch")
        self._st_method = p.add_combo("method","Method",
            ["veralux","auto_stf","linear","hyperbolic","asinh","log","midtone","statistical","power"],
            "veralux",
            "veralux   — Veralux HyperMetric Stretch (HMS) — fizik tabanli\n"
            "auto_stf  — PixInsight Auto STF\n"
            "linear    — Percentile clip\n"
            "hyperbolic— GHS (Generalised Hyperbolic)\n"
            "asinh     — Arcsinh\n"
            "statistical—mean±k·σ\n"
            "power     — x^α")
        p.add_slider("low","Clip Low (%)",0,20,2.0,1)
        p.add_slider("high","Clip High (%)",80,100,98.0,1)
        p.add_slider("gamma","Gamma",0.1,3.0,1.0,2)
        p.add_sep()
        vl_t=QLabel("── Veralux HMS Parameters ──"); vl_t.setStyleSheet(f"color:#7af0c0;font-size:10px;font-weight:600;")
        vl_t.setAlignment(Qt.AlignmentFlag.AlignCenter); p._bl.addWidget(vl_t)
        self._vl_log_d      = p.add_slider("vl_log_d","Log D",0.5,20.0,2.0,1,"Stretch yogunlugu (Auto-Calc ile hesaplanir)")
        self._vl_tb         = p.add_slider("vl_target_bg","Target BG",0.05,0.50,0.20,2,"Hedef arka plan parlakligi")
        self._vl_cg         = p.add_slider("vl_color_grip","Color Grip",0.0,1.0,1.0,2,"1.0=tam renk koruma  0.0=klasik stretch")
        self._vl_scr        = p.add_slider("vl_star_core","Star Core Rec.",0.0,1.0,0.5,2,"Yildiz cekirdek kurtarma")
        self._vl_sha        = p.add_slider("vl_shadow","Shadow Auth.",0.0,1.0,0.0,2,"Golge korumasi")
        vl_row = QHBoxLayout(); vl_row.setSpacing(6)
        vl_mode_lbl = QLabel("Mode:"); vl_mode_lbl.setStyleSheet(LBL_CSS); vl_mode_lbl.setFixedWidth(60)
        self._vl_mode = QComboBox(); self._vl_mode.addItems(["preserve","scientific"])
        self._vl_mode.setStyleSheet(COMBO_CSS); self._vl_mode.setFixedWidth(110)
        self._vl_mode.setToolTip("preserve=renk koru (default)  scientific=fotometrik")
        vl_sensor_lbl = QLabel("Sensor:"); vl_sensor_lbl.setStyleSheet(LBL_CSS); vl_sensor_lbl.setFixedWidth(55)
        self._vl_sensor = QComboBox()
        self._vl_sensor.addItems(["Rec.709","IMX294","IMX571","IMX533","ASI2600MM","Canon DSLR","Nikon DSLR","Equal (Mono)"])
        self._vl_sensor.setStyleSheet(COMBO_CSS); self._vl_sensor.setFixedWidth(120)
        b_autod = QPushButton("⚡ Auto-Calc Log D"); b_autod.setStyleSheet(_btn(h=22)); b_autod.setFixedHeight(22)
        b_autod.setToolTip("Mevcut resim icin optimal Log D degerini hesapla")
        b_autod.clicked.connect(self._veralux_auto_calc)
        vl_row.addWidget(vl_mode_lbl); vl_row.addWidget(self._vl_mode)
        vl_row.addWidget(vl_sensor_lbl); vl_row.addWidget(self._vl_sensor)
        p._bl.addLayout(vl_row)
        p._bl.addWidget(b_autod)
        self._vl_widgets = [vl_t, self._vl_log_d, self._vl_tb, self._vl_cg,
                            self._vl_scr, self._vl_sha]
        p.add_sep()
        ghs_t=QLabel("── GHS Parameters ──"); ghs_t.setStyleSheet(f"color:{GOLD};font-size:10px;font-weight:600;")
        ghs_t.setAlignment(Qt.AlignmentFlag.AlignCenter); p._bl.addWidget(ghs_t)
        self._hs_D  = p.add_slider("hs_D","D — Intensity",0,100000,5000,0,"Typical: 1000–20000")
        self._hs_b  = p.add_slider("hs_b","b — Shape",-5,15,0.0,2)
        self._hs_SP = p.add_slider("hs_SP","SP — Stretch Pt",0,1,0.1,3)
        self._hs_LP = p.add_slider("hs_LP","LP — Lower Limit",0,0.5,0.0,3)
        self._hs_HP = p.add_slider("hs_HP","HP — Upper Limit",0.5,1,1.0,3)
        self._ghs_widgets=[ghs_t,self._hs_D,self._hs_b,self._hs_SP,self._hs_LP,self._hs_HP]
        stat_t=QLabel("── Statistical Parameters ──"); stat_t.setStyleSheet(f"color:{PURPLE};font-size:10px;font-weight:600;")
        stat_t.setAlignment(Qt.AlignmentFlag.AlignCenter); p._bl.addWidget(stat_t)
        self._stat_k=p.add_slider("stat_k","k (sigma factor)",0.5,10,2.5,1,"Smaller k → more aggressive")
        self._stat_widgets=[stat_t,self._stat_k]
        pow_t=QLabel("── Power Parameters ──"); pow_t.setStyleSheet(f"color:{GREEN};font-size:10px;font-weight:600;")
        pow_t.setAlignment(Qt.AlignmentFlag.AlignCenter); p._bl.addWidget(pow_t)
        self._pow_alpha=p.add_slider("power_alpha","α (exponent)",0.1,3.0,0.5,2,"α<1=brighten  α>1=darken")
        self._pow_widgets=[pow_t,self._pow_alpha]
        stf_t=QLabel("── Auto STF Parameters ──"); stf_t.setStyleSheet(f"color:{ACCENT2};font-size:10px;font-weight:600;")
        stf_t.setAlignment(Qt.AlignmentFlag.AlignCenter); p._bl.addWidget(stf_t)
        self._stf_target=p.add_slider("stf_target","Target Background",0.05,0.5,0.25,2,"STF target median")
        self._stf_clip=p.add_slider("stf_clip","Shadow Clip",-5.0,0.0,-2.8,1)
        self._stf_widgets=[stf_t,self._stf_target,self._stf_clip]
        self._update_stretch_vis()
        self._st_method.cb.currentTextChanged.connect(lambda _: self._update_stretch_vis())
        p.run_requested.connect(lambda s,k="stretch": self._run_key(k,s))

        # Store order
        self._panel_order   = list(self._panels.keys())
        self._panel_visible = {k: True for k in self._panels}


    def _hist_apply(self, img):
        """Called when histogram Apply is pressed — bakes into history."""
        self._set_image(img, "Histogram Adjust")
        self.status.showMessage("✅  Histogram adjustment applied")


    # ── Stretch auto-analysis ─────────────────────────────────────────────
    def _veralux_auto_calc(self):
        """Mevcut resim icin Veralux Log D degerini otomatik hesapla."""
        if self._current is None:
            self.status.showMessage("Resim yok — once bir goruntu acin")
            return
        try:
            from processing.veralux_hms import auto_calc_log_d
            sensor = self._vl_sensor.currentText() if hasattr(self,"_vl_sensor") else "Rec.709"
            tb     = float(self._vl_tb.sp.value()) if hasattr(self,"_vl_tb") else 0.20
            d = auto_calc_log_d(self._current, target_background=tb, sensor=sensor)
            if hasattr(self,"_vl_log_d"):
                self._vl_log_d.sp.setValue(d)
            self.status.showMessage(f"Veralux Auto-Calc: Log D = {d:.3f}")
        except Exception as e:
            self.status.showMessage(f"Auto-Calc hatasi: {e}")

    def _auto_analyse_stretch(self):
        """\n        Analyse current image and auto-set Histogram Stretch parameters.\n        Called when the Stretch flyout panel is opened.\n        Detects if image is:\n          - linear (dark, needs aggressive stretch) → sets auto_stf with calculated params\n          - already stretched (bright) → suggests gentle gamma / midtone\n          - noisy → prefers statistical\n        Updates all stretch sliders in-place.\n        """
        img = self._current
        if img is None: return

        import numpy as np

        flat  = img.ravel().astype(np.float64)
        median = float(np.median(flat))
        mad    = float(np.median(np.abs(flat - median))) * 1.4826  # robust sigma
        mean   = float(np.mean(flat))
        p1     = float(np.percentile(flat, 1))
        p99    = float(np.percentile(flat, 99))
        snr    = mean / max(mad, 1e-9)
        dynamic_range = p99 - p1

        # ── Decide best method ──────────────────────────────────────────────
        if median < 0.08:
            # Very dark linear image → Auto STF
            best_method = "auto_stf"
            # PixInsight STF shadow clip: c0 = median + shadow_clip * 1.4826 * mad
            # Target: background at 0.25
            shadow_clip = -2.8 if mad > 0.001 else -1.5
            target_bkg  = 0.25

        elif median < 0.25 and dynamic_range > 0.3:
            # Moderately dark → GHS hyperbolic
            best_method = "hyperbolic"
            # SP = normalised sky background
            sp = max(0.05, min(0.3, (median - p1) / max(dynamic_range, 1e-9)))
            # D proportional to how dark the image is
            D  = max(500, min(30000, int(5000 * (0.25 / max(median, 0.01)))))

        elif snr < 10:
            # Low SNR / noisy → statistical with robust k
            best_method = "statistical"
            stat_k = max(1.5, min(4.0, snr / 3.0))

        elif median > 0.4:
            # Already bright / over-stretched → gentle midtone or linear
            best_method = "linear"
        else:
            # Middle range → asinh
            best_method = "asinh"

        # ── Apply best method and parameters ───────────────────────────────
        # Set method combo
        self._st_method.set(best_method)
        self._update_stretch_vis()

        # Clip points from percentiles
        clip_low  = max(0.0, min(5.0, round(float(np.percentile(flat,  1)) * 100, 1)))
        clip_high = max(95.0, min(100.0, round(100.0 - float(np.percentile(flat, 0.5)) * 100, 1)))
        # Find the panel's low/high sliders
        panel = self._panels.get("stretch")
        if panel:
            for key, (w, _) in panel._params.items():
                if key == "low":   w.set(clip_low)
                if key == "high":  w.set(max(clip_high, 95.0))
                if key == "gamma": w.set(1.0)

        # Method-specific parameters
        if best_method == "auto_stf":
            self._stf_clip.set(round(shadow_clip, 1))
            self._stf_target.set(round(target_bkg, 2))

        elif best_method == "hyperbolic":
            self._hs_D.set(int(D))
            self._hs_b.set(0.0)
            self._hs_SP.set(round(float(sp), 3))
            self._hs_LP.set(round(float(p1), 3))
            self._hs_HP.set(round(min(1.0, float(p99) + 0.05), 3))

        elif best_method == "statistical":
            self._stat_k.set(round(float(stat_k), 1))

        # Build info message
        lines = [
            f"Method: {best_method}",
            f"Median: {median:.4f}",
            f"MAD σ:  {mad:.4f}",
            f"SNR:    {snr:.1f}",
            f"Range:  {p1:.3f} – {p99:.3f}",
        ]
        self.status.showMessage(
            "📊  Stretch auto-analysed — "
            + f"median={median:.4f}  σ={mad:.4f}  SNR={snr:.1f}  → {best_method}")


    # ── Star Recomposition ───────────────────────────────────────────────
    def _open_recomposition(self):
        """Open the Star Recomposition dialog directly — no questions asked."""
        # Pre-load from last StarNet run if available
        starless   = None
        stars_only = None
        if (self._starnet_fn is not None and
                hasattr(self._starnet_fn, "_last_result") and
                self._starnet_fn._last_result is not None):
            res        = self._starnet_fn._last_result
            starless   = res.get("starless")
            stars_only = res.get("stars_only")

        def _on_apply(result):
            self._set_image(result, "Recomposition")
            self.status.showMessage("✅  Star recomposition applied")

        # Open dialog immediately — files can be loaded inside the dialog
        dlg = RecompositionDialog(
            starless   = starless,
            stars      = stars_only,
            on_apply   = _on_apply,
            parent     = self,
        )
        dlg.exec()

    def _load_recomp_files(self):
        """Let user pick starless and stars-only files."""
        from core.loader import load_image
        QMessageBox.information(self, "Load Starless",
            "First select the STARLESS file:")
        p1, _ = QFileDialog.getOpenFileName(
            self, "Select Starless Image", "",
            "Images (*.fits *.tif *.tiff *.png *.jpg);;All (*)")
        if not p1: return None, None

        QMessageBox.information(self, "Load Stars Only",
            "Now select the STARS ONLY file:")
        p2, _ = QFileDialog.getOpenFileName(
            self, "Select Stars-Only Image", "",
            "Images (*.fits *.tif *.tiff *.png *.jpg);;All (*)")
        if not p2: return None, None

        try:
            sl = load_image(p1)
            st = load_image(p2)
            return sl, st
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))
            return None, None


    # ── Settings ────────────────────────────────────────────────────────
    def _open_update_dialog(self):
        """Güncelleme dialogunu açar."""
        from gui.update_dialog import UpdateDialog
        dlg = UpdateDialog(parent=self, auto_check=True)
        dlg.exec()

    def _load_settings(self):
        from gui.settings import load as _load_cfg
        self._settings = _load_cfg()
        fs = int(self._settings.get("font_size", 10))
        if fs != 10:
            font = self.font(); font.setPointSize(fs)
            self.setFont(font)
        # Startup güncelleme kontrolü (arka planda, 3 saniye sonra)
        if self._settings.get("check_updates_on_startup", False):
            QTimer.singleShot(3000, self._bg_update_check)

    def _bg_update_check(self):
        """Arka planda güncelleme kontrolü — bildirim varsa göster."""
        from gui.update_dialog import UpdateDialog
        UpdateDialog.check_on_startup(
            parent   = self,
            settings = self._settings,
        )

    def _open_settings(self):
        dlg = SettingsDialog(self._settings, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._settings = dlg.get_settings()
            from gui.settings import save as _save_cfg
            _save_cfg(self._settings)
            QMessageBox.information(
                self, "Settings Saved",
                "Settings saved.\nSome changes (font size, theme) take effect on next launch.")

    def _show_workflow(self):
        """Workflow rehber panelini göster/gizle."""
        btn = self._btn_workflow
        if not btn.isChecked():
            return
        wf = WorkflowPanel(btn, parent=self)
        wf.step_clicked.connect(self._workflow_step)
        wf.closed.connect(lambda: btn.setChecked(False))
        self._active_workflow = wf
        wf.show()
        wf._position()

    def _workflow_step(self, key: str):
        """Workflow adımına tıklanınca ilgili paneli/diyaloğu aç."""
        self._btn_workflow.setChecked(False)
        if key == "stack":
            self._open_stacking()
        elif key == "recomp":
            self._open_recomposition()
        elif key == "hist":
            # Histogram editörünü görünür yap
            if hasattr(self, "_hist_panel_wrap"):
                self._hist_panel_wrap.show()
                self._hist_panel_wrap.setMinimumWidth(280)
            self.status.showMessage("📈  Histogram editörü açıldı — sağdaki paneli kullanın")
        else:
            # İşlem paneli flyout olarak aç
            btn = self._proc_btns.get(key)
            if btn:
                # Önce tüm butonları sıfırla
                for b in self._proc_btns.values(): b.setChecked(False)
                btn.setChecked(True)
                self._show_process_flyout(key, btn)

    def _show_original(self):
        if self._orig is None: return
        self.viewer.show_image(self._orig,"Original")
        self.status.showMessage("🖼  Showing original")






    # ── StarNet++ one-click run & save ──────────────────────────────────
    def _run_starnet_and_save(self):
        """\n        One-click StarNet++:\n          1. Checks settings for exe path\n          2. Asks user for output folder\n          3. Runs StarNet++ in background thread\n          4. Saves  <name>_starless.tif  and  <name>_stars_only.tif\n          5. Shows starless in viewer\n        """
        if self._current is None:
            QMessageBox.information(self, "Info", "Please open an image first.")
            return

        # Check exe configured
        exe = self._settings.get("starnet_exe", "").strip()
        if not exe or not os.path.isfile(exe):
            reply = QMessageBox.question(
                self, "StarNet++ Not Configured",
                "StarNet++ executable is not set or not found.\n\n"
                "Would you like to open Settings to configure it?\n\n"
                "(Without a path, AI fallback will be used instead)",
                QMessageBox.StandardButton.Yes |
                QMessageBox.StandardButton.No |
                QMessageBox.StandardButton.Cancel)
            if reply == QMessageBox.StandardButton.Cancel:
                return
            if reply == QMessageBox.StandardButton.Yes:
                self._open_settings()
                # Re-read after settings dialog
                exe = self._settings.get("starnet_exe", "").strip()

        # Ask for output folder
        default_dir = (self._settings.get("last_save_dir","") or
                       self._settings.get("last_open_dir","") or
                       os.path.expanduser("~"))
        out_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Folder for StarNet++ Results", default_dir)
        if not out_dir:
            return   # user cancelled

        # Determine base filename
        lbl = self.lbl_file.text().strip()
        if lbl and lbl != "No file":
            base = os.path.splitext(lbl)[0]
        else:
            base = "image"

        # Build output paths
        starless_path   = os.path.join(out_dir, f"{base}_starless.tif")
        stars_only_path = os.path.join(out_dir, f"{base}_stars_only.tif")

        # Confirm
        exe_name = os.path.basename(exe) if exe and os.path.isfile(exe) else "AI fallback"
        stride   = int(self._settings.get("starnet_stride", 256))
        use_gpu  = bool(self._settings.get("starnet_use_gpu", False))

        reply2 = QMessageBox.question(
            self, "Run StarNet++",
            f"Engine:  {exe_name}\n"
            f"Stride:  {stride}\n"
            f"Output folder: {out_dir}\n\n"
            f"Will save:\n"
            f"  • {os.path.basename(starless_path)}\n"
            f"  • {os.path.basename(stars_only_path)}\n\n"
            "Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply2 != QMessageBox.StandardButton.Yes:
            return

        # Disable button while running
        self._tb_starless.setEnabled(False)
        self._tb_starless.setText("⏳\nRunning…")
        self.pbar.show(); self.pbar.setRange(0, 0)
        self.status.showMessage("🌌  StarNet++ running…")

        # Launch worker thread
        worker = _StarNetWorker(
            image    = self._current.copy(),
            exe      = exe,
            stride   = stride,
            use_gpu  = use_gpu,
            starless_path   = starless_path,
            stars_only_path = stars_only_path,
        )
        self._workers.append(worker)

        def _on_prog(step, total, msg):
            if total > 0:
                self.pbar.setRange(0, total); self.pbar.setValue(step)
            self.status.showMessage(f"🌌  {msg}")

        def _on_done(result):
            self._workers_cleanup(worker)
            self._tb_starless.setEnabled(True)
            self._tb_starless.setText("🌌\nStarless")
            self.pbar.hide(); self.pbar.setRange(0, 100)
            if result.get("error"):
                err = result["error"]
                self.status.showMessage(f"❌  StarNet++ failed")
                dlg = QMessageBox(self)
                dlg.setWindowTitle("StarNet++ Error")
                dlg.setIcon(QMessageBox.Icon.Critical)
                dlg.setText("StarNet++ failed:\n\n" + err[:300])
                dlg.setDetailedText(err)
                dlg.exec()
                return
            # Show starless in viewer and add to history
            starless = result["starless"]
            self._set_image(starless, "Starless (StarNet++)")
            # Status with file paths
            saved = result.get("saved", [])
            self.status.showMessage(
                f"✅  StarNet++ done — saved: {', '.join(os.path.basename(p) for p in saved)}")
            # Notify user
            QMessageBox.information(
                self, "StarNet++ Complete",
                f"✅  Star removal complete!\n\n"
                f"Saved to:\n"
                f"  {starless_path}\n"
                f"  {stars_only_path}\n\n"
                f"Starless image is now active in the viewer.")

        worker.progress.connect(_on_prog)
        worker.finished.connect(_on_done)
        worker.finished.connect(lambda _: self._cleanup_worker(worker))
        worker.start()

    def _cleanup_worker(self, worker):
        try: self._workers.remove(worker)
        except ValueError: pass

    def _workers_cleanup(self, worker):
        try: self._workers.remove(worker)
        except ValueError: pass


    # ── Panel navigation & customization ────────────────────────────────
    def _jump_to_panel(self, key: str):
        """Scroll to and expand a specific panel, highlight its button."""
        panel = self._panels.get(key)
        if panel is None: return
        # Expand it
        panel.expand()
        # Scroll to it
        scroll = self._left_scroll
        if scroll:
            scroll.ensureWidgetVisible(panel)
        # Highlight button briefly
        for k, b in self._proc_btns.items():
            b.setChecked(k == key)
        # Reset highlight after 1.5s
        QTimer.singleShot(1500, lambda: [b.setChecked(False)
                                          for b in self._proc_btns.values()])

    def _expand_all_panels(self, expand: bool):
        for panel in self._panels.values():
            if expand: panel.expand()
            else:      panel.collapse()

    def _open_panel_customize(self):
        """Dialog to reorder and show/hide process panels."""
        dlg = PanelCustomizeDialog(
            panels      = self._panels,
            order       = self._panel_order,
            visible     = self._panel_visible,
            parent      = self,
        )
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._panel_order   = dlg.get_order()
            self._panel_visible = dlg.get_visible()
            self._apply_panel_layout()

    def _apply_panel_layout(self):
        """Re-order and show/hide panels according to current settings."""
        return  # panel layout not used in flyout mode
        # Remove all from layout
        while self._panel_vb.count() > 0:
            item = self._panel_vb.takeAt(0)
            if item.widget():
                item.widget().setParent(None)
        # Re-add crop first
        if self._crop_panel_widget:
            self._panel_vb.addWidget(self._crop_panel_widget)
        # Add panels in order
        for key in self._panel_order:
            panel = self._panels.get(key)
            if panel:
                vis = self._panel_visible.get(key, True)
                panel.setVisible(vis)
                self._panel_vb.addWidget(panel)
        self._panel_vb.addStretch()


    def _undo(self):
        if len(self._history)>1:
            self._history.pop(); lbl,img=self._history[-1]
            self._current=img.copy()
            idx=len(self._history)-1
            self.viewer.show_image(img,lbl)
            self.lbl_step.setText(f"Step: {lbl}")
            self.hist_panel.truncate_to(idx); self.hist_panel.select(idx)
            self.status.showMessage(f"↺  Geri alındı → {lbl}")

    def _reset_all(self):
        """Reset: önce son işlemi geri al, tam sıfırlama için tekrar bas."""
        if self._orig is None: return
        # Eğer son işlem öncesi snapshot varsa oraya git (soft reset)
        if (hasattr(self, "_before_process") and
                self._before_process is not None and
                len(self._history) > 1):
            msg = "Son islemi geri al (Evet) / Tamamen sifirla (Hayir)?"
            reply = QMessageBox.question(
                self, "Reset",
                msg,
                QMessageBox.StandardButton.Yes |
                QMessageBox.StandardButton.No  |
                QMessageBox.StandardButton.Cancel)
            if reply == QMessageBox.StandardButton.Cancel: return
            if reply == QMessageBox.StandardButton.Yes:
                # Son işlemi geri al
                self._set_image(self._before_process.copy(),
                                "↺ Reset (önceki)", reset=False)
                self._before_process = None
                self.status.showMessage("↺  Son işlem geri alındı")
                return
        # Tam sıfırlama
        if QMessageBox.question(self,"Reset","Tüm işlemleri sıfırla?",
            QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No
        )!=QMessageBox.StandardButton.Yes: return
        self._set_image(self._orig.copy(),"Original",reset=True)
        self._before_process = None
        self.status.showMessage("🔄  Sıfırlandı — orijinal görüntü yüklendi")

    def _jump_to(self, index):
        if 0<=index<len(self._history):
            lbl,img=self._history[index]; self._current=img.copy()
            self._history=self._history[:index+1]
            self.hist_panel.truncate_to(index)
            self.viewer.show_image(img,lbl)
            self.lbl_step.setText(f"Step: {lbl}  ({index})")
            self.status.showMessage(f"⏮  #{index} → {lbl}")

    def _set_image(self, img, label, reset=False):
        self._current=img
        if reset:
            self._history=[(label,img.copy())]
            self.hist_panel.lst.clear(); self.hist_panel.push(label,0)
            self.lbl_step.setText(f"Step: {label}")
        else:
            self._history.append((label,img.copy()))
            idx=len(self._history)-1
            self.hist_panel.push(label,idx)
            self.lbl_step.setText(f"Step: {label}  ({idx})")
        # Use selected slot
        slot = int(self._slot_combo.currentIndex()) if hasattr(self,"_slot_combo") else 0
        if reset: slot = 0
        self.viewer.show_image(img, label, slot=slot)

    # ── Stacking ────────────────────────────────────────────────────────
    def _open_stacking(self):
        dlg=StackingDialog(self)
        if dlg.exec()==QDialog.DialogCode.Accepted:
            result=dlg.get_result()
            if result and result.get("result") is not None:
                img=result["result"]
                label=f"Stack×{result['n_lights']}({result['method']})"
                self._set_image(img,label,reset=(self._orig is None))
                if self._orig is None: self._orig=img.copy()
                h,w=img.shape[:2]
                self.status.showMessage(
                    f"✅  Yığınlama — {result['n_lights']} frame, {result['method']}, {w}×{h}")

    # ── Process runner (CRASH-SAFE) ─────────────────────────────────────
    # ── Process key→function dispatch ───────────────────────────────────
    def _run_key(self, key, params):
        """Dispatch process key to the correct function (no forward-ref issue)."""
        _dispatch = {
            "bg":      ("processing.background",       "remove_gradient_dispatch"),
            "noise":   ("processing.noise_reduction",   "reduce_noise"),
            "stars":   ("processing.star_removal",      "remove_stars"),
            "deconv":  ("processing.deconvolution", "deconvolve_dispatch"),
            "sharp":   ("processing.sharpening",        "sharpen"),
            "nebula":  ("ai.nebula_enhancer",           "enhance_nebula"),
            "color":   ("processing.color_calibration", "calibrate_color"),
            "morph":   ("processing.morphology",        "morphological"),
            "galaxy":  ("ai.galaxy_detector",           "detect_galaxies"),
            "stretch": ("processing.stretch",           "stretch"),  # veralux overrides below
        }
        entry = _dispatch.get(key)
        if entry is None:
            QMessageBox.critical(self,"Error",f"Unknown process: {key}"); return
        mod_name, fn_name = entry
        try:
            import importlib
            mod = importlib.import_module(mod_name)
            fn  = getattr(mod, fn_name)
        except Exception as e:
            QMessageBox.critical(self,"Import Error",str(e)); return

        if key == "sharp":
            method = params.get("method","revela")
            if method == "revela":
                def _rev_fn(img, **kw):
                    import numpy as _np
                    from processing.veralux_revela import StructureCore
                    texture = float(kw.get("amount", 1.0))
                    struct  = float(kw.get("radius", 2.0)) / 10.0
                    shadow  = float(kw.get("threshold", 0.0))
                    r = StructureCore.process_structure(img, texture, struct, shadow, True, False)
                    return _np.clip(r, 0, 1).astype("float32")
                self._run_worker(key, _rev_fn, params); return
            self._run_worker(key, fn, params); return

        if key == "color":
            method = params.get("method","vectra")
            if method == "vectra":
                def _vec_fn(img, **kw):
                    import numpy as _np
                    from processing.veralux_vectra import VectraCore
                    if img.ndim != 3: return img
                    # vectors: dict {"color_key": (hue_shift, sat_boost)}
                    # Empty dict = neutral (no color shift, passthrough)
                    r = VectraCore.process_vectors(img, {}, 0.0, True)
                    return _np.clip(r, 0, 1).astype("float32")
                self._run_worker(key, _vec_fn, params); return
            if method == "alchemy":
                def _alch_fn(img, **kw):
                    import numpy as _np
                    from processing.veralux_alchemy import VeraLuxNBCore
                    if img.ndim != 3: return img
                    # Alchemy expects (3,H,W) CHW — rgb_img[0]=R channel
                    img_chw = img.transpose(2,0,1).astype(_np.float32)
                    r = VeraLuxNBCore.linear_fit_channels(img_chw,
                            align_bg=True, auto_gain=True, manual_boost=1.0)
                    if r is None: return img
                    # output (3,H,W) → (H,W,3)
                    if isinstance(r, _np.ndarray) and r.ndim==3 and r.shape[0]==3:
                        r = r.transpose(1,2,0)
                    return _np.clip(r, 0, 1).astype("float32")
                self._run_worker(key, _alch_fn, params); return
            self._run_worker(key, fn, params); return

        if key == "stars":
            method = params.get("method","starsmalerx")
            if method == "starcomposer":
                def _sc_fn(img, **kw):
                    import numpy as _np
                    from processing.veralux_starcomposer import process_star_pipeline
                    strength = float(kw.get("strength", 0.9))
                    d = strength * 5 + 0.5
                    grip = float(kw.get("sensitivity", 0.5))
                    # StarComposer expects (3,H,W) CHW
                    img_chw = img.transpose(2,0,1).astype(_np.float32) if img.ndim==3 else img
                    r = process_star_pipeline(img_chw, D=d, b=0.0, grip=grip,
                                              shadow=0.0, reduction=0.0, healing=0.0,
                                              lsr=0.0, weights=(0.2126,0.7152,0.0722),
                                              use_adaptive=True)
                    if r is not None and isinstance(r, _np.ndarray):
                        # output (3,H,W) → (H,W,3)
                        if r.ndim==3 and r.shape[0]==3: r=r.transpose(1,2,0)
                        return _np.clip(r,0,1).astype("float32")
                    return img
                self._run_worker(key, _sc_fn, params); return
            if method == "starsmalerx":
                def _sxt_fn(img, **kw):
                    from processing.starsmalerx import reduce_stars
                    strength    = float(kw.get("strength",    0.9))
                    sensitivity = float(kw.get("sensitivity", 0.5))
                    feather     = float(kw.get("feather",     3.0))
                    if img.ndim == 2:
                        img3 = img[:,:,None]
                        r, _ = reduce_stars(img3, strength=strength,
                                            sensitivity=sensitivity,
                                            feather_radius=feather)
                        return r[:,:,0]
                    r, _ = reduce_stars(img, strength=strength,
                                        sensitivity=sensitivity,
                                        feather_radius=feather)
                    import numpy as _np
                    return _np.clip(r, 0, 1).astype("float32")
                self._run_worker(key, _sxt_fn, params)
                return
            # fallback: eski remove_stars
            self._run_worker(key, fn, params)
            return

        if key == "stretch":
            method = params.get("method","veralux")
            if method == "veralux":
                sensor = params.get("vl_sensor", "Rec.709")
                tb     = float(params.get("vl_target_bg", 0.20))
                log_d  = float(params.get("vl_log_d", 0.0))
                cg     = float(params.get("vl_color_grip", 1.0))
                scr    = float(params.get("vl_star_core", 0.5))
                sha    = float(params.get("vl_shadow", 0.0))
                mode   = params.get("vl_mode", "preserve")
                auto   = (log_d < 0.6)  # very low = use auto
                def _vl_fn(img, **kw):
                    import numpy as _np
                    from processing.veralux_hms import process_veralux_v6, SENSOR_PROFILES
                    ws = sensor if sensor in SENSOR_PROFILES else "Rec.709 (Recommended)"
                    # normalize_input handles HWC→CHW automatically
                    result = process_veralux_v6(
                        img.astype(_np.float32),
                        log_D             = None if auto else log_d,
                        protect_b         = 0.0,
                        convergence_power = 1.0,
                        working_space     = ws,
                        processing_mode   = mode,
                        target_bg         = tb,
                        color_grip        = cg,
                        shadow_convergence= sha,
                        use_adaptive_anchor=auto,
                    )
                    # output is (3,H,W) CHW → convert back to (H,W,3)
                    if isinstance(result, _np.ndarray):
                        if result.ndim == 3 and result.shape[0] == 3:
                            result = result.transpose(1,2,0)
                    return _np.clip(result, 0, 1).astype("float32")
                self._run_worker(key, _vl_fn, params)
                return
            # diger stretch metodlari
            self._run_worker(key, fn, params)
            return

        if key == "bg":
            method = params.get("method","dbe_spline")
            if method == "nox":
                def _nox_fn(img, **kw):
                    import numpy as _np
                    import cv2 as _cv2
                    from processing.veralux_nox import NoxCore
                    pcb = kw.get("_progress_cb")
                    stiffness  = float(kw.get("grid_size", 0.5))   # 0-1
                    aggression = float(kw.get("poly_degree", 50.0)) # percentile
                    if img.ndim == 2:
                        img_in = img
                    else:
                        img_in = img
                    h, w = img_in.shape[:2]
                    num_channels = img_in.shape[2] if img_in.ndim == 3 else 1
                    fwhm_avg = 4.0
                    mask_map = _np.ones((h, w), dtype=bool)
                    # Variance pyramid
                    max_grid = 50
                    scale = min(1.0, max_grid / max(h, w))
                    h_grid = max(5, int(h * scale))
                    w_grid = max(5, int(w * scale))
                    if num_channels == 3:
                        img_master = _np.max(img_in, axis=2)
                    else:
                        img_master = img_in
                    v_map = NoxCore.compute_pyramid_variance(img_master, h_grid, w_grid, fwhm_avg)
                    # Membrane solve per channel
                    model_out = _np.zeros_like(img_in)
                    if num_channels == 1:
                        model_out = NoxCore.membrane_solve_channel(img_in, mask_map, v_map, stiffness, aggression, max_grid)
                    else:
                        for c in range(3):
                            model_out[:,:,c] = NoxCore.membrane_solve_channel(
                                img_in[:,:,c], mask_map, v_map, stiffness, aggression, max_grid)
                    corrected = img_in - model_out
                    # Pedestal
                    mn = corrected.min()
                    if mn < 0: corrected -= mn
                    return _np.clip(corrected, 0, 1).astype("float32")
                self._run_worker(key, _nox_fn, params)
                return
            if method == "graxpert":
                exe = self._settings.get("graxpert_exe","").strip()
                if not exe or not os.path.isfile(exe):
                    reply = QMessageBox.question(
                        self, "GraXpert Ayarlanmamis",
                        "GraXpert executable bulunamadi.\n\n"
                        "Settings'i acmak ister misiniz?",
                        QMessageBox.StandardButton.Yes |
                        QMessageBox.StandardButton.No |
                        QMessageBox.StandardButton.Cancel)
                    if reply == QMessageBox.StandardButton.Cancel: return
                    if reply == QMessageBox.StandardButton.Yes:
                        self._open_settings()
                        exe = self._settings.get("graxpert_exe","").strip()
                    if not exe or not os.path.isfile(exe):
                        params = dict(params); params["method"] = "nox"
                        self._run_worker(key, fn, params); return

                # Panel parametrelerini al
                smoothing  = float(params.get("gx_smoothing",  self._settings.get("graxpert_smoothing", 0.5)))
                correction = params.get("gx_correction", self._settings.get("graxpert_correction","Subtraction"))
                ai_version = params.get("gx_model",      self._settings.get("graxpert_ai_version","1.0.0"))
                keep_bg    = bool(params.get("gx_keep_bg", False))
                use_gpu    = bool(params.get("gx_gpu", True))
                batch_size = int(params.get("gx_batch", 4))

                # Cikti klasoru — mevcut resim klasoru
                lbl  = self.lbl_file.text().strip()
                base = os.path.splitext(lbl)[0] if lbl and lbl != "No file" else "image"
                # Cikti dosyalarini resmin yanına kaydet
                src_dir = self._settings.get("last_open_dir","") or os.path.expanduser("~")
                bg_removed_path = os.path.join(src_dir, f"{os.path.basename(base)}_GraXpert.tif")
                bg_model_path   = os.path.join(src_dir, f"{os.path.basename(base)}_background.tif")

                _exe=exe; _sm=smoothing; _av=ai_version; _cor=correction
                _brp=bg_removed_path; _bmp=bg_model_path; _keep=keep_bg

                def graxpert_fn(img, **kw):
                    import cv2 as _cv2
                    from ai.graxpert_bridge import run_graxpert
                    pcb = kw.get("_progress_cb")
                    def _cb(msg):
                        if pcb:
                            m = re.match(r"\[(\d+)/(\d+)\]", msg)
                            if m: pcb(msg, step=int(m.group(1)), total=int(m.group(2)))
                            else: pcb(msg)
                    result = run_graxpert(img, exe_path=_exe,
                                          command="background-extraction",
                                          smoothing=_sm, ai_version=_av,
                                          correction=_cor,
                                          progress_cb=_cb)
                    br = result["background_removed"]
                    # TIFF kaydet
                    def _save_tif(arr, path):
                        img16 = (np.clip(arr,0,1)*65535).astype(np.uint16)
                        try:
                            import tifffile as _tf
                            _tf.imwrite(path, img16,
                                        photometric="rgb" if arr.ndim==3 else "minisblack",
                                        compression=None)
                        except ImportError:
                            if arr.ndim==3: img16=_cv2.cvtColor(img16,_cv2.COLOR_RGB2BGR)
                            _cv2.imwrite(path, img16, [_cv2.IMWRITE_TIFF_COMPRESSION,1])
                    _save_tif(br, _brp)
                    if _keep and result.get("background_model") is not None:
                        _save_tif(result["background_model"], _bmp)
                    return br

                self._run_worker(key, graxpert_fn, params)
                return
            self._run_worker(key, fn, params)
            return

        if key == "noise":
            method = params.get("method","bilateral")
            if method == "silentium":
                def _sil_fn(img, **kw):
                    import numpy as _np
                    from processing.veralux_silentium import SilentiumCore
                    # Silentium sliders expect 0-100 range
                    strength = float(kw.get("strength", 0.7)) * 100.0
                    detail   = float(kw.get("detail",   0.5)) * 100.0
                    # Silentium expects (3,H,W) CHW format
                    if img.ndim == 2:
                        img_chw = _np.stack([img,img,img], 0).astype(_np.float32)
                        r = SilentiumCore.apply_noise_reduction(
                            img_chw, strength, detail, True, None, False)
                        if isinstance(r,_np.ndarray):
                            if r.ndim==3 and r.shape[0]==3:
                                r = r.transpose(1,2,0).mean(axis=2)
                            elif r.ndim==3:
                                r = r.mean(axis=2)
                    else:
                        img_chw = img.transpose(2,0,1).astype(_np.float32)
                        r = SilentiumCore.apply_noise_reduction(
                            img_chw, strength, detail, True, None, True)
                        if isinstance(r,_np.ndarray) and r.ndim==3 and r.shape[0]==3:
                            r = r.transpose(1,2,0)
                    return _np.clip(r,0,1).astype("float32")
                self._run_worker(key, _sil_fn, params)
                return
            if method == "noisexterminator":
                def _nxt_fn(img, **kw):
                    from processing.noisexterminator import noisexterminator
                    strength = float(kw.get("strength", 0.7))
                    detail   = float(kw.get("detail",   0.5))
                    pcb = kw.get("_progress_cb")
                    def _log(msg):
                        if pcb: pcb(msg)
                    # noisexterminator (H,W,C) float32 bekliyor
                    if img.ndim == 2:
                        img3 = img[:,:,None]
                        r, _ = noisexterminator(img3, strength=strength, detail=detail)
                        return r[:,:,0]
                    r, _ = noisexterminator(img, strength=strength, detail=detail)
                    import numpy as _np
                    return _np.clip(r, 0, 1).astype("float32")
                self._run_worker(key, _nxt_fn, params)
                return
            if method == "graxpert":
                exe = self._settings.get("graxpert_exe","").strip()
                if not exe or not os.path.isfile(exe):
                    reply = QMessageBox.question(
                        self, "GraXpert Ayarlanmamis",
                        "GraXpert executable ayarli degil.\n\n"
                        "Settings'i acmak ister misiniz?\n"
                        "(Hayir = bilateral filter kullanilir)",
                        QMessageBox.StandardButton.Yes |
                        QMessageBox.StandardButton.No)
                    if reply == QMessageBox.StandardButton.Yes:
                        self._open_settings()
                        exe = self._settings.get("graxpert_exe","").strip()
                    if not exe or not os.path.isfile(exe):
                        params = dict(params); params["method"] = "bilateral"
                        self._run_worker(key, fn, params); return

                # Klasör seç
                default_dir = (self._settings.get("last_save_dir","") or
                               self._settings.get("last_open_dir","") or
                               os.path.expanduser("~"))
                out_dir = QFileDialog.getExistingDirectory(
                    self, "GraXpert Denoise Ciktisi Icin Klasor Sec", default_dir)
                if not out_dir: return

                lbl   = self.lbl_file.text().strip()
                base  = os.path.splitext(lbl)[0] if lbl and lbl != "No file" else "image"
                out_path = os.path.join(out_dir, f"{base}_denoised.tif")

                strength   = float(self._settings.get("graxpert_denoise_strength", 0.8))
                ai_version = self._settings.get("graxpert_ai_version","latest")
                _exe=exe; _str=strength; _av=ai_version; _op=out_path

                reply2 = QMessageBox.question(
                    self, "GraXpert Denoise",
                    f"Engine:  {os.path.basename(exe)}\n"
                    f"Strength: {strength}\n"
                    f"Kayit:    {out_path}\n\nDevam?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                if reply2 != QMessageBox.StandardButton.Yes: return

                def graxpert_denoise_fn(img, **kw):
                    import cv2 as _cv2
                    from ai.graxpert_bridge import run_graxpert
                    pcb = kw.get("_progress_cb")
                    def _cb(msg):
                        if pcb:
                            m = re.match(r"\[(\d+)/(\d+)\]", msg)
                            if m: pcb(msg, step=int(m.group(1)), total=int(m.group(2)))
                            else: pcb(msg)
                    result = run_graxpert(img, exe_path=_exe,
                                          command="denoising",
                                          denoise_strength=_str,
                                          ai_version=_av,
                                          progress_cb=_cb)
                    dn = result["denoised"]
                    # Dosyaya kaydet
                    img16 = (np.clip(dn,0,1)*65535).astype(np.uint16)
                    try:
                        import tifffile as _tf
                        _tf.imwrite(_op, img16,
                                    photometric="rgb" if dn.ndim==3 else "minisblack",
                                    compression=None)
                    except ImportError:
                        if dn.ndim==3: img16=_cv2.cvtColor(img16,_cv2.COLOR_RGB2BGR)
                        _cv2.imwrite(_op, img16, [_cv2.IMWRITE_TIFF_COMPRESSION,1])
                    return dn

                self._run_worker(key, graxpert_denoise_fn, params)
                return
            self._run_worker(key, fn, params)
            return

        if key == "starnet":
            # Check if real StarNet++ exe is configured
            exe = self._settings.get("starnet_exe","").strip()
            if exe and os.path.isfile(exe):
                # Real StarNet++ bridge
                stride   = int(self._settings.get("starnet_stride", 256))
                use_gpu  = bool(self._settings.get("starnet_use_gpu", False))
                def starnet_fn(img, **kw):
                    from ai.starnet_bridge import run_starnet
                    # Wire _progress_cb from Worker to run_starnet's progress_cb
                    pcb = kw.get("_progress_cb")
                    def sn_cb(msg):
                        if pcb:
                            # Parse "[N/4]" prefix for step info
                            import re
                            m = re.match(r"\[(\d+)/(\d+)\]", msg)
                            if m:
                                pcb(msg, step=int(m.group(1)), total=int(m.group(2)))
                            else:
                                pcb(msg)
                    result = run_starnet(img, exe_path=exe,
                                        stride=stride, use_gpu=use_gpu,
                                        progress_cb=sn_cb)
                    starnet_fn._last_result = result
                    return result["starless"]
            else:
                # AI fallback (no exe configured)
                def starnet_fn(img, **kw):
                    pcb = kw.pop("_progress_cb", None)
                    if pcb: pcb("[1/3] AI Star Separation starting…", step=1, total=3)
                    # Remove _progress_cb before passing to underlying fn
                    kw.pop("_cb", None)
                    if pcb: pcb("[2/3] Detecting and removing stars…", step=2, total=3)
                    result = fn(img, **kw)
                    if pcb: pcb("[3/3] Done", step=3, total=3)
                    if isinstance(result, dict):
                        starnet_fn._last_result = result
                        return result["starless"]
                    starnet_fn._last_result = {"starless": result, "stars_only": None, "star_mask": None}
                    return result
            starnet_fn._last_result = None
            self._starnet_fn = starnet_fn
            self._run_worker(key, starnet_fn, params)
        else:
            self._run_worker(key, fn, params)

    def _run_worker(self, key, fn, params):
        if self._current is None:
            QMessageBox.information(self,"Info","Please open an image first."); return

        active = [w for w in self._workers if w.isRunning()]
        if active:
            self.status.showMessage("⚠  A process is already running. Please wait…"); return

        panel = self._panels.get(key)
        if panel: panel.set_running(True)

        # ── "Before" snapshot — reset bu noktaya döner ───────────────────────
        self._before_process = self._current.copy()

        # Status bar progress
        self.pbar.show(); self.pbar.setRange(0, 0)
        self.pbar.setTextVisible(False)
        self.status.showMessage(f"⏳  {key}  başlıyor…")

        worker = Worker(fn, self._current.copy(), params)
        self._workers.append(worker)

        # Progress bar & status bar — % gösterim
        def _on_progress(step, total, msg, p=panel, k=key):
            if total > 0:
                self.pbar.setRange(0, total)
                self.pbar.setValue(step)
                self.pbar.setTextVisible(True)
                pct = int(step * 100 / max(total, 1))
                self.pbar.setFormat(f"{k}  %p%")
                short = msg if len(msg) <= 70 else msg[-68:]
                self.status.showMessage(f"⏳  {k}  {pct}%  —  {short}")
            else:
                self.pbar.setRange(0, 0)
                self.pbar.setTextVisible(False)
                short = msg if len(msg) <= 80 else msg[-78:]
                self.status.showMessage(f"⏳  {short}")
            if p: p.set_progress(step, total, msg)

        # Ara önizleme — viewer'a anlık göster (history'e gitmez)
        def _on_preview(preview_img):
            if preview_img is not None and isinstance(preview_img, np.ndarray):
                import numpy as _np
                pimg = _np.clip(preview_img, 0, 1).astype(_np.float32)
                self.viewer._preview_slot(pimg, f"⏳ {key}…")

        worker.progress.connect(_on_progress)
        worker.preview.connect(_on_preview)
        worker.finished.connect(lambda img, k=key, p=panel: self._on_done(img, k, p))
        worker.error.connect(lambda e, p=panel: self._on_error(e, p))
        worker.finished.connect(lambda _, w=worker: self._cleanup_worker(w))
        worker.error.connect(lambda _, w=worker: self._cleanup_worker(w))
        worker.start()

    def _cleanup_worker(self, worker):
        """Biten worker'ı listeden çıkar."""
        try: self._workers.remove(worker)
        except ValueError: pass

    def _on_done(self, img, key, panel):
        import numpy as _np
        result = _np.clip(img, 0, 1).astype(_np.float32) if isinstance(img, _np.ndarray) else img
        # Label oluştur
        method_names = {
            "bg":"BG Extract","noise":"Noise Reduce","deconv":"Deconvolve",
            "stars":"Star Remove","sharp":"Sharpen","color":"Color Cal",
            "stretch":"Stretch","nebula":"Nebula","morph":"Morph",
            "galaxy":"Galaxy","crop":"Crop",
        }
        label = method_names.get(key, key)
        self._set_image(result, label)
        if panel: panel.set_running(False)
        self.pbar.hide()
        self.pbar.setRange(0, 100)
        self.pbar.setTextVisible(False)
        h, w = result.shape[:2] if hasattr(result,"shape") else (0,0)
        self.status.showMessage(f"✅  {label} tamamlandı  —  {w}×{h}px")

    def _on_error(self, msg, panel):
        if panel: panel.set_running(False)
        self.pbar.hide()
        self.pbar.setRange(0, 100)
        self.pbar.setTextVisible(False)
        # Extract first meaningful RuntimeError/FileNotFoundError line
        lines = [l.strip() for l in msg.split("\n") if l.strip()]
        # Find the actual error message (last non-traceback line)
        error_lines = [l for l in lines if not l.startswith("File ") and not l.startswith("Traceback")]
        short = error_lines[-1] if error_lines else (lines[-1] if lines else "Unknown error")
        short = short[:120]
        self.status.showMessage(f"❌  {short}")
        # Full error dialog with scrollable details
        err_dlg = QMessageBox(self)
        err_dlg.setWindowTitle("Processing Error")
        err_dlg.setIcon(QMessageBox.Icon.Critical)
        err_dlg.setText(short)
        err_dlg.setDetailedText(msg)
        err_dlg.setStyleSheet(f"QLabel{{min-width:500px;}}")
        err_dlg.exec()

    # ── Crop ────────────────────────────────────────────────────────────
    def _open_script_editor(self):
        """Script Editor dialogunu ac."""
        if self._current is None:
            QMessageBox.information(self, "Script Editor",
                                    "Lutfen once bir resim acin.")
            return
        dlg = ScriptEditorDialog(
            image     = self._current.copy(),
            apply_cb  = self._script_apply,
            parent    = self,
        )
        dlg.exec()

    def _script_apply(self, result):
        """Script sonucunu history'e ekle."""
        import numpy as np
        result = np.clip(result, 0, 1).astype(np.float32)
        self._set_image(result, "Python Script")

    def _inline_crop_apply(self):
        """Toolbar crop ikonu üzerinden uygula."""
        cropped = self.viewer.apply_inline_crop()
        if cropped is None:
            QMessageBox.information(self, "Crop", "Once bir bolge secin."); return
        h, w = cropped.shape[:2]
        self.viewer.exit_crop_mode()
        self._crop_active = False
        b = self._proc_btns.get("crop")
        if b: b.setChecked(False)
        self._set_image(cropped, f"Crop {w}x{h}")
        self.status.showMessage(f"Crop: {w}x{h} px")

    def _inline_crop_cancel(self):
        """Toolbar crop modundan çık."""
        self.viewer.exit_crop_mode()
        self._crop_active = False
        b = self._proc_btns.get("crop")
        if b: b.setChecked(False)
        self.status.showMessage("Crop iptal edildi")

    def _crop_start(self):
        if self._current is None:
            QMessageBox.information(self,"Info","Please open an image first."); return
        self._crop_src = self._current.copy()
        # Callback'leri bağla
        self.viewer._crop_apply_cb  = self._crop_apply
        self.viewer._crop_cancel_cb = self._crop_reset
        self.viewer.enter_crop_mode()
        pass  # crop buttons removed
        self.status.showMessage("✂  Sol tik surukle sec  |  Sag tik menue")

    def _crop_apply(self):
        cropped=self.viewer.apply_crop()
        if cropped is None: QMessageBox.information(self,"Info","Draw a crop region first."); return
        h,w=cropped.shape[:2]; self._set_image(cropped,f"Kırpma {w}×{h}")
        self.status.showMessage(f"✅  Kırpıldı: {w}×{h} piksel")

    def _crop_reset(self):
        self.viewer.exit_crop_mode()
        pass  # crop buttons removed
        self.status.showMessage("Crop iptal edildi")

    def closeEvent(self, e):
        """Kapanırken tüm worker'ları durdur."""
        for w in self._workers:
            if w.isRunning():
                w.requestInterruption()
                w.wait(1000)
        super().closeEvent(e)


# ═══════════════════════════════ PROCESS FUNCTIONS ═══════════════════════════
def _bg(img,**kw):
    from processing.background import remove_gradient; return remove_gradient(img,**kw)
def _noise(img,**kw):
    from processing.noise_reduction import reduce_noise; return reduce_noise(img,**kw)
def _stars(img,**kw):
    from processing.star_removal import remove_stars; return remove_stars(img,**kw)
def _deconv(img,**kw):
    from processing.deconvolution import deconvolve; return deconvolve(img,**kw)
def _sharp(img,**kw):
    from processing.sharpening import sharpen; return sharpen(img,**kw)
def _nebula(img,**kw):
    from ai.nebula_enhancer import enhance_nebula; return enhance_nebula(img,**kw)
def _color(img,**kw):
    from processing.color_calibration import calibrate_color; return calibrate_color(img,**kw)
def _morph(img,**kw):
    from processing.morphology import morphological; return morphological(img,**kw)
def _galaxy(img,**kw):
    from ai.galaxy_detector import detect_galaxies; return detect_galaxies(img,**kw)


def _stretch(img,**kw):
    from processing.stretch import stretch; return stretch(img,**kw)


def start_app():
    # Global exception handler — crash yerine dialog göster
    sys.excepthook = _global_exception_handler

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    QToolTip.setFont(QFont("Segoe UI", 10))
    w = AstroApp()
    w.show()
    sys.exit(app.exec())
