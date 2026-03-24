"""
Astro Mastro Pro — Photoshop-Style Histogram Editor
====================================================
• Levels panel  : B / M / W üçgen sürükleme + output clipping
• Curves panel  : her kanalda çoklu kontrol noktası, sürüklenebilir eğri
• Adjustments   : Brightness/Contrast, Vibrance, Saturation, Hue Shift,
                  Shadows/Midtones/Highlights, Exposure, Color Temp
• Per-channel   : L / R / G / B (bağımsız ya da bağlantılı)
• Apply / Reset / Copy/Paste channel settings
"""

import numpy as np
import cv2
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QDoubleSpinBox, QCheckBox, QComboBox, QSlider, QFrame,
    QSizePolicy, QTabWidget, QGridLayout, QScrollArea, QSpacerItem
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QPoint, QPointF, QRect
from PyQt6.QtGui import (
    QPainter, QColor, QPen, QBrush, QLinearGradient,
    QFont, QPainterPath, QPixmap, QPolygonF
)

# ── Colours (match app.py — SC2 Light + Red) ─────────────────────────────────
BG      = "#0c1018"
BG2     = "#141e2c"
BG3     = "#1c2a3c"
BG4     = "#253850"
BORDER  = "#2a4060"
BORDER2 = "#3a6090"
ACCENT  = "#e04040"
ACCENT2 = "#ff6060"
GOLD    = "#f0b830"
GREEN   = "#50dd66"
RED     = "#ff3333"
PURPLE  = "#cc77ff"
TEXT    = "#e8f0ff"
MUTED   = "#80a8c8"
HEAD    = "#c0e0ff"
SUBTEXT = "#506880"

CH_COLORS = {"L": ACCENT2, "R": RED, "G": GREEN, "B": "#6699ff", "ALL": ACCENT}

SPIN_CSS = (
    f"QDoubleSpinBox,QSpinBox{{background:{BG};color:{TEXT};"
    f"border:1px solid {BORDER};border-radius:3px;padding:1px 4px;font-size:10px;}}"
    f"QDoubleSpinBox:focus,QSpinBox:focus{{border:1px solid {ACCENT};}}"
)
SLIDER_CSS = (
    f"QSlider::groove:horizontal{{height:3px;background:{BORDER};border-radius:1px;}}"
    f"QSlider::handle:horizontal{{width:12px;height:12px;margin:-5px 0;"
    f"background: qlineargradient(x1:0,y1:0,x2:0,y2:1,"
    f"  stop:0 {ACCENT2}, stop:1 {ACCENT});"
    f"border-radius:6px;border:1px solid {ACCENT2};}}"
    f"QSlider::sub-page:horizontal{{background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
    f"  stop:0 {BORDER2}, stop:1 {ACCENT});border-radius:1px;}}"
)
BTN_CSS = (
    f"QPushButton{{"
    f"  background: qlineargradient(x1:0,y1:0,x2:0,y2:1,"
    f"    stop:0 {BG3}, stop:1 {BG});"
    f"  color:{TEXT}; border:1px solid {BORDER};"
    f"  border-top:1px solid {BORDER2};"
    f"  border-radius:2px; padding:3px 10px; font-size:10px; font-weight:600;}}"
    f"QPushButton:hover{{"
    f"  background: qlineargradient(x1:0,y1:0,x2:0,y2:1,"
    f"    stop:0 {BG4}, stop:1 {BG3});"
    f"  border:1px solid {ACCENT}; border-top:1px solid {ACCENT2};}}"
    f"QPushButton:pressed{{background:{BG};}}"
)
CHECK_CSS = (
    f"QCheckBox{{color:{HEAD};font-size:10px;spacing:5px;}}"
    f"QCheckBox::indicator{{width:13px;height:13px;border-radius:2px;"
    f"border:1px solid {BORDER};background:{BG};}}"
    f"QCheckBox::indicator:checked{{background:{ACCENT};border:1px solid {ACCENT2};}}"
)
TAB_CSS = (
    f"QTabWidget::pane{{background:{BG2};border:1px solid {BORDER};"
    f"border-top:1px solid {BORDER2};"
    f"border-radius:2px;margin-top:-1px;}}"
    f"QTabBar::tab{{"
    f"  background: qlineargradient(x1:0,y1:0,x2:0,y2:1,"
    f"    stop:0 {BG4}, stop:1 {BG3});"
    f"  color:{MUTED}; border:1px solid {BORDER};"
    f"  border-bottom:none; border-top:1px solid {BORDER2};"
    f"  padding:5px 14px; font-size:10px; font-weight:700;"
    f"  min-width:65px; border-radius:2px 2px 0 0;}}"
    f"QTabBar::tab:selected{{"
    f"  background: qlineargradient(x1:0,y1:0,x2:0,y2:1,"
    f"    stop:0 {BG3}, stop:1 {BG2});"
    f"  color:{ACCENT2}; border-bottom:2px solid {ACCENT};"
    f"  border-top:1px solid {ACCENT};}}"
    f"QTabBar::tab:hover{{color:{TEXT};background:{BG4};}}"
)


# ─────────────────────────────────────────────────────────────────────────────
#  HistogramWidget — draws channel histograms + Levels sliders
# ─────────────────────────────────────────────────────────────────────────────
class HistogramWidget(QWidget):
    """
    Custom QPainter widget:
      • Filled RGB/L histogram bars
      • Three draggable triangles below (Black / Midtone / White)
      • Two output sliders (top bar) — clipping output range
    Signals: levels_changed(channel, black, mid, white, out_lo, out_hi)
    """
    levels_changed = pyqtSignal(str, float, float, float, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(260, 160)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

        # Per-channel state: {ch: [black, mid, white, out_lo, out_hi]}
        self._state = {ch: [0.0, 0.5, 1.0, 0.0, 1.0]
                       for ch in ("L","R","G","B")}
        self._ch    = "L"
        self._hdata = {}    # {ch: ndarray[256]}
        self._drag  = None  # ("handle", channel) or ("out_lo"|"out_hi")
        self._hover = None

        # Layout constants (recalculated in paintEvent)
        self._pad_l  = 10
        self._pad_r  = 10
        self._pad_top = 14   # output bar
        self._tri_h   = 12
        self._pad_bot = self._tri_h + 6

    # ── Public API ────────────────────────────────────────────────────────────
    def set_image(self, img: np.ndarray):
        """Compute histograms from image."""
        if img is None:
            self._hdata = {}; self.update(); return
        img = np.clip(img, 0, 1).astype(np.float32)
        if img.ndim == 2:
            h, _ = np.histogram(img.ravel(), bins=256, range=(0,1))
            self._hdata = {"L": h.astype(float)}
        else:
            gray = img.mean(axis=2)
            hl, _ = np.histogram(gray.ravel(), bins=256, range=(0,1))
            self._hdata["L"] = hl.astype(float)
            for i, ch in enumerate(("R","G","B")):
                h, _ = np.histogram(img[:,:,i].ravel(), bins=256, range=(0,1))
                self._hdata[ch] = h.astype(float)
        self.update()

    def set_channel(self, ch: str):
        self._ch = ch; self.update()

    def get_state(self, ch=None):
        return list(self._state[ch or self._ch])

    def set_state(self, ch, black, mid, white, out_lo=0.0, out_hi=1.0):
        self._state[ch] = [
            float(black), float(mid), float(white),
            float(out_lo), float(out_hi)
        ]
        self.update()

    def reset_channel(self, ch=None):
        for c in (self._state if ch is None else [ch]):
            self._state[c] = [0.0, 0.5, 1.0, 0.0, 1.0]
        self.update()

    # ── Geometry helpers ──────────────────────────────────────────────────────
    def _hist_rect(self):
        w, h = self.width(), self.height()
        return QRect(self._pad_l, self._pad_top,
                     w - self._pad_l - self._pad_r,
                     h - self._pad_top - self._pad_bot - 2)

    def _val_to_x(self, v, rect=None):
        r = rect or self._hist_rect()
        return r.left() + int(v * r.width())

    def _x_to_val(self, x, rect=None):
        r = rect or self._hist_rect()
        return float(np.clip((x - r.left()) / max(r.width(), 1), 0, 1))

    # ── Paint ─────────────────────────────────────────────────────────────────
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self._hist_rect()
        w, h = self.width(), self.height()

        # Background
        p.fillRect(0, 0, w, h, QColor(BG))
        p.fillRect(r, QColor(BG2))

        # Grid lines
        pen = QPen(QColor(BORDER)); pen.setWidth(1); p.setPen(pen)
        for frac in (0.25, 0.5, 0.75):
            x = self._val_to_x(frac, r)
            p.drawLine(x, r.top(), x, r.bottom())

        # Histogram bars — L modunda tüm kanalları kendi renginde göster
        ch   = self._ch
        show = {"L": [("R", RED, 0.35), ("G", GREEN, 0.35), ("B", "#6699ff", 0.35), ("L", "#cccccc", 0.5)],
                "R": [("L", ACCENT, 0.12), ("R", RED,   0.85)],
                "G": [("L", ACCENT, 0.12), ("G", GREEN, 0.85)],
                "B": [("L", ACCENT, 0.12), ("B", "#6699ff", 0.85)],
                }.get(ch, [("L", ACCENT, 0.6)])

        for data_ch, color, alpha in show:
            hdata = self._hdata.get(data_ch)
            if hdata is None: continue
            mx = hdata.max()
            if mx == 0: continue
            norm = hdata / mx
            bar_w = max(1, r.width() / 256)
            col = QColor(color); col.setAlphaF(alpha)
            p.setPen(Qt.PenStyle.NoPen); p.setBrush(QBrush(col))
            for i, v in enumerate(norm):
                bh = int(v * r.height())
                if bh < 1: continue
                bx = r.left() + int(i * bar_w)
                p.drawRect(bx, r.bottom() - bh, max(1, int(bar_w)), bh)

        # Border
        pen = QPen(QColor(BORDER)); pen.setWidth(1); p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(r)

        # ── Output bar (top) ──────────────────────────────────────────────
        st     = self._state[ch]
        out_lo, out_hi = st[3], st[4]
        bar_y  = 4; bar_h = 7

        grad = QLinearGradient(self._pad_l, 0,
                               w - self._pad_r, 0)
        grad.setColorAt(0, QColor("#000000"))
        grad.setColorAt(1, QColor("#ffffff"))
        p.fillRect(self._pad_l, bar_y,
                   w - self._pad_l - self._pad_r, bar_h,
                   QBrush(grad))

        # Output handles (small squares)
        for val, col in [(out_lo, "#44aaff"), (out_hi, "#ffffff")]:
            ox = self._val_to_x(val)
            pen2 = QPen(QColor(col)); pen2.setWidth(2); p.setPen(pen2)
            p.setBrush(QBrush(QColor(col)))
            p.drawRect(ox - 5, bar_y - 1, 10, bar_h + 2)

        # ── Level triangles (bottom) ──────────────────────────────────────
        tri_y  = r.bottom() + 4
        for handle, val, color in [
            ("black", st[0], "#ffffff"),
            ("mid",   st[1], "#888888"),
            ("white", st[2], "#ffffff"),
        ]:
            if handle == "black":  col = "#ccccff"
            elif handle == "mid":  col = "#888888"
            else:                  col = "#ffffcc"

            tx  = self._val_to_x(val)
            tri = QPolygonF([
                QPointF(tx,      tri_y),
                QPointF(tx - 7,  tri_y + self._tri_h),
                QPointF(tx + 7,  tri_y + self._tri_h),
            ])
            pen3 = QPen(QColor(col)); pen3.setWidth(1); p.setPen(pen3)
            fill = QColor(col); fill.setAlphaF(0.85)
            p.setBrush(QBrush(fill))
            p.drawPolygon(tri)

            # Highlight dragged / hovered
            if self._drag and self._drag[0] == handle:
                pen4 = QPen(QColor(ACCENT2)); pen4.setWidth(2); p.setPen(pen4)
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawPolygon(tri)

        # Value readout
        p.setPen(QColor(MUTED))
        p.setFont(QFont("Courier New", 8))
        p.drawText(r.left(), h - 2,
                   f"B:{st[0]:.3f}  M:{st[1]:.3f}  W:{st[2]:.3f}  "
                   f"out:[{st[3]:.2f}–{st[4]:.2f}]")

        p.end()

    # ── Mouse ─────────────────────────────────────────────────────────────────
    def _hit_handle(self, x, y):
        r     = self._hist_rect()
        tri_y = r.bottom() + 4
        st    = self._state[self._ch]
        # Triangle handles
        if tri_y <= y <= tri_y + self._tri_h + 4:
            for handle, val in [("black",st[0]),("mid",st[1]),("white",st[2])]:
                if abs(x - self._val_to_x(val)) < 10:
                    return ("handle", handle)
        # Output handles (top bar)
        if 2 <= y <= 16:
            for key, val in [("out_lo", st[3]), ("out_hi", st[4])]:
                if abs(x - self._val_to_x(val)) < 10:
                    return ("output", key)
        return None

    def mousePressEvent(self, ev):
        hit = self._hit_handle(ev.pos().x(), ev.pos().y())
        if hit: self._drag = hit

    def mouseMoveEvent(self, ev):
        x   = ev.pos().x()
        val = self._x_to_val(x)
        ch  = self._ch
        st  = list(self._state[ch])

        if self._drag:
            kind, name = self._drag
            if kind == "handle":
                if name == "black":
                    st[0] = float(np.clip(val, 0.0, st[2] - 0.01))
                    st[1] = float(np.clip(st[1], st[0], st[2]))
                elif name == "white":
                    st[2] = float(np.clip(val, st[0] + 0.01, 1.0))
                    st[1] = float(np.clip(st[1], st[0], st[2]))
                elif name == "mid":
                    st[1] = float(np.clip(val, st[0], st[2]))
            else:  # output
                if name == "out_lo":
                    st[3] = float(np.clip(val, 0.0, st[4] - 0.01))
                else:
                    st[4] = float(np.clip(val, st[3] + 0.01, 1.0))
            self._state[ch] = st
            self.update()
            self.levels_changed.emit(ch, st[0], st[1], st[2], st[3], st[4])
        else:
            self._hover = self._hit_handle(x, ev.pos().y())
            cur = Qt.CursorShape.SizeHorCursor if self._hover else Qt.CursorShape.ArrowCursor
            self.setCursor(cur)

    def mouseReleaseEvent(self, ev):
        self._drag = None


# ─────────────────────────────────────────────────────────────────────────────
#  CurvesWidget — per-channel curve editor  (v2 — aesthetic & precise)
# ─────────────────────────────────────────────────────────────────────────────
class CurvesWidget(QWidget):
    """
    Bezier-like tone curve editor with up to 8 control points per channel.
    Signals: curve_changed(channel, lut)  lut = ndarray[256] in [0,1]
    """
    curve_changed = pyqtSignal(str, object)

    _GLOW = {
        "L": ("#5bb8f0", "#2a7ec0", "#1a5080"),
        "R": ("#ff6666", "#cc3333", "#801a1a"),
        "G": ("#66ff88", "#33cc55", "#1a802a"),
        "B": ("#6699ff", "#3366cc", "#1a3380"),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(220, 220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

        self._pts  = {ch: [(0.0,0.0),(1.0,1.0)] for ch in ("L","R","G","B")}
        self._ch   = "L"
        self._drag_idx = None
        self._hdata    = {}
        self._hover_idx = None
        self._pad = 24
        self._coord_label = None   # (wx, wy, text)

    def set_image(self, img):
        if img is None: self._hdata = {}; self.update(); return
        img = np.clip(img,0,1).astype(np.float32)
        if img.ndim == 2:
            h,_ = np.histogram(img.ravel(), bins=128, range=(0,1))
            self._hdata = {"L": h.astype(float)}
        else:
            gray = img.mean(axis=2)
            hl,_ = np.histogram(gray.ravel(), bins=128, range=(0,1))
            self._hdata["L"] = hl.astype(float)
            for i, ch in enumerate(("R","G","B")):
                h,_ = np.histogram(img[:,:,i].ravel(), bins=128, range=(0,1))
                self._hdata[ch] = h.astype(float)
        self.update()

    def set_channel(self, ch):
        self._ch = ch; self.update()

    def get_lut(self, ch=None):
        pts = sorted(self._pts[ch or self._ch], key=lambda p:p[0])
        return self._pts_to_lut(pts)

    def reset_channel(self, ch=None):
        for c in (self._pts if ch is None else [ch]):
            self._pts[c] = [(0.0,0.0),(1.0,1.0)]
        self.update()
        self._emit()

    def _pts_to_lut(self, pts):
        if len(pts) < 2:
            return np.linspace(0,1,256)
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        x_in = np.linspace(0,1,256)
        from scipy.interpolate import PchipInterpolator
        try:
            lut = PchipInterpolator(xs, ys)(x_in)
        except Exception:
            lut = np.interp(x_in, xs, ys)
        return np.clip(lut, 0, 1)

    def _emit(self):
        lut = self.get_lut(self._ch)
        self.curve_changed.emit(self._ch, lut)

    def _to_widget(self, x, y):
        p = self._pad
        W = self.width()  - 2*p
        H = self.height() - 2*p
        return QPointF(p + x*W, p + (1-y)*H)

    def _from_widget(self, px, py):
        p  = self._pad
        W  = self.width()  - 2*p
        H  = self.height() - 2*p
        return (float(np.clip((px-p)/W,0,1)), float(np.clip(1-(py-p)/H,0,1)))

    def paintEvent(self, event):
        p    = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        pad  = self._pad
        ch   = self._ch
        glow = self._GLOW.get(ch, self._GLOW["L"])

        # ── Background gradient ──
        bg_grad = QLinearGradient(0, 0, 0, H)
        bg_grad.setColorAt(0, QColor("#060e18"))
        bg_grad.setColorAt(1, QColor("#0a1a2e"))
        p.fillRect(0, 0, W, H, QBrush(bg_grad))

        inner = QRect(pad, pad, W-2*pad, H-2*pad)
        inner_grad = QLinearGradient(pad, pad, pad, H-pad)
        inner_grad.setColorAt(0, QColor("#081520"))
        inner_grad.setColorAt(0.5, QColor("#0c1c30"))
        inner_grad.setColorAt(1, QColor("#081520"))
        p.fillRect(inner, QBrush(inner_grad))

        # ── Grid — subtle ──
        pen = QPen(QColor("#152535")); pen.setWidth(1); p.setPen(pen)
        for i in range(1, 8):
            frac = i / 8
            x = pad + int(frac * (W - 2*pad))
            y = pad + int(frac * (H - 2*pad))
            p.drawLine(x, pad, x, H-pad)
            p.drawLine(pad, y, W-pad, y)
        # Major grid (quarters)
        pen2 = QPen(QColor("#1e3548")); pen2.setWidth(1); p.setPen(pen2)
        for i in (2, 4, 6):
            frac = i / 8
            x = pad + int(frac * (W - 2*pad))
            y = pad + int(frac * (H - 2*pad))
            p.drawLine(x, pad, x, H-pad)
            p.drawLine(pad, y, W-pad, y)

        # ── Diagonal reference ──
        diag_pen = QPen(QColor("#2a4a6a")); diag_pen.setWidth(1)
        diag_pen.setStyle(Qt.PenStyle.DashLine); p.setPen(diag_pen)
        p.drawLine(pad, H-pad, W-pad, pad)

        # ── Histogram background (smooth filled) ──
        # L modunda: R/G/B kanallarını kendi renginde göster
        if ch == "L":
            _ch_layers = [("R", RED, 0.18), ("G", GREEN, 0.18), ("B", "#6699ff", 0.18)]
        else:
            _ch_layers = [(ch, glow[0], 0.25)]
        for _lch, _lcol, _lalpha in _ch_layers:
            hdata = self._hdata.get(_lch)
            if hdata is None: continue
            mx = hdata.max()
            if mx <= 0: continue
            hist_path = QPainterPath()
            hist_path.moveTo(pad, H - pad)
            n = len(hdata)
            for i in range(n):
                bh = (hdata[i] / mx) * (H - 2*pad)
                hx = pad + (i / n) * (W - 2*pad)
                hist_path.lineTo(hx, H - pad - bh)
            hist_path.lineTo(W - pad, H - pad)
            hist_path.closeSubpath()
            _hcol = QColor(_lcol); _hcol.setAlphaF(_lalpha)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(_hcol))
            p.drawPath(hist_path)

        # ── Curve fill (area under curve with gradient) ──
        pts  = sorted(self._pts[ch], key=lambda q:q[0])
        lut  = self._pts_to_lut(pts)
        curve_path = QPainterPath()
        w_pts = []
        for i, y_val in enumerate(lut):
            wp = self._to_widget(i/255, y_val)
            w_pts.append(wp)
            if i == 0: curve_path.moveTo(wp)
            else:      curve_path.lineTo(wp)

        # Area fill
        fill_path = QPainterPath(curve_path)
        fill_path.lineTo(self._to_widget(1.0, 0.0))
        fill_path.lineTo(self._to_widget(0.0, 0.0))
        fill_path.closeSubpath()
        fill_grad = QLinearGradient(0, pad, 0, H - pad)
        fill_top = QColor(glow[0]); fill_top.setAlphaF(0.12)
        fill_bot = QColor(glow[2]); fill_bot.setAlphaF(0.02)
        fill_grad.setColorAt(0, fill_top)
        fill_grad.setColorAt(1, fill_bot)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(fill_grad))
        p.drawPath(fill_path)

        # ── Glow layer (wide soft line) ──
        glow_col = QColor(glow[0]); glow_col.setAlphaF(0.15)
        glow_pen = QPen(glow_col); glow_pen.setWidth(8); glow_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(glow_pen); p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(curve_path)

        # ── Main curve line ──
        main_col = QColor(glow[0])
        main_pen = QPen(main_col); main_pen.setWidth(2); main_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(main_pen); p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(curve_path)

        # ── Control points ──
        for idx, (cx, cy) in enumerate(self._pts[ch]):
            wp = self._to_widget(cx, cy)
            is_drag  = (idx == self._drag_idx)
            is_hover = (idx == self._hover_idx)

            # Outer glow
            if is_drag or is_hover:
                glow_r = 14 if is_drag else 11
                gc = QColor(glow[0]); gc.setAlphaF(0.25 if is_drag else 0.15)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QBrush(gc))
                p.drawEllipse(wp, glow_r, glow_r)

            # Ring
            r = 7 if is_drag else (6 if is_hover else 5)
            ring_pen = QPen(QColor(glow[0]))
            ring_pen.setWidth(2 if is_drag else 2)
            p.setPen(ring_pen)

            # Fill gradient for point
            pt_grad = QLinearGradient(wp.x()-r, wp.y()-r, wp.x()+r, wp.y()+r)
            if is_drag:
                pt_grad.setColorAt(0, QColor("#ffffff"))
                pt_grad.setColorAt(1, QColor(glow[0]))
            elif is_hover:
                pt_grad.setColorAt(0, QColor(glow[0]))
                pt_grad.setColorAt(1, QColor(glow[1]))
            else:
                pt_grad.setColorAt(0, QColor("#0e2238"))
                pt_grad.setColorAt(1, QColor("#081828"))
            p.setBrush(QBrush(pt_grad))
            p.drawEllipse(wp, r, r)

            # Center dot
            if is_drag:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QBrush(QColor("#ffffff")))
                p.drawEllipse(wp, 2, 2)

        # ── Coordinate tooltip near dragged point ──
        if self._coord_label and self._drag_idx is not None:
            wx, wy, txt = self._coord_label
            label_bg = QColor("#000000"); label_bg.setAlphaF(0.75)
            fm = p.fontMetrics()
            tw = fm.horizontalAdvance(txt) + 10
            th = fm.height() + 4
            lx = min(wx + 12, W - tw - 4)
            ly = max(wy - th - 4, 4)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(label_bg))
            p.drawRoundedRect(int(lx), int(ly), tw, th, 3, 3)
            p.setPen(QColor(glow[0]))
            p.setFont(QFont("Consolas", 8))
            p.drawText(int(lx) + 5, int(ly) + th - 4, txt)

        # ── Scale labels ──
        p.setPen(QColor("#3a5a7a"))
        p.setFont(QFont("Consolas", 7))
        for i in range(0, 5):
            v = i / 4
            x = pad + int(v * (W - 2*pad))
            y = pad + int((1-v) * (H - 2*pad))
            p.drawText(x - 8, H - pad + 12, f"{v:.1f}"[1:])
            p.drawText(2, y + 4, f"{v:.0%}"[:-1] if v < 1 else "1")

        # ── Border ──
        border_pen = QPen(QColor("#1a3a5c")); border_pen.setWidth(1)
        p.setPen(border_pen); p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(inner)

        p.end()

    def _nearest(self, px, py, tol=16):
        best_i, best_d = None, tol
        for i, (cx, cy) in enumerate(self._pts[self._ch]):
            wp  = self._to_widget(cx, cy)
            d   = ((wp.x()-px)**2 + (wp.y()-py)**2)**0.5
            if d < best_d: best_i, best_d = i, d
        return best_i

    def mousePressEvent(self, ev):
        px, py = ev.pos().x(), ev.pos().y()
        idx = self._nearest(px, py)
        if idx is not None:
            if ev.button() == Qt.MouseButton.RightButton and len(self._pts[self._ch]) > 2:
                self._pts[self._ch].pop(idx)
                self._coord_label = None
                self.update(); self._emit(); return
            self._drag_idx = idx
            cx, cy = self._pts[self._ch][idx]
            wp = self._to_widget(cx, cy)
            self._coord_label = (wp.x(), wp.y(), f"({cx:.3f}, {cy:.3f})")
            self.update()
        else:
            if ev.button() == Qt.MouseButton.LeftButton:
                x, y = self._from_widget(px, py)
                self._pts[self._ch].append((x,y))
                self._pts[self._ch].sort(key=lambda q:q[0])
                self._drag_idx = next(
                    (i for i,pt in enumerate(self._pts[self._ch])
                     if abs(pt[0]-x)<0.001), None)
                wp = self._to_widget(x, y)
                self._coord_label = (wp.x(), wp.y(), f"({x:.3f}, {y:.3f})")
                self.update(); self._emit()

    def mouseMoveEvent(self, ev):
        px, py = ev.pos().x(), ev.pos().y()
        if self._drag_idx is not None:
            x, y = self._from_widget(px, py)
            pts  = self._pts[self._ch]
            lo = pts[self._drag_idx-1][0]+0.005 if self._drag_idx>0 else 0.0
            hi = pts[self._drag_idx+1][0]-0.005 if self._drag_idx<len(pts)-1 else 1.0
            x  = float(np.clip(x, lo, hi))
            y  = float(np.clip(y, 0, 1))
            pts[self._drag_idx] = (x, y)
            wp = self._to_widget(x, y)
            self._coord_label = (wp.x(), wp.y(), f"({x:.3f}, {y:.3f})")
            self.update(); self._emit()
        else:
            self._hover_idx = self._nearest(px, py)
            if self._hover_idx is not None:
                self.setCursor(Qt.CursorShape.SizeAllCursor)
            else:
                self.setCursor(Qt.CursorShape.CrossCursor)
            self._coord_label = None
            self.update()

    def mouseReleaseEvent(self, ev):
        self._drag_idx = None
        self._coord_label = None
        self.update()


# ─────────────────────────────────────────────────────────────────────────────
#  SliderRow — label + slider + spinbox
# ─────────────────────────────────────────────────────────────────────────────
class SliderRow(QWidget):
    value_changed = pyqtSignal(float)

    def __init__(self, label, lo, hi, default, decimals=2, step=None, parent=None):
        super().__init__(parent)
        self._lo = lo; self._hi = hi; self._dec = decimals
        self._scale = 10**decimals
        lay = QHBoxLayout(self); lay.setContentsMargins(0,0,0,0); lay.setSpacing(4)

        lbl = QLabel(label); lbl.setFixedWidth(78)
        lbl.setStyleSheet(f"color:{MUTED};font-size:10px;")

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(int(lo*self._scale), int(hi*self._scale))
        self._slider.setValue(int(default*self._scale))
        self._slider.setStyleSheet(SLIDER_CSS)

        self._spin = QDoubleSpinBox()
        self._spin.setRange(lo, hi)
        self._spin.setValue(default)
        self._spin.setDecimals(decimals)
        self._spin.setSingleStep(step or 10**-decimals)
        self._spin.setFixedWidth(65)
        self._spin.setStyleSheet(SPIN_CSS)

        lay.addWidget(lbl); lay.addWidget(self._slider, 1); lay.addWidget(self._spin)

        self._slider.valueChanged.connect(self._from_slider)
        self._spin.valueChanged.connect(self._from_spin)
        self._block = False

    def _from_slider(self, v):
        if self._block: return
        self._block = True
        val = v / self._scale
        self._spin.setValue(val)
        self._block = False
        self.value_changed.emit(val)

    def _from_spin(self, v):
        if self._block: return
        self._block = True
        self._slider.setValue(int(v * self._scale))
        self._block = False
        self.value_changed.emit(v)

    def value(self): return self._spin.value()

    def setValue(self, v):
        self._block = True
        self._spin.setValue(v)
        self._slider.setValue(int(v*self._scale))
        self._block = False

    def reset(self):
        default = (self._lo + self._hi) / 2
        self.setValue(default)


# ─────────────────────────────────────────────────────────────────────────────
#  HistogramEditorPanel — full Photoshop-style editor
# ─────────────────────────────────────────────────────────────────────────────
class HistogramEditorPanel(QWidget):
    """
    Main panel embedded in ImageViewer's Histogram tab.
    Signals:
      preview_changed(ndarray)  — live preview, don't add to history
      apply_requested(ndarray)  — user clicked Apply → add to history
    """
    preview_changed = pyqtSignal(object)
    apply_requested = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{BG};")
        self._img      = None   # current image
        self._orig_img = None   # pristine original for non-destructive editing
        self._ch       = "L"
        self._linked   = True
        self._live     = True
        self._debounce = QTimer(); self._debounce.setSingleShot(True)
        self._debounce.timeout.connect(self._emit_preview)
        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6,6,6,6); root.setSpacing(4)

        # ── Channel selector ──────────────────────────────────────────────
        ch_row = QHBoxLayout(); ch_row.setSpacing(3)
        ch_lbl = QLabel("Channel")
        ch_lbl.setStyleSheet(f"color:{MUTED};font-size:9px;font-weight:600;letter-spacing:0.5px;")
        ch_row.addWidget(ch_lbl)
        self._ch_btns = {}
        _ch_cols = {"L": ACCENT2, "R": RED, "G": GREEN, "B": "#6699ff"}
        for ch, col in _ch_cols.items():
            b = QPushButton(ch); b.setFixedSize(30, 24); b.setCheckable(True)
            b.setStyleSheet(
                f"QPushButton{{background:{BG3};color:{SUBTEXT};border:1px solid {BORDER};"
                f"border-radius:4px;font-size:11px;font-weight:800;}}"
                f"QPushButton:checked{{background:{col}40;color:{col};"
                f"border:1px solid {col};border-bottom:2px solid {col};}}"
                f"QPushButton:hover{{color:{col};background:{col}18;border-color:{col}66;}}")
            b.clicked.connect(lambda _, c=ch: self._set_channel(c))
            ch_row.addWidget(b); self._ch_btns[ch] = b
        self._ch_btns["L"].setChecked(True)
        ch_row.addSpacing(8)
        self._chk_link = QCheckBox("Link RGB")
        self._chk_link.setChecked(True)
        self._chk_link.setStyleSheet(
            f"QCheckBox{{color:{MUTED};font-size:9px;spacing:4px;}}"
            f"QCheckBox::indicator{{width:14px;height:14px;border-radius:3px;"
            f"border:1px solid {BORDER};background:{BG3};}}"
            f"QCheckBox::indicator:checked{{background:{ACCENT}88;border:1px solid {ACCENT};}}")
        ch_row.addWidget(self._chk_link)
        ch_row.addStretch()
        self._chk_live = QCheckBox("Live")
        self._chk_live.setChecked(True)
        self._chk_live.setStyleSheet(
            f"QCheckBox{{color:{MUTED};font-size:9px;spacing:4px;}}"
            f"QCheckBox::indicator{{width:14px;height:14px;border-radius:3px;"
            f"border:1px solid {BORDER};background:{BG3};}}"
            f"QCheckBox::indicator:checked{{background:{GREEN}88;border:1px solid {GREEN};}}")
        ch_row.addWidget(self._chk_live)
        root.addLayout(ch_row)

        # ── Tab widget ────────────────────────────────────────────────────
        self._tabs = QTabWidget(); self._tabs.setStyleSheet(TAB_CSS)
        root.addWidget(self._tabs, 1)

        # Tab 1: Levels
        self._levels_tab()
        # Tab 2: Curves
        self._curves_tab()
        # Tab 3: Adjustments
        self._adjustments_tab()

        # ── Bottom buttons ────────────────────────────────────────────────
        btn_row = QHBoxLayout(); btn_row.setSpacing(8)
        b_reset = QPushButton("↺ Reset All")
        b_reset.setFixedHeight(28)
        b_reset.setStyleSheet(
            f"QPushButton{{background:{BG3};color:{MUTED};"
            f"border:1px solid {BORDER};border-radius:5px;"
            f"padding:3px 12px;font-size:10px;font-weight:600;}}"
            f"QPushButton:hover{{color:{TEXT};background:{BG4};"
            f"border-color:{ACCENT};}}"
            f"QPushButton:pressed{{background:{BG};}}")
        b_reset.clicked.connect(self._reset_all)
        btn_row.addWidget(b_reset)
        btn_row.addStretch()
        b_apply = QPushButton("✅  Apply to Image")
        b_apply.setFixedHeight(30)
        b_apply.setStyleSheet(
            f"QPushButton{{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            f"stop:0 {GREEN},stop:1 #2aaa55);color:#ffffff;"
            f"border:none;border-radius:5px;padding:4px 18px;"
            f"font-size:11px;font-weight:700;letter-spacing:0.3px;}}"
            f"QPushButton:hover{{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            f"stop:0 #5ad48a,stop:1 #3dbd6e);}}"
            f"QPushButton:pressed{{background:{BG4};}}")
        b_apply.clicked.connect(lambda: self._apply(emit=True))
        btn_row.addWidget(b_apply)
        root.addLayout(btn_row)

    def _levels_tab(self):
        w = QWidget(); w.setStyleSheet(f"background:{BG2};")
        lay = QVBoxLayout(w); lay.setContentsMargins(6,6,6,4); lay.setSpacing(6)

        # Histogram + triangle widget
        self._hist_wgt = HistogramWidget()
        self._hist_wgt.levels_changed.connect(self._on_levels_changed)
        lay.addWidget(self._hist_wgt, 1)

        # Input spinboxes
        inp_row = QHBoxLayout(); inp_row.setSpacing(4)
        inp_row.addWidget(QLabel("In:"))
        self._sp_black = self._spinbox(0, 1, 0.0, 3)
        self._sp_mid   = self._spinbox(0, 1, 0.5, 3)
        self._sp_white = self._spinbox(0, 1, 1.0, 3)
        for sp, tip in [(self._sp_black,"Shadows (Black point)"),
                        (self._sp_mid,  "Midtones (Gamma)"),
                        (self._sp_white,"Highlights (White point)")]:
            sp.setToolTip(tip)
            sp.valueChanged.connect(self._on_spin_levels)
            inp_row.addWidget(sp)
        inp_row.addStretch()
        b_auto = QPushButton("Auto")
        b_auto.setStyleSheet(BTN_CSS); b_auto.setFixedHeight(22); b_auto.setFixedWidth(44)
        b_auto.setToolTip("Auto Levels: stretch to 0.1%–99.9%")
        b_auto.clicked.connect(self._auto_levels)
        inp_row.addWidget(b_auto)
        lay.addLayout(inp_row)

        # Output spinboxes
        out_row = QHBoxLayout(); out_row.setSpacing(4)
        out_row.addWidget(QLabel("Out:"))
        self._sp_out_lo = self._spinbox(0, 1, 0.0, 3)
        self._sp_out_hi = self._spinbox(0, 1, 1.0, 3)
        self._sp_out_lo.setToolTip("Output black (lift shadows)")
        self._sp_out_hi.setToolTip("Output white (compress highlights)")
        self._sp_out_lo.valueChanged.connect(self._on_spin_levels)
        self._sp_out_hi.valueChanged.connect(self._on_spin_levels)
        out_row.addWidget(self._sp_out_lo); out_row.addWidget(self._sp_out_hi)
        out_row.addStretch()
        self._tabs.addTab(w, "Levels")
        lay.addLayout(out_row)

    def _curves_tab(self):
        w = QWidget(); w.setStyleSheet(f"background:{BG2};")
        lay = QVBoxLayout(w); lay.setContentsMargins(4,4,4,4); lay.setSpacing(3)

        self._curves_wgt = CurvesWidget()
        self._curves_wgt.curve_changed.connect(self._on_curve_changed)
        lay.addWidget(self._curves_wgt, 1)

        hint = QLabel("🖱 Left: add point  •  Right: remove  •  Drag: adjust")
        hint.setStyleSheet(f"color:{SUBTEXT};font-size:8px;letter-spacing:0.3px;")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(hint)

        b_reset_curve = QPushButton("↺ Reset Curve")
        b_reset_curve.setFixedHeight(24)
        b_reset_curve.setStyleSheet(
            f"QPushButton{{background:{BG3};color:{MUTED};"
            f"border:1px solid {BORDER};border-radius:4px;"
            f"font-size:10px;font-weight:600;}}"
            f"QPushButton:hover{{color:{ACCENT2};border-color:{ACCENT};"
            f"background:{BG4};}}"
            f"QPushButton:pressed{{background:{BG};}}")
        b_reset_curve.clicked.connect(lambda: self._curves_wgt.reset_channel(self._ch))
        lay.addWidget(b_reset_curve)
        self._tabs.addTab(w, "Curves")

    def _adjustments_tab(self):
        w = QWidget(); w.setStyleSheet(f"background:{BG2};")
        scroll = QScrollArea(); scroll.setWidget(w)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"background:{BG2};border:none;")
        lay = QVBoxLayout(w); lay.setContentsMargins(8,8,8,8); lay.setSpacing(2)

        def _sep(title):
            lbl = QLabel(title)
            lbl.setStyleSheet(
                f"color:{HEAD};font-size:10px;font-weight:700;"
                f"border-bottom:1px solid {BORDER};padding-bottom:2px;margin-top:6px;")
            lay.addWidget(lbl)

        def _row(label, lo, hi, default, dec=2, step=None):
            sr = SliderRow(label, lo, hi, default, dec, step)
            sr.value_changed.connect(self._on_adjustment)
            lay.addWidget(sr)
            return sr

        _sep("Exposure & Tone")
        self._adj_exposure    = _row("Exposure",    -3.0,  3.0,  0.0, 2, 0.05)
        self._adj_brightness  = _row("Brightness",  -1.0,  1.0,  0.0, 2, 0.01)
        self._adj_contrast    = _row("Contrast",    -1.0,  1.0,  0.0, 2, 0.01)
        self._adj_highlights  = _row("Highlights",  -1.0,  1.0,  0.0, 2, 0.01)
        self._adj_shadows     = _row("Shadows",     -1.0,  1.0,  0.0, 2, 0.01)
        self._adj_whites      = _row("Whites",      -1.0,  1.0,  0.0, 2, 0.01)
        self._adj_blacks      = _row("Blacks",      -1.0,  1.0,  0.0, 2, 0.01)

        _sep("Color")
        self._adj_temp        = _row("Color Temp",  -1.0,  1.0,  0.0, 2, 0.01)
        self._adj_tint        = _row("Tint",        -1.0,  1.0,  0.0, 2, 0.01)
        self._adj_vibrance    = _row("Vibrance",    -1.0,  1.0,  0.0, 2, 0.01)
        self._adj_saturation  = _row("Saturation",  -1.0,  1.0,  0.0, 2, 0.01)
        self._adj_hue         = _row("Hue Shift",  -0.5,  0.5,  0.0, 2, 0.005)

        _sep("Detail")
        self._adj_clarity     = _row("Clarity",      0.0,  1.0,  0.0, 2, 0.01)
        self._adj_dehaze      = _row("Dehaze",        0.0,  1.0,  0.0, 2, 0.01)

        lay.addStretch()

        b_reset_adj = QPushButton("↺ Reset Adjustments")
        b_reset_adj.setStyleSheet(BTN_CSS); b_reset_adj.setFixedHeight(22)
        b_reset_adj.clicked.connect(self._reset_adjustments)
        lay.addWidget(b_reset_adj)

        self._tabs.addTab(scroll, "Adjustments")

    def _spinbox(self, lo, hi, val, dec):
        sp = QDoubleSpinBox()
        sp.setRange(lo, hi); sp.setValue(val)
        sp.setDecimals(dec); sp.setSingleStep(10**-dec)
        sp.setFixedWidth(66); sp.setStyleSheet(SPIN_CSS)
        return sp

    # ── Data ──────────────────────────────────────────────────────────────────
    def set_image(self, img: np.ndarray, reset: bool = False):
        self._img = np.clip(img, 0, 1).astype(np.float32) if img is not None else None
        self._orig_img = self._img.copy() if self._img is not None else None
        self._hist_wgt.set_image(self._img)
        self._curves_wgt.set_image(self._img)
        if reset:
            self._reset_all_silent()

    def _set_channel(self, ch):
        self._ch = ch
        for c, b in self._ch_btns.items(): b.setChecked(c==ch)
        self._hist_wgt.set_channel(ch)
        self._curves_wgt.set_channel(ch)
        # Sync spinboxes to channel state
        st = self._hist_wgt.get_state(ch)
        self._sync_spins_from_state(st)

    def _sync_spins_from_state(self, st):
        for sp, v in [(self._sp_black, st[0]), (self._sp_mid, st[1]),
                      (self._sp_white, st[2]), (self._sp_out_lo, st[3]),
                      (self._sp_out_hi, st[4])]:
            sp.blockSignals(True); sp.setValue(v); sp.blockSignals(False)

    # ── Callbacks ─────────────────────────────────────────────────────────────
    def _on_levels_changed(self, ch, b, m, w, ol, oh):
        """Histogram widget drag → sync spinboxes."""
        channels = self._linked_channels(ch)
        for c in channels:
            self._hist_wgt.set_state(c, b, m, w, ol, oh)
        self._sync_spins_from_state([b, m, w, ol, oh])
        self._schedule_preview()

    def _on_spin_levels(self):
        b  = float(self._sp_black.value())
        m  = float(self._sp_mid.value())
        w  = float(self._sp_white.value())
        ol = float(self._sp_out_lo.value())
        oh = float(self._sp_out_hi.value())
        # Clamp
        b = min(b, w-0.01); w = max(w, b+0.01)
        m = float(np.clip(m, b, w))
        channels = self._linked_channels(self._ch)
        for c in channels:
            self._hist_wgt.set_state(c, b, m, w, ol, oh)
        self._schedule_preview()

    def _on_curve_changed(self, ch, lut):
        channels = self._linked_channels(ch)
        if len(channels) > 1:
            for c in channels:
                self._curves_wgt._pts[c] = list(self._curves_wgt._pts[ch])
        self._schedule_preview()

    def _on_adjustment(self, _):
        self._schedule_preview()

    def _linked_channels(self, ch):
        if not self._chk_link.isChecked():
            return [ch]
        if ch == "L":
            return ["L","R","G","B"]
        return ["R","G","B","L"]

    def _schedule_preview(self):
        if self._chk_live.isChecked():
            self._debounce.start(50)

    def _emit_preview(self):
        if self._img is None: return
        result = self._apply(emit=False)
        if result is not None:
            self.preview_changed.emit(result)

    def _auto_levels(self):
        if self._img is None: return
        ch = self._ch
        if self._img.ndim == 2:
            data = self._img.ravel()
        elif ch == "L":
            data = self._img.mean(axis=2).ravel()
        else:
            ci = {"R":0,"G":1,"B":2}[ch]
            data = self._img[:,:,ci].ravel()
        lo = float(np.percentile(data, 0.1))
        hi = float(np.percentile(data, 99.9))
        mid = 0.5
        for c in self._linked_channels(ch):
            self._hist_wgt.set_state(c, lo, mid, hi, 0.0, 1.0)
        self._sync_spins_from_state([lo, mid, hi, 0.0, 1.0])
        self._schedule_preview()

    def _reset_all(self):
        self._hist_wgt.reset_channel()
        self._curves_wgt.reset_channel()
        self._reset_adjustments()
        self._sync_spins_from_state([0.0,0.5,1.0,0.0,1.0])
        self._schedule_preview()

    def _reset_all_silent(self):
        """Reset all controls without emitting preview (used after Apply).
        Image stays as-is; only the editor UI resets to defaults."""
        # Stop any pending preview timer
        self._debounce.stop()
        # Block signals so reset doesn't trigger preview chain
        self._hist_wgt.blockSignals(True)
        self._curves_wgt.blockSignals(True)
        # Reset levels: all channels → [0, 0.5, 1, 0, 1]
        for c in list(self._hist_wgt._state.keys()):
            self._hist_wgt._state[c] = [0.0, 0.5, 1.0, 0.0, 1.0]
        # Reset curves: all channels → diagonal line
        for c in list(self._curves_wgt._pts.keys()):
            self._curves_wgt._pts[c] = [(0.0, 0.0), (1.0, 1.0)]
        # Unblock signals
        self._hist_wgt.blockSignals(False)
        self._curves_wgt.blockSignals(False)
        # Reset adjustment sliders
        self._reset_adjustments()
        # Sync spinboxes to default
        self._sync_spins_from_state([0.0, 0.5, 1.0, 0.0, 1.0])
        # Force immediate repaint (not deferred)
        self._hist_wgt.repaint()
        self._curves_wgt.repaint()

    def _reset_adjustments(self):
        defaults = {
            "_adj_exposure":0.0,"_adj_brightness":0.0,"_adj_contrast":0.0,
            "_adj_highlights":0.0,"_adj_shadows":0.0,"_adj_whites":0.0,
            "_adj_blacks":0.0,"_adj_temp":0.0,"_adj_tint":0.0,
            "_adj_vibrance":0.0,"_adj_saturation":0.0,"_adj_hue":0.0,
            "_adj_clarity":0.0,"_adj_dehaze":0.0,
        }
        for attr, val in defaults.items():
            getattr(self, attr).setValue(val)

    # ── Apply ─────────────────────────────────────────────────────────────────
    def _apply(self, emit=True):
        if self._img is None: return None
        # Always apply from ORIGINAL — prevents cumulative degradation on live preview
        base = self._orig_img if self._orig_img is not None else self._img
        img = base.astype(np.float64)

        # ① Levels per channel
        def _apply_levels_1ch(c, st):
            b, m, w, ol, oh = st
            rng = max(w - b, 1e-9)
            c = np.clip((c - b) / rng, 0, 1)
            if abs(m - 0.5) > 0.005:
                eps = 1e-9
                c = np.where(c<=0, 0, np.where(c>=1, 1,
                    (m-1)*c / ((2*m-1)*c - m + eps)))
                c = np.clip(c, 0, 1)
            c = ol + c * (oh - ol)
            return c

        if img.ndim == 2:
            img = _apply_levels_1ch(img, self._hist_wgt.get_state("L"))
        else:
            # Önce L kanalı — tüm RGB'ye eşit uygula
            st_L = self._hist_wgt.get_state("L")
            if st_L != [0.0, 0.5, 1.0, 0.0, 1.0]:
                for i in range(3):
                    img[:,:,i] = _apply_levels_1ch(img[:,:,i], st_L)
            # Sonra ayrı R, G, B kanalları
            for i, ch in enumerate(("R","G","B")):
                st = self._hist_wgt.get_state(ch)
                if st != [0.0, 0.5, 1.0, 0.0, 1.0]:
                    img[:,:,i] = _apply_levels_1ch(img[:,:,i], st)

        # ② Curves per channel (apply LUT)
        pts_L = self._curves_wgt._pts.get("L", [(0,0),(1,1)])
        is_flat_L = (len(pts_L)==2 and pts_L[0]==(0,0) and pts_L[1]==(1,1))
        if img.ndim == 2:
            if not is_flat_L:
                lut = self._curves_wgt.get_lut("L")
                img = np.clip(np.interp(img, np.linspace(0,1,256), lut), 0, 1)
        else:
            # Önce L curve — tüm RGB'ye uygula
            if not is_flat_L:
                lut_L = self._curves_wgt.get_lut("L")
                xs = np.linspace(0,1,256)
                for i in range(3):
                    img[:,:,i] = np.interp(img[:,:,i], xs, lut_L)
            # Sonra ayrı R, G, B curves
            for i, ch in enumerate(("R","G","B")):
                pts = self._curves_wgt._pts.get(ch, [(0,0),(1,1)])
                is_flat = (len(pts)==2 and pts[0]==(0,0) and pts[1]==(1,1))
                if not is_flat:
                    lut = self._curves_wgt.get_lut(ch)
                    img[:,:,i] = np.interp(img[:,:,i], np.linspace(0,1,256), lut)

        img = np.clip(img, 0, 1).astype(np.float32)

        # ③ Adjustments
        img = self._apply_adjustments(img)
        img = np.clip(img, 0, 1).astype(np.float32)

        if emit:
            self.apply_requested.emit(img)
            # After apply: update baseline and reset all controls to defaults
            self._img = img.copy()
            self._orig_img = img.copy()
            self._hist_wgt.set_image(self._img)
            self._curves_wgt.set_image(self._img)
            self._reset_all_silent()
        return img

    def _apply_adjustments(self, img: np.ndarray) -> np.ndarray:
        img = img.astype(np.float32)

        # Exposure (EV stops)
        exp = float(self._adj_exposure.value())
        if abs(exp) > 0.001:
            img = np.clip(img * (2.0 ** exp), 0, 1)

        # Brightness (additive)
        br = float(self._adj_brightness.value())
        if abs(br) > 0.001:
            img = np.clip(img + br * 0.5, 0, 1)

        # Contrast (S-curve)
        ct = float(self._adj_contrast.value())
        if abs(ct) > 0.001:
            mid = 0.5
            img = np.clip(mid + (img - mid) * (1 + ct), 0, 1)

        # Highlights recovery
        hl = float(self._adj_highlights.value())
        if abs(hl) > 0.001 and img.ndim == 3:
            gray = img.mean(axis=2, keepdims=True)
            mask = np.clip((gray - 0.5) * 2, 0, 1)
            img  = np.clip(img + hl * mask * (-0.5), 0, 1)

        # Shadows boost
        sh = float(self._adj_shadows.value())
        if abs(sh) > 0.001 and img.ndim == 3:
            gray = img.mean(axis=2, keepdims=True)
            mask = np.clip((0.5 - gray) * 2, 0, 1)
            img  = np.clip(img + sh * mask * 0.5, 0, 1)

        # Whites / Blacks
        wh = float(self._adj_whites.value())
        bl = float(self._adj_blacks.value())
        if abs(wh) > 0.001:
            img = np.clip(img + wh * (img**2), 0, 1)
        if abs(bl) > 0.001:
            img = np.clip(img + bl * ((1-img)**2) * (-0.3), 0, 1)

        if img.ndim == 3:
            # Color Temp (blue↔orange)
            temp = float(self._adj_temp.value())
            if abs(temp) > 0.001:
                img[:,:,0] = np.clip(img[:,:,0] + temp * 0.15, 0, 1)
                img[:,:,2] = np.clip(img[:,:,2] - temp * 0.15, 0, 1)

            # Tint (green↔magenta)
            tint = float(self._adj_tint.value())
            if abs(tint) > 0.001:
                img[:,:,1] = np.clip(img[:,:,1] + tint * 0.10, 0, 1)
                img[:,:,0] = np.clip(img[:,:,0] - tint * 0.05, 0, 1)

            # Saturation + Vibrance
            sat = float(self._adj_saturation.value())
            vib = float(self._adj_vibrance.value())
            if abs(sat) > 0.001 or abs(vib) > 0.001:
                gray = img.mean(axis=2, keepdims=True)
                chroma = img - gray
                if abs(sat) > 0.001:
                    img = np.clip(gray + chroma * (1 + sat), 0, 1)
                if abs(vib) > 0.001:
                    # Vibrance: protect already-saturated pixels
                    sat_map = np.clip(np.abs(chroma).max(axis=2, keepdims=True) * 2, 0, 1)
                    protect = 1 - sat_map
                    img = np.clip(img + vib * chroma * protect, 0, 1)

            # Hue Shift (via HSV)
            hue = float(self._adj_hue.value())
            if abs(hue) > 0.001:
                img8 = (np.clip(img,0,1)*255).astype(np.uint8)
                hsv  = cv2.cvtColor(img8, cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[:,:,0] = (hsv[:,:,0] + hue * 180) % 180
                img8 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
                img  = img8.astype(np.float32) / 255.0

            # Clarity (local contrast via unsharp)
            clar = float(self._adj_clarity.value())
            if clar > 0.001:
                blur = cv2.GaussianBlur(img, (0,0), 30)
                img  = np.clip(img + clar * (img - blur), 0, 1)

            # Dehaze
            dh = float(self._adj_dehaze.value())
            if dh > 0.001:
                dark = img.min(axis=2, keepdims=True)
                img  = np.clip((img - dark * dh) / max(1 - dark.mean()*dh, 0.1), 0, 1)

        return np.clip(img, 0, 1).astype(np.float32)
