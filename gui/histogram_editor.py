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
            bar_w = max(1.0, r.width() / 256)
            col = QColor(color); col.setAlphaF(alpha)
            p.setPen(Qt.PenStyle.NoPen); p.setBrush(QBrush(col))
            path = QPainterPath()
            path.moveTo(r.left(), r.bottom())
            rh = r.height()
            rl = r.left()
            rb = r.bottom()
            for i, v in enumerate(norm):
                bh = v * rh
                bx = rl + i * bar_w
                path.lineTo(bx, rb - bh)
                path.lineTo(bx + bar_w, rb - bh)
            path.lineTo(rl + 256 * bar_w, rb)
            path.closeSubpath()
            p.drawPath(path)

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
        self._cached_bg_size = None  # (W, H) for gradient cache
        self._cached_bg_grad = None
        self._cached_inner_grad = None

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

        # ── Background gradient (cached) ──
        if self._cached_bg_size != (W, H):
            self._cached_bg_size = (W, H)
            self._cached_bg_grad = QLinearGradient(0, 0, 0, H)
            self._cached_bg_grad.setColorAt(0, QColor("#060e18"))
            self._cached_bg_grad.setColorAt(1, QColor("#0a1a2e"))
            self._cached_inner_grad = QLinearGradient(pad, pad, pad, H-pad)
            self._cached_inner_grad.setColorAt(0, QColor("#081520"))
            self._cached_inner_grad.setColorAt(0.5, QColor("#0c1c30"))
            self._cached_inner_grad.setColorAt(1, QColor("#081520"))
        p.fillRect(0, 0, W, H, QBrush(self._cached_bg_grad))

        inner = QRect(pad, pad, W-2*pad, H-2*pad)
        p.fillRect(inner, QBrush(self._cached_inner_grad))

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

        lbl = QLabel(label); lbl.setFixedWidth(92)
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
      preview_changed(ndarray)  - live preview, don't add to history
      apply_requested(ndarray)  - user clicked Apply -> add to history
    """
    preview_changed = pyqtSignal(object)
    apply_requested = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{BG};")
        self._img = None
        self._orig_img = None
        self._preview_img = None
        self._preview_orig = None
        self._preview_scale = 1.0
        self._ch = "L"
        self._linked = True
        self._live = True
        self._debounce = QTimer()
        self._debounce.setSingleShot(True)
        self._debounce.timeout.connect(self._emit_preview)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        hist_card = QFrame()
        hist_card.setStyleSheet(
            f"QFrame{{background:{BG2};border:1px solid {BORDER};border-radius:6px;}}"
        )
        hist_lay = QVBoxLayout(hist_card)
        hist_lay.setContentsMargins(8, 8, 8, 8)
        hist_lay.setSpacing(6)
        self._hist_wgt = HistogramWidget()
        self._hist_wgt.setMinimumHeight(152)
        self._hist_wgt.setMaximumHeight(184)
        self._hist_wgt.levels_changed.connect(self._on_levels_changed)
        hist_lay.addWidget(self._hist_wgt)
        root.addWidget(hist_card)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet(
            f"QScrollArea{{background:{BG2};border:1px solid {BORDER};border-radius:6px;}}"
        )
        self._panel_host = QWidget()
        self._panel_host.setStyleSheet(f"background:{BG2};")
        self._panel_lay = QVBoxLayout(self._panel_host)
        self._panel_lay.setContentsMargins(8, 8, 8, 8)
        self._panel_lay.setSpacing(8)
        self._scroll.setWidget(self._panel_host)
        root.addWidget(self._scroll, 1)

        top = QHBoxLayout()
        top.setSpacing(6)
        title = QLabel("Düzenle")
        title.setStyleSheet(f"color:{HEAD};font-size:11px;font-weight:800;")
        top.addWidget(title)
        top.addStretch()
        b_auto = QPushButton("Otomatik")
        b_auto.setStyleSheet(BTN_CSS)
        b_auto.setFixedHeight(24)
        b_auto.clicked.connect(self._auto_raw_adjustments)
        top.addWidget(b_auto)
        self._raw_bw_btn = QPushButton("Siyah Beyaz")
        self._raw_bw_btn.setCheckable(True)
        self._raw_bw_btn.setFixedHeight(24)
        self._raw_bw_btn.setStyleSheet(BTN_CSS)
        self._raw_bw_btn.toggled.connect(self._toggle_bw_profile)
        top.addWidget(self._raw_bw_btn)
        self._panel_lay.addLayout(top)

        profile_row = QHBoxLayout()
        profile_row.setSpacing(6)
        profile_lbl = QLabel("Profil")
        profile_lbl.setStyleSheet(f"color:{MUTED};font-size:10px;")
        self._raw_profile = self._styled_combo()
        self._raw_profile.addItems(["Renk", "Siyah Beyaz"])
        self._raw_profile.currentTextChanged.connect(self._on_profile_changed)
        profile_row.addWidget(profile_lbl)
        profile_row.addWidget(self._raw_profile, 1)
        self._panel_lay.addLayout(profile_row)

        curve_frame, curve_lay = self._make_section("Eğri", section_key="curve", expanded=True)
        self._panel_lay.addWidget(curve_frame)
        self._build_curve_section(curve_lay)

        basic_frame, basic_lay = self._make_section("Temel", section_key="basic", expanded=True)
        self._panel_lay.addWidget(basic_frame)
        self._build_basic_section(basic_lay)

        detail_frame, detail_lay = self._make_section("Ayrıntı", section_key="detail", expanded=False)
        self._panel_lay.addWidget(detail_frame)
        self._build_detail_section(detail_lay)

        levels_frame, levels_lay = self._make_section("Histogram", section_key="levels", expanded=False)
        self._panel_lay.addWidget(levels_frame)
        self._build_levels_section(levels_lay)

        channels_frame, channels_lay = self._make_section("Kanallar", section_key="channels", expanded=False)
        self._panel_lay.addWidget(channels_frame)
        self._build_channel_section(channels_lay)
        self._panel_lay.addStretch()

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        b_reset = QPushButton("Reset All")
        b_reset.setFixedHeight(28)
        b_reset.setStyleSheet(
            f"QPushButton{{background:{BG3};color:{MUTED};"
            f"border:1px solid {BORDER};border-radius:5px;"
            f"padding:3px 12px;font-size:10px;font-weight:600;}}"
            f"QPushButton:hover{{color:{TEXT};background:{BG4};border-color:{ACCENT};}}"
            f"QPushButton:pressed{{background:{BG};}}"
        )
        b_reset.clicked.connect(self._reset_all)
        btn_row.addWidget(b_reset)
        btn_row.addStretch()
        b_apply = QPushButton("Apply to Image")
        b_apply.setFixedHeight(30)
        b_apply.setStyleSheet(
            f"QPushButton{{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            f"stop:0 {GREEN},stop:1 #2aaa55);color:#ffffff;"
            f"border:none;border-radius:5px;padding:4px 18px;"
            f"font-size:11px;font-weight:700;letter-spacing:0.3px;}}"
            f"QPushButton:hover{{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            f"stop:0 #5ad48a,stop:1 #3dbd6e);}}"
            f"QPushButton:pressed{{background:{BG4};}}"
        )
        b_apply.clicked.connect(lambda: self._apply(emit=True))
        btn_row.addWidget(b_apply)
        root.addLayout(btn_row)

    def _make_section(self, title, section_key=None, expanded=True):
        frame = QFrame()
        if section_key:
            frame.setObjectName(f"hist_section_{section_key}")
        frame.setStyleSheet(
            f"QFrame{{background:{BG2};border:1px solid {BORDER};border-radius:4px;}}"
        )
        outer = QVBoxLayout(frame)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        button = QPushButton()
        button.setCheckable(True)
        button.setChecked(expanded)
        button.setStyleSheet(
            f"QPushButton{{background:{BG3};color:{HEAD};border:none;text-align:left;"
            f"padding:7px 10px;font-size:10px;font-weight:700;}}"
            f"QPushButton:hover{{background:{BG4};}}"
        )
        outer.addWidget(button)

        body = QWidget()
        if section_key:
            body.setObjectName(f"hist_section_body_{section_key}")
        body.setStyleSheet(f"background:{BG2};")
        body_lay = QVBoxLayout(body)
        body_lay.setContentsMargins(8, 8, 8, 8)
        body_lay.setSpacing(4)
        outer.addWidget(body)

        def _sync(checked):
            button.setText(f"{'v' if checked else '>'} {title}")
            body.setVisible(checked)

        button.toggled.connect(_sync)
        _sync(expanded)
        return frame, body_lay

    def _styled_combo(self):
        combo = QComboBox()
        combo.setStyleSheet(
            f"QComboBox{{background:{BG3};color:{TEXT};border:1px solid {BORDER};"
            f"border-radius:4px;padding:3px 8px;font-size:10px;min-height:24px;}}"
            f"QComboBox::drop-down{{border:none;width:18px;}}"
            f"QComboBox QAbstractItemView{{background:{BG3};color:{TEXT};selection-background-color:{ACCENT};}}"
        )
        return combo

    def _build_basic_section(self, lay):
        wb_row = QHBoxLayout()
        wb_row.setSpacing(6)
        wb_lbl = QLabel("Beyaz Dengesi")
        wb_lbl.setStyleSheet(f"color:{MUTED};font-size:10px;")
        self._wb_preset = self._styled_combo()
        self._wb_preset.addItems(["Çekildiği Gibi", "Gün Işığı", "Bulutlu", "Gölge", "Tungsten"])
        self._wb_preset.currentTextChanged.connect(self._on_wb_preset_changed)
        self._wb_sample_btn = QPushButton("N")
        self._wb_sample_btn.setFixedSize(24, 24)
        self._wb_sample_btn.setStyleSheet(BTN_CSS)
        self._wb_sample_btn.setToolTip("Otomatik beyaz dengesi")
        self._wb_sample_btn.clicked.connect(self._auto_white_balance)
        wb_row.addWidget(wb_lbl)
        wb_row.addWidget(self._wb_preset, 1)
        wb_row.addWidget(self._wb_sample_btn)
        lay.addLayout(wb_row)

        def _row(label, lo, hi, default, dec=2, step=None):
            sr = SliderRow(label, lo, hi, default, dec, step)
            sr.value_changed.connect(self._on_adjustment)
            lay.addWidget(sr)
            return sr

        self._adj_temp = _row("Sıcaklık", -100.0, 100.0, 0.0, 0, 1)
        self._adj_tint = _row("Renk Tonu", -100.0, 100.0, 0.0, 0, 1)
        self._adj_exposure = _row("Pozlama", -5.0, 5.0, 0.0, 2, 0.05)
        self._adj_contrast = _row("Kontrast", -100.0, 100.0, 0.0, 0, 1)
        self._adj_highlights = _row("Açık Tonlar", -100.0, 100.0, 0.0, 0, 1)
        self._adj_shadows = _row("Gölgeler", -100.0, 100.0, 0.0, 0, 1)
        self._adj_whites = _row("Beyazlar", -100.0, 100.0, 0.0, 0, 1)
        self._adj_blacks = _row("Siyahlar", -100.0, 100.0, 0.0, 0, 1)
        self._adj_vibrance = _row("Titreşim", -100.0, 100.0, 0.0, 0, 1)
        self._adj_saturation = _row("Doygunluk", -100.0, 100.0, 0.0, 0, 1)

    def _build_curve_section(self, lay):
        self._curves_wgt = CurvesWidget()
        self._curves_wgt.curve_changed.connect(self._on_curve_changed)
        self._curves_wgt.setMinimumHeight(320)
        lay.addWidget(self._curves_wgt)

        hint = QLabel("Left: add point  |  Right: remove  |  Drag: adjust")
        hint.setStyleSheet(f"color:{SUBTEXT};font-size:8px;letter-spacing:0.3px;")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(hint)

        b_reset_curve = QPushButton("Reset Curve")
        b_reset_curve.setFixedHeight(24)
        b_reset_curve.setStyleSheet(BTN_CSS)
        b_reset_curve.clicked.connect(lambda: self._curves_wgt.reset_channel(self._ch))
        lay.addWidget(b_reset_curve)

    def _build_detail_section(self, lay):
        def _row(label, lo, hi, default, dec=2, step=None):
            sr = SliderRow(label, lo, hi, default, dec, step)
            sr.value_changed.connect(self._on_adjustment)
            lay.addWidget(sr)
            return sr

        self._adj_texture = _row("Doku", -100.0, 100.0, 0.0, 0, 1)
        self._adj_sharpen = _row("Netlik", -100.0, 100.0, 0.0, 0, 1)
        self._adj_clarity = _row("Mikro Kontrast", -100.0, 100.0, 0.0, 0, 1)
        self._adj_dehaze = _row("Sis Kaldır", -100.0, 100.0, 0.0, 0, 1)

    def _build_levels_section(self, lay):
        inp_row = QHBoxLayout()
        inp_row.setSpacing(4)
        inp_row.addWidget(QLabel("In:"))
        self._sp_black = self._spinbox(0, 1, 0.0, 3)
        self._sp_mid = self._spinbox(0, 1, 0.5, 3)
        self._sp_white = self._spinbox(0, 1, 1.0, 3)
        for sp, tip in [
            (self._sp_black, "Shadows (Black point)"),
            (self._sp_mid, "Midtones (Gamma)"),
            (self._sp_white, "Highlights (White point)"),
        ]:
            sp.setToolTip(tip)
            sp.valueChanged.connect(self._on_spin_levels)
            inp_row.addWidget(sp)
        b_auto = QPushButton("Auto")
        b_auto.setStyleSheet(BTN_CSS)
        b_auto.setFixedHeight(22)
        b_auto.setFixedWidth(44)
        b_auto.clicked.connect(self._auto_levels)
        inp_row.addWidget(b_auto)
        lay.addLayout(inp_row)

        out_row = QHBoxLayout()
        out_row.setSpacing(4)
        out_row.addWidget(QLabel("Out:"))
        self._sp_out_lo = self._spinbox(0, 1, 0.0, 3)
        self._sp_out_hi = self._spinbox(0, 1, 1.0, 3)
        self._sp_out_lo.valueChanged.connect(self._on_spin_levels)
        self._sp_out_hi.valueChanged.connect(self._on_spin_levels)
        out_row.addWidget(self._sp_out_lo)
        out_row.addWidget(self._sp_out_hi)
        lay.addLayout(out_row)

    def _build_channel_section(self, lay):
        ch_row = QHBoxLayout()
        ch_row.setSpacing(4)
        self._ch_btns = {}
        for ch, col in {"L": ACCENT2, "R": RED, "G": GREEN, "B": "#6699ff"}.items():
            b = QPushButton(ch)
            b.setFixedSize(30, 24)
            b.setCheckable(True)
            b.setStyleSheet(
                f"QPushButton{{background:{BG3};color:{SUBTEXT};border:1px solid {BORDER};"
                f"border-radius:4px;font-size:11px;font-weight:800;}}"
                f"QPushButton:checked{{background:{col}40;color:{col};"
                f"border:1px solid {col};border-bottom:2px solid {col};}}"
                f"QPushButton:hover{{color:{col};background:{col}18;border-color:{col}66;}}"
            )
            b.clicked.connect(lambda _, c=ch: self._set_channel(c))
            ch_row.addWidget(b)
            self._ch_btns[ch] = b
        self._ch_btns["L"].setChecked(True)
        ch_row.addStretch()
        lay.addLayout(ch_row)

        opts = QHBoxLayout()
        opts.setSpacing(10)
        self._chk_link = QCheckBox("RGB Bağla")
        self._chk_link.setChecked(False)
        self._chk_link.setStyleSheet(CHECK_CSS)
        self._chk_live = QCheckBox("Canlı")
        self._chk_live.setChecked(True)
        self._chk_live.setStyleSheet(CHECK_CSS)
        opts.addWidget(self._chk_link)
        opts.addWidget(self._chk_live)
        opts.addStretch()
        lay.addLayout(opts)

    def _levels_tab(self):
        w = QWidget()
        w.setStyleSheet(f"background:{BG2};")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(6, 6, 6, 4)
        lay.setSpacing(6)

        self._hist_wgt = HistogramWidget()
        self._hist_wgt.levels_changed.connect(self._on_levels_changed)
        lay.addWidget(self._hist_wgt, 1)

        inp_row = QHBoxLayout()
        inp_row.setSpacing(4)
        inp_row.addWidget(QLabel("In:"))
        self._sp_black = self._spinbox(0, 1, 0.0, 3)
        self._sp_mid = self._spinbox(0, 1, 0.5, 3)
        self._sp_white = self._spinbox(0, 1, 1.0, 3)
        for sp, tip in [
            (self._sp_black, "Shadows (Black point)"),
            (self._sp_mid, "Midtones (Gamma)"),
            (self._sp_white, "Highlights (White point)"),
        ]:
            sp.setToolTip(tip)
            sp.valueChanged.connect(self._on_spin_levels)
            inp_row.addWidget(sp)
        inp_row.addStretch()
        b_auto = QPushButton("Auto")
        b_auto.setStyleSheet(BTN_CSS)
        b_auto.setFixedHeight(22)
        b_auto.setFixedWidth(44)
        b_auto.setToolTip("Auto Levels: stretch to 0.1%-99.9%")
        b_auto.clicked.connect(self._auto_levels)
        inp_row.addWidget(b_auto)
        lay.addLayout(inp_row)

        out_row = QHBoxLayout()
        out_row.setSpacing(4)
        out_row.addWidget(QLabel("Out:"))
        self._sp_out_lo = self._spinbox(0, 1, 0.0, 3)
        self._sp_out_hi = self._spinbox(0, 1, 1.0, 3)
        self._sp_out_lo.setToolTip("Output black (lift shadows)")
        self._sp_out_hi.setToolTip("Output white (compress highlights)")
        self._sp_out_lo.valueChanged.connect(self._on_spin_levels)
        self._sp_out_hi.valueChanged.connect(self._on_spin_levels)
        out_row.addWidget(self._sp_out_lo)
        out_row.addWidget(self._sp_out_hi)
        out_row.addStretch()
        self._tabs.addTab(w, "Levels")
        lay.addLayout(out_row)

    def _curves_tab(self):
        w = QWidget()
        w.setStyleSheet(f"background:{BG2};")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(3)

        self._curves_wgt = CurvesWidget()
        self._curves_wgt.curve_changed.connect(self._on_curve_changed)
        lay.addWidget(self._curves_wgt, 1)

        hint = QLabel("Left: add point  |  Right: remove  |  Drag: adjust")
        hint.setStyleSheet(f"color:{SUBTEXT};font-size:8px;letter-spacing:0.3px;")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(hint)

        b_reset_curve = QPushButton("Reset Curve")
        b_reset_curve.setFixedHeight(24)
        b_reset_curve.setStyleSheet(
            f"QPushButton{{background:{BG3};color:{MUTED};"
            f"border:1px solid {BORDER};border-radius:4px;font-size:10px;font-weight:600;}}"
            f"QPushButton:hover{{color:{ACCENT2};border-color:{ACCENT};background:{BG4};}}"
            f"QPushButton:pressed{{background:{BG};}}"
        )
        b_reset_curve.clicked.connect(lambda: self._curves_wgt.reset_channel(self._ch))
        lay.addWidget(b_reset_curve)
        self._tabs.addTab(w, "Curves")

    def _adjustments_tab(self):
        w = QWidget()
        w.setStyleSheet(f"background:{BG2};")
        scroll = QScrollArea()
        scroll.setWidget(w)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"background:{BG2};border:none;")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(2)

        top = QHBoxLayout()
        top.setSpacing(6)
        title = QLabel("Düzenle")
        title.setStyleSheet(f"color:{HEAD};font-size:11px;font-weight:800;")
        top.addWidget(title)
        top.addStretch()

        b_auto = QPushButton("Otomatik")
        b_auto.setStyleSheet(BTN_CSS)
        b_auto.setFixedHeight(24)
        b_auto.clicked.connect(self._auto_raw_adjustments)
        top.addWidget(b_auto)

        self._raw_bw_btn = QPushButton("Siyah Beyaz")
        self._raw_bw_btn.setCheckable(True)
        self._raw_bw_btn.setFixedHeight(24)
        self._raw_bw_btn.setStyleSheet(BTN_CSS)
        self._raw_bw_btn.toggled.connect(self._toggle_bw_profile)
        top.addWidget(self._raw_bw_btn)
        lay.addLayout(top)

        profile_row = QHBoxLayout()
        profile_row.setSpacing(6)
        profile_lbl = QLabel("Profil")
        profile_lbl.setStyleSheet(f"color:{MUTED};font-size:10px;")
        self._raw_profile = QComboBox()
        self._raw_profile.addItems(["Renk", "Siyah Beyaz"])
        self._raw_profile.setStyleSheet(
            f"QComboBox{{background:{BG3};color:{TEXT};border:1px solid {BORDER};"
            f"border-radius:4px;padding:3px 8px;font-size:10px;min-height:22px;}}"
            f"QComboBox::drop-down{{border:none;width:18px;}}"
            f"QComboBox QAbstractItemView{{background:{BG3};color:{TEXT};selection-background-color:{ACCENT};}}"
        )
        self._raw_profile.currentTextChanged.connect(self._on_profile_changed)
        profile_row.addWidget(profile_lbl)
        profile_row.addWidget(self._raw_profile, 1)
        lay.addLayout(profile_row)

        def _sep(title_text):
            lbl = QLabel(title_text)
            lbl.setStyleSheet(
                f"color:{HEAD};font-size:10px;font-weight:700;"
                f"border-bottom:1px solid {BORDER};padding-bottom:4px;margin-top:8px;"
            )
            lay.addWidget(lbl)

        def _row(label, lo, hi, default, dec=2, step=None):
            sr = SliderRow(label, lo, hi, default, dec, step)
            sr.value_changed.connect(self._on_adjustment)
            lay.addWidget(sr)
            return sr

        _sep("Temel")
        self._adj_temp = _row("Sıcaklık", -100.0, 100.0, 0.0, 0, 1)
        self._adj_tint = _row("Renk Tonu", -100.0, 100.0, 0.0, 0, 1)
        self._adj_exposure = _row("Pozlama", -5.0, 5.0, 0.0, 2, 0.05)
        self._adj_contrast = _row("Kontrast", -100.0, 100.0, 0.0, 0, 1)
        self._adj_highlights = _row("Açık Tonlar", -100.0, 100.0, 0.0, 0, 1)
        self._adj_shadows = _row("Gölgeler", -100.0, 100.0, 0.0, 0, 1)
        self._adj_whites = _row("Beyazlar", -100.0, 100.0, 0.0, 0, 1)
        self._adj_blacks = _row("Siyahlar", -100.0, 100.0, 0.0, 0, 1)

        _sep("Ayrıntı")
        self._adj_texture = _row("Doku", -100.0, 100.0, 0.0, 0, 1)
        self._adj_sharpen = _row("Netlik", -100.0, 100.0, 0.0, 0, 1)
        self._adj_clarity = _row("Mikro Kontrast", -100.0, 100.0, 0.0, 0, 1)
        self._adj_dehaze = _row("Sis Kaldır", -100.0, 100.0, 0.0, 0, 1)

        _sep("Renk")
        self._adj_vibrance = _row("Titreşim", -100.0, 100.0, 0.0, 0, 1)
        self._adj_saturation = _row("Doygunluk", -100.0, 100.0, 0.0, 0, 1)

        lay.addStretch()

        b_reset_adj = QPushButton("Reset RAW")
        b_reset_adj.setStyleSheet(BTN_CSS)
        b_reset_adj.setFixedHeight(22)
        b_reset_adj.clicked.connect(self._reset_adjustments)
        lay.addWidget(b_reset_adj)

        self._tabs.addTab(scroll, "RAW")

    def _spinbox(self, lo, hi, val, dec):
        sp = QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setValue(val)
        sp.setDecimals(dec)
        sp.setSingleStep(10 ** -dec)
        sp.setFixedWidth(66)
        sp.setStyleSheet(SPIN_CSS)
        return sp

    def set_image(self, img: np.ndarray, reset: bool = False):
        self._img = np.clip(img, 0, 1).astype(np.float32) if img is not None else None
        self._orig_img = self._img.copy() if self._img is not None else None
        self._preview_img, self._preview_scale = build_preview_proxy(self._img)
        self._preview_orig = self._preview_img.copy() if self._preview_img is not None else None
        hist_src = self._preview_img if self._preview_img is not None else self._img
        self._hist_wgt.set_image(hist_src)
        self._curves_wgt.set_image(hist_src)
        if reset:
            self._reset_all_silent()

    def _set_channel(self, ch):
        self._ch = ch
        for c, b in self._ch_btns.items():
            b.setChecked(c == ch)
        self._hist_wgt.set_channel(ch)
        self._curves_wgt.set_channel(ch)
        self._sync_spins_from_state(self._hist_wgt.get_state(ch))

    def _sync_spins_from_state(self, st):
        for sp, v in [
            (self._sp_black, st[0]),
            (self._sp_mid, st[1]),
            (self._sp_white, st[2]),
            (self._sp_out_lo, st[3]),
            (self._sp_out_hi, st[4]),
        ]:
            sp.blockSignals(True)
            sp.setValue(v)
            sp.blockSignals(False)

    def _on_levels_changed(self, ch, b, m, w, ol, oh):
        channels = self._linked_channels(ch)
        for c in channels:
            self._hist_wgt.set_state(c, b, m, w, ol, oh)
        self._sync_spins_from_state([b, m, w, ol, oh])
        self._schedule_preview()

    def _on_spin_levels(self):
        b = float(self._sp_black.value())
        m = float(self._sp_mid.value())
        w = float(self._sp_white.value())
        ol = float(self._sp_out_lo.value())
        oh = float(self._sp_out_hi.value())
        b = min(b, w - 0.01)
        w = max(w, b + 0.01)
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

    def _on_profile_changed(self, text):
        self._raw_bw_btn.blockSignals(True)
        self._raw_bw_btn.setChecked(text == "Siyah Beyaz")
        self._raw_bw_btn.blockSignals(False)
        self._schedule_preview()

    def _toggle_bw_profile(self, checked):
        self._raw_profile.blockSignals(True)
        self._raw_profile.setCurrentText("Siyah Beyaz" if checked else "Renk")
        self._raw_profile.blockSignals(False)
        self._schedule_preview()

    def _on_wb_preset_changed(self, text):
        presets = {
            "Çekildiği Gibi": (0.0, 0.0),
            "Gün Işığı": (6.0, 0.0),
            "Bulutlu": (16.0, 2.0),
            "Gölge": (28.0, 4.0),
            "Tungsten": (-32.0, -6.0),
        }
        if text not in presets:
            return
        temp, tint = presets[text]
        self._set_adjustment_values(
            {
                "_adj_temp": temp,
                "_adj_tint": tint,
            },
            schedule=True,
        )

    def _auto_white_balance(self):
        base = self._preview_orig if self._preview_orig is not None else self._orig_img
        if base is None or base.ndim != 3:
            return
        med = np.median(base.reshape(-1, 3), axis=0).astype(np.float32)
        mean_med = float(np.mean(med))
        if mean_med <= 1e-6:
            return
        temp = float(np.clip((med[2] - med[0]) / mean_med * 60.0, -100.0, 100.0))
        tint = float(np.clip((med[1] - 0.5 * (med[0] + med[2])) / mean_med * 90.0, -100.0, 100.0))
        self._set_adjustment_values(
            {
                "_adj_temp": temp,
                "_adj_tint": tint,
                "wb_preset": "Çekildiği Gibi",
            },
            schedule=True,
        )

    def _linked_channels(self, ch):
        if not self._chk_link.isChecked():
            return [ch]
        if ch in ("R", "G", "B"):
            return ["R", "G", "B"]
        return ["L"]

    def _schedule_preview(self):
        if self._chk_live.isChecked():
            self._debounce.start(24)

    def _emit_preview(self):
        if self._img is None:
            return
        result = self._apply(emit=False, preview=True)
        if result is not None:
            if self._orig_img is not None and result.shape != self._orig_img.shape:
                result = resize_to_shape(result, self._orig_img.shape)
            self.preview_changed.emit(result)

    def _auto_levels(self):
        base = self._preview_orig if self._preview_orig is not None else self._img
        if base is None:
            return
        ch = self._ch
        if base.ndim == 2:
            data = base.ravel()
        elif ch == "L":
            data = base.mean(axis=2).ravel()
        else:
            ci = {"R": 0, "G": 1, "B": 2}[ch]
            data = base[:, :, ci].ravel()
        lo = float(np.percentile(data, 0.1))
        hi = float(np.percentile(data, 99.9))
        mid = 0.5
        for c in self._linked_channels(ch):
            self._hist_wgt.set_state(c, lo, mid, hi, 0.0, 1.0)
        self._sync_spins_from_state([lo, mid, hi, 0.0, 1.0])
        self._schedule_preview()

    def _set_adjustment_values(self, values, schedule=False):
        payload = dict(values)
        profile_value = payload.pop("profile", None)
        wb_value = payload.pop("wb_preset", None)
        for attr, value in payload.items():
            widget = getattr(self, attr, None)
            if widget is None:
                continue
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)
        if profile_value is not None:
            self._raw_profile.blockSignals(True)
            self._raw_profile.setCurrentText(profile_value)
            self._raw_profile.blockSignals(False)
            self._raw_bw_btn.blockSignals(True)
            self._raw_bw_btn.setChecked(profile_value == "Siyah Beyaz")
            self._raw_bw_btn.blockSignals(False)
        if wb_value is not None and hasattr(self, "_wb_preset"):
            self._wb_preset.blockSignals(True)
            self._wb_preset.setCurrentText(wb_value)
            self._wb_preset.blockSignals(False)
        if schedule:
            self._schedule_preview()

    def _auto_raw_adjustments(self):
        base = self._preview_orig if self._preview_orig is not None else self._orig_img
        if base is None:
            return
        luma = image_luma(base)
        p1, p10, p50, p90, p99 = [float(v) for v in np.percentile(luma, [1, 10, 50, 90, 99])]
        dynamic = max(p99 - p1, 1e-3)
        exposure = float(np.clip(np.log2(0.30 / max(p50, 1e-3)), -1.75, 1.75))
        contrast = float(np.clip((0.56 - dynamic) * 140.0, -20.0, 42.0))
        shadows = float(np.clip((0.18 - p10) * 240.0, -20.0, 48.0))
        highlights = float(np.clip((0.82 - p90) * 220.0, -45.0, 28.0))
        whites = float(np.clip((0.94 - p99) * 180.0, -25.0, 30.0))
        blacks = float(np.clip((0.03 - p1) * 320.0, -38.0, 24.0))
        self._set_adjustment_values(
            {
                "_adj_exposure": exposure,
                "_adj_contrast": contrast,
                "_adj_highlights": highlights,
                "_adj_shadows": shadows,
                "_adj_whites": whites,
                "_adj_blacks": blacks,
                "_adj_texture": 12.0,
                "_adj_sharpen": 10.0,
                "_adj_clarity": 10.0,
                "_adj_dehaze": 8.0,
                "_adj_vibrance": 14.0,
                "_adj_saturation": 0.0,
                "_adj_temp": 0.0,
                "_adj_tint": 0.0,
                "profile": "Renk",
                "wb_preset": "Çekildiği Gibi",
            },
            schedule=True,
        )

    def _reset_all(self):
        self._hist_wgt.reset_channel()
        self._curves_wgt.reset_channel()
        self._reset_adjustments()
        self._sync_spins_from_state([0.0, 0.5, 1.0, 0.0, 1.0])
        self._schedule_preview()

    def _reset_all_silent(self):
        self._debounce.stop()
        self._hist_wgt.blockSignals(True)
        self._curves_wgt.blockSignals(True)
        for c in list(self._hist_wgt._state.keys()):
            self._hist_wgt._state[c] = [0.0, 0.5, 1.0, 0.0, 1.0]
        for c in list(self._curves_wgt._pts.keys()):
            self._curves_wgt._pts[c] = [(0.0, 0.0), (1.0, 1.0)]
        self._hist_wgt.blockSignals(False)
        self._curves_wgt.blockSignals(False)
        self._reset_adjustments()
        self._sync_spins_from_state([0.0, 0.5, 1.0, 0.0, 1.0])
        self._hist_wgt.repaint()
        self._curves_wgt.repaint()

    def _reset_adjustments(self):
        self._set_adjustment_values(
            {
                "_adj_exposure": 0.0,
                "_adj_contrast": 0.0,
                "_adj_highlights": 0.0,
                "_adj_shadows": 0.0,
                "_adj_whites": 0.0,
                "_adj_blacks": 0.0,
                "_adj_temp": 0.0,
                "_adj_tint": 0.0,
                "_adj_texture": 0.0,
                "_adj_sharpen": 0.0,
                "_adj_clarity": 0.0,
                "_adj_dehaze": 0.0,
                "_adj_vibrance": 0.0,
                "_adj_saturation": 0.0,
                "profile": "Renk",
                "wb_preset": "Çekildiği Gibi",
            },
            schedule=False,
        )

    def _apply(self, emit=True, preview=False):
        if self._img is None:
            return None
        base = (
            self._preview_orig if preview and self._preview_orig is not None
            else self._orig_img if self._orig_img is not None
            else self._img
        )
        img = base.astype(np.float32, copy=True)

        def _apply_levels_1ch(channel, st):
            b, m, w, ol, oh = st
            rng = max(w - b, 1e-9)
            channel = np.clip((channel - b) / rng, 0, 1)
            if abs(m - 0.5) > 0.005:
                eps = 1e-9
                channel = np.where(
                    channel <= 0,
                    0,
                    np.where(
                        channel >= 1,
                        1,
                        (m - 1) * channel / ((2 * m - 1) * channel - m + eps),
                    ),
                )
                channel = np.clip(channel, 0, 1)
            channel = ol + channel * (oh - ol)
            return channel

        if img.ndim == 2:
            img = _apply_levels_1ch(img, self._hist_wgt.get_state("L"))
        else:
            st_l = self._hist_wgt.get_state("L")
            if st_l != [0.0, 0.5, 1.0, 0.0, 1.0]:
                for i in range(3):
                    img[:, :, i] = _apply_levels_1ch(img[:, :, i], st_l)
            for i, ch in enumerate(("R", "G", "B")):
                st = self._hist_wgt.get_state(ch)
                if st != [0.0, 0.5, 1.0, 0.0, 1.0]:
                    img[:, :, i] = _apply_levels_1ch(img[:, :, i], st)

        pts_l = self._curves_wgt._pts.get("L", [(0, 0), (1, 1)])
        is_flat_l = len(pts_l) == 2 and pts_l[0] == (0, 0) and pts_l[1] == (1, 1)
        xs = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        if img.ndim == 2:
            if not is_flat_l:
                lut = self._curves_wgt.get_lut("L")
                img = np.clip(np.interp(img, xs, lut), 0, 1)
        else:
            if not is_flat_l:
                lut_l = self._curves_wgt.get_lut("L")
                for i in range(3):
                    img[:, :, i] = np.interp(img[:, :, i], xs, lut_l)
            for i, ch in enumerate(("R", "G", "B")):
                pts = self._curves_wgt._pts.get(ch, [(0, 0), (1, 1)])
                is_flat = len(pts) == 2 and pts[0] == (0, 0) and pts[1] == (1, 1)
                if not is_flat:
                    lut = self._curves_wgt.get_lut(ch)
                    img[:, :, i] = np.interp(img[:, :, i], xs, lut)

        img = np.clip(img, 0, 1).astype(np.float32)
        img = self._apply_adjustments(img)
        img = np.clip(img, 0, 1).astype(np.float32)

        if emit:
            self.apply_requested.emit(img)
            self._img = img.copy()
            self._orig_img = img.copy()
            self._preview_img, self._preview_scale = build_preview_proxy(self._img)
            self._preview_orig = self._preview_img.copy() if self._preview_img is not None else None
            hist_src = self._preview_img if self._preview_img is not None else self._img
            self._hist_wgt.set_image(hist_src)
            self._curves_wgt.set_image(hist_src)
            self._reset_all_silent()
        return img

    def _apply_adjustments(self, img: np.ndarray) -> np.ndarray:
        params = {
            "profile": self._raw_profile.currentText() if hasattr(self, "_raw_profile") else "Renk",
            "exposure": float(self._adj_exposure.value()),
            "contrast": float(self._adj_contrast.value()),
            "highlights": float(self._adj_highlights.value()),
            "shadows": float(self._adj_shadows.value()),
            "whites": float(self._adj_whites.value()),
            "blacks": float(self._adj_blacks.value()),
            "temp": float(self._adj_temp.value()),
            "tint": float(self._adj_tint.value()),
            "texture": float(self._adj_texture.value()),
            "sharpen": float(self._adj_sharpen.value()),
            "clarity": float(self._adj_clarity.value()),
            "dehaze": float(self._adj_dehaze.value()),
            "vibrance": float(self._adj_vibrance.value()),
            "saturation": float(self._adj_saturation.value()),
        }
        return apply_camera_raw_adjustments(img, params)


def image_luma(img: np.ndarray) -> np.ndarray:
    if img is None:
        return None
    if img.ndim == 2:
        return img.astype(np.float32, copy=False)
    return (
        0.2126 * img[:, :, 0]
        + 0.7152 * img[:, :, 1]
        + 0.0722 * img[:, :, 2]
    ).astype(np.float32, copy=False)


def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    width = max(edge1 - edge0, 1e-6)
    t = np.clip((x - edge0) / width, 0.0, 1.0).astype(np.float32)
    return t * t * (3.0 - 2.0 * t)


def build_preview_proxy(img: np.ndarray, max_side: int = 1440):
    if img is None:
        return None, 1.0
    h, w = img.shape[:2]
    scale = min(1.0, float(max_side) / float(max(h, w, 1)))
    if scale >= 0.999:
        return img.copy(), 1.0
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(
        img.astype(np.float32, copy=False),
        (new_w, new_h),
        interpolation=cv2.INTER_AREA,
    )
    return np.clip(resized, 0.0, 1.0).astype(np.float32), scale


def resize_to_shape(img: np.ndarray, shape) -> np.ndarray:
    if img is None:
        return None
    target_h, target_w = shape[:2]
    if img.shape[:2] == (target_h, target_w):
        return np.clip(img, 0.0, 1.0).astype(np.float32, copy=False)
    resized = cv2.resize(
        img.astype(np.float32, copy=False),
        (int(target_w), int(target_h)),
        interpolation=cv2.INTER_LINEAR,
    )
    return np.clip(resized, 0.0, 1.0).astype(np.float32)


def apply_camera_raw_adjustments(img: np.ndarray, params: dict) -> np.ndarray:
    img = np.clip(img.astype(np.float32, copy=True), 0.0, 1.0)

    exposure = float(params.get("exposure", 0.0))
    if abs(exposure) > 1e-4:
        img = np.clip(img * (2.0 ** exposure), 0.0, 1.0)

    contrast = float(params.get("contrast", 0.0)) / 100.0
    if abs(contrast) > 1e-4:
        img = np.clip(0.5 + (img - 0.5) * (1.0 + contrast * 1.65), 0.0, 1.0)

    luma = image_luma(img)
    shadow_mask = 1.0 - smoothstep(0.16, 0.58, luma)
    highlight_mask = smoothstep(0.42, 0.90, luma)
    white_mask = smoothstep(0.72, 0.98, luma)
    black_mask = 1.0 - smoothstep(0.02, 0.28, luma)
    if img.ndim == 3:
        shadow_mask = shadow_mask[:, :, None]
        highlight_mask = highlight_mask[:, :, None]
        white_mask = white_mask[:, :, None]
        black_mask = black_mask[:, :, None]

    shadows = float(params.get("shadows", 0.0)) / 100.0
    if abs(shadows) > 1e-4:
        if shadows >= 0.0:
            img = img + shadows * shadow_mask * (1.0 - img) * 0.78
        else:
            img = img + shadows * shadow_mask * img * 0.72
        img = np.clip(img, 0.0, 1.0)

    highlights = float(params.get("highlights", 0.0)) / 100.0
    if abs(highlights) > 1e-4:
        if highlights >= 0.0:
            img = img + highlights * highlight_mask * (1.0 - img) * 0.52
        else:
            img = img + highlights * highlight_mask * img * 0.82
        img = np.clip(img, 0.0, 1.0)

    whites = float(params.get("whites", 0.0)) / 100.0
    if abs(whites) > 1e-4:
        if whites >= 0.0:
            img = img + whites * white_mask * (1.0 - img) * 0.95
        else:
            img = img + whites * white_mask * img * 0.96
        img = np.clip(img, 0.0, 1.0)

    blacks = float(params.get("blacks", 0.0)) / 100.0
    if abs(blacks) > 1e-4:
        if blacks >= 0.0:
            img = img + blacks * black_mask * np.maximum(0.30 - img, 0.0) * 1.15
        else:
            img = img + blacks * black_mask * img * 1.08
        img = np.clip(img, 0.0, 1.0)

    if img.ndim == 3:
        profile = str(params.get("profile", "Renk"))
        if profile != "Siyah Beyaz":
            temp = float(params.get("temp", 0.0)) / 100.0
            tint = float(params.get("tint", 0.0)) / 100.0
            if abs(temp) > 1e-4 or abs(tint) > 1e-4:
                gains = np.array(
                    [
                        1.0 + temp * 0.20 + tint * 0.07,
                        1.0 + temp * 0.05 - tint * 0.12,
                        1.0 - temp * 0.24 + tint * 0.05,
                    ],
                    dtype=np.float32,
                )
                img = np.clip(img * gains.reshape(1, 1, 3), 0.0, 1.0)

            gray = image_luma(img)[:, :, None]
            chroma = img - gray
            saturation = float(params.get("saturation", 0.0)) / 100.0
            vibrance = float(params.get("vibrance", 0.0)) / 100.0
            if abs(saturation) > 1e-4:
                img = np.clip(gray + chroma * (1.0 + saturation * 1.35), 0.0, 1.0)
                chroma = img - image_luma(img)[:, :, None]
            if abs(vibrance) > 1e-4:
                sat_map = np.clip(
                    np.max(np.abs(chroma), axis=2, keepdims=True)
                    / np.maximum(gray + 1e-3, 1e-3),
                    0.0,
                    1.0,
                )
                protect = 1.0 - sat_map * 0.8
                img = np.clip(img + vibrance * chroma * protect * 1.25, 0.0, 1.0)
        else:
            gray = image_luma(img)
            img = np.repeat(gray[:, :, None], 3, axis=2)

    texture = float(params.get("texture", 0.0)) / 100.0
    if abs(texture) > 1e-4:
        fine = img - cv2.GaussianBlur(img, (0, 0), 1.1)
        img = np.clip(img + texture * fine * 0.90, 0.0, 1.0)

    sharpen = float(params.get("sharpen", 0.0)) / 100.0
    if abs(sharpen) > 1e-4:
        sharp_base = cv2.GaussianBlur(img, (0, 0), 0.75)
        img = np.clip(img + sharpen * (img - sharp_base) * 1.18, 0.0, 1.0)

    clarity = float(params.get("clarity", 0.0)) / 100.0
    if abs(clarity) > 1e-4:
        local = cv2.GaussianBlur(img, (0, 0), 7.5)
        luma = image_luma(img)
        mid_mask = 1.0 - np.clip(np.abs(luma - 0.5) / 0.5, 0.0, 1.0)
        if img.ndim == 3:
            mid_mask = mid_mask[:, :, None]
        img = np.clip(img + clarity * (img - local) * (0.6 + mid_mask * 0.9), 0.0, 1.0)

    dehaze = float(params.get("dehaze", 0.0)) / 100.0
    if abs(dehaze) > 1e-4:
        haze = cv2.GaussianBlur(img, (0, 0), 28.0)
        img = np.clip(img + dehaze * (img - haze) * 1.35, 0.0, 1.0)
        if dehaze > 0.0:
            img = np.clip(0.5 + (img - 0.5) * (1.0 + dehaze * 0.18), 0.0, 1.0)

    return np.clip(img, 0.0, 1.0).astype(np.float32)
