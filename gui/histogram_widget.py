"""
Astro Mastro Pro — Histogram Widget
Photoshop tarzı tam özellikli histogram paneli:
  • R / G / B / Luminance kanal seçimi
  • Log / Linear ölçek
  • Giriş/çıkış seviyeleri (sürüklenebilir üçgenler)
  • Kırpma uyarısı (siyah/beyaz yüzde)
  • İstatistikler: Ort, Medyan, Std, Min, Max, Piksel sayısı
  • Gamma orta nokta kaydırıcısı
  • Uygula sinyali → ana pencere image günceller
"""

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QCheckBox, QPushButton, QSizePolicy, QGroupBox, QDoubleSpinBox,
    QFrame
)
from PyQt6.QtCore import Qt, QRect, QPointF, pyqtSignal, QTimer
from PyQt6.QtGui import (
    QPainter, QColor, QPen, QBrush, QLinearGradient,
    QPolygonF, QPainterPath, QFont, QFontMetrics
)


# ─────────────────────────────────────────────────────────────────────────────
#  Histogram Çizim Widget'ı
# ─────────────────────────────────────────────────────────────────────────────

class HistogramView(QWidget):
    """
    Histogram çizim alanı.
    Sürüklenebilir:
      • sol alt üçgen  → black point  (0–1)
      • sağ alt üçgen  → white point  (0–1)
      • orta alt üçgen → gamma (0.1–9.9)
      • sol üst üçgen  → output black (0–1)
      • sağ üst üçgen  → output white (0–1)
    """

    # (black_in, gamma, white_in, black_out, white_out)
    levels_changed = pyqtSignal(float, float, float, float, float)

    # Renk haritası: kanal adı → (R,G,B)
    CHANNEL_COLORS = {
        "Luminance": (200, 200, 200),
        "R":         (220,  60,  60),
        "G":         ( 60, 200,  60),
        "B":         ( 60, 100, 220),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(260, 160)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(200)

        # Histogram verisi {kanal: array[256]}
        self._histograms: dict[str, np.ndarray] = {}
        self._channel = "Luminance"
        self._log_scale = False

        # Seviyeleri: giriş [0,1], gamma [0.1,9.9], çıkış [0,1]
        self._black_in  = 0.0
        self._white_in  = 1.0
        self._gamma     = 1.0
        self._black_out = 0.0
        self._white_out = 1.0

        # Sürükleme durumu
        self._drag = None   # "black_in" | "white_in" | "gamma" | "black_out" | "white_out"
        self._drag_start_x = 0

        # Kırpma istatistikleri
        self._clip_black = 0.0
        self._clip_white = 0.0

        self.setMouseTracking(True)

    # ── Veri ─────────────────────────────────────────────────────────────────

    def set_histograms(self, histograms: dict, clip_black=0.0, clip_white=0.0):
        self._histograms = histograms
        self._clip_black = clip_black
        self._clip_white = clip_white
        self.update()

    def set_channel(self, channel: str):
        self._channel = channel
        self.update()

    def set_log_scale(self, enabled: bool):
        self._log_scale = enabled
        self.update()

    def get_levels(self):
        return (self._black_in, self._gamma, self._white_in,
                self._black_out, self._white_out)

    def set_levels(self, black_in, gamma, white_in, black_out, white_out):
        self._black_in  = float(np.clip(black_in,  0.0, 0.999))
        self._gamma     = float(np.clip(gamma,     0.1, 9.9))
        self._white_in  = float(np.clip(white_in,  self._black_in + 0.001, 1.0))
        self._black_out = float(np.clip(black_out, 0.0, 0.999))
        self._white_out = float(np.clip(white_out, self._black_out + 0.001, 1.0))
        self.update()

    def reset_levels(self):
        self.set_levels(0.0, 1.0, 1.0, 0.0, 1.0)
        self.levels_changed.emit(0.0, 1.0, 1.0, 0.0, 1.0)

    # ── Çizim ────────────────────────────────────────────────────────────────

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()

        # Bölgeler
        BOTTOM_MARGIN  = 22   # giriş kaydırıcı üçgenleri
        TOP_MARGIN     = 18   # çıkış üçgenleri
        SIDE_MARGIN    = 6
        HIST_HEIGHT    = h - BOTTOM_MARGIN - TOP_MARGIN
        hist_rect = QRect(SIDE_MARGIN, TOP_MARGIN,
                          w - 2 * SIDE_MARGIN, HIST_HEIGHT)

        # ── Arka plan ──
        p.fillRect(0, 0, w, h, QColor(20, 20, 20))
        p.fillRect(hist_rect, QColor(30, 30, 30))

        # ── Kılavuz çizgileri ──
        p.setPen(QPen(QColor(50, 50, 50), 1))
        for i in range(1, 4):
            x = hist_rect.left() + i * hist_rect.width() // 4
            p.drawLine(x, hist_rect.top(), x, hist_rect.bottom())
        for i in range(1, 4):
            y = hist_rect.top() + i * hist_rect.height() // 4
            p.drawLine(hist_rect.left(), y, hist_rect.right(), y)

        # ── Histogram ──
        data = self._histograms.get(self._channel)
        if data is not None and len(data) > 0:
            self._draw_histogram(p, hist_rect, data)

        # ── Giriş tonu eğrisi önizleme (gradient band) ──
        self._draw_gradient_band(p, hist_rect)

        # ── Çerçeve ──
        p.setPen(QPen(QColor(70, 70, 70), 1))
        p.drawRect(hist_rect)

        # ── Kırpma uyarı çizgileri ──
        if self._black_in > 0:
            xb = hist_rect.left() + int(self._black_in * hist_rect.width())
            p.setPen(QPen(QColor(255, 100, 0, 160), 1, Qt.PenStyle.DashLine))
            p.drawLine(xb, hist_rect.top(), xb, hist_rect.bottom())

        if self._white_in < 1:
            xw = hist_rect.left() + int(self._white_in * hist_rect.width())
            p.setPen(QPen(QColor(255, 255, 100, 160), 1, Qt.PenStyle.DashLine))
            p.drawLine(xw, hist_rect.top(), xw, hist_rect.bottom())

        # ── Çıkış üçgenleri (üst) ──
        self._draw_triangle(p, hist_rect, self._black_out, TOP_MARGIN, "out_black", above=True)
        self._draw_triangle(p, hist_rect, self._white_out, TOP_MARGIN, "out_white", above=True)

        # ── Giriş üçgenleri (alt) ──
        self._draw_triangle(p, hist_rect, self._black_in,  BOTTOM_MARGIN, "in_black",  above=False)
        self._draw_triangle(p, hist_rect, self._white_in,  BOTTOM_MARGIN, "in_white",  above=False)
        # Gamma üçgeni — orta, beyaz
        gamma_pos = self._gamma_screen_pos()
        self._draw_gamma_triangle(p, hist_rect, gamma_pos, BOTTOM_MARGIN)

        # ── Kırpma yüzdeleri ──
        self._draw_clip_labels(p, hist_rect)

        p.end()

    def _draw_histogram(self, p: QPainter, rect: QRect, data: np.ndarray):
        bins = len(data)
        if self._log_scale:
            plot = np.log1p(data.astype(np.float64))
        else:
            plot = data.astype(np.float64)

        max_val = plot.max()
        if max_val < 1e-9:
            return

        rw = rect.width()
        rh = rect.height()
        rx = rect.left()
        ry = rect.top()

        color = self.CHANNEL_COLORS.get(self._channel, (200, 200, 200))
        bar_color   = QColor(*color, 160)
        frame_color = QColor(*color, 220)

        # Dolgulu histogram
        path = QPainterPath()
        path.moveTo(rx, ry + rh)
        for i in range(bins):
            x = rx + (i / bins) * rw
            y = ry + rh - (plot[i] / max_val) * rh
            path.lineTo(x, y)
        path.lineTo(rx + rw, ry + rh)
        path.closeSubpath()

        p.fillPath(path, QBrush(bar_color))

        # Üst kenar
        p.setPen(QPen(frame_color, 1))
        p.drawPath(path)

    def _draw_gradient_band(self, p: QPainter, rect: QRect):
        """Giriş seviyeleri tonu önizleme bandı."""
        rx = rect.left()
        ry = rect.top()
        rw = rect.width()
        rh = rect.height()

        grad = QLinearGradient(rx, 0, rx + rw, 0)
        grad.setColorAt(0.0, QColor(0, 0, 0, 0))
        grad.setColorAt(float(self._black_in), QColor(0, 0, 0, 30))
        mid = (self._black_in + self._white_in) / 2
        grad.setColorAt(float(mid), QColor(128, 128, 128, 20))
        grad.setColorAt(float(self._white_in), QColor(255, 255, 255, 30))
        grad.setColorAt(1.0, QColor(255, 255, 255, 0))

        p.fillRect(rx, ry, rw, rh, QBrush(grad))

    def _draw_triangle(self, p: QPainter, rect: QRect, pos: float,
                        zone_h: int, handle_type: str, above: bool):
        SIZE = 8
        x = rect.left() + pos * rect.width()

        if above:
            tip_y = rect.top()
            pts = [QPointF(x, tip_y),
                   QPointF(x - SIZE // 2, tip_y - SIZE),
                   QPointF(x + SIZE // 2, tip_y - SIZE)]
        else:
            tip_y = rect.bottom()
            pts   = [QPointF(x, tip_y),
                     QPointF(x - SIZE // 2, tip_y + SIZE),
                     QPointF(x + SIZE // 2, tip_y + SIZE)]

        poly = QPolygonF(pts)
        if "black" in handle_type:
            fill = QColor(30, 30, 30)
            border = QColor(160, 160, 160)
        else:
            fill = QColor(240, 240, 240)
            border = QColor(80, 80, 80)

        p.setBrush(QBrush(fill))
        p.setPen(QPen(border, 1))
        p.drawPolygon(poly)

    def _draw_gamma_triangle(self, p: QPainter, rect: QRect,
                              gamma_pos: float, zone_h: int):
        SIZE = 8
        x = rect.left() + gamma_pos * rect.width()
        tip_y = rect.bottom()
        pts = [QPointF(x, tip_y),
               QPointF(x - SIZE // 2, tip_y + SIZE),
               QPointF(x + SIZE // 2, tip_y + SIZE)]
        poly = QPolygonF(pts)
        p.setBrush(QBrush(QColor(150, 150, 150)))
        p.setPen(QPen(QColor(60, 60, 60), 1))
        p.drawPolygon(poly)

    def _draw_clip_labels(self, p: QPainter, rect: QRect):
        font = QFont("monospace", 8)
        p.setFont(font)

        if self._clip_black > 0:
            col = QColor(255, 80, 0) if self._clip_black > 0.1 else QColor(255, 180, 0)
            p.setPen(col)
            p.drawText(rect.left() + 2, rect.top() - 4, f"B {self._clip_black:.2f}%")

        if self._clip_white > 0:
            col = QColor(255, 80, 0) if self._clip_white > 0.1 else QColor(255, 180, 0)
            p.setPen(col)
            fm = QFontMetrics(font)
            txt = f"W {self._clip_white:.2f}%"
            tw  = fm.horizontalAdvance(txt)
            p.drawText(rect.right() - tw - 2, rect.top() - 4, txt)

    # ── Mouse etkileşimi ─────────────────────────────────────────────────────

    def _hit_zone(self, mx: int, my: int):
        """Hangi üçgene tıklandı?"""
        w   = self.width()
        h   = self.height()
        SIDE = 6
        hist_w = w - 2 * SIDE
        rx   = SIDE

        BOTTOM_MARGIN = 22
        TOP_MARGIN    = 18
        BOTTOM_Y      = h - BOTTOM_MARGIN
        TOP_Y         = TOP_MARGIN

        RANGE = 12

        def x_of(pos): return rx + pos * hist_w

        checks = [
            ("in_black",  x_of(self._black_in),            BOTTOM_Y),
            ("in_white",  x_of(self._white_in),             BOTTOM_Y),
            ("in_gamma",  x_of(self._gamma_screen_pos()),   BOTTOM_Y),
            ("out_black", x_of(self._black_out),            TOP_Y),
            ("out_white", x_of(self._white_out),            TOP_Y),
        ]
        for name, cx, cy in checks:
            if abs(mx - cx) < RANGE and abs(my - cy) < RANGE + 6:
                return name
        return None

    def _gamma_screen_pos(self) -> float:
        """Gamma üçgeninin ekran pozisyonu [0,1] (siyah-beyaz arası log)."""
        bi = self._black_in
        wi = self._white_in
        rng = wi - bi
        if rng < 1e-9:
            return bi
        log_g = np.log(self._gamma) / np.log(10)  # log10 gamma
        # [-1, 1] → [bi, wi]
        norm = (log_g + 1) / 2  # gamma=1.0 → norm=0.5 (orta)
        return bi + norm * rng

    def _pos_to_value(self, mx: int) -> float:
        SIDE   = 6
        hist_w = self.width() - 2 * SIDE
        val    = (mx - SIDE) / max(hist_w, 1)
        return float(np.clip(val, 0.0, 1.0))

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            hit = self._hit_zone(int(event.position().x()), int(event.position().y()))
            if hit:
                self._drag = hit
                self._drag_start_x = int(event.position().x())

    def mouseMoveEvent(self, event):
        if self._drag is None:
            hit = self._hit_zone(int(event.position().x()), int(event.position().y()))
            self.setCursor(Qt.CursorShape.SizeHorCursor if hit else Qt.CursorShape.ArrowCursor)
            return

        val = self._pos_to_value(int(event.position().x()))

        if self._drag == "in_black":
            self._black_in = float(np.clip(val, 0.0, self._white_in - 0.005))
        elif self._drag == "in_white":
            self._white_in = float(np.clip(val, self._black_in + 0.005, 1.0))
        elif self._drag == "in_gamma":
            bi, wi = self._black_in, self._white_in
            rng = wi - bi
            if rng > 1e-9:
                norm = (val - bi) / rng           # 0..1 içinde
                norm = np.clip(norm, 0.01, 0.99)
                log_g = norm * 2 - 1              # -1..1
                self._gamma = float(np.clip(10 ** log_g, 0.1, 9.9))
        elif self._drag == "out_black":
            self._black_out = float(np.clip(val, 0.0, self._white_out - 0.005))
        elif self._drag == "out_white":
            self._white_out = float(np.clip(val, self._black_out + 0.005, 1.0))

        self.update()
        self.levels_changed.emit(self._black_in, self._gamma, self._white_in,
                                  self._black_out, self._white_out)

    def mouseReleaseEvent(self, event):
        self._drag = None

    def mouseDoubleClickEvent(self, event):
        """Çift tık → o üçgeni sıfırla."""
        hit = self._hit_zone(int(event.position().x()), int(event.position().y()))
        if hit == "in_black":    self._black_in  = 0.0
        elif hit == "in_white":  self._white_in  = 1.0
        elif hit == "in_gamma":  self._gamma     = 1.0
        elif hit == "out_black": self._black_out = 0.0
        elif hit == "out_white": self._white_out = 1.0
        self.update()
        self.levels_changed.emit(self._black_in, self._gamma, self._white_in,
                                  self._black_out, self._white_out)


# ─────────────────────────────────────────────────────────────────────────────
#  İstatistik Etiketi
# ─────────────────────────────────────────────────────────────────────────────

class StatsLabel(QWidget):
    """Mini istatistik tablosu."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 0, 2, 0)
        layout.setSpacing(12)

        self._labels = {}
        for key in ["Ort", "Medyan", "Std", "Min", "Max"]:
            col = QVBoxLayout()
            col.setSpacing(1)
            title = QLabel(key)
            title.setStyleSheet("color: #888; font-size: 8pt;")
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            val = QLabel("—")
            val.setStyleSheet("color: #ddd; font-size: 9pt; font-weight: bold;")
            val.setAlignment(Qt.AlignmentFlag.AlignCenter)
            col.addWidget(title)
            col.addWidget(val)
            layout.addLayout(col)
            self._labels[key] = val

    def update_stats(self, stats: dict):
        fmt = {
            "Ort":    f"{stats.get('mean', 0):.4f}",
            "Medyan": f"{stats.get('median', 0):.4f}",
            "Std":    f"{stats.get('std', 0):.4f}",
            "Min":    f"{stats.get('min', 0):.4f}",
            "Max":    f"{stats.get('max', 0):.4f}",
        }
        for k, v in fmt.items():
            self._labels[k].setText(v)


# ─────────────────────────────────────────────────────────────────────────────
#  Ana Histogram Panel (QDockWidget içine konacak)
# ─────────────────────────────────────────────────────────────────────────────

class HistogramPanel(QWidget):
    """
    Photoshop tarzı Levels + Histogram paneli.

    Sinyaller
    ---------
    apply_levels(black_in, gamma, white_in, black_out, white_out)
        → ana pencere bu sinyali alır ve görüntüye uygular
    auto_levels_requested()
        → otomatik seviyeler
    """

    apply_levels    = pyqtSignal(float, float, float, float, float)
    auto_levels_req = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image: np.ndarray | None = None
        self._compute_timer = QTimer(self)
        self._compute_timer.setSingleShot(True)
        self._compute_timer.setInterval(80)
        self._compute_timer.timeout.connect(self._compute_and_refresh)

        self._build_ui()

    # ── UI ──────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # ── Üst satır: kanal + log + auto + reset ──
        top_row = QHBoxLayout()

        self._channel_cb = QComboBox()
        self._channel_cb.addItems(["Luminance", "R", "G", "B"])
        self._channel_cb.setFixedWidth(100)
        self._channel_cb.currentTextChanged.connect(self._on_channel_changed)
        top_row.addWidget(QLabel("Kanal:"))
        top_row.addWidget(self._channel_cb)

        self._log_chk = QCheckBox("Log")
        self._log_chk.setToolTip("Logaritmik ölçek")
        self._log_chk.toggled.connect(self._on_log_toggled)
        top_row.addWidget(self._log_chk)

        top_row.addStretch()

        btn_auto = QPushButton("Oto Seviye")
        btn_auto.setToolTip("Siyah/Beyaz noktayı otomatik belirle")
        btn_auto.clicked.connect(self._on_auto_levels)
        btn_auto.setFixedHeight(24)
        top_row.addWidget(btn_auto)

        btn_reset = QPushButton("Sıfırla")
        btn_reset.setToolTip("Tüm seviyeleri sıfırla")
        btn_reset.clicked.connect(self._on_reset)
        btn_reset.setFixedHeight(24)
        top_row.addWidget(btn_reset)

        root.addLayout(top_row)

        # ── Histogram görünümü ──
        self._view = HistogramView()
        self._view.levels_changed.connect(self._on_levels_dragged)
        root.addWidget(self._view)

        # ── Giriş seviyeleri (sayısal) ──
        inp_grp = QGroupBox("Giriş Seviyeleri")
        inp_lay = QHBoxLayout(inp_grp)
        inp_lay.setSpacing(4)

        self._spin_black_in = self._make_spin(0.0, 0.999, 0.0,  "Siyah Giriş")
        self._spin_gamma    = self._make_spin(0.1, 9.9,   1.0,  "Gamma",  step=0.05, dec=2)
        self._spin_white_in = self._make_spin(0.001, 1.0, 1.0,  "Beyaz Giriş")

        for lbl, sp in [("Siyah", self._spin_black_in),
                         ("Gamma", self._spin_gamma),
                         ("Beyaz", self._spin_white_in)]:
            col = QVBoxLayout()
            col.setSpacing(1)
            l = QLabel(lbl)
            l.setStyleSheet("color:#aaa;font-size:8pt;")
            l.setAlignment(Qt.AlignmentFlag.AlignCenter)
            col.addWidget(l)
            col.addWidget(sp)
            inp_lay.addLayout(col)

        self._spin_black_in.valueChanged.connect(self._on_spin_changed)
        self._spin_gamma.valueChanged.connect(self._on_spin_changed)
        self._spin_white_in.valueChanged.connect(self._on_spin_changed)
        root.addWidget(inp_grp)

        # ── Çıkış seviyeleri ──
        out_grp = QGroupBox("Çıkış Seviyeleri")
        out_lay = QHBoxLayout(out_grp)
        out_lay.setSpacing(4)

        self._spin_black_out = self._make_spin(0.0, 0.999, 0.0,  "Çıkış Siyah")
        self._spin_white_out = self._make_spin(0.001, 1.0, 1.0,  "Çıkış Beyaz")

        for lbl, sp in [("Siyah Çıkış", self._spin_black_out),
                         ("Beyaz Çıkış", self._spin_white_out)]:
            col = QVBoxLayout()
            col.setSpacing(1)
            l = QLabel(lbl)
            l.setStyleSheet("color:#aaa;font-size:8pt;")
            l.setAlignment(Qt.AlignmentFlag.AlignCenter)
            col.addWidget(l)
            col.addWidget(sp)
            out_lay.addLayout(col)

        self._spin_black_out.valueChanged.connect(self._on_spin_changed)
        self._spin_white_out.valueChanged.connect(self._on_spin_changed)
        root.addWidget(out_grp)

        # ── İstatistikler ──
        self._stats = StatsLabel()
        root.addWidget(self._stats)

        # ── Kırpma uyarısı ──
        self._clip_lbl = QLabel("")
        self._clip_lbl.setStyleSheet("color: #ff8800; font-size: 8pt;")
        self._clip_lbl.setWordWrap(True)
        root.addWidget(self._clip_lbl)

        # ── Uygula düğmesi ──
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #444;")
        root.addWidget(sep)

        btn_apply = QPushButton("Seviyeleri Uygula")
        btn_apply.setStyleSheet(
            "QPushButton{background:#285299;color:#fff;font-weight:bold;padding:6px;}"
            "QPushButton:hover{background:#355ea1;}"
        )
        btn_apply.clicked.connect(self._on_apply)
        root.addWidget(btn_apply)

        root.addStretch()

        # İlk kez güncelle
        self._block_spin = False

    # ── Spin helper ──────────────────────────────────────────────────────────

    def _make_spin(self, mn, mx, val, tip, step=0.001, dec=3) -> QDoubleSpinBox:
        sp = QDoubleSpinBox()
        sp.setRange(mn, mx)
        sp.setSingleStep(step)
        sp.setDecimals(dec)
        sp.setValue(val)
        sp.setToolTip(tip)
        sp.setFixedWidth(72)
        return sp

    # ── Görüntü güncelleme ───────────────────────────────────────────────────

    def set_image(self, image: np.ndarray | None):
        """Ana pencere görüntü değiştiğinde bunu çağırır."""
        self._image = image
        self._compute_timer.start()

    def _compute_and_refresh(self):
        if self._image is None:
            self._view.set_histograms({})
            self._stats.update_stats({})
            return

        img = self._image
        histograms = {}
        bins = 256

        # Luminance
        if img.ndim == 3:
            lum = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
        else:
            lum = img
        histograms["Luminance"] = np.histogram(lum.ravel(), bins=bins, range=(0, 1))[0].astype(np.float64)

        # RGB kanalları
        if img.ndim == 3:
            for i, ch in enumerate(["R", "G", "B"]):
                histograms[ch] = np.histogram(img[:, :, i].ravel(), bins=bins, range=(0, 1))[0].astype(np.float64)

        # İstatistikler (aktif kanal)
        ch = self._channel_cb.currentText()
        data = img[:, :, ["R","G","B"].index(ch)].ravel() if (ch in ("R","G","B") and img.ndim == 3) else lum.ravel()
        stats = {
            "mean":   float(np.mean(data)),
            "median": float(np.median(data)),
            "std":    float(np.std(data)),
            "min":    float(np.min(data)),
            "max":    float(np.max(data)),
        }

        # Kırpma hesabı
        eps = 1e-6
        total = data.size
        clip_b = float(np.sum(data <= eps) / total * 100)
        clip_w = float(np.sum(data >= 1.0 - eps) / total * 100)

        self._view.set_histograms(histograms, clip_b, clip_w)
        self._stats.update_stats(stats)

        # Kırpma uyarısı
        parts = []
        if clip_b > 0.01:
            parts.append(f"⬛ Siyah kırpma: {clip_b:.3f}%")
        if clip_w > 0.01:
            parts.append(f"⬜ Beyaz kırpma: {clip_w:.3f}%")
        self._clip_lbl.setText("  ".join(parts))

    # ── Sinyaller ────────────────────────────────────────────────────────────

    def _on_channel_changed(self, ch: str):
        self._view.set_channel(ch)
        self._compute_timer.start()

    def _on_log_toggled(self, val: bool):
        self._view.set_log_scale(val)

    def _on_levels_dragged(self, bi, g, wi, bo, wo):
        """Görünümden gelen sürükleme → spin kutularını güncelle."""
        self._block_spin = True
        self._spin_black_in.setValue(bi)
        self._spin_gamma.setValue(g)
        self._spin_white_in.setValue(wi)
        self._spin_black_out.setValue(bo)
        self._spin_white_out.setValue(wo)
        self._block_spin = False

    def _on_spin_changed(self):
        """Spin kutusu değişince histogram üçgenlerini güncelle."""
        if getattr(self, "_block_spin", False):
            return
        bi  = self._spin_black_in.value()
        g   = self._spin_gamma.value()
        wi  = self._spin_white_in.value()
        bo  = self._spin_black_out.value()
        wo  = self._spin_white_out.value()
        # Geçerlilik kontrolü
        if bi >= wi:
            wi = bi + 0.001
            self._spin_white_in.setValue(wi)
        if bo >= wo:
            wo = bo + 0.001
            self._spin_white_out.setValue(wo)
        self._view.set_levels(bi, g, wi, bo, wo)

    def _on_auto_levels(self):
        """Histogramdan otomatik siyah/beyaz noktası belirle."""
        if self._image is None:
            return
        img = self._image
        ch = self._channel_cb.currentText()
        if ch in ("R", "G", "B") and img.ndim == 3:
            data = img[:, :, ["R","G","B"].index(ch)].ravel()
        elif img.ndim == 3:
            data = (0.2126*img[:,:,0]+0.7152*img[:,:,1]+0.0722*img[:,:,2]).ravel()
        else:
            data = img.ravel()

        # %0.1 ve %99.9 yüzdelik
        lo = float(np.percentile(data, 0.1))
        hi = float(np.percentile(data, 99.9))
        lo = float(np.clip(lo, 0.0, 0.99))
        hi = float(np.clip(hi, lo + 0.001, 1.0))

        self._view.set_levels(lo, 1.0, hi, 0.0, 1.0)
        self._spin_black_in.setValue(lo)
        self._spin_gamma.setValue(1.0)
        self._spin_white_in.setValue(hi)
        self._spin_black_out.setValue(0.0)
        self._spin_white_out.setValue(1.0)
        self.auto_levels_req.emit()

    def _on_reset(self):
        self._view.reset_levels()
        self._block_spin = True
        self._spin_black_in.setValue(0.0)
        self._spin_gamma.setValue(1.0)
        self._spin_white_in.setValue(1.0)
        self._spin_black_out.setValue(0.0)
        self._spin_white_out.setValue(1.0)
        self._block_spin = False

    def _on_apply(self):
        bi, g, wi, bo, wo = self._view.get_levels()
        self.apply_levels.emit(bi, g, wi, bo, wo)
