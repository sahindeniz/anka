"""
Astro Mastro Pro — History Panel
İşlem geçmişi: undo/redo, thumbnail listesi
"""
import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QListWidget, QListWidgetItem,
                              QLabel, QHBoxLayout, QPushButton)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QIcon


class HistoryPanel(QWidget):
    """
    Sağ panelde gösterilen işlem geçmişi.
    state_selected(index) sinyali: kullanıcı bir geçmiş durumu seçtiğinde
    """
    state_selected = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(180)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Başlık
        title = QLabel("Geçmiş")
        title.setStyleSheet("font-weight: bold; color: #88aaff; font-size: 11pt;")
        layout.addWidget(title)

        # Liste
        self._list = QListWidget()
        self._list.setSpacing(2)
        self._list.currentRowChanged.connect(self._on_row_changed)
        layout.addWidget(self._list)

        # Düğmeler
        btn_row = QHBoxLayout()
        self._btn_undo = QPushButton("↩ Geri")
        self._btn_undo.setEnabled(False)
        self._btn_undo.clicked.connect(self._undo)
        btn_row.addWidget(self._btn_undo)

        self._btn_redo = QPushButton("↪ İleri")
        self._btn_redo.setEnabled(False)
        self._btn_redo.clicked.connect(self._redo)
        btn_row.addWidget(self._btn_redo)
        layout.addLayout(btn_row)

        self._states = []   # list of (label, image_float32)
        self._current = -1
        self._updating = False

    # ── API ─────────────────────────────────────────────────────────────────

    def push(self, label: str, image: np.ndarray):
        """Yeni bir durum ekle. Mevcut pozisyondan sonrasını sil."""
        if self._current < len(self._states) - 1:
            self._states = self._states[:self._current + 1]

        # Thumbnail için küçük kopya sakla
        thumb = _make_thumb(image)
        self._states.append((label, image.copy(), thumb))
        self._current = len(self._states) - 1
        self._rebuild_list()
        self._update_buttons()

    def current_image(self) -> np.ndarray:
        if 0 <= self._current < len(self._states):
            return self._states[self._current][1]
        return None

    def undo(self):
        self._undo()

    def redo(self):
        self._redo()

    def clear(self):
        self._states.clear()
        self._current = -1
        self._list.clear()
        self._update_buttons()

    # ── İç ──────────────────────────────────────────────────────────────────

    def _undo(self):
        if self._current > 0:
            self._current -= 1
            self._update_list_selection()
            self._update_buttons()
            self.state_selected.emit(self._current)

    def _redo(self):
        if self._current < len(self._states) - 1:
            self._current += 1
            self._update_list_selection()
            self._update_buttons()
            self.state_selected.emit(self._current)

    def _on_row_changed(self, row):
        if self._updating:
            return
        if row >= 0 and row != self._current:
            self._current = row
            self._update_buttons()
            self.state_selected.emit(self._current)

    def _rebuild_list(self):
        self._updating = True
        self._list.clear()
        for i, (label, _, thumb) in enumerate(self._states):
            item = QListWidgetItem()
            item.setText(label)
            if thumb is not None:
                item.setIcon(QIcon(QPixmap.fromImage(thumb)))
            item.setSizeHint(item.sizeHint().__class__(160, 36))
            self._list.addItem(item)
        self._list.setCurrentRow(self._current)
        self._updating = False

    def _update_list_selection(self):
        self._updating = True
        self._list.setCurrentRow(self._current)
        self._updating = False

    def _update_buttons(self):
        self._btn_undo.setEnabled(self._current > 0)
        self._btn_redo.setEnabled(self._current < len(self._states) - 1)


def _make_thumb(image: np.ndarray, size: int = 48) -> QImage:
    """Küçük thumbnail QImage oluştur."""
    try:
        import cv2
        img = np.clip(image, 0, 1)
        h, w = img.shape[:2]
        scale = size / max(h, w)
        th = max(1, int(h * scale))
        tw = max(1, int(w * scale))
        small = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        disp = (small * 255).astype(np.uint8)
        if disp.ndim == 2:
            return QImage(disp.data.tobytes(), tw, th, tw, QImage.Format.Format_Grayscale8).copy()
        else:
            disp = np.ascontiguousarray(disp)
            return QImage(disp.data.tobytes(), tw, th, tw * 3, QImage.Format.Format_RGB888).copy()
    except Exception:
        return None
