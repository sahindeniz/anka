"""
Astro Mastro Pro — Canvas Widget
Görüntü gösterimi: zoom, pan, histogram preview
"""
import numpy as np
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PyQt6.QtCore import Qt, QPointF, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QWheelEvent, QMouseEvent


class ImageCanvas(QGraphicsView):
    """
    Sürükle-bırak, zoom, pan destekli görüntü canvas'ı.
    image_dropped(path) sinyali: sürükle-bırak ile dosya açıldığında
    """
    image_dropped = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pix_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pix_item)

        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setStyleSheet("background: #111111; border: none;")
        self.setAcceptDrops(True)

        self._zoom = 0
        self._interp = Qt.TransformationMode.FastTransformation
        self._current_image = None  # float32 numpy

    # ── Görüntü güncelle ────────────────────────────────────────────────────

    def set_image(self, image: np.ndarray, fit: bool = False):
        """float32 [0,1] numpy array'i göster."""
        if image is None:
            self._pix_item.setPixmap(QPixmap())
            self._current_image = None
            return

        self._current_image = image
        qimg = _numpy_to_qimage(image, self._interp)
        self._pix_item.setPixmap(QPixmap.fromImage(qimg))
        self._scene.setSceneRect(self._pix_item.boundingRect())

        if fit:
            self.fit_in_view()

    def set_interpolation(self, mode: str = "nearest"):
        if mode == "smooth":
            self._interp = Qt.TransformationMode.SmoothTransformation
        else:
            self._interp = Qt.TransformationMode.FastTransformation
        self._pix_item.setTransformationMode(self._interp)

    def fit_in_view(self):
        if self._pix_item.pixmap().isNull():
            return
        self.fitInView(self._pix_item, Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom = 0

    # ── Zoom ────────────────────────────────────────────────────────────────

    def wheelEvent(self, event: QWheelEvent):
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self._zoom += 1 if event.angleDelta().y() > 0 else -1
        self._zoom = max(-10, min(20, self._zoom))
        self.scale(factor, factor)

    def zoom_in(self):
        self.scale(1.25, 1.25)
        self._zoom += 1

    def zoom_out(self):
        self.scale(0.8, 0.8)
        self._zoom -= 1

    def zoom_reset(self):
        self.resetTransform()
        self._zoom = 0

    # ── Drag & Drop ─────────────────────────────────────────────────────────

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                if path:
                    self.image_dropped.emit(path)
                    break
            event.acceptProposedAction()
        else:
            super().dropEvent(event)


# ── Yardımcı ────────────────────────────────────────────────────────────────

def _numpy_to_qimage(image: np.ndarray, interp=None) -> QImage:
    """float32 [0,1] numpy → QImage (8-bit display)."""
    img = np.clip(image, 0, 1)
    disp = (img * 255).astype(np.uint8)

    if disp.ndim == 2:
        # Mono → grayscale
        h, w = disp.shape
        disp = np.ascontiguousarray(disp)
        qimg = QImage(disp.data, w, h, w, QImage.Format.Format_Grayscale8)
    elif disp.ndim == 3 and disp.shape[2] == 3:
        h, w, _ = disp.shape
        disp = np.ascontiguousarray(disp)
        qimg = QImage(disp.data, w, h, w * 3, QImage.Format.Format_RGB888)
    else:
        # Fallback
        h, w = disp.shape[:2]
        disp = np.ascontiguousarray(disp[:, :, :3] if disp.ndim == 3 else disp)
        qimg = QImage(disp.data, w, h, w * 3, QImage.Format.Format_RGB888)

    return qimg.copy()
