"""
Astro Mastro Pro — Worker Thread
UI donmasını engellemek için işlemleri arka planda çalıştırır.
"""
import traceback
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal


class ProcessWorker(QThread):
    """
    Genel amaçlı işlem thread'i.
    
    Kullanım:
        worker = ProcessWorker(fn, image, **kwargs)
        worker.finished.connect(on_done)
        worker.error.connect(on_error)
        worker.progress.connect(on_progress)
        worker.start()
    
    finished(result) — numpy array veya None
    error(message)   — hata mesajı
    progress(pct, msg)
    """
    finished = pyqtSignal(object)
    error    = pyqtSignal(str)
    progress = pyqtSignal(int, str)

    def __init__(self, fn, image: np.ndarray, **kwargs):
        super().__init__()
        self._fn = fn
        self._image = image
        self._kwargs = kwargs

    def run(self):
        try:
            result = self._fn(self._image, **self._kwargs)
            # Bazı fonksiyonlar (noisexterminator, starsmalerx) tuple döner
            if isinstance(result, tuple):
                result = result[0]
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
