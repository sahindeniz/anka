"""
Astro Mastro Pro — Ana Pencere
Toolbar, canvas, panel dock, geçmiş paneli, menüler
"""
import os
import numpy as np
from core.loader import FILE_FILTER as _FILE_FILTER
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QToolBar, QDockWidget, QFileDialog, QMessageBox,
    QStatusBar, QProgressBar, QLabel, QSplitter, QApplication,
    QTabWidget, QTabBar
)
from PyQt6.QtCore import Qt, QSize, QTimer
from PyQt6.QtGui import QAction, QKeySequence, QIcon

from gui import settings_manager as SM
from gui.canvas import ImageCanvas
from gui.history_panel import HistoryPanel
from gui.histogram_widget import HistogramPanel
from gui.histogram_editor import HistogramEditorPanel
from gui.panels import (BgPanel, StretchPanel, NoisePanel, SharpenPanel,
                         ColorPanel, DeconvPanel, StarShrinkPanel, RecompPanel)
from gui.settings_dialog import SettingsDialog
from gui.theme import get_stylesheet
from gui.worker import ProcessWorker


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Astro Mastro Pro")
        self.setMinimumSize(900, 600)
        self.resize(1280, 800)

        self._current_image = None    # float32 numpy, gösterilen
        self._base_image = None       # yıldızsız base (recomp için)
        self._starmask_image = None   # yıldız maskesi (recomp için)
        self._current_path = None     # son açılan dosya
        self._worker = None           # aktif ProcessWorker
        self._live_base_image = None  # live preview öncesi orijinal görüntü

        self._init_ui()
        self._setup_menubar()
        self._setup_toolbar()
        self._setup_dock_panels()
        self._setup_statusbar()
        self._apply_theme()

    # ── UI Kurulumu ──────────────────────────────────────────────────────────

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Splitter: sol=canvas, sağ=aktif panel
        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(self._splitter)

        # Sekmeli canvas alanı
        self._tab_widget = QTabWidget()
        self._tab_widget.setTabsClosable(True)
        self._tab_widget.tabCloseRequested.connect(self._on_tab_close)
        self._tab_widget.currentChanged.connect(self._on_tab_changed)
        self._tab_widget.setStyleSheet(
            "QTabWidget::pane { border: none; }"
            "QTabBar::tab { background: #1a1a2e; color: #aaa; padding: 5px 14px;"
            "  border: 1px solid #333; border-bottom: none; border-radius: 4px 4px 0 0;"
            "  min-width: 80px; font-size: 10px; }"
            "QTabBar::tab:selected { background: #0a0a0a; color: #fff;"
            "  border-bottom: 2px solid #3d9bd4; }"
            "QTabBar::tab:hover { background: #252540; color: #ddd; }"
            "QTabBar::close-button { image: none; width: 14px; height: 14px;"
            "  subcontrol-position: right; padding: 2px; }"
        )
        self._splitter.addWidget(self._tab_widget)

        # Ana canvas (kapatılamaz)
        self._canvas = ImageCanvas()
        self._canvas.image_dropped.connect(self.open_file)
        self._tab_widget.addTab(self._canvas, "Ana Görüntü")
        # Ana sekme kapatılamasın
        self._tab_widget.tabBar().setTabButton(0, QTabBar.ButtonPosition.RightSide, None)

        # Ek canvas'lar: Yıldızsız ve Yıldız Maskesi
        self._starless_canvas = ImageCanvas()
        self._starmask_canvas = ImageCanvas()
        self._starless_tab_idx = -1
        self._starmask_tab_idx = -1

        # Aktif panel alanı (başta gizli)
        self._panel_container = QWidget()
        self._panel_container.setMaximumWidth(SM.get("panel_width", 415))
        self._panel_container.setMinimumWidth(200)
        self._panel_layout = QVBoxLayout(self._panel_container)
        self._panel_layout.setContentsMargins(0, 0, 0, 0)
        self._panel_container.setVisible(False)
        self._splitter.addWidget(self._panel_container)

        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 0)

        self._active_panel = None

    def _setup_menubar(self):
        mb = self.menuBar()

        # FILE
        fm = mb.addMenu("Dosya")
        self._act_open = QAction("Aç…", self)
        self._act_open.setShortcut(QKeySequence.StandardKey.Open)
        self._act_open.triggered.connect(self.open_file_dialog)
        fm.addAction(self._act_open)

        self._act_save = QAction("Kaydet…", self)
        self._act_save.setShortcut(QKeySequence.StandardKey.Save)
        self._act_save.triggered.connect(self.save_file_dialog)
        fm.addAction(self._act_save)

        fm.addSeparator()

        self._act_open_starless = QAction("Yıldızsız Aç…", self)
        self._act_open_starless.triggered.connect(self._open_starless)
        fm.addAction(self._act_open_starless)

        self._act_open_starmask = QAction("Yıldız Maskesi Aç…", self)
        self._act_open_starmask.triggered.connect(self._open_starmask)
        fm.addAction(self._act_open_starmask)

        fm.addSeparator()
        act_quit = QAction("Çıkış", self)
        act_quit.setShortcut(QKeySequence.StandardKey.Quit)
        act_quit.triggered.connect(self.close)
        fm.addAction(act_quit)

        # EDIT
        em = mb.addMenu("Düzenle")
        self._act_undo = QAction("Geri Al", self)
        self._act_undo.setShortcut(QKeySequence.StandardKey.Undo)
        self._act_undo.triggered.connect(self._undo)
        em.addAction(self._act_undo)

        self._act_redo = QAction("Yinele", self)
        self._act_redo.setShortcut(QKeySequence.StandardKey.Redo)
        self._act_redo.triggered.connect(self._redo)
        em.addAction(self._act_redo)

        em.addSeparator()
        act_settings = QAction("Ayarlar…", self)
        act_settings.triggered.connect(self._open_settings)
        em.addAction(act_settings)

        # VIEW
        vm = mb.addMenu("Görünüm")
        act_fit = QAction("Ekrana Sığdır", self)
        act_fit.setShortcut(QKeySequence("Ctrl+0"))
        act_fit.triggered.connect(self._canvas.fit_in_view)
        vm.addAction(act_fit)

        act_100 = QAction("Gerçek Boyut (100%)", self)
        act_100.setShortcut(QKeySequence("Ctrl+1"))
        act_100.triggered.connect(self._canvas.zoom_reset)
        vm.addAction(act_100)

    def _setup_toolbar(self):
        tb = QToolBar("Ana Toolbar")
        tb.setMovable(False)
        tb.setIconSize(QSize(22, 22))
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb)

        def tool_action(label, icon_text, slot, checkable=True):
            act = QAction(f"{icon_text} {label}", self)
            act.setCheckable(checkable)
            act.triggered.connect(slot)
            tb.addAction(act)
            return act

        # Açma / kaydetme
        act_open = QAction("📂 Aç", self)
        act_open.setShortcut(QKeySequence.StandardKey.Open)
        act_open.triggered.connect(self.open_file_dialog)
        tb.addAction(act_open)

        act_save = QAction("💾 Kaydet", self)
        act_save.setShortcut(QKeySequence.StandardKey.Save)
        act_save.triggered.connect(self.save_file_dialog)
        tb.addAction(act_save)

        tb.addSeparator()

        # İşlem panelleri
        self._panel_actions = {}
        panels_def = [
            ("BG Çıkar",  "🌌", "bg",     self._toggle_panel_bg),
            ("Stretch",   "📊", "stretch", self._toggle_panel_stretch),
            ("Noise",     "🔊", "noise",   self._toggle_panel_noise),
            ("Sharpen",   "🔬", "sharpen", self._toggle_panel_sharpen),
            ("Color",     "🎨", "color",   self._toggle_panel_color),
            ("Deconv",    "🌀", "deconv",  self._toggle_panel_deconv),
            ("StarShrink","✦↓", "star_shrink", self._toggle_panel_star_shrink),
            ("Recomp",    "✦+", "recomp",  self._toggle_panel_recomp),
        ]
        for label, icon, key, slot in panels_def:
            act = tool_action(label, icon, slot)
            self._panel_actions[key] = act

        tb.addSeparator()

        # StarNet
        act_starnet = QAction("⭐ StarNet", self)
        act_starnet.triggered.connect(self._run_starnet)
        tb.addAction(act_starnet)

        # ASTAP
        act_astap = QAction("🔭 Plate Solve", self)
        act_astap.triggered.connect(self._run_astap)
        tb.addAction(act_astap)

    def _setup_dock_panels(self):
        # History dock — sağ
        if SM.get("show_history", True):
            self._history = HistoryPanel()
            self._history.state_selected.connect(self._on_history_select)

            dock = QDockWidget("Geçmiş", self)
            dock.setWidget(self._history)
            dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea |
                                  Qt.DockWidgetArea.LeftDockWidgetArea)
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        else:
            self._history = HistoryPanel()  # gizli ama aktif

        # Histogram dock — alt (Photoshop tarzı: Levels + Curves + Adjustments)
        self._histogram_editor = HistogramEditorPanel()
        self._histogram_editor.preview_changed.connect(self._on_hist_preview)
        self._histogram_editor.apply_requested.connect(self._on_hist_apply)

        hist_dock = QDockWidget("Histogram / Levels / Curves", self)
        hist_dock.setWidget(self._histogram_editor)
        hist_dock.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea |
            Qt.DockWidgetArea.RightDockWidgetArea  |
            Qt.DockWidgetArea.LeftDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, hist_dock)

        # Panel örnekleri (lazy create)
        self._panels = {}

    def _setup_statusbar(self):
        sb = QStatusBar()
        self.setStatusBar(sb)

        self._status_label = QLabel("Hazır")
        sb.addWidget(self._status_label, 1)

        self._progress = QProgressBar()
        self._progress.setMaximumWidth(200)
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        sb.addPermanentWidget(self._progress)

        self._info_label = QLabel("")
        sb.addPermanentWidget(self._info_label)

    def _apply_theme(self):
        theme = SM.get("theme", "dark")
        self.setStyleSheet(get_stylesheet(theme))
        self._canvas.setStyleSheet(
            "background: #0a0a0a; border: none;" if theme == "dark"
            else "background: #888; border: none;"
        )

    # ── Dosya İşlemleri ──────────────────────────────────────────────────────

    def open_file_dialog(self):
        last_dir = SM.get("last_open_dir", "")
        path, _ = QFileDialog.getOpenFileName(
            self, "Görüntü Aç", last_dir,
            _FILE_FILTER
        )
        if path:
            self.open_file(path)

    def open_file(self, path: str):
        try:
            from core.loader import load_image
            self._status("Yükleniyor: " + os.path.basename(path))
            image = load_image(path)
            self._current_path = path
            self._set_image(image, label="Aç: " + os.path.basename(path), fit=True)
            SM.set("last_open_dir", os.path.dirname(path))
            SM.set("last_open_file", path)
            SM.save()
            self._update_info(image)
        except Exception as e:
            QMessageBox.critical(self, "Yükleme Hatası", str(e))

    def save_file_dialog(self):
        if self._current_image is None:
            QMessageBox.warning(self, "Kaydet", "Kaydedilecek görüntü yok.")
            return

        last_dir = SM.get("last_save_dir", SM.get("last_open_dir", ""))
        fmt = SM.get("output_format", "FITS")
        ext_map = {"FITS": "*.fits", "TIFF": "*.tiff", "PNG": "*.png", "JPG": "*.jpg"}
        filt = f"{fmt} ({ext_map.get(fmt, '*.*')});;Tümü (*)"

        path, _ = QFileDialog.getSaveFileName(self, "Kaydet", last_dir, filt)
        if path:
            try:
                from core.loader import save_image
                save_image(self._current_image, path, fmt)
                SM.set("last_save_dir", os.path.dirname(path))
                SM.save()
                self._status(f"Kaydedildi: {os.path.basename(path)}")
            except Exception as e:
                QMessageBox.critical(self, "Kaydetme Hatası", str(e))

    def _open_starless(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Yıldızsız Görüntü Aç", SM.get("last_open_dir", ""),
            _FILE_FILTER
        )
        if path:
            try:
                from core.loader import load_image
                self._base_image = load_image(path)
                self._show_image_tab(
                    "starless", self._starless_canvas, self._base_image,
                    f"⭐ Yıldızsız — {os.path.basename(path)}")
                self._status(f"Yıldızsız yüklendi: {os.path.basename(path)}")
            except Exception as e:
                QMessageBox.critical(self, "Hata", str(e))

    def _open_starmask(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Yıldız Maskesi Aç", SM.get("last_open_dir", ""),
            _FILE_FILTER
        )
        if path:
            try:
                from core.loader import load_image
                self._starmask_image = load_image(path)
                self._show_image_tab(
                    "starmask", self._starmask_canvas, self._starmask_image,
                    f"✦ Yıldız Maskesi — {os.path.basename(path)}")
                self._status(f"Yıldız maskesi yüklendi: {os.path.basename(path)}")
            except Exception as e:
                QMessageBox.critical(self, "Hata", str(e))

    # ── Sekme Yönetimi ─────────────────────────────────────────────────────

    def _show_image_tab(self, key: str, canvas: ImageCanvas,
                        image: np.ndarray, title: str):
        """Bir görüntüyü yeni veya mevcut sekmede göster."""
        # Mevcut sekmeyi bul
        idx = -1
        for i in range(self._tab_widget.count()):
            if self._tab_widget.widget(i) is canvas:
                idx = i
                break

        if idx == -1:
            # Yeni sekme ekle
            idx = self._tab_widget.addTab(canvas, title)
        else:
            # Başlığı güncelle
            self._tab_widget.setTabText(idx, title)

        canvas.set_image(image, fit=True)

        if key == "starless":
            self._starless_tab_idx = idx
        elif key == "starmask":
            self._starmask_tab_idx = idx

        # Yeni sekmeye geç
        self._tab_widget.setCurrentIndex(idx)

    def _on_tab_close(self, index: int):
        """Sekme kapatma — ana sekme kapatılamaz."""
        if index == 0:
            return  # Ana görüntü sekmesi kapatılamaz
        widget = self._tab_widget.widget(index)
        self._tab_widget.removeTab(index)

        if widget is self._starless_canvas:
            self._starless_tab_idx = -1
        elif widget is self._starmask_canvas:
            self._starmask_tab_idx = -1

    def _on_tab_changed(self, index: int):
        """Sekme değiştiğinde bilgi güncelle."""
        if index == 0 and self._current_image is not None:
            self._update_info(self._current_image)
        elif index >= 0:
            widget = self._tab_widget.widget(index)
            if widget is self._starless_canvas and self._base_image is not None:
                self._update_info(self._base_image)
            elif widget is self._starmask_canvas and self._starmask_image is not None:
                self._update_info(self._starmask_image)

    # ── Görüntü Güncelleme ───────────────────────────────────────────────────

    def _set_image(self, image: np.ndarray, label: str = "İşlem",
                   fit: bool = False, push_history: bool = True):
        self._current_image = image
        self._canvas.set_image(image, fit=fit)
        if push_history:
            self._history.push(label, image)
        self._update_info(image)
        # Histogram'ı güncelle
        self._histogram_editor.set_image(image)

    def _update_info(self, image: np.ndarray):
        if image is None:
            self._info_label.setText("")
            return
        h, w = image.shape[:2]
        c = image.shape[2] if image.ndim == 3 else 1
        ctype = "RGB" if c == 3 else "Mono"
        self._info_label.setText(f"{w}×{h}  {ctype}  float32")

    # ── Geçmiş ───────────────────────────────────────────────────────────────

    def _undo(self):
        self._history.undo()

    def _redo(self):
        self._history.redo()

    def _on_history_select(self, index: int):
        img = self._history.current_image()
        if img is not None:
            self._current_image = img
            self._canvas.set_image(img)
            self._histogram_editor.set_image(img)

    # ── Panel Toggle ─────────────────────────────────────────────────────────

    def _toggle_panel(self, key: str, panel_class):
        # Diğer panel butonlarını kapat
        for k, act in self._panel_actions.items():
            if k != key:
                act.setChecked(False)

        if self._panel_actions[key].isChecked():
            # Mevcut paneli kaldır
            if self._active_panel is not None:
                self._panel_layout.removeWidget(self._active_panel)
                self._active_panel.setParent(None)
                self._active_panel = None

            # Yeni panel oluştur (cache)
            if key not in self._panels:
                p = panel_class()
                p.apply_requested.connect(self._on_apply)
                p.preview_requested.connect(self._on_preview)
                p.live_preview_requested.connect(self._on_live_preview)
                self._panels[key] = p

            panel = self._panels[key]
            self._panel_layout.addWidget(panel)
            self._active_panel = panel
            self._panel_container.setVisible(True)
        else:
            # Panel kapatılırken live preview'u iptal et, orijinale dön
            if self._live_base_image is not None:
                self._current_image = self._live_base_image
                self._live_base_image = None
                self._canvas.set_image(self._current_image)
                self._histogram_editor.set_image(self._current_image)
            if self._active_panel is not None:
                self._panel_layout.removeWidget(self._active_panel)
                self._active_panel.setParent(None)
                self._active_panel = None
            self._panel_container.setVisible(False)

    def _toggle_panel_bg(self):      self._toggle_panel("bg",     BgPanel)
    def _toggle_panel_stretch(self): self._toggle_panel("stretch", StretchPanel)
    def _toggle_panel_noise(self):   self._toggle_panel("noise",   NoisePanel)
    def _toggle_panel_sharpen(self): self._toggle_panel("sharpen", SharpenPanel)
    def _toggle_panel_color(self):   self._toggle_panel("color",   ColorPanel)
    def _toggle_panel_deconv(self):  self._toggle_panel("deconv",  DeconvPanel)
    def _toggle_panel_star_shrink(self): self._toggle_panel("star_shrink", StarShrinkPanel)
    def _toggle_panel_recomp(self):  self._toggle_panel("recomp",  RecompPanel)

    # ── İşlem Çalıştırma ─────────────────────────────────────────────────────

    def _on_apply(self, params: dict):
        if self._current_image is None:
            QMessageBox.warning(self, "Uyarı", "Önce bir görüntü açın.")
            return
        # Live preview varsa orijinalden uygula
        if self._live_base_image is not None:
            self._current_image = self._live_base_image.copy()
            self._live_base_image = None
        self._run_processing(params, apply=True)

    def _on_preview(self, params: dict):
        if self._current_image is None:
            return
        self._run_processing(params, apply=False)

    def _on_live_preview(self, params: dict):
        """Live preview — parametre değişikliklerinde anlık önizleme.
        Her zaman orijinal görüntü üstüne uygulanır (katlanma olmaz)."""
        if self._current_image is None:
            return
        # İlk live preview çağrısında orijinali sakla
        if self._live_base_image is None:
            self._live_base_image = self._current_image.copy()
        # Zaten çalışan bir worker varsa iptal et
        if self._worker and self._worker.isRunning():
            self._worker.quit()
            self._worker.wait(200)
        # Orijinal görüntüyü kullan (katlanmayı önle)
        self._current_image = self._live_base_image.copy()
        self._run_processing(params, apply=False)

    def _run_processing(self, params: dict, apply: bool = True):
        # Aktif panel anahtarına göre işlev belirle
        panel_key = None
        for k, act in self._panel_actions.items():
            if act.isChecked():
                panel_key = k
                break

        fn = self._get_processing_fn(panel_key, params)
        if fn is None:
            return

        # Önceki worker'ı iptal et
        if self._worker and self._worker.isRunning():
            self._worker.quit()
            self._worker.wait(500)

        # Live preview sırasında paneli kilitleme (slider kullanımı devam etsin)
        is_live = self._live_base_image is not None
        self._progress.setVisible(True)
        self._status("İşleniyor…")
        if not is_live:
            self._set_processing_lock(True)

        self._worker = ProcessWorker(fn, self._current_image, **params)
        self._worker.finished.connect(
            lambda img, a=apply: self._on_worker_done(img, a, panel_key)
        )
        self._worker.error.connect(self._on_worker_error)
        self._worker.start()

    def _get_processing_fn(self, panel_key: str, params: dict):
        """Panel anahtarına göre uygun işlev döner."""
        try:
            if panel_key == "bg":
                from processing.background import remove_gradient_dispatch
                return remove_gradient_dispatch

            elif panel_key == "stretch":
                from processing.stretch import stretch
                return stretch

            elif panel_key == "noise":
                method = params.get("method", "bilateral")
                if method == "wavelet":
                    from processing.noisexterminator import noisexterminator
                    def wavelet_fn(img, **kw):
                        result, _ = noisexterminator(img, **kw)
                        return result
                    return wavelet_fn
                else:
                    from processing.noise_reduction import reduce_noise
                    return reduce_noise

            elif panel_key == "sharpen":
                from processing.sharpening import sharpen
                return sharpen

            elif panel_key == "color":
                from processing.color_calibration import calibrate_color
                return calibrate_color

            elif panel_key == "deconv":
                from processing.deconvolution import deconvolve_dispatch
                return deconvolve_dispatch

            elif panel_key == "star_shrink":
                from processing.star_shrink import star_shrink
                return star_shrink

            elif panel_key == "recomp":
                return self._recompose_fn

        except Exception as e:
            QMessageBox.critical(self, "Modül Hatası", str(e))
        return None

    def _recompose_fn(self, image: np.ndarray, **params):
        """Yıldızsız + maske birleştirme."""
        base = self._base_image if self._base_image is not None else image
        mask = self._starmask_image

        if mask is None:
            # Eğer maske yok, orijinal görüntüyü baz al
            return image

        # Boyut uyumu
        h, w = base.shape[:2]
        import cv2
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

        opacity = params.get("opacity", 1.0)
        mode = params.get("mode", "add")
        mask_op = mask * opacity

        if mode == "screen":
            result = 1.0 - (1.0 - base) * (1.0 - mask_op)
        elif mode == "lighten":
            result = np.maximum(base, mask_op)
        else:  # add
            result = np.clip(base + mask_op, 0, 1)

        return result.astype(np.float32)

    def _on_worker_done(self, image, apply: bool, panel_key: str):
        self._set_processing_lock(False)
        self._progress.setVisible(False)

        if image is None:
            self._status("İşlem başarısız.")
            return

        label_map = {
            "bg": "BG Çıkarma", "stretch": "Stretch", "noise": "Gürültü Azaltma",
            "sharpen": "Keskinleştirme", "color": "Renk Kalibrasyonu",
            "deconv": "Dekonvolüsyon", "recomp": "Yıldız Birleştirme",
        }
        label = label_map.get(panel_key, "İşlem")

        if apply:
            self._live_base_image = None  # Apply sonrası live base temizle
            self._set_image(image, label=label)
            self._status(f"✓ {label} uygulandı")
        else:
            # Sadece görüntüle, geçmişe ekleme
            self._current_image = image
            self._canvas.set_image(image)
            self._histogram_editor.set_image(image)
            self._status(f"Önizleme: {label}")

    def _on_hist_preview(self, image: np.ndarray):
        """Histogram editor'dan gelen canlı önizleme — geçmişe eklenmez."""
        if image is None:
            return
        self._canvas.set_image(image)

    def _on_hist_apply(self, image: np.ndarray):
        """Histogram editor'dan gelen Apply — geçmişe eklenir."""
        if image is None:
            return
        self._set_image(image, label="Histogram / Curves / Adj")

    def _on_worker_error(self, msg: str):
        self._set_processing_lock(False)
        self._progress.setVisible(False)
        self._status("Hata!")
        QMessageBox.critical(self, "İşlem Hatası", msg)

    # ── Harici Araçlar ───────────────────────────────────────────────────────

    def _run_starnet(self):
        if self._current_image is None:
            QMessageBox.warning(self, "StarNet", "Önce bir görüntü açın.")
            return
        starnet_exe = SM.get("starnet_exe", "")
        if not starnet_exe or not os.path.exists(starnet_exe):
            QMessageBox.warning(
                self, "StarNet",
                "StarNet++ yolu ayarlanmamış.\n"
                "Düzenle → Ayarlar → StarNet++ sekmesinden yolu girin."
            )
            return

        # Geçici dosyaya yaz, StarNet'i çalıştır
        import tempfile, subprocess
        from core.loader import save_image, load_image

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                in_path  = os.path.join(tmpdir, "input.tif")
                out_path = os.path.join(tmpdir, "output.tif")
                mask_path= os.path.join(tmpdir, "starmask.tif")

                save_image(self._current_image, in_path, "TIFF")
                stride = str(SM.get("starnet_stride", 256))

                cmd = [starnet_exe, in_path, out_path, stride]
                if SM.get("starnet_use_gpu", False):
                    cmd.append("1")

                self._status("StarNet++ çalışıyor…")
                QApplication.processEvents()
                result = subprocess.run(cmd, capture_output=True, timeout=300)

                if result.returncode != 0:
                    raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))

                if os.path.exists(out_path):
                    starless = load_image(out_path)
                    self._base_image = starless
                    self._set_image(starless, label="StarNet: Yıldızsız")

                    # Yıldızsız sekme
                    self._show_image_tab(
                        "starless", self._starless_canvas, starless,
                        "⭐ Yıldızsız (StarNet)")

                    # Yıldız maskesi = orijinal - yıldızsız
                    orig = self._history.current_image()
                    if orig is not None:
                        self._starmask_image = np.clip(orig - starless, 0, 1).astype(np.float32)
                        self._show_image_tab(
                            "starmask", self._starmask_canvas,
                            self._starmask_image, "✦ Yıldız Maskesi (StarNet)")

                    # Ana sekmeye geri dön
                    self._tab_widget.setCurrentIndex(0)
                    self._status("✓ StarNet++ tamamlandı")
                else:
                    raise FileNotFoundError("StarNet çıktı dosyası oluşturulamadı.")

        except Exception as e:
            QMessageBox.critical(self, "StarNet Hatası", str(e))
            self._status("StarNet hatası")

    def _run_astap(self):
        if self._current_image is None:
            QMessageBox.warning(self, "ASTAP", "Önce bir görüntü açın.")
            return
        from ai.astap_bridge import solve_image
        astap_exe = SM.get("astap_exe", "")
        if not astap_exe or not os.path.exists(astap_exe):
            QMessageBox.warning(
                self, "ASTAP",
                "ASTAP yolu ayarlanmamış.\n"
                "Düzenle → Ayarlar → ASTAP sekmesinden yolu girin."
            )
            return

        self._status("Plate solving…")
        QApplication.processEvents()
        try:
            result = solve_image(
                self._current_image,
                astap_exe=astap_exe,
                db_path=SM.get("astap_db", ""),
                radius=SM.get("astap_radius", 30.0),
                min_stars=SM.get("astap_min_stars", 10),
                timeout=SM.get("astap_timeout", 120),
            )
            if result:
                ra  = result.get("ra",  "?")
                dec = result.get("dec", "?")
                self._status(f"✓ Plate solved — RA: {ra}  Dec: {dec}")
                QMessageBox.information(
                    self, "Plate Solve Sonucu",
                    f"RA:  {ra}\nDec: {dec}\n\n"
                    + "\n".join(f"{k}: {v}" for k, v in result.items()
                                if k not in ("ra", "dec"))
                )
            else:
                self._status("Plate solve başarısız")
                QMessageBox.warning(self, "ASTAP", "Çözüm bulunamadı.")
        except Exception as e:
            self._status("ASTAP hatası")
            QMessageBox.critical(self, "ASTAP Hatası", str(e))

    # ── Ayarlar ──────────────────────────────────────────────────────────────

    def _open_settings(self):
        dlg = SettingsDialog(self)
        if dlg.exec():
            self._apply_theme()
            self._canvas.set_interpolation(SM.get("canvas_interp", "nearest"))

    # ── İşlem Kilidi (canvas zoom/pan hâlâ çalışır) ────────────────────────

    def _set_processing_lock(self, locked: bool):
        """Toolbar ve panelleri kilitle/aç ama canvas'ı etkileşime açık bırak."""
        disabled = locked
        self.menuBar().setEnabled(not disabled)
        for act in self._panel_actions.values():
            act.setEnabled(not disabled)
        if self._panel_container is not None:
            self._panel_container.setEnabled(not disabled)

    # ── Yardımcılar ──────────────────────────────────────────────────────────

    def _status(self, msg: str):
        self._status_label.setText(msg)

    def closeEvent(self, event):
        SM.save()
        super().closeEvent(event)
