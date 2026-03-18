"""
Astro Maestro Pro — Update Dialog
===================================
Güncelleme kontrol ve indirme paneli.

Özellikler:
  • Mevcut / yeni sürüm karşılaştırması
  • Changelog gösterimi
  • Arka planda indirme + progress bar
  • İndirme tamamlanınca klasörü aç / kurulumu başlat
  • Otomatik başlatma (startup check)
"""

import os
import sys
import threading
import zipfile
import tempfile
import shutil

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QProgressBar, QWidget, QFrame, QMessageBox,
    QGroupBox, QCheckBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QUrl
from PyQt6.QtGui import QFont, QDesktopServices

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
    f"border-radius:4px;padding:4px 12px;font-size:10px;}}"
    f"QPushButton:hover{{background:{BG4};border:1px solid {ACCENT};}}"
    f"QPushButton:pressed{{background:{BG};}}"
    f"QPushButton:disabled{{color:{SUBTEXT};}}"
)
_RUN = (
    f"QPushButton{{background:{GREEN};color:#000;border:none;"
    f"border-radius:4px;padding:5px 18px;font-size:11px;font-weight:700;}}"
    f"QPushButton:hover{{background:#5ad48a;}}"
    f"QPushButton:pressed{{background:#2a9a50;}}"
    f"QPushButton:disabled{{background:{BG3};color:{SUBTEXT};}}"
)
_GRP = (
    f"QGroupBox{{background:{BG3};border:1px solid {BORDER};"
    f"border-radius:6px;margin-top:14px;padding:8px;}}"
    f"QGroupBox::title{{color:{HEAD};font-size:10px;font-weight:700;"
    f"subcontrol-origin:margin;left:8px;padding:0 4px;}}"
)


# ─────────────────────────────────────────────────────────────────────────────
#  Kontrol worker'ı
# ─────────────────────────────────────────────────────────────────────────────
class _CheckWorker(QThread):
    done = pyqtSignal(dict)

    def run(self):
        from core.version import check_for_updates
        self.done.emit(check_for_updates())


# ─────────────────────────────────────────────────────────────────────────────
#  İndirme worker'ı
# ─────────────────────────────────────────────────────────────────────────────
class _DownloadWorker(QThread):
    progress  = pyqtSignal(int, int, str)   # downloaded, total, msg
    finished  = pyqtSignal(str)             # indirilen dosya yolu
    error     = pyqtSignal(str)

    def __init__(self, url: str, dest_dir: str):
        super().__init__()
        self.url      = url
        self.dest_dir = dest_dir

    def run(self):
        try:
            import urllib.request
            from core.version import APP_VERSION

            fname   = f"AstroMaestroPro_update.zip"
            outpath = os.path.join(self.dest_dir, fname)

            self.progress.emit(0, 0, "Bağlanıyor…")

            def _hook(count, block, total):
                downloaded = count * block
                pct_msg = (f"{downloaded/1024/1024:.1f} MB"
                           + (f" / {total/1024/1024:.1f} MB"
                              if total > 0 else ""))
                if not self.isInterruptionRequested():
                    self.progress.emit(downloaded, max(total, 1), pct_msg)

            urllib.request.urlretrieve(self.url, outpath, reporthook=_hook)
            self.finished.emit(outpath)

        except Exception as e:
            self.error.emit(str(e))


# ─────────────────────────────────────────────────────────────────────────────
#  Ana Update Dialog
# ─────────────────────────────────────────────────────────────────────────────
class UpdateDialog(QDialog):
    """
    Güncelleme kontrol + indirme dialogu.

    Kullanım:
        dlg = UpdateDialog(parent=self)
        dlg.exec()

        # Otomatik kontrol (başlangıçta):
        UpdateDialog.check_on_startup(parent=self)
    """

    def __init__(self, parent=None, auto_check: bool = True):
        super().__init__(parent)
        self.setWindowTitle("🔄  Güncelleme Kontrolü")
        self.setMinimumSize(520, 560)
        self.resize(540, 600)
        self.setStyleSheet(f"background:{BG};color:{TEXT};font-size:11px;")

        self._check_worker    = None
        self._download_worker = None
        self._update_info     = None
        self._download_path   = None

        self._build()
        if auto_check:
            QTimer.singleShot(300, self._start_check)

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        # Başlık
        hdr = QLabel("🔄  Güncelleme Yöneticisi")
        hdr.setStyleSheet(
            f"color:{ACCENT2};font-size:15px;font-weight:700;"
            f"letter-spacing:1px;padding:4px 0 8px 0;")
        root.addWidget(hdr)

        # ── Sürüm bilgi kartı ────────────────────────────────────────────────
        ver_grp = QGroupBox("Sürüm Bilgisi")
        ver_grp.setStyleSheet(_GRP)
        vgl = QHBoxLayout(ver_grp)

        from core.version import APP_VERSION, APP_BUILD_DATE
        cur_col = QVBoxLayout()
        cur_col.addWidget(self._lbl("Mevcut Sürüm", MUTED))
        self._lbl_current = QLabel(f"v{APP_VERSION}")
        self._lbl_current.setStyleSheet(
            f"color:{ACCENT2};font-size:18px;font-weight:700;")
        cur_col.addWidget(self._lbl_current)
        cur_col.addWidget(self._lbl(f"Yapı: {APP_BUILD_DATE}", SUBTEXT))
        vgl.addLayout(cur_col)

        arrow = QLabel("→")
        arrow.setStyleSheet(f"color:{SUBTEXT};font-size:22px;padding:0 16px;")
        vgl.addWidget(arrow)

        new_col = QVBoxLayout()
        new_col.addWidget(self._lbl("Yeni Sürüm", MUTED))
        self._lbl_remote = QLabel("Kontrol ediliyor…")
        self._lbl_remote.setStyleSheet(
            f"color:{MUTED};font-size:18px;font-weight:700;")
        new_col.addWidget(self._lbl_remote)
        self._lbl_status = QLabel("")
        self._lbl_status.setStyleSheet(f"color:{MUTED};font-size:9px;")
        new_col.addWidget(self._lbl_status)
        vgl.addLayout(new_col)
        vgl.addStretch()

        root.addWidget(ver_grp)

        # ── Changelog ────────────────────────────────────────────────────────
        cl_grp = QGroupBox("Yenilikler / Changelog")
        cl_grp.setStyleSheet(_GRP)
        cgl = QVBoxLayout(cl_grp)
        self._txt_changelog = QTextEdit()
        self._txt_changelog.setReadOnly(True)
        self._txt_changelog.setFixedHeight(200)
        self._txt_changelog.setStyleSheet(
            f"QTextEdit{{background:{BG2};color:{TEXT};"
            f"border:1px solid {BORDER};border-radius:4px;"
            f"font-family:'Consolas','Courier New',monospace;"
            f"font-size:10px;padding:6px;}}")

        from core.version import CHANGELOG
        self._txt_changelog.setPlainText(CHANGELOG.strip())
        cgl.addWidget(self._txt_changelog)
        root.addWidget(cl_grp)

        # ── İndirme ilerleme ─────────────────────────────────────────────────
        dl_grp = QGroupBox("İndirme")
        dl_grp.setStyleSheet(_GRP)
        dll = QVBoxLayout(dl_grp)

        self._pbar = QProgressBar()
        self._pbar.setRange(0, 100)
        self._pbar.setValue(0)
        self._pbar.setFixedHeight(18)
        self._pbar.setTextVisible(True)
        self._pbar.setFormat("Hazır")
        self._pbar.setStyleSheet(
            f"QProgressBar{{background:{BG2};border:1px solid {BORDER};"
            f"border-radius:4px;color:{TEXT};font-size:9px;text-align:center;}}"
            f"QProgressBar::chunk{{background:qlineargradient("
            f"x1:0,y1:0,x2:1,y2:0,stop:0 {BORDER2},stop:1 {ACCENT});"
            f"border-radius:3px;}}")
        dll.addWidget(self._pbar)

        self._lbl_dl_status = QLabel("")
        self._lbl_dl_status.setStyleSheet(f"color:{MUTED};font-size:9px;")
        dll.addWidget(self._lbl_dl_status)

        root.addWidget(dl_grp)

        # ── Seçenekler ───────────────────────────────────────────────────────
        self._chk_startup = QCheckBox(
            "Program başlarken güncelleme kontrolü yap")
        self._chk_startup.setChecked(True)
        self._chk_startup.setStyleSheet(
            f"QCheckBox{{color:{HEAD};font-size:10px;spacing:5px;}}"
            f"QCheckBox::indicator{{width:13px;height:13px;border-radius:2px;"
            f"border:1px solid {BORDER};background:{BG};}}"
            f"QCheckBox::indicator:checked{{background:{ACCENT};"
            f"border:1px solid {ACCENT2};}}")
        root.addWidget(self._chk_startup)

        root.addStretch()

        # ── Butonlar ─────────────────────────────────────────────────────────
        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"color:{BORDER};")
        root.addWidget(sep)

        btn_row = QHBoxLayout(); btn_row.setSpacing(8)

        self._btn_check = QPushButton("🔍 Tekrar Kontrol Et")
        self._btn_check.setStyleSheet(_BTN)
        self._btn_check.setFixedHeight(30)
        self._btn_check.clicked.connect(self._start_check)
        btn_row.addWidget(self._btn_check)

        self._btn_release = QPushButton("🌐 Release Sayfası")
        self._btn_release.setStyleSheet(_BTN)
        self._btn_release.setFixedHeight(30)
        self._btn_release.setEnabled(False)
        self._btn_release.clicked.connect(self._open_release_page)
        btn_row.addWidget(self._btn_release)

        btn_row.addStretch()

        b_close = QPushButton("Kapat")
        b_close.setStyleSheet(_BTN)
        b_close.setFixedHeight(32)
        b_close.clicked.connect(self.reject)
        btn_row.addWidget(b_close)

        self._btn_download = QPushButton("⬇ Güncellemeyi İndir")
        self._btn_download.setStyleSheet(_RUN)
        self._btn_download.setFixedHeight(34)
        self._btn_download.setEnabled(False)
        self._btn_download.clicked.connect(self._start_download)
        btn_row.addWidget(self._btn_download)

        root.addLayout(btn_row)

    def _lbl(self, text, color=None):
        l = QLabel(text)
        c = color or MUTED
        l.setStyleSheet(f"color:{c};font-size:9px;")
        return l

    # ── Güncelleme kontrolü ─────────────────────────────────────────────────
    def _start_check(self):
        self._btn_check.setEnabled(False)
        self._btn_check.setText("⏳ Kontrol ediliyor…")
        self._btn_download.setEnabled(False)
        self._lbl_remote.setText("Kontrol ediliyor…")
        self._lbl_remote.setStyleSheet(
            f"color:{MUTED};font-size:18px;font-weight:700;")
        self._lbl_status.setText("")
        self._pbar.setFormat("Hazır"); self._pbar.setValue(0)
        self._lbl_dl_status.setText("")

        from core.version import GITHUB_REPO
        if not GITHUB_REPO:
            # GitHub deposu ayarlı değil — sadece mevcut changelog göster
            self._lbl_remote.setText("—")
            self._lbl_status.setText("GitHub deposu ayarlı değil.\nGüncelleme kontrolü devre dışı.")
            self._lbl_status.setStyleSheet(f"color:{SUBTEXT};font-size:9px;")
            self._btn_check.setEnabled(True)
            self._btn_check.setText("🔍 Tekrar Kontrol Et")
            return

        self._check_worker = _CheckWorker()
        self._check_worker.done.connect(self._on_check_done)
        self._check_worker.start()

    def _on_check_done(self, info: dict):
        self._btn_check.setEnabled(True)
        self._btn_check.setText("🔍 Tekrar Kontrol Et")
        self._update_info = info

        if info.get("error"):
            self._lbl_remote.setText("?")
            self._lbl_remote.setStyleSheet(
                f"color:{GOLD};font-size:18px;font-weight:700;")
            self._lbl_status.setText(f"Kontrol hatası: {info['error'][:80]}")
            self._lbl_status.setStyleSheet(f"color:{GOLD};font-size:9px;")
            return

        remote_ver = info.get("version", "?")
        self._lbl_remote.setText(f"v{remote_ver}")

        if info.get("available"):
            self._lbl_remote.setStyleSheet(
                f"color:{GREEN};font-size:18px;font-weight:700;")
            self._lbl_status.setText("✅ Yeni sürüm mevcut!")
            self._lbl_status.setStyleSheet(f"color:{GREEN};font-size:10px;font-weight:700;")
            self._btn_download.setEnabled(True)
            if info.get("url"):
                self._btn_release.setEnabled(True)
            # Release notes'u göster
            if info.get("notes"):
                self._txt_changelog.setPlainText(
                    f"v{remote_ver} — Yenilikler:\n\n{info['notes']}")
        else:
            self._lbl_remote.setStyleSheet(
                f"color:{ACCENT2};font-size:18px;font-weight:700;")
            self._lbl_status.setText("✅  Güncel sürümü kullanıyorsunuz.")
            self._lbl_status.setStyleSheet(f"color:{MUTED};font-size:9px;")

    # ── İndirme ──────────────────────────────────────────────────────────────
    def _start_download(self):
        if not self._update_info:
            return
        url = self._update_info.get("download_url", "")
        if not url:
            QMessageBox.warning(self, "İndirme", "İndirme linki bulunamadı.")
            return

        # İndirme klasörü — masaüstü
        dest = (os.path.expanduser("~/Desktop") if
                os.path.isdir(os.path.expanduser("~/Desktop"))
                else tempfile.gettempdir())

        self._btn_download.setEnabled(False)
        self._btn_download.setText("⏳ İndiriliyor…")
        self._pbar.setRange(0, 100)
        self._pbar.setValue(0)
        self._pbar.setFormat("Bağlanıyor…")

        self._download_worker = _DownloadWorker(url, dest)
        self._download_worker.progress.connect(self._on_dl_progress)
        self._download_worker.finished.connect(self._on_dl_done)
        self._download_worker.error.connect(self._on_dl_error)
        self._download_worker.start()

    def _on_dl_progress(self, downloaded: int, total: int, msg: str):
        if total > 0:
            pct = int(downloaded * 100 / total)
            self._pbar.setValue(pct)
            self._pbar.setFormat(f"%p%  —  {msg}")
        else:
            self._pbar.setRange(0, 0)
            self._pbar.setFormat(msg)
        self._lbl_dl_status.setText(msg)

    def _on_dl_done(self, path: str):
        self._download_path = path
        self._pbar.setRange(0, 100)
        self._pbar.setValue(100)
        self._pbar.setFormat("✅  İndirme tamamlandı!")
        self._btn_download.setText("✅ İndirildi")
        self._lbl_dl_status.setText(f"Konum: {path}")

        ver = self._update_info.get("version", "?")
        reply = QMessageBox.question(
            self,
            "İndirme Tamamlandı",
            f"v{ver} başarıyla indirildi!\n\n"
            f"Konum: {path}\n\n"
            "Klasörü açmak ister misiniz?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            folder = os.path.dirname(path)
            QDesktopServices.openUrl(QUrl.fromLocalFile(folder))

    def _on_dl_error(self, msg: str):
        self._pbar.setRange(0, 100)
        self._pbar.setValue(0)
        self._pbar.setFormat("❌  Hata!")
        self._btn_download.setEnabled(True)
        self._btn_download.setText("⬇ Güncellemeyi İndir")
        self._lbl_dl_status.setText(f"Hata: {msg[:100]}")
        QMessageBox.critical(self, "İndirme Hatası", msg[:400])

    def _open_release_page(self):
        if self._update_info and self._update_info.get("url"):
            QDesktopServices.openUrl(
                QUrl(self._update_info["url"]))

    # ── Statik başlangıç kontrolü ────────────────────────────────────────────
    @staticmethod
    def check_on_startup(parent=None, settings: dict = None):
        """
        Program başlarken çağrılır. Güncelleme varsa bildirim gösterir.
        settings["check_updates_on_startup"] = False ise çalışmaz.
        """
        if settings and not settings.get("check_updates_on_startup", True):
            return

        from core.version import GITHUB_REPO
        if not GITHUB_REPO:
            return

        def _bg_check():
            from core.version import check_for_updates
            info = check_for_updates(timeout=5)
            if info.get("available"):
                # Ana thread'e geri dön
                QTimer.singleShot(0, lambda: _show_notification(info))

        def _show_notification(info):
            ver = info.get("version", "?")
            reply = QMessageBox.question(
                parent,
                "🔄  Güncelleme Mevcut",
                f"Astro Maestro Pro v{ver} mevcut!\n\n"
                f"Şu an v{_cur()} kullanıyorsunuz.\n\n"
                "Güncelleme detaylarını görmek ister misiniz?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                dlg = UpdateDialog(parent=parent, auto_check=False)
                dlg._update_info = info
                dlg._on_check_done(info)
                dlg.exec()

        t = threading.Thread(target=_bg_check, daemon=True)
        t.start()


def _cur() -> str:
    from core.version import APP_VERSION
    return APP_VERSION
