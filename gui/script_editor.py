"""
Astro Mastro Pro — Python Script Editor
========================================
Kullanici kendi Python scriptlerini yazip calistirabiliir.

Script API:
  image  : numpy float32 [0,1]  — islenen resim (okuma/yazma)
  result : sonuc degiskeni — script bunu set etmeli
  np     : numpy
  cv2    : opencv

Ornek:
  result = np.clip(image * 1.2, 0, 1)
"""

import os, sys, traceback, json
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QMessageBox, QSplitter, QListWidget, QListWidgetItem,
    QTextEdit, QWidget, QToolBar, QComboBox, QInputDialog, QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import (
    QFont, QColor, QTextCharFormat, QSyntaxHighlighter,
    QTextDocument, QKeySequence, QShortcut
)

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
TEXT    = "#e8f0ff"
MUTED   = "#80a8c8"
HEAD    = "#c0e0ff"
SUBTEXT = "#506880"

EDITOR_CSS = f"""
    QTextEdit {{
        background: {BG};
        color: #d4e8ff;
        border: 1px solid {BORDER};
        border-top: 1px solid {BORDER2};
        border-radius: 2px;
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 12px;
        selection-background-color: {BG4};
        padding: 6px;
    }}
"""

BTN_CSS = (
    f"QPushButton{{"
    f"  background: qlineargradient(x1:0,y1:0,x2:0,y2:1,"
    f"    stop:0 {BG3}, stop:1 {BG});"
    f"  color:{TEXT}; border:1px solid {BORDER};"
    f"  border-top:1px solid {BORDER2};"
    f"  border-radius:2px; padding:4px 12px; font-size:10px; font-weight:600;}}"
    f"QPushButton:hover{{"
    f"  background: qlineargradient(x1:0,y1:0,x2:0,y2:1,"
    f"    stop:0 {BG4}, stop:1 {BG3});"
    f"  border:1px solid {ACCENT}; border-top:1px solid {ACCENT2};}}"
    f"QPushButton:pressed{{background:{BG};}}"
    f"QPushButton:disabled{{color:{SUBTEXT};}}"
)
RUN_CSS = (
    f"QPushButton{{"
    f"  background: qlineargradient(x1:0,y1:0,x2:0,y2:1,"
    f"    stop:0 #0a4a0a, stop:1 #0a2a0a);"
    f"  color:{GREEN}; border:1px solid {GREEN};"
    f"  border-radius:2px; padding:4px 16px;"
    f"  font-size:11px; font-weight:700;}}"
    f"QPushButton:hover{{"
    f"  background: qlineargradient(x1:0,y1:0,x2:0,y2:1,"
    f"    stop:0 #0f5f0f, stop:1 #0a3a0a);"
    f"  border:1px solid #5ad48a;}}"
    f"QPushButton:pressed{{background:{BG};}}"
)

TEMPLATE_SCRIPTS = {
    "Bos Script": (
        "# Astro Mastro Pro — Python Script\n"
        "# image : float32 numpy array [0,1], shape (H,W,3) or (H,W)\n"
        "# result: bu degiskene sonucu atayın\n"
        "# np, cv2 otomatik yuklenir\n\n"
        "result = image.copy()\n"
    ),
    "Parlaklik Artir": (
        "# Parlaklik artir\n"
        "result = np.clip(image * 1.3, 0, 1)\n"
    ),
    "Kontrast Artir": (
        "# Kontrast — S-curve\n"
        "mid = 0.5\n"
        "strength = 0.5\n"
        "result = np.clip(mid + (image - mid) * (1 + strength), 0, 1)\n"
    ),
    "Gamma Duzeltme": (
        "# Gamma duzeltme\n"
        "gamma = 0.5  # <1 aciklas, >1 koyulas\n"
        "result = np.power(np.clip(image, 1e-9, 1), gamma)\n"
    ),
    "Renk Dengesi": (
        "# R/G/B kanallarini ayarla\n"
        "result = image.copy()\n"
        "result[:,:,0] = np.clip(result[:,:,0] * 1.05, 0, 1)  # R +5%\n"
        "result[:,:,1] = np.clip(result[:,:,1] * 1.00, 0, 1)  # G\n"
        "result[:,:,2] = np.clip(result[:,:,2] * 0.95, 0, 1)  # B -5%\n"
    ),
    "Gaussian Blur": (
        "# Gaussian blur\n"
        "import cv2\n"
        "sigma = 2.0\n"
        "if image.ndim == 3:\n"
        "    result = cv2.GaussianBlur(image, (0,0), sigma)\n"
        "else:\n"
        "    result = cv2.GaussianBlur(image, (0,0), sigma)\n"
        "result = np.clip(result, 0, 1)\n"
    ),
    "Histogram Gerdirme": (
        "# Manuel percentile stretch\n"
        "lo = np.percentile(image, 1)\n"
        "hi = np.percentile(image, 99)\n"
        "result = np.clip((image - lo) / max(hi - lo, 1e-9), 0, 1)\n"
    ),
    "Gri Tonlamaya Cevir": (
        "# RGB'yi gri tonlamaya cevir\n"
        "if image.ndim == 3:\n"
        "    gray = image.mean(axis=2)\n"
        "    result = np.stack([gray, gray, gray], axis=2)\n"
        "else:\n"
        "    result = image.copy()\n"
    ),
    "Keskinlestir (Unsharp)": (
        "# Unsharp mask ile keskinlestir\n"
        "import cv2\n"
        "radius = 2.0\n"
        "amount = 1.5\n"
        "if image.ndim == 3:\n"
        "    blur = cv2.GaussianBlur(image, (0,0), radius)\n"
        "else:\n"
        "    blur = cv2.GaussianBlur(image, (0,0), radius)\n"
        "result = np.clip(image + (image - blur) * amount, 0, 1)\n"
    ),
}


# ── Syntax Highlighter ────────────────────────────────────────────────────────
class PythonHighlighter(QSyntaxHighlighter):
    def __init__(self, doc):
        super().__init__(doc)
        self._rules = []

        def fmt(color, bold=False, italic=False):
            f = QTextCharFormat()
            f.setForeground(QColor(color))
            if bold:   f.setFontWeight(700)
            if italic: f.setFontItalic(True)
            return f

        import re
        kw_fmt  = fmt("#569cd6", bold=True)
        str_fmt = fmt("#ce9178")
        com_fmt = fmt("#6a9955", italic=True)
        num_fmt = fmt("#b5cea8")
        fn_fmt  = fmt("#dcdcaa")
        cls_fmt = fmt("#4ec9b0", bold=True)
        self_fmt= fmt("#9cdcfe")
        bi_fmt  = fmt("#4ec9b0")

        keywords = r"\b(def|class|return|import|from|if|elif|else|for|while|" \
                   r"try|except|finally|with|as|in|not|and|or|is|True|False|None|" \
                   r"lambda|yield|global|nonlocal|pass|break|continue|raise|del|" \
                   r"assert|async|await)\b"
        builtins = r"\b(print|len|range|int|float|str|list|dict|tuple|set|bool|" \
                   r"type|isinstance|hasattr|getattr|setattr|enumerate|zip|map|" \
                   r"filter|sorted|min|max|sum|abs|round|open|super|object)\b"

        self._rules = [
            (re.compile(keywords),   kw_fmt),
            (re.compile(builtins),   bi_fmt),
            (re.compile(r"\bself\b"),self_fmt),
            (re.compile(r"\bdef\s+(\w+)"), fn_fmt),
            (re.compile(r"\bclass\s+(\w+)"),cls_fmt),
            (re.compile(r'"[^"\\]*(?:\\.[^"\\]*)*"'),  str_fmt),
            (re.compile(r"'[^'\\]*(?:\\.[^'\\]*)*'"),  str_fmt),
            (re.compile(r'"""[\s\S]*?"""'),              str_fmt),
            (re.compile(r"'''[\s\S]*?'''"),              str_fmt),
            (re.compile(r"#.*$", re.MULTILINE),         com_fmt),
            (re.compile(r"\b\d+\.?\d*\b"),              num_fmt),
        ]

    def highlightBlock(self, text):
        import re
        for pattern, fmt in self._rules:
            for m in pattern.finditer(text):
                self.setFormat(m.start(), m.end()-m.start(), fmt)


# ── Script Runner Thread ──────────────────────────────────────────────────────
class ScriptRunner(QThread):
    finished = pyqtSignal(object)   # result ndarray
    error    = pyqtSignal(str)
    log      = pyqtSignal(str)

    def __init__(self, code, image):
        super().__init__()
        self.code  = code
        self.image = image

    def run(self):
        import numpy as np
        import io, contextlib
        try:
            import cv2
        except ImportError:
            cv2 = None

        log_buf = io.StringIO()
        # Ekstra kutuphaneler (varsa yukle)
        extra = {}
        try:
            from skimage import exposure as _exposure
            extra["exposure"] = _exposure
        except ImportError:
            pass
        try:
            from scipy import ndimage as _ndimage
            extra["ndimage"] = _ndimage
        except ImportError:
            pass
        ns = {
            "image":  self.image.copy(),
            "result": None,
            "np":     np,
            "numpy":  np,
            "cv2":    cv2,
            "os":     __import__("os"),
            "math":   __import__("math"),
            "print":  lambda *a, **kw: (log_buf.write(" ".join(str(x) for x in a) + "\n"),
                                        self.log.emit(" ".join(str(x) for x in a))),
            **extra,
        }
        try:
            exec(compile(self.code, "<script>", "exec"), ns)
            out = log_buf.getvalue()
            if out: self.log.emit(out)
            result = ns.get("result")
            if result is None:
                self.error.emit(
                    "Script 'result' degiskenini set etmedi.\n\n"
                    "Ornekler:\n"
                    "  result = image.copy()\n"
                    "  result = np.clip(image * 1.5, 0, 1)")
                return
            import numpy as _np
            result = _np.clip(_np.array(result, dtype=_np.float32), 0, 1)
            if result.shape[:2] != self.image.shape[:2]:
                self.error.emit(
                    f"Sonuc resim boyutu uyusmuyor:\n"
                    f"  Beklenen: {self.image.shape}\n"
                    f"  Gelen:    {result.shape}")
                return
            self.finished.emit(result)
        except Exception:
            tb = traceback.format_exc()
            self.error.emit(tb)


# ── Main Dialog ───────────────────────────────────────────────────────────────
class ScriptEditorDialog(QDialog):
    """
    Script editoru dialog.
    apply_cb(ndarray) — kullanici Apply tikladiginda cagirilir
    """

    SCRIPTS_FILE = os.path.join(
        os.path.dirname(__file__), "..", "user_scripts.json")

    def __init__(self, image, apply_cb, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Astro Mastro Pro — Python Script Editor")
        self.resize(1000, 680)
        self.setStyleSheet(f"background:{BG};color:{TEXT};")
        self._image    = image.copy() if image is not None else None
        self._apply_cb = apply_cb
        self._result   = None
        self._runner   = None
        self._scripts  = {}   # name → code
        self._current  = None
        self._apply_count = 0

        self._load_scripts()
        self._build_ui()
        self._refresh_list()

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0,0,0,0); root.setSpacing(0)

        # ── Toolbar ───────────────────────────────────────────────────────────
        tb = QWidget(); tb.setFixedHeight(38)
        tb.setStyleSheet(f"background:{BG3};border-bottom:1px solid {BORDER};")
        tbl = QHBoxLayout(tb); tbl.setContentsMargins(8,4,8,4); tbl.setSpacing(6)

        # Template combo
        tbl.addWidget(QLabel("Sablon:")); tbl.itemAt(0).widget().setStyleSheet(f"color:{MUTED};font-size:10px;")
        self._tmpl_cb = QComboBox()
        self._tmpl_cb.addItems(list(TEMPLATE_SCRIPTS.keys()))
        self._tmpl_cb.setStyleSheet(
            f"QComboBox{{background:{BG};color:{TEXT};border:1px solid {BORDER};"
            f"border-radius:3px;padding:2px 6px;font-size:10px;}}"
            f"QComboBox QAbstractItemView{{background:{BG2};color:{TEXT};"
            f"selection-background-color:{BG4};}}")
        self._tmpl_cb.setFixedWidth(180)
        b_load_tmpl = QPushButton("Yukle")
        b_load_tmpl.setStyleSheet(BTN_CSS); b_load_tmpl.setFixedHeight(26)
        b_load_tmpl.clicked.connect(self._load_template)
        tbl.addWidget(self._tmpl_cb); tbl.addWidget(b_load_tmpl)
        tbl.addSpacing(12)

        b_new   = QPushButton("+ Yeni");   b_new.setStyleSheet(BTN_CSS);   b_new.setFixedHeight(26)
        b_save  = QPushButton("💾 Kaydet"); b_save.setStyleSheet(BTN_CSS);  b_save.setFixedHeight(26)
        b_del   = QPushButton("🗑 Sil");    b_del.setStyleSheet(BTN_CSS);   b_del.setFixedHeight(26)
        b_imp   = QPushButton("📂 Ac");     b_imp.setStyleSheet(BTN_CSS);   b_imp.setFixedHeight(26)
        b_exp   = QPushButton("💾 Dosyaya Kaydet"); b_exp.setStyleSheet(BTN_CSS); b_exp.setFixedHeight(26)
        for b in (b_new, b_save, b_del, b_imp, b_exp): tbl.addWidget(b)
        tbl.addStretch()

        self._btn_run   = QPushButton("▶  Calistir")
        self._btn_run.setStyleSheet(RUN_CSS)
        self._btn_run.setFixedHeight(28)

        APPLY_CSS = (
            f"QPushButton{{background:#0a2a0a;color:{GREEN};border:1px solid {GREEN};"
            f"border-radius:4px;padding:4px 16px;font-size:11px;font-weight:700;}}"
            f"QPushButton:hover{{background:#0f4f0f;border:1px solid #5dd88a;}}"
            f"QPushButton:pressed{{background:{BG};}}"
            f"QPushButton:disabled{{color:{SUBTEXT};border-color:{BORDER};}}"
        )
        self._btn_apply = QPushButton("Resme Uygula")
        self._btn_apply.setStyleSheet(APPLY_CSS)
        self._btn_apply.setFixedHeight(28)
        self._btn_apply.setEnabled(False)
        self._btn_apply.setToolTip("Sonucu ana penceredeki resme uygula (dialog acik kalir)")

        CLOSE_CSS = (
            f"QPushButton{{background:{BG3};color:{MUTED};border:1px solid {BORDER};"
            f"border-radius:4px;padding:4px 12px;font-size:10px;}}"
            f"QPushButton:hover{{background:{RED};color:#fff;border:1px solid {RED};}}"
        )
        self._btn_close = QPushButton("Kapat")
        self._btn_close.setStyleSheet(CLOSE_CSS)
        self._btn_close.setFixedHeight(28)

        tbl.addWidget(self._btn_run)
        tbl.addWidget(self._btn_apply)
        tbl.addWidget(self._btn_close)
        root.addWidget(tb)

        b_new.clicked.connect(self._new_script)
        b_save.clicked.connect(self._save_script)
        b_del.clicked.connect(self._delete_script)
        b_imp.clicked.connect(self._import_script)
        b_exp.clicked.connect(self._export_script)
        self._btn_run.clicked.connect(self._run)
        self._btn_apply.clicked.connect(self._apply)
        self._btn_close.clicked.connect(self.accept)

        # ── Body: sol list + sağ editör+log ───────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStyleSheet(f"QSplitter::handle{{background:{BORDER};width:3px;}}")

        # Sol: script listesi
        left = QWidget(); left.setStyleSheet(f"background:{BG2};")
        ll = QVBoxLayout(left); ll.setContentsMargins(6,6,6,6); ll.setSpacing(4)
        lbl = QLabel("Scriptlerim"); lbl.setStyleSheet(f"color:{HEAD};font-size:10px;font-weight:700;")
        ll.addWidget(lbl)
        self._list = QListWidget()
        self._list.setStyleSheet(
            f"QListWidget{{background:{BG};color:{TEXT};border:1px solid {BORDER};"
            f"border-radius:4px;font-size:11px;}}"
            f"QListWidget::item{{padding:5px 8px;}}"
            f"QListWidget::item:selected{{background:{BG4};color:{ACCENT2};}}"
            f"QListWidget::item:hover{{background:{BG3};}}")
        self._list.currentTextChanged.connect(self._on_select)
        ll.addWidget(self._list, 1)
        left.setMinimumWidth(160); left.setMaximumWidth(240)
        splitter.addWidget(left)

        # Sağ: editör + log
        right_split = QSplitter(Qt.Orientation.Vertical)
        right_split.setStyleSheet(f"QSplitter::handle{{background:{BORDER};height:3px;}}")

        # Editör
        self._editor = QTextEdit()
        self._editor.setStyleSheet(EDITOR_CSS)
        self._editor.setAcceptRichText(False)
        self._editor.setTabStopDistance(32)
        self._hl = PythonHighlighter(self._editor.document())
        # Ctrl+Enter = run
        QShortcut(QKeySequence("Ctrl+Return"), self._editor, self._run)
        right_split.addWidget(self._editor)

        # Log / output
        log_wrap = QWidget(); log_wrap.setStyleSheet(f"background:{BG2};")
        lw = QVBoxLayout(log_wrap); lw.setContentsMargins(0,0,0,0); lw.setSpacing(0)
        log_hdr = QWidget(); log_hdr.setFixedHeight(24)
        log_hdr.setStyleSheet(f"background:{BG3};border-top:1px solid {BORDER};")
        lhdr_l = QHBoxLayout(log_hdr); lhdr_l.setContentsMargins(8,2,8,2)
        lhdr_l.addWidget(QLabel("Cikti / Hatalar"))
        lhdr_l.itemAt(0).widget().setStyleSheet(f"color:{SUBTEXT};font-size:9px;")
        lhdr_l.addStretch()
        b_clear_log = QPushButton("Temizle"); b_clear_log.setStyleSheet(BTN_CSS); b_clear_log.setFixedHeight(20)
        b_clear_log.clicked.connect(lambda: self._log.clear())
        lhdr_l.addWidget(b_clear_log)
        lw.addWidget(log_hdr)
        self._log = QTextEdit(); self._log.setReadOnly(True)
        self._log.setStyleSheet(
            f"QTextEdit{{background:#020810;color:{GREEN};border:none;"
            f"font-family:'Consolas','Courier New',monospace;font-size:11px;padding:4px;}}")
        lw.addWidget(self._log, 1)
        right_split.addWidget(log_wrap)
        right_split.setSizes([10000, 180])

        splitter.addWidget(right_split)
        splitter.setSizes([200, 10000])
        root.addWidget(splitter, 1)

        # Status bar
        self._status = QLabel("Hazir")
        self._status.setFixedHeight(22)
        self._status.setStyleSheet(
            f"background:{BG3};color:{MUTED};font-size:10px;"
            f"padding:0 8px;border-top:1px solid {BORDER};")
        root.addWidget(self._status)

    # ── Script management ─────────────────────────────────────────────────────
    def _load_scripts(self):
        try:
            if os.path.isfile(self.SCRIPTS_FILE):
                with open(self.SCRIPTS_FILE, encoding="utf-8") as f:
                    self._scripts = json.load(f)
        except Exception:
            self._scripts = {}
        if not self._scripts:
            self._scripts["Ornek Script"] = TEMPLATE_SCRIPTS["Bos Script"]

    def _save_scripts_file(self):
        try:
            with open(self.SCRIPTS_FILE, "w", encoding="utf-8") as f:
                json.dump(self._scripts, f, ensure_ascii=False, indent=2)
        except Exception as e:
            QMessageBox.warning(self, "Kayit Hatasi", str(e))

    def _refresh_list(self):
        self._list.blockSignals(True)
        self._list.clear()
        for name in sorted(self._scripts.keys()):
            self._list.addItem(name)
        self._list.blockSignals(False)
        if self._current and self._current in self._scripts:
            items = self._list.findItems(self._current, Qt.MatchFlag.MatchExactly)
            if items: self._list.setCurrentItem(items[0])
        elif self._list.count():
            self._list.setCurrentRow(0)

    def _on_select(self, name):
        if not name: return
        self._current = name
        code = self._scripts.get(name, "")
        self._editor.blockSignals(True)
        self._editor.setPlainText(code)
        self._editor.blockSignals(False)
        self._status.setText(f"Script: {name}")

    def _new_script(self):
        name, ok = QInputDialog.getText(self, "Yeni Script", "Script adi:")
        if not ok or not name.strip(): return
        name = name.strip()
        self._scripts[name] = TEMPLATE_SCRIPTS["Bos Script"]
        self._save_scripts_file()
        self._current = name
        self._refresh_list()

    def _save_script(self):
        if not self._current:
            name, ok = QInputDialog.getText(self, "Kaydet", "Script adi:")
            if not ok or not name.strip(): return
            self._current = name.strip()
        self._scripts[self._current] = self._editor.toPlainText()
        self._save_scripts_file()
        self._refresh_list()
        self._status.setText(f"Kaydedildi: {self._current}")

    def _delete_script(self):
        if not self._current: return
        r = QMessageBox.question(self, "Sil", f"'{self._current}' silinsin mi?")
        if r != QMessageBox.StandardButton.Yes: return
        self._scripts.pop(self._current, None)
        self._save_scripts_file()
        self._current = None
        self._refresh_list()

    def _import_script(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Python Script Ac", "", "Python Files (*.py);;All Files (*)")
        if not path: return
        name = os.path.splitext(os.path.basename(path))[0]
        with open(path, encoding="utf-8", errors="replace") as f:
            code = f.read()
        self._scripts[name] = code
        self._save_scripts_file()
        self._current = name
        self._refresh_list()

    def _export_script(self):
        if not self._current: return
        path, _ = QFileDialog.getSaveFileName(
            self, "Python Dosyasina Kaydet",
            f"{self._current}.py", "Python Files (*.py)")
        if not path: return
        with open(path, "w", encoding="utf-8") as f:
            f.write(self._editor.toPlainText())

    def _load_template(self):
        name = self._tmpl_cb.currentText()
        code = TEMPLATE_SCRIPTS.get(name, "")
        self._editor.setPlainText(code)

    # ── Run / Apply ──────────────────────────────────────────────────────────
    def _run(self):
        code = self._editor.toPlainText().strip()
        if not code:
            self._log.append("Script bos!"); return
        if self._image is None:
            self._log.append("Resim yuklenmemis!"); return
        self._log.append(f">>> Calistiriliyor...")
        self._btn_run.setEnabled(False)
        self._btn_apply.setEnabled(False)
        self._status.setText("Calistiriliyor...")
        self._result = None

        self._runner = ScriptRunner(code, self._image)
        self._runner.finished.connect(self._on_done)
        self._runner.error.connect(self._on_error)
        self._runner.log.connect(lambda m: self._log.append(m))
        self._runner.start()

    def _on_done(self, result):
        import numpy as np
        self._result = result
        h, w = result.shape[:2]
        ch = "RGB" if result.ndim == 3 else "Gray"
        self._log.append(
            f"OK  {w}x{h} {ch}  "
            f"min={result.min():.4f}  max={result.max():.4f}")
        self._btn_run.setEnabled(True)
        self._btn_apply.setEnabled(True)
        self._status.setText("Tamamlandi — 'Resme Uygula' butonuna basin")

    def _on_error(self, msg):
        self._log.append(f"\n--- HATA ---\n{msg}\n")
        self._btn_run.setEnabled(True)
        self._status.setText("Hata!")

    def _apply(self):
        """Sonucu ana penceredeki resme uygula — dialog acik kalir."""
        if self._result is None:
            self._log.append("Henuz sonuc yok — once Calistir'a basin.")
            return
        import numpy as np
        result = np.clip(self._result, 0, 1).astype(np.float32)
        # Ana pencereye uygula
        if self._apply_cb:
            self._apply_cb(result)
        self._apply_count += 1
        # Uygulanan sonucu yeni baseline olarak ayarla
        # Boylece tekrar calistirinca SON hali uzerinden calisir
        self._image = result.copy()
        self._result = None
        self._btn_apply.setEnabled(False)
        self._log.append(
            f"--- Resme uygulandi (#{self._apply_count}) ---\n"
            f"    Simdi tekrar calistirabilir veya baska script deneyebilirsiniz.\n"
            f"    Sonraki calistirma SON uygulanan resim uzerinden yapilir.")
        self._status.setText(
            f"Uygulandi (#{self._apply_count}) — tekrar calistirilabilir")
