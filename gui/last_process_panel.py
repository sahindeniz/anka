# gui/last_process_panel.py
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QFormLayout, QDoubleSpinBox, QComboBox,
    QTextEdit, QCheckBox
)
from PyQt6.QtCore import Qt
import json
import os
from processing.auto_pipeline import load_settings, run_auto_pipeline


class LastProcessPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent  # ana pencereye erişim için (current_file_path vs.)
        self.settings = load_settings()
        self.last_run_path = None
        self.last_run_time = "Henüz çalıştırılmadı"

        self.init_ui()
        self.load_last_info()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Başlık
        title = QLabel("Last Process (Son İşlem)")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        # Durum bilgisi
        self.status_label = QLabel(f"Son çalıştırma: {self.last_run_time}")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        if hasattr(self.main_window, 'current_file_path') and self.main_window.current_file_path:
            file_info = QLabel(f"Dosya: {os.path.basename(self.main_window.current_file_path)}")
            file_info.setWordWrap(True)
            layout.addWidget(file_info)

        layout.addSpacing(10)

        # Ayarlar grubu
        group = QGroupBox("Pipeline Ayarları (Son Kullanılan)")
        group_layout = QFormLayout()
        group.setLayout(group_layout)

        # Örnek ayar alanları (daha fazlası eklenebilir)
        self.cb_enabled = QCheckBox("Pipeline aktif")
        self.cb_enabled.setChecked(self.settings.get("pipeline_enabled", True))
        group_layout.addRow("Durum:", self.cb_enabled)

        self.combo_stretch = QComboBox()
        self.combo_stretch.addItems(["arcsinh", "clahe", "percentile"])
        self.combo_stretch.setCurrentText(self.settings.get("stretch_type", "arcsinh"))
        group_layout.addRow("Stretch türü:", self.combo_stretch)

        self.spin_arcsinh = QDoubleSpinBox()
        self.spin_arcsinh.setRange(1.0, 10.0)
        self.spin_arcsinh.setSingleStep(0.5)
        self.spin_arcsinh.setValue(self.settings.get("arcsinh_factor", 4.0))
        group_layout.addRow("Arcsinh faktör:", self.spin_arcsinh)

        self.spin_saturation = QDoubleSpinBox()
        self.spin_saturation.setRange(0.8, 2.0)
        self.spin_saturation.setSingleStep(0.05)
        self.spin_saturation.setValue(self.settings.get("color_saturation_boost", 1.30))
        group_layout.addRow("Renk doygunluk:", self.spin_saturation)

        self.spin_sharpen = QDoubleSpinBox()
        self.spin_sharpen.setRange(0.0, 2.0)
        self.spin_sharpen.setSingleStep(0.05)
        self.spin_sharpen.setValue(self.settings.get("sharpen_amount", 1.15))
        group_layout.addRow("Keskinleştirme:", self.spin_sharpen)

        layout.addWidget(group)

        # Butonlar
        btn_layout = QHBoxLayout()
        self.btn_run = QPushButton("Şimdi Tekrar Çalıştır (Last Process)")
        self.btn_run.clicked.connect(self.run_last_pipeline)
        btn_layout.addWidget(self.btn_run)

        self.btn_save = QPushButton("Ayarları Kaydet")
        self.btn_save.clicked.connect(self.save_settings)
        btn_layout.addWidget(self.btn_save)

        layout.addLayout(btn_layout)
        layout.addStretch()

        # Log alanı (isteğe bağlı)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(120)
        layout.addWidget(QLabel("Son işlem log:"))
        layout.addWidget(self.log_text)

    def load_last_info(self):
        # İstersen settings.json'a son çalıştırma bilgisi de eklenebilir
        # Şimdilik basitçe varsayılan göster
        pass

    def run_last_pipeline(self):
        if not hasattr(self.main_window, 'current_file_path') or not self.main_window.current_file_path:
            self.log_text.append("❌ Açık bir dosya yok!")
            return

        input_path = self.main_window.current_file_path

        # Güncel ayarları topla
        current_settings = self.settings.copy()
        current_settings["pipeline_enabled"] = self.cb_enabled.isChecked()
        current_settings["stretch_type"] = self.combo_stretch.currentText()
        current_settings["arcsinh_factor"] = self.spin_arcsinh.value()
        current_settings["color_saturation_boost"] = self.spin_saturation.value()
        current_settings["sharpen_amount"] = self.spin_sharpen.value()

        self.log_text.clear()
        self.log_text.append(f"İşlem başlatılıyor: {os.path.basename(input_path)}")

        try:
            new_path = run_auto_pipeline(input_path, custom_settings=current_settings)
            self.log_text.append(f"✓ Tamamlandı → {os.path.basename(new_path)}")
            self.last_run_path = new_path
            self.last_run_time = "Az önce"
            self.status_label.setText(f"Son çalıştırma: {self.last_run_time}")

            # Ana pencerede yeni dosyayı aç/göster
            if hasattr(self.main_window, 'load_image'):
                self.main_window.load_image(new_path)

        except Exception as e:
            self.log_text.append(f"✗ Hata: {str(e)}")

    def save_settings(self):
        # Değişiklikleri settings.json'a geri yaz
        try:
            settings_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "settings.json")
            with open(settings_path, 'r', encoding='utf-8') as f:
                full_settings = json.load(f)

            # pipeline bölümünü güncelle
            pipeline = {
                "enabled": self.cb_enabled.isChecked(),
                "stretch_type": self.combo_stretch.currentText(),
                "arcsinh_factor": self.spin_arcsinh.value(),
                "color_saturation_boost": self.spin_saturation.value(),
                "sharpen_amount": self.spin_sharpen.value(),
                # diğer ayarlar değişmediyse korunur
            }
            full_settings["pipeline"] = pipeline

            with open(settings_path, 'w', encoding='utf-8') as f:
                json.dump(full_settings, f, indent=2, ensure_ascii=False)

            self.log_text.append("Ayarlar kaydedildi.")
            self.settings = load_settings()  # yenile

        except Exception as e:
            self.log_text.append(f"Kaydetme hatası: {str(e)}")
