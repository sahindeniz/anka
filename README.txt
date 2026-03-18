╔══════════════════════════════════════════════════════════════════════════════╗
║                 ASTRO MAESTRO PRO  —  Kullanım Kılavuzu                    ║
║                         by Deniz  |  v1.0.0                                ║
╚══════════════════════════════════════════════════════════════════════════════╝


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  HIZLI BAŞLANGIÇ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  İLK KEZ KURULUM:
    1. install.bat dosyasına çift tıklayın
    2. Kurulum otomatik tamamlanır (~3-8 dakika, internet gerekir)
    3. Masaüstündeki "Astro Maestro Pro" kısayolundan başlatın

  SONRAKI BAŞLATMALAR:
    • Masaüstündeki kısayola çift tıklayın
    • veya: setup_and_run.bat'a çift tıklayın

  KALDIRMA:
    • uninstall.bat'ı çalıştırın


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  SİSTEM GEREKSİNİMLERİ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  İŞLETİM SİSTEMİ:
    • Windows 10 / 11 (64-bit) — ÖNERİLEN
    • macOS 12+ (Monterey ve üzeri)
    • Linux Ubuntu 20.04+ / Debian 11+

  YAZILIM:
    • Python 3.10, 3.11 veya 3.12  →  python.org/downloads
      ⚠  Kurulumda "Add Python to PATH" seçeneğini işaretleyin!

  DONANIM:
    • RAM: minimum 4 GB (8 GB önerilir)
    • Disk: minimum 2 GB boş alan
    • GPU: opsiyonel (StarNet++ GPU modu için)

  PYTHON PAKETLERI (install.bat otomatik kurar):
    numpy, opencv-python, astropy, scikit-image, scipy, matplotlib,
    PyQt6, astroquery, pyyaml, tifffile, PyWavelets


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  OPSIYONEL ARAÇLAR (ayrıca kurulur)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ASTAP  (Plate Solving):
    İndir: https://www.hnsky.org/astap.htm
    Kurulumdan sonra: Araçlar → Settings → ASTAP → Yolu ayarla
    Katalog: D80 (önerilen) veya G17/H17 — ASTAP ile aynı klasöre koy

  GraXpert  (AI Arka Plan Çıkarma):
    İndir: https://www.graxpert.com
    Kurulumdan sonra: Settings → GraXpert → Yolu ayarla

  StarNet++  (Yıldız Ayırma):
    İndir: https://www.starnetastro.com
    Kurulumdan sonra: Settings → StarNet++ → Yolu ayarla


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ÖNERILEN İŞLEM SIRASI (Workflow)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  LİNEER AŞAMA (Stretch öncesi):
    1. 🗂 Stacking      — Bias/Dark/Flat kalibrasyon + hizalama + yığınlama
    2. 🌌 BG Extract   — Arka plan gradyanı çıkar (GraXpert AI önerilir)
    3. ✨ Noise         — Lineer gürültü azaltma (Silentium önerilir)
    4. 🔭 Deconv       — PSF düzeltme / optik bulanıklık (Blur Exterminator)
    5. 🎨 Color Cal    — Renk kalibrasyonu (SPCC G2V önerilir)

  STRETCH (Lineer → Non-lineer):
    6. 📊 Stretch      — VeraLux HMS önerilir (Auto Log D ile)

  NON-LİNEER AŞAMA:
    7. ⭐ StarNet++    — Yıldızları ayır
    8. 🔪 Sharpen     — Yapı detayları (Revela önerilir)
    9. 🎨 Color Grade — LCH renk cerrahi (Vectra)
   10. ✦+ Recompose  — Yıldızları geri birleştir
   11. 📈 Histogram  — Final tonlama (Levels/Curves/Adjustments)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  KLAVYE KISAYOLLARI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Ctrl+O     Dosya aç
  Ctrl+S     Kaydet
  Ctrl+Z     Geri al
  Ctrl+H     Orijinali göster
  Ctrl+0     Ekrana sığdır
  Ctrl+=     Yakınlaştır
  Ctrl+-     Uzaklaştır
  Ctrl+1     1 panel görünüm
  Ctrl+2     2 panel görünüm
  Ctrl+4     4 panel görünüm
  Ctrl+T     Stacking aç
  Ctrl+P     Plate Solve aç
  Ctrl+E     Script editörü


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  SORUN GİDERME
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Program açılmıyor:
    → install.bat'ı tekrar çalıştırın
    → Python'un PATH'te olduğundan emin olun

  "ModuleNotFoundError" hatası:
    → install.bat'ı tekrar çalıştırın

  Plate Solve çalışmıyor:
    → Settings → ASTAP → exe yolunu kontrol edin
    → D80 kataloğunun ASTAP klasöründe olduğunu kontrol edin
    → Search radius'u artırın (90°)
    → RA/Dec ipucu girin

  GraXpert / StarNet++ çalışmıyor:
    → İlgili programın yüklendiğinden emin olun
    → Settings'te yolu doğru girdiğinizden emin olun

  Performans yavaş:
    → Düşük çözünürlüklü önizleme kullanın (canvas_interp: nearest)
    → Gereksiz processleri kapatın


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  DOSYA YAPISI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  AstroMastroPro/
  ├── main.py               ← Başlangıç noktası
  ├── install.bat           ← Kurulum (ilk kez)
  ├── setup_and_run.bat     ← Başlatma
  ├── uninstall.bat         ← Kaldırma
  ├── settings.json         ← Kullanıcı ayarları (otomatik oluşur)
  ├── requirements.txt      ← Python paket listesi
  ├── gui/                  ← Arayüz modülleri
  │   ├── app.py            ← Ana pencere
  │   ├── plate_solve_dialog.py
  │   ├── update_dialog.py
  │   └── ...
  ├── processing/           ← İşlem algoritmaları
  │   ├── veralux_hms.py    ← Stretch
  │   ├── stacking.py       ← Yığınlama
  │   └── ...
  ├── ai/                   ← AI modülleri
  │   ├── astap_bridge.py   ← Plate solving
  │   ├── starnet_bridge.py ← StarNet++
  │   └── ...
  └── astrometry/           ← WCS / annotation
