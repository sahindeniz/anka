# Astro Maestro Pro Kullanım Kılavuzu

Bu belge, Astro Maestro Pro'yu ilk kurulumdan günlük kullanıma kadar hızlıca kullanabilmeniz için hazırlanmıştır.

## 1. Kurulum

Windows için en kolay yol:

1. `install.bat` dosyasını çalıştırın.
2. Kurulum tamamlanınca `setup_and_run.bat` ile uygulamayı açın.
3. Sonraki kullanımlarda aynı başlatıcıyı veya masaüstü kısayolunu kullanın.

Manuel kurulum:

```bash
pip install -r requirements.txt
python main.py
```

## 2. İlk Açılış

Uygulama açıldığında ana bölümler:

- Üst menü: `Dosya`, `Düzenle`, `Görünüm`, `İşlem`, `Araçlar`, `Yardım`
- Orta alan: ana görüntüleyici
- Sağ veya alt paneller: histogram, işlem panelleri, geçmiş
- Toolbar: hızlı erişim düğmeleri

İlk önerilen adımlar:

1. `Ayarlar` menüsünden isteğe bağlı araç yollarını girin.
2. `Çalışma Klasörü` seçin.
3. Bir FITS/TIFF/PNG/JPG görüntü açın.

## 3. Desteklenen Dosyalar

- FITS
- TIFF
- PNG
- JPG / JPEG

## 4. Önerilen İş Akışı

Lineer aşama:

1. `Stacking`
2. `BG Extract`
3. `Noise`
4. `Deconv`
5. `Color Calibration`

Geçiş:

6. `Stretch`

Non-lineer aşama:

7. `StarNet++` veya `Mastro Starless`
8. `Sharpen`
9. `Nebula / Color / Morphology`
10. `Recompose`
11. Final histogram ve renk düzeltmeleri

Not:

- `AutoSTF` artık sadece görüntüleme önizlemesidir.
- Kapatıldığında görüntü orijinale dönmez; mevcut lineer işlenmiş hali gösterilir.
- Starless işlemleri lineer kaynak görüntü üzerinden çalışır.

## 5. Görüntü Açma ve Kaydetme

Dosya açmak için:

- `Dosya > Aç`
- veya sürükle-bırak

Kaydetmek için:

- `Dosya > Kaydet`

Çalışma klasörü seçici yalnızca klasör seçer; dosya seçmez.

## 6. Stacking Kullanımı

`İşlem > Stacking` veya `Ctrl+T` ile stacking penceresini açın.

Önerilen temel kullanım:

1. Light karelerini ekleyin.
2. Varsa `Dark`, `Flat`, `Bias` karelerini ekleyin.
3. Önce `Hizala` çalıştırın.
4. Sonra `Stacking Başlat` ile yığma işlemini başlatın.

Önemli ayarlar:

- `Stacking Metodu`: Genelde `auto`
- `Kare Ağırlıklandırma`: Genelde `snr`
- `Normalizasyon`: Genelde `additive_scaling`
- `Low RAM fallback`: RAM daralırsa güvenli fallback

Performans notu:

- Büyük setlerde `auto` artık daha hızlı rejection yolunu seçer.
- Hizalama ve kalite analizi hız için optimize edilmiştir.

## 7. AutoSTF ve Stretch

`AutoSTF`:

- Ekranda geçici bir auto-stretch önizlemesi açar
- History'ye yeni adım eklemez
- İşleme verisini değiştirmez

Kalıcı stretch için:

- `Stretch` panelini kullanın
- `auto_stf`, `VeraLux`, `asinh`, `hyperbolic` gibi yöntemlerden birini seçin

## 8. Starless İşlemleri

İki ana yol vardır:

- `StarNet++`
- `Mastro Starless`

Her ikisi için de:

- Giriş görüntüsü mevcut lineer kaynak üzerinden alınır
- Sonuç ayrı sekme veya history adımı olarak gelir
- İsterseniz yıldızsız ve yıldız-only çıktıları kaydedilir

## 9. Plate Solve

Plate solve için ASTAP gerekir.

Kurulum:

1. ASTAP'ı kurun
2. `Ayarlar > ASTAP` bölümünden exe yolunu girin
3. Gerekli katalogları ekleyin

Kullanım:

- `Araçlar > Plate Solve`
- veya `Ctrl+P`

## 10. Geçmiş ve Geri Alma

- `Undo / Redo` desteklenir
- History panelinden eski adıma dönebilirsiniz
- `Reset`, son işlemi geri alma veya tam sıfırlama için kullanılır

## 11. Görünüm Özellikleri

- `1 / 2 / 4` panel görünümü
- RGB / R / G / B / L kanal görüntüleme
- Çift tıklama ile büyük görüntüleyici
- Fullscreen viewer artık live preview ile senkron çalışır

## 12. Kısayollar

- `Ctrl+O`: Aç
- `Ctrl+S`: Kaydet
- `Ctrl+Z`: Geri al
- `Ctrl+Y`: İleri al
- `Ctrl+0`: Ekrana sığdır
- `Ctrl+1`: 1 panel
- `Ctrl+2`: 2 panel
- `Ctrl+4`: 4 panel
- `Ctrl+T`: Stacking
- `Ctrl+P`: Plate Solve

## 13. Sık Karşılaşılan Sorunlar

Uygulama açılmıyorsa:

- `install.bat` dosyasını yeniden çalıştırın
- Python bağımlılıklarının kurulu olduğundan emin olun

StarNet++ çalışmıyorsa:

- `Ayarlar` içinden exe yolunu kontrol edin

ASTAP çalışmıyorsa:

- Exe yolu ve katalogları kontrol edin

Stacking yavaşsa:

- Önce hizalama yapın
- `auto` metodunu kullanın
- Çok büyük setlerde gereksiz ağır rejection yöntemlerini seçmeyin

## 14. İlgili Dosyalar

- `main.py`: uygulama giriş noktası
- `gui/app.py`: ana arayüz
- `processing/stacking.py`: stacking motoru
- `settings.json`: kullanıcı ayarları
