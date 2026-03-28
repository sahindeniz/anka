"""
Astro Maestro Pro 2 - Akıllı Arka Plan Temizleyici
=====================================================
Özellikler:
- Yıldızları korur (parlak noktalar dokunulmaz)
- Galaksileri korur (geniş alanlı yapılar)
- Deep space nesnelerini (bulutsular) korur
- Sadece arka plan gradyanlarını ve ışık kirliliğini temizler
- Homojen, düzgün bir arka plan oluşturur
"""

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, median_filter
from scipy.interpolate import RBFInterpolator
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures

# ============================================================================
# YARDIMCI FONKSİYONLAR
# ============================================================================

def create_protection_mask(img, star_threshold=98, galaxy_smooth=15):
    """
    Korunacak bölgelerin maskesini oluşturur.
    - Yıldızlar (parlak noktalar)
    - Galaksiler (geniş alanlı yapılar)
    - Deep space nesneleri (bulutsular)
    """
    if img.ndim == 3:
        lum = 0.2126 * img[:,:,0] + 0.7152 * img[:,:,1] + 0.0722 * img[:,:,2]
    else:
        lum = img
    
    h, w = lum.shape
    mask = np.ones((h, w), dtype=np.float32)
    
    # 1. YILDIZ MASKESİ (parlak noktalar)
    star_thr = np.percentile(lum, star_threshold)
    star_mask = (lum > star_thr).astype(np.float32)
    
    # Yıldız maskesini genişlet (haloları da kapsasın)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    star_mask = cv2.dilate(star_mask, kernel, iterations=2)
    star_mask = cv2.GaussianBlur(star_mask, (9, 9), 2)
    
    # 2. GALAKSİ / BULUTSU MASKESİ (geniş alanlı yapılar)
    smoothed = cv2.GaussianBlur(lum, (galaxy_smooth, galaxy_smooth), galaxy_smooth/3)
    galaxy_thr = np.percentile(smoothed, 85)
    galaxy_mask = (smoothed > galaxy_thr).astype(np.float32)
    galaxy_mask = cv2.GaussianBlur(galaxy_mask, (21, 21), 5)
    
    # 3. TOPLAM KORUMA MASKESİ
    protection = np.maximum(star_mask, galaxy_mask)
    protection = np.clip(protection, 0, 1)
    
    # Maskeyi yumuşat (keskin geçişleri önle)
    protection = cv2.GaussianBlur(protection, (15, 15), 3)
    
    return protection


def estimate_background_polynomial(img, mask, degree=3, grid_size=32):
    """Polinomal yüzey uydurma ile arka plan tahmini"""
    h, w = img.shape[:2]
    is_color = (img.ndim == 3)
    
    if is_color:
        lum = 0.2126 * img[:,:,0] + 0.7152 * img[:,:,1] + 0.0722 * img[:,:,2]
    else:
        lum = img
    
    xs, ys = np.meshgrid(np.arange(0, w, grid_size), np.arange(0, h, grid_size))
    xs = xs.flatten()
    ys = ys.flatten()
    
    values = []
    points = []
    for x, y in zip(xs, ys):
        if 0 <= y < h and 0 <= x < w and mask[y, x] > 0.8:
            values.append(lum[y, x])
            points.append([x, y])
    
    if len(points) < 10:
        return estimate_background_polynomial(img, mask, degree, grid_size*2)
    
    points = np.array(points)
    values = np.array(values)
    
    poly = PolynomialFeatures(degree=degree)
    X = poly.fit_transform(points)
    
    ransac = RANSACRegressor(random_state=42)
    ransac.fit(X, values)
    
    X_full = poly.fit_transform(np.column_stack((np.arange(w).repeat(h), np.tile(np.arange(h), w))))
    bg = ransac.predict(X_full).reshape(h, w)
    bg = gaussian_filter(bg, sigma=3)
    
    return bg


def estimate_background_rbf(img, mask, smoothing=0.3, grid_size=48):
    """RBF interpolasyonu ile arka plan tahmini"""
    h, w = img.shape[:2]
    is_color = (img.ndim == 3)
    
    if is_color:
        lum = 0.2126 * img[:,:,0] + 0.7152 * img[:,:,1] + 0.0722 * img[:,:,2]
    else:
        lum = img
    
    xs, ys = np.meshgrid(np.arange(0, w, grid_size), np.arange(0, h, grid_size))
    xs = xs.flatten()
    ys = ys.flatten()
    
    values = []
    points = []
    for x, y in zip(xs, ys):
        if 0 <= y < h and 0 <= x < w and mask[y, x] > 0.8:
            values.append(lum[y, x])
            points.append([x, y])
    
    if len(points) < 20:
        return estimate_background_polynomial(img, mask, degree=3, grid_size=grid_size)
    
    points = np.array(points)
    values = np.array(values)
    
    rbf = RBFInterpolator(points, values, kernel='thin_plate_spline', smoothing=smoothing)
    X_full = np.column_stack((np.arange(w).repeat(h), np.tile(np.arange(h), w)))
    bg = rbf(X_full).reshape(h, w)
    bg = gaussian_filter(bg, sigma=2)
    
    return bg


def apply_background_correction(img, bg, protection_mask):
    """Arka plan düzeltmesini uygula"""
    is_color = (img.ndim == 3)
    
    if is_color:
        result = np.zeros_like(img)
        for c in range(3):
            corrected = img[:,:,c] - bg
            result[:,:,c] = img[:,:,c] * protection_mask + corrected * (1 - protection_mask)
            result[:,:,c] = np.clip(result[:,:,c], 0, 1)
    else:
        corrected = img - bg
        result = img * protection_mask + corrected * (1 - protection_mask)
        result = np.clip(result, 0, 1)
    
    return result


def normalize_background(img, target_bg=0.05):
    """Arka plan seviyesini hedef değere normalize et"""
    if img.ndim == 3:
        lum = 0.2126 * img[:,:,0] + 0.7152 * img[:,:,1] + 0.0722 * img[:,:,2]
    else:
        lum = img
    
    current_bg = np.median(lum[lum < np.percentile(lum, 20)])
    if current_bg > 0:
        scale = target_bg / current_bg
        return np.clip(img * scale, 0, 1)
    
    return img


def homogenize_background(img, sigma=5):
    """Arka planı homojenleştir"""
    if img.ndim == 3:
        result = np.zeros_like(img)
        for c in range(3):
            bg_part = gaussian_filter(img[:,:,c], sigma=sigma)
            result[:,:,c] = bg_part
        return result
    else:
        return gaussian_filter(img, sigma=sigma)


# ============================================================================
# ANA FONKSİYON
# ============================================================================

def process(Image, method='auto', star_threshold=98, galaxy_smooth=15, 
            bg_method='polynomial', homogenize=True, target_bg=0.05,
            verbose=True):
    """
    Akıllı arka plan temizleyici
    
    Parameters:
    -----------
    Image : np.ndarray
        Giriş görüntüsü (float32, [0,1])  <-- Büyük I ile Image
    method : str
        'auto', 'polynomial', 'rbf'
    star_threshold : float
        Yıldız tespiti yüzdelik eşiği (95-99 arası)
    galaxy_smooth : int
        Galaksi tespiti için yumuşatma boyutu
    bg_method : str
        Arka plan tahmin yöntemi ('polynomial' veya 'rbf')
    homogenize : bool
        Arka planı homojenleştir
    target_bg : float
        Hedef arka plan seviyesi
    """
    
    print("=" * 50)
    print("🌌 AKILLI ARKA PLAN TEMİZLEYİCİ")
    print("=" * 50)
    
    h, w = Image.shape[:2]
    print(f"Görüntü boyutu: {w} x {h} = {w*h/1e6:.1f} MP")
    
    # 1. KORUMA MASKESİ OLUŞTUR
    print("\n📌 1. Koruma maskesi oluşturuluyor...")
    print(f"   - Yıldız eşiği: %{star_threshold}")
    print(f"   - Galaksi yumuşatma: {galaxy_smooth} px")
    
    protection_mask = create_protection_mask(Image, star_threshold, galaxy_smooth)
    protected_percent = np.mean(protection_mask < 0.5) * 100
    print(f"   - Korunan alan: %{protected_percent:.1f} (yıldızlar + galaksiler)")
    
    # 2. ARKA PLAN TAHMİNİ
    print("\n📌 2. Arka plan tahmini yapılıyor...")
    
    if method == 'auto':
        if Image.ndim == 3:
            lum = 0.2126 * Image[:,:,0] + 0.7152 * Image[:,:,1] + 0.0722 * Image[:,:,2]
        else:
            lum = Image
        
        grad_h = np.abs(np.diff(lum, axis=0)).mean()
        grad_v = np.abs(np.diff(lum, axis=1)).mean()
        total_grad = (grad_h + grad_v) / 2
        
        if total_grad > 0.01:
            use_method = 'rbf'
            print(f"   - Yöntem: RBF (karmaşık gradyan: {total_grad:.5f})")
        else:
            use_method = 'polynomial'
            print(f"   - Yöntem: Polinom (basit gradyan: {total_grad:.5f})")
    else:
        use_method = method
        print(f"   - Yöntem: {use_method}")
    
    if use_method == 'polynomial':
        bg = estimate_background_polynomial(Image, protection_mask, degree=3, grid_size=48)
    else:
        bg = estimate_background_rbf(Image, protection_mask, smoothing=0.3, grid_size=48)
    
    print(f"   - Arka plan seviyesi: {np.median(bg):.4f}")
    
    # 3. ARKA PLAN DÜZELTMESİ
    print("\n📌 3. Arka plan düzeltmesi uygulanıyor...")
    result = apply_background_correction(Image, bg, protection_mask)
    
    # 4. HOMOJENLEŞTİRME
    if homogenize:
        print("\n📌 4. Arka plan homojenleştiriliyor...")
        bg_part = homogenize_background(result, sigma=3)
        result = bg_part * (1 - protection_mask) + result * protection_mask
        result = np.clip(result, 0, 1)
    
    # 5. NORMALİZASYON
    print("\n📌 5. Arka plan normalize ediliyor...")
    result = normalize_background(result, target_bg)
    
    # 6. SON KONTROL
    if result.ndim == 3:
        final_bg = np.median(result[protection_mask > 0.8])
    else:
        final_bg = np.median(result[protection_mask > 0.8])
    
    print(f"\n✅ İŞLEM TAMAMLANDI!")
    print(f"   - Yeni arka plan seviyesi: {final_bg:.4f}")
    print(f"   - Hedef seviye: {target_bg:.4f}")
    print("=" * 50)
    
    return result.astype(np.float32)


# ============================================================================
# SCRIPT ÇALIŞTIRMA
# ============================================================================

# Script Editörü'nden çağrıldığında çalışacak kod
# DİKKAT: Image büyük I ile yazıldı!
result = process(
    Image=Image,                    # Görüntü (büyük I ile Image)
    method='auto',                  # auto, polynomial, rbf
    star_threshold=98,              # Yıldız eşiği (%98 üstü yıldız)
    galaxy_smooth=15,               # Galaksi tespiti yumuşatma
    bg_method='polynomial',         # polynomial veya rbf
    homogenize=True,                # Arka planı homojenleştir
    target_bg=0.05,                 # Hedef arka plan seviyesi
    verbose=True                    # Detaylı çıktı
)

print("\n🎯 Sonuç: Arka plan temizlendi, yıldızlar ve galaksiler korundu!")
