"""
StarXterminator — Siril algoritması
Cok olcekli Starlet donusumuyle yildiz kaldirilmasi.
"""
import numpy as np
from scipy.ndimage import (
    gaussian_filter, minimum_filter, maximum_filter,
    binary_dilation, binary_erosion, distance_transform_edt,
)

def build_star_mask(gray: np.ndarray, sensitivity: float = 0.5) -> np.ndarray:
    """
    Çok ölçekli Starlet dönüşümüyle yıldız maskesi oluşturur.

    Adımlar
    -------
    1. Gaussian piramitiyle wavelet düzlemleri hesapla.
    2. İnce ölçeklerde güçlü olan piksel = yıldız çekirdeği.
    3. Kaba ölçekte de parlak olan piksel = nebula → koru.
    4. İkili genişletme ile tam yıldız diskini yakala.
    """
    # Wavelet düzlemleri (Difference of Gaussians)
    s1 = gaussian_filter(gray, sigma=1.0)
    s2 = gaussian_filter(gray, sigma=2.0)
    s4 = gaussian_filter(gray, sigma=4.0)
    s8 = gaussian_filter(gray, sigma=8.0)

    w1 = np.clip(gray - s1, 0, None)   # en ince ölçek
    w2 = np.clip(s1  - s2, 0, None)   # ikinci ölçek
    w3 = np.clip(s2  - s4, 0, None)   # üçüncü ölçek

    star_signal = w1 + w2 * 0.6 + w3 * 0.3

    # Duyarlılığa göre eşik
    pct = 100 * (1 - sensitivity * 0.3) - 2   # ~88–95 arası
    pct = np.clip(pct, 70, 99)
    threshold = np.percentile(star_signal[star_signal > 0], pct)

    star_binary = (star_signal > threshold) & (gray > 0.05)

    # Nebula koruması: kaba ölçekte parlak → nebula
    nebula_protect = s8 > (0.10 + (1 - sensitivity) * 0.15)
    star_binary = star_binary & ~nebula_protect

    # Yıldız diskini genişlet (dilation)
    dil_iter = max(1, int(3 * sensitivity))
    star_mask = binary_dilation(star_binary, iterations=dil_iter)

    return star_mask.astype(np.float32)

def feather_mask(mask_binary: np.ndarray, radius: float = 3.0) -> np.ndarray:
    """Maske kenarlarını yumuşatır (feathering)."""
    dist_in  = distance_transform_edt(mask_binary > 0.5)
    feathered = np.clip(dist_in / radius, 0, 1)
    return feathered

def estimate_background(channel: np.ndarray, sigma: float = 14.0) -> np.ndarray:
    """Yıldıkların altındaki arka planı (nebula + gök) tahmin eder."""
    return gaussian_filter(channel, sigma=sigma)

def reduce_stars(
    image: np.ndarray,
    strength: float = 0.9,
    sensitivity: float = 0.5,
    feather_radius: float = 3.0,
    bg_sigma: float = 14.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    StarXterminator ana fonksiyonu.

    Parametreler
    ------------
    image        : float32 ndarray (H, W, C), değerler [0,1]
    strength     : Yıldız küçültme gücü 0–1 (1 = maksimum)
    sensitivity  : Yıldız tespiti hassasiyeti 0–1
    feather_radius: Kenar yumuşatma yarıçapı (piksel)
    bg_sigma     : Arka plan tahmin bulanıklığı (piksel)

    Döndürür
    --------
    result  : İşlenmiş görüntü (float32)
    mask    : Yıldız maskesi görselleştirme (float32)
    """
    # Mono görüntü desteği
    if image.ndim == 2:
        H, W = image.shape
        C = 1
        work = image[:, :, np.newaxis]
    else:
        H, W, C = image.shape
        work = image

    # Luminans (maksimum kanal)
    gray = work.max(axis=2)

    star_mask_bin = build_star_mask(gray, sensitivity)
    soft_mask = feather_mask(star_mask_bin, feather_radius) * strength

    result = work.copy()

    # Çekirdek maskesini döngü dışına al (her kanal için aynı)
    core_mask = binary_erosion(star_mask_bin > 0.5, iterations=1).astype(np.float32)
    core_soft = gaussian_filter(core_mask, sigma=1.0) * strength

    for c in range(C):
        bg = estimate_background(work[:, :, c], bg_sigma)

        # Önce dış halka → arka planla değiştir
        result[:, :, c] = work[:, :, c] * (1 - soft_mask) + bg * soft_mask
        # Sonra çekirdek → hafif karart
        result[:, :, c] = result[:, :, c] * (1 - core_soft * 0.4) + bg * (core_soft * 0.4)

    result = np.clip(result, 0, 1)

    # Mono girdi ise mono çıktı döndür
    if image.ndim == 2:
        return result[:, :, 0], soft_mask

    return result, soft_mask
