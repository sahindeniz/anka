"""
Astro Maestro Pro — NoiseXterminator  (optimised v4)
Wavelet-based noise suppression — fully vectorised, parallel channels.
"""
import cv2
import numpy as np
import pywt
from concurrent.futures import ThreadPoolExecutor


def noisexterminator(image, strength=0.7, detail=0.5, **kw):
    img = np.ascontiguousarray(image, dtype=np.float32)
    np.clip(img, 0, 1, out=img)
    s = float(np.clip(strength, 0, 1))
    d = float(np.clip(detail, 0, 1))

    if img.ndim == 2:
        denoised = _wavelet_denoise(img, s, d)
    else:
        n_ch = img.shape[2]
        def _do(c):
            return _wavelet_denoise(img[:, :, c], s, d)
        with ThreadPoolExecutor(max_workers=min(n_ch, 3)) as pool:
            channels = list(pool.map(_do, range(n_ch)))
        denoised = np.stack(channels, axis=2)

    np.clip(denoised, 0, 1, out=denoised)
    return denoised.astype(np.float32), {}


def _wavelet_denoise(channel, strength, detail):
    h, w = channel.shape

    # Downsample büyük görüntüler için hız optimizasyonu
    MAX = 2048
    scale = min(1.0, MAX / max(h, w, 1))
    if scale < 0.99:
        sw, sh = max(4, int(w * scale)), max(4, int(h * scale))
        ch = cv2.resize(channel, (sw, sh), interpolation=cv2.INTER_AREA)
    else:
        ch = channel.copy()

    # Wavelet dönüşümü
    wavelet = 'db4'
    level   = min(4, pywt.dwt_max_level(min(ch.shape), wavelet))
    coeffs  = pywt.wavedec2(ch, wavelet, level=level)

    # Adaptive threshold
    sigma = _estimate_noise(coeffs[-1][0])
    thr   = sigma * np.sqrt(2 * np.log(max(ch.size, 1))) * strength

    # Soft threshold uygula
    new_coeffs = [coeffs[0]]
    for i, c_tuple in enumerate(coeffs[1:], 1):
        # Detail katsayısı: yüksek seviyelerde daha az thresh
        level_scale = 1.0 - detail * (i - 1) / max(level, 1)
        thr_lv = thr * level_scale
        new_c  = tuple(pywt.threshold(c, thr_lv, mode='soft') for c in c_tuple)
        new_coeffs.append(new_c)

    denoised = pywt.waverec2(new_coeffs, wavelet).astype(np.float32)
    # Boyut uyumu
    denoised = denoised[:ch.shape[0], :ch.shape[1]]

    if scale < 0.99:
        denoised = cv2.resize(denoised, (w, h), interpolation=cv2.INTER_LINEAR)

    return np.clip(denoised, 0, 1)


def _estimate_noise(detail_band):
    """MAD-based noise estimate from finest detail coefficients."""
    return float(np.median(np.abs(detail_band))) / 0.6745


def denoise(image, strength=0.7, detail_preserve=0.5, **kw):
    """Compatibility wrapper used by generic noise dispatch."""
    return noisexterminator(image, strength=strength, detail=detail_preserve, **kw)
