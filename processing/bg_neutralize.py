"""
Astro Maestro Pro — Arka Plan Siyahlaştırma Modülü

Stacking sonrası veya herhangi bir aşamada arka planı siyah yapar.
Yıldız ve nebula sinyallerini korur, sadece arka plan seviyesini düşürür.

Yöntemler:
  - percentile : Alt yüzdelik dilimden arka plan tahmini (hızlı, varsayılan)
  - sigma_clip : Sigma-kırpmalı istatistiksel arka plan tahmini (hassas)
  - grid       : Grid tabanlı yerel arka plan çıkarma (gradient varsa)
"""

import cv2
import numpy as np


def neutralize_background(image: np.ndarray,
                           method: str = "percentile",
                           strength: float = 1.0,
                           bg_percentile: float = 5.0,
                           sigma: float = 2.5,
                           grid_size: int = 8,
                           protect_signal: float = 0.3,
                           per_channel: bool = True,
                           **kwargs) -> np.ndarray:
    """
    Arka planı siyahlaştırır, sinyal renklerini korur.

    Parameters
    ----------
    image : ndarray (float32, 0-1)
    method : "percentile" | "sigma_clip" | "grid"
    strength : 0.0 - 1.0 → ne kadar siyahlaştırılacak
    bg_percentile : Arka plan olarak kabul edilecek en karanlık yüzde
    sigma : sigma_clip yöntemi için sigma değeri
    grid_size : grid yöntemi için ızgara boyutu
    protect_signal : Sinyal koruma eşiği (0-1)
    per_channel : Her kanalı ayrı mı işle (renk dengesi için)
    """
    if image is None or image.size == 0:
        return image

    img = np.ascontiguousarray(image, dtype=np.float32).copy()
    strength = np.clip(strength, 0.0, 1.0)

    if strength == 0:
        return img

    is_color = img.ndim == 3 and img.shape[2] >= 3
    nc = img.shape[2] if is_color else 1

    if method == "sigma_clip":
        bg_levels = _estimate_bg_sigma_clip(img, sigma, per_channel)
    elif method == "grid":
        bg_levels = _estimate_bg_grid(img, grid_size, per_channel)
    else:  # percentile
        bg_levels = _estimate_bg_percentile(img, bg_percentile, per_channel)

    # Güç ayarı
    bg_levels = bg_levels * strength

    # Çıkarma
    if is_color and per_channel:
        for ch in range(min(nc, 3)):
            img[:, :, ch] -= bg_levels[ch]
    else:
        # Tüm kanallara aynı seviye
        level = bg_levels[0] if isinstance(bg_levels, np.ndarray) else bg_levels
        img -= level

    np.clip(img, 0, 1, out=img)

    # Sinyal koruması: çok parlak bölgeleri orijinalden karıştır
    if protect_signal > 0 and is_color:
        gray_orig = np.mean(image, axis=2) if is_color else image
        # Yüksek sinyal bölgelerinde orijinali koru
        signal_mask = gray_orig > np.percentile(gray_orig, 100 * (1 - protect_signal))
        if signal_mask.any():
            # Yumuşak geçiş maskesi
            mask_f = signal_mask.astype(np.float32)
            mask_f = cv2.GaussianBlur(mask_f, (15, 15), 3)
            # Sinyal bölgelerinde sadece minimum düzeltme uygula
            # (orijinal ile sonuç arası karıştır — sinyal korunsun)
            for ch in range(min(nc, 3)):
                orig_shifted = image[:, :, ch] - bg_levels[ch] * 0.5
                np.clip(orig_shifted, 0, 1, out=orig_shifted)
                img[:, :, ch] = img[:, :, ch] * (1 - mask_f) + orig_shifted * mask_f

    np.clip(img, 0, 1, out=img)
    return img.astype(np.float32)


def _estimate_bg_percentile(img: np.ndarray, pct: float = 5.0,
                             per_channel: bool = True) -> np.ndarray:
    """En karanlık yüzdelik dilimden arka plan tahmini."""
    is_color = img.ndim == 3 and img.shape[2] >= 3

    if is_color:
        gray = np.mean(img[:, :, :3], axis=2)
    else:
        gray = img if img.ndim == 2 else img[:, :, 0]

    threshold = np.percentile(gray, pct)
    bg_mask = gray <= threshold

    if bg_mask.sum() < 50:
        # Çok az piksel — fallback
        bg_mask = gray <= np.percentile(gray, max(pct * 2, 15))

    if bg_mask.sum() < 10:
        return np.zeros(3 if is_color else 1, dtype=np.float32)

    if is_color and per_channel:
        levels = np.array([
            float(np.median(img[:, :, ch][bg_mask]))
            for ch in range(min(img.shape[2], 3))
        ], dtype=np.float32)
    else:
        med = float(np.median(gray[bg_mask]))
        levels = np.array([med] * (3 if is_color else 1), dtype=np.float32)

    return levels


def _estimate_bg_sigma_clip(img: np.ndarray, sigma: float = 2.5,
                             per_channel: bool = True) -> np.ndarray:
    """Sigma-kırpmalı arka plan tahmini — daha hassas."""
    is_color = img.ndim == 3 and img.shape[2] >= 3

    def _sigma_median(data, sigma_val, iters=3):
        d = data.copy()
        for _ in range(iters):
            med = np.median(d)
            std = np.std(d)
            if std < 1e-9:
                break
            mask = np.abs(d - med) < sigma_val * std
            if mask.sum() < 10:
                break
            d = d[mask]
        return float(np.median(d))

    if is_color and per_channel:
        levels = np.array([
            _sigma_median(img[:, :, ch].ravel(), sigma)
            for ch in range(min(img.shape[2], 3))
        ], dtype=np.float32)
    else:
        gray = np.mean(img[:, :, :3], axis=2) if is_color else img.ravel()
        med = _sigma_median(gray.ravel(), sigma)
        levels = np.array([med] * (3 if is_color else 1), dtype=np.float32)

    return levels


def _estimate_bg_grid(img: np.ndarray, grid_size: int = 8,
                       per_channel: bool = True) -> np.ndarray:
    """Grid tabanlı yerel arka plan — gradient varsa etkili."""
    h, w = img.shape[:2]
    is_color = img.ndim == 3 and img.shape[2] >= 3
    nc = min(img.shape[2], 3) if is_color else 1

    gh = max(1, h // grid_size)
    gw = max(1, w // grid_size)

    # Her grid hücresinin medyan değerini al
    cell_values = [[] for _ in range(nc)]

    for gy in range(grid_size):
        for gx in range(grid_size):
            y0, y1 = gy * gh, min((gy + 1) * gh, h)
            x0, x1 = gx * gw, min((gx + 1) * gw, w)
            if y1 <= y0 or x1 <= x0:
                continue

            patch = img[y0:y1, x0:x1]
            if is_color:
                gray_patch = np.mean(patch[:, :, :3], axis=2)
            else:
                gray_patch = patch if patch.ndim == 2 else patch[:, :, 0]

            # Sadece en karanlık hücreleri al
            cell_values[0].append((float(np.median(gray_patch)), gy, gx))

    if not cell_values[0]:
        return np.zeros(nc, dtype=np.float32)

    # En karanlık %25 hücreleri seç
    cell_values[0].sort(key=lambda x: x[0])
    n_dark = max(1, len(cell_values[0]) // 4)
    dark_cells = cell_values[0][:n_dark]

    levels = np.zeros(nc, dtype=np.float32)
    for _, gy, gx in dark_cells:
        y0, y1 = gy * gh, min((gy + 1) * gh, h)
        x0, x1 = gx * gw, min((gx + 1) * gw, w)
        patch = img[y0:y1, x0:x1]
        if is_color and per_channel:
            for ch in range(nc):
                levels[ch] += float(np.median(patch[:, :, ch]))
        else:
            gray_p = np.mean(patch[:, :, :3], axis=2) if is_color else patch
            med = float(np.median(gray_p))
            levels[:] += med

    levels /= len(dark_cells)
    return levels
