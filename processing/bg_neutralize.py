"""
Astro Maestro Pro — Arka Plan Nötralizasyon Modülü

PixInsight BackgroundNeutralization mantığı:
  1. Arka plan piksellerini tespit et (koyu bölgeler)
  2. Per-channel medyan hesapla
  3. Çıkar + ölçekle → tüm kanalları eşitle
  4. Sinyal renklerini koru (sadece arka plan nötralize edilir)

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
    Arka planı nötralize eder — RGB kanallarını eşitler.

    PixInsight BackgroundNeutralization yaklaşımı:
    1. Arka plan seviyelerini per-channel tespit et
    2. En düşük kanala göre diğerlerini hizala (çıkarma)
    3. Sinyal bölgelerinde renk oranlarını koru

    Parameters
    ----------
    image : ndarray (float32, 0-1)
    method : "percentile" | "sigma_clip" | "grid"
    strength : 0.0 - 1.0 → ne kadar nötralize edilecek
    bg_percentile : Arka plan olarak kabul edilecek en karanlık yüzde
    sigma : sigma_clip yöntemi için sigma değeri
    grid_size : grid yöntemi için ızgara boyutu
    protect_signal : Sinyal koruma eşiği (0-1, 0=koruma yok)
    per_channel : Her kanalı ayrı mı işle
    """
    if image is None or image.size == 0:
        return image

    img = np.ascontiguousarray(image, dtype=np.float32).copy()
    strength = float(np.clip(strength, 0.0, 1.0))

    if strength == 0:
        return img

    is_color = img.ndim == 3 and img.shape[2] >= 3
    if not is_color:
        # Mono resim — sadece arka plan çıkar
        bg = _estimate_bg_percentile(img, bg_percentile, False)
        img -= bg[0] * strength
        np.clip(img, 0, 1, out=img)
        return img.astype(np.float32)

    nc = min(img.shape[2], 3)

    # ── 1. Arka plan seviyelerini tespit et ──────────────────────────────
    if method == "sigma_clip":
        bg_levels = _estimate_bg_sigma_clip(img, sigma, per_channel)
    elif method == "grid":
        bg_levels = _estimate_bg_grid(img, grid_size, per_channel)
    else:  # percentile
        bg_levels = _estimate_bg_percentile(img, bg_percentile, per_channel)

    # ── 2. Nötralizasyon — PixInsight yaklaşımı ─────────────────────────
    # Per-channel: her kanalı bağımsız sıfıra çek, renk oranları korunsun
    if per_channel and bg_levels.max() > 1e-6:
        # Sinyal koruma maskesi oluştur
        if protect_signal > 0:
            gray = np.mean(img[:, :, :3], axis=2)
            bg_threshold = np.percentile(gray, max(15, (1 - protect_signal) * 100))
            # Daha yumuşak geçiş — geniş blur ile homojen maske
            signal_mask = np.clip((gray - bg_threshold) / max(bg_threshold * 0.3, 0.005), 0, 1)
            signal_mask = cv2.GaussianBlur(signal_mask, (51, 51), 12)

            for ch in range(nc):
                # Arka planda tam düzeltme, sinyalde korumalı düzeltme
                ch_correction = bg_levels[ch] * strength * (1 - signal_mask * 0.85)
                img[:, :, ch] -= ch_correction
        else:
            for ch in range(nc):
                img[:, :, ch] -= bg_levels[ch] * strength
    else:
        # Per-channel kapalı — aynı seviye tüm kanallara
        level = float(bg_levels.mean()) * strength
        img[:, :, :3] -= level

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
    cell_data = []

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

            cell_data.append((float(np.median(gray_patch)), gy, gx))

    if not cell_data:
        return np.zeros(nc, dtype=np.float32)

    # En karanlık %25 hücreleri seç
    cell_data.sort(key=lambda x: x[0])
    n_dark = max(1, len(cell_data) // 4)
    dark_cells = cell_data[:n_dark]

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
