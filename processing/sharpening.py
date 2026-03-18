"""
Astro Maestro Pro — Sharpening  (optimised v3)
- _MAX_PX 1600 (was 1200) — daha iyi detay
- uint8 işlemler
- multiscale_vlc: vectorised wavelet, no Python loop overhead
"""
import cv2
import numpy as np

_MAX_PX = 1600

def sharpen(image, method="multiscale_vlc", radius=2.0, amount=1.0,
            threshold=0.0, scale_levels=4, iterations=1, **kw):
    img = np.ascontiguousarray(image, dtype=np.float32)
    np.clip(img, 0, 1, out=img)
    h, w = img.shape[:2]

    scale = min(1.0, _MAX_PX / max(h, w, 1))
    if scale < 0.99:
        sw, sh = max(4, int(w * scale)), max(4, int(h * scale))
        small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    else:
        small = img.copy()

    fn_map = {
        "multiscale_vlc": lambda x: _multiscale_vlc(x, int(scale_levels), float(amount)),
        "unsharp_mask":   lambda x: _unsharp(x, float(radius), float(amount), float(threshold)),
        "laplacian_ai":   lambda x: _laplacian(x, float(amount)),
        "high_pass":      lambda x: _high_pass(x, float(radius), float(amount)),
    }
    fn = fn_map.get(method, fn_map["unsharp_mask"])

    for _ in range(max(1, int(iterations))):
        small = np.clip(fn(small), 0, 1)

    if scale < 0.99:
        result = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        result = small

    np.clip(result, 0, 1, out=result)
    return result.astype(np.float32)


def _unsharp(img, radius, amount, threshold):
    ks = max(3, int(radius * 2 + 1) | 1)
    u8 = (img * 255).astype(np.uint8)
    blurred_u8 = cv2.GaussianBlur(u8, (ks, ks), radius)
    blurred = blurred_u8.astype(np.float32) / 255.0
    diff = img - blurred
    if threshold > 0:
        diff = np.where(np.abs(diff) >= threshold, diff, 0)
    return img + diff * amount


def _multiscale_vlc(img, levels, amount):
    """Wavelet-like multiscale — fully vectorised, uint8 gaussian."""
    residual = img.copy()
    u8 = (img * 255).astype(np.uint8)
    for lv in range(min(levels, 4)):
        sigma = 2.0 ** lv
        ks = max(3, int(sigma * 3) | 1)
        smooth_u8 = cv2.GaussianBlur(u8, (ks, ks), sigma)
        smooth = smooth_u8.astype(np.float32) / 255.0
        detail = residual - smooth
        mad = float(np.median(np.abs(detail))) * 1.4826 + 1e-9
        weight = np.clip(np.abs(detail) / (mad * 2), 0, 1)
        residual = residual + detail * weight * (amount / levels)
        u8 = np.clip(residual * 255, 0, 255).astype(np.uint8)
    return residual


def _laplacian(img, amount):
    gray = img if img.ndim == 2 else img.mean(axis=2)
    g8 = (np.clip(gray, 0, 1) * 255).astype(np.uint8)
    lap = cv2.Laplacian(g8, cv2.CV_32F, ksize=3) / 255.0
    std = gray.std() + 1e-9
    snr = np.abs(gray) / std
    w = np.clip(snr / (snr.max() + 1e-9), 0, 1)
    if img.ndim == 3:
        lap = lap[:, :, np.newaxis]
        w   = w[:, :, np.newaxis]
    return img + lap * w * amount * 0.3


def _high_pass(img, radius, amount):
    ks = max(3, int(radius * 4 + 1) | 1)
    u8 = (img * 255).astype(np.uint8)
    blurred_u8 = cv2.GaussianBlur(u8, (ks, ks), radius * 2)
    blurred = blurred_u8.astype(np.float32) / 255.0
    return 0.5 + (img - blurred) * amount
