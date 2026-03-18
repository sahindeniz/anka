"""
Astro Maestro Pro — Starsmalerx  (optimised v3)
Fast star detection + inpaint — vectorised, uint8 OpenCV.
"""
import cv2
import numpy as np


def reduce_stars(image, strength=0.9, sensitivity=0.5, feather=3,
                 max_sigma=6, min_sigma=1, threshold=0.03, **kw):
    img = np.ascontiguousarray(image, dtype=np.float32)
    np.clip(img, 0, 1, out=img)
    h, w = img.shape[:2]

    gray = img if img.ndim == 2 else img.mean(axis=2)

    # Hızlı star mask: Gaussian diff (LoG yaklaşımı)
    mask = _fast_star_mask(gray, float(sensitivity), float(threshold),
                           int(max_sigma), int(min_sigma))

    # Feather (dilate + blur)
    feather = max(1, int(feather))
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (feather*2+1, feather*2+1))
    mask    = cv2.dilate(mask, kernel)
    mask_f  = cv2.GaussianBlur(mask.astype(np.float32), (feather*2+1, feather*2+1), feather)

    # Inpaint (telea — hızlı)
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    mask_u8 = (mask_f > 0.15).astype(np.uint8) * 255
    inpainted = cv2.inpaint(u8, mask_u8, 3, cv2.INPAINT_TELEA).astype(np.float32) / 255.0

    # Blend strength ile
    s = float(np.clip(strength, 0, 1))
    mask_3d = mask_f[:, :, None] if img.ndim == 3 else mask_f
    result = img * (1 - mask_3d * s) + inpainted * (mask_3d * s)

    np.clip(result, 0, 1, out=result)
    return result.astype(np.float32), mask_f


def _fast_star_mask(gray, sensitivity, threshold, max_sigma, min_sigma):
    """DoG-based star detection — hızlı uint8."""
    g8 = (np.clip(gray, 0, 1) * 255).astype(np.uint8)
    h, w = g8.shape
    mask = np.zeros((h, w), np.uint8)

    for sigma in range(min_sigma, max_sigma + 1, max(1, (max_sigma - min_sigma) // 3)):
        ks1 = max(3, sigma * 2 + 1) | 1
        ks2 = max(3, sigma * 4 + 1) | 1
        g1  = cv2.GaussianBlur(g8, (ks1, ks1), sigma)
        g2  = cv2.GaussianBlur(g8, (ks2, ks2), sigma * 2)
        dog = cv2.subtract(g1, g2)
        thr_val = max(1, int(threshold * 255 * (1.1 - sensitivity)))
        _, bw = cv2.threshold(dog, thr_val, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_or(mask, bw)

    return mask
