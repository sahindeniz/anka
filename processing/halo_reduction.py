"""
Astro Maestro Pro — Fast Halo Reduction v2
===========================================
Direct morphological halo suppression — no full star separation needed.

Algorithm:
1. Fast star detection (threshold + connected components)
2. Per-star halo ring mask (core excluded)
3. Local background estimation via morphological opening
4. Halo suppression by pulling halos toward local background
5. Core preservation with smooth feathering

Much faster and more natural than the v1 separate→shrink→recompose pipeline.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import cv2
import numpy as np


def _emit(cb: Optional[Callable], msg: str, step: int, total: int,
          preview: Optional[np.ndarray] = None) -> None:
    if cb is None:
        return
    try:
        cb(msg, step=step, total=total, preview=preview)
    except TypeError:
        try:
            cb(msg)
        except Exception:
            pass


def _luminance(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img.astype(np.float32)
    return (0.2126 * img[:, :, 0] +
            0.7152 * img[:, :, 1] +
            0.0722 * img[:, :, 2]).astype(np.float32)


def _detect_stars_fast(lum: np.ndarray, sensitivity: float = 0.5
                       ) -> list[tuple[int, int, float]]:
    """Fast star detection: threshold + connected components.

    Returns list of (cy, cx, radius) for each detected star.
    """
    h, w = lum.shape

    # Adaptive threshold based on image statistics
    med = float(np.median(lum))
    mad = float(np.median(np.abs(lum - med))) * 1.4826
    thr = med + max(3.0 - sensitivity * 2.0, 0.8) * max(mad, 1e-4)
    thr = float(np.clip(thr, med + 0.01, 1.0))

    bw = (lum > thr).astype(np.uint8)

    # Clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bw)

    max_area = max(600, h * w * 0.0008)
    stars = []
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 2 or area > max_area:
            continue
        bw_i = stats[i, cv2.CC_STAT_WIDTH]
        bh_i = stats[i, cv2.CC_STAT_HEIGHT]
        # Reject elongated objects (not stars)
        if max(bw_i, bh_i) > min(bw_i, bh_i) * 3.0:
            continue
        cy, cx = centroids[i][1], centroids[i][0]
        radius = max(1.5, np.sqrt(area / np.pi))
        stars.append((int(round(cy)), int(round(cx)), float(radius)))

    return stars


def _build_halo_mask(shape: tuple[int, int],
                     stars: list[tuple[int, int, float]],
                     halo_multiplier: float = 3.5,
                     core_fraction: float = 0.6,
                     feather_sigma: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Build halo ring mask and core protection mask.

    halo_multiplier: how far out the halo extends (in star radii)
    core_fraction: fraction of star radius considered 'core' (protected)
    """
    h, w = shape
    halo_mask = np.zeros((h, w), dtype=np.float32)
    core_mask = np.zeros((h, w), dtype=np.float32)

    for cy, cx, r in stars:
        outer_r = r * halo_multiplier
        inner_r = r * core_fraction
        pad = int(np.ceil(outer_r)) + 2

        y0, y1 = max(0, cy - pad), min(h, cy + pad + 1)
        x0, x1 = max(0, cx - pad), min(w, cx + pad + 1)
        if y1 <= y0 or x1 <= x0:
            continue

        yy = np.arange(y0, y1, dtype=np.float32) - cy
        xx = np.arange(x0, x1, dtype=np.float32) - cx
        dy, dx = np.meshgrid(yy, xx, indexing='ij')
        dist = np.sqrt(dy * dy + dx * dx)

        # Halo ring: between core and outer radius, smooth falloff
        ring = np.clip((dist - inner_r) / max(outer_r - inner_r, 0.5), 0, 1)
        ring *= np.clip(1.0 - (dist - outer_r) / max(r * 0.5, 0.5), 0, 1)
        np.maximum(halo_mask[y0:y1, x0:x1], ring, out=halo_mask[y0:y1, x0:x1])

        # Core mask: strong at center, fading out
        core = np.clip(1.0 - dist / max(inner_r, 1.0), 0, 1)
        np.maximum(core_mask[y0:y1, x0:x1], core, out=core_mask[y0:y1, x0:x1])

    # Smooth both masks
    if feather_sigma > 0:
        ksize = int(np.ceil(feather_sigma * 3)) * 2 + 1
        halo_mask = cv2.GaussianBlur(halo_mask, (ksize, ksize), feather_sigma)
        core_mask = cv2.GaussianBlur(core_mask, (ksize, ksize), feather_sigma * 0.5)

    return np.clip(halo_mask, 0, 1), np.clip(core_mask, 0, 1)


def _estimate_local_background(img: np.ndarray, stars: list,
                                scale: float = 6.0) -> np.ndarray:
    """Estimate local background using morphological opening.

    This gives us what the image would look like without the halos.
    """
    # Kernel size based on typical star halo extent
    if stars:
        median_r = float(np.median([s[2] for s in stars]))
        ksize = max(5, int(np.ceil(median_r * scale)) * 2 + 1)
    else:
        ksize = 15

    # Cap kernel for performance
    ksize = min(ksize, 51)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

    if img.ndim == 2:
        bg = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        bg = cv2.GaussianBlur(bg, (ksize, ksize), ksize / 4)
    else:
        bg = np.empty_like(img)
        for c in range(img.shape[2]):
            opened = cv2.morphologyEx(img[:, :, c], cv2.MORPH_OPEN, kernel)
            bg[:, :, c] = cv2.GaussianBlur(opened, (ksize, ksize), ksize / 4)

    return bg


def reduce_halos(
    image: np.ndarray,
    halo_strength: float = 0.50,
    core_protect: float = 0.85,
    halo_radius: float = 3.5,
    sensitivity: float = 0.50,
    chroma_reduce: float = 0.30,
    _progress_cb: Optional[Callable[..., None]] = None,
    **_kw: Any,
) -> Dict[str, np.ndarray]:
    """
    Fast, natural halo reduction.

    Parameters
    ----------
    image : float32 [0,1], HxW or HxWx3
    halo_strength : 0-1, how aggressively to suppress halos
    core_protect : 0-1, how much to protect star cores
    halo_radius : multiplier for halo extent (in star radii)
    sensitivity : 0-1, star detection sensitivity
    chroma_reduce : 0-1, reduce color fringing in halos
    """
    img = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    halo_strength = float(np.clip(halo_strength, 0.0, 1.0))
    core_protect = float(np.clip(core_protect, 0.0, 1.0))
    halo_radius = float(np.clip(halo_radius, 1.5, 8.0))
    sensitivity = float(np.clip(sensitivity, 0.0, 1.0))
    chroma_reduce = float(np.clip(chroma_reduce, 0.0, 1.0))

    total = 4
    _emit(_progress_cb, "Yıldızlar tespit ediliyor", step=1, total=total)

    # ── 1. Detect stars ──
    lum = _luminance(img)
    stars = _detect_stars_fast(lum, sensitivity)

    if not stars:
        _emit(_progress_cb, "Yıldız bulunamadı", step=total, total=total,
              preview=img)
        return {"result": img.copy(), "n_stars": 0}

    _emit(_progress_cb, f"{len(stars)} yıldız bulundu, halo maskesi oluşturuluyor",
          step=2, total=total)

    # ── 2. Build masks ──
    h, w = img.shape[:2]
    core_frac = 0.4 + core_protect * 0.5  # 0.4-0.9
    feather = max(1.5, np.median([s[2] for s in stars]) * 0.6)
    halo_mask, core_mask = _build_halo_mask(
        (h, w), stars,
        halo_multiplier=halo_radius,
        core_fraction=core_frac,
        feather_sigma=feather,
    )

    _emit(_progress_cb, "Yerel arka plan hesaplanıyor", step=3, total=total)

    # ── 3. Estimate local background ──
    local_bg = _estimate_local_background(img, stars, scale=halo_radius * 1.5)

    # ── 4. Suppress halos ──
    _emit(_progress_cb, "Halolar bastırılıyor", step=4, total=total)

    # Pull halo regions toward local background
    # strength controls how much we pull: 0 = no change, 1 = fully replace with bg
    pull = halo_mask * halo_strength
    # Protect cores
    pull = pull * (1.0 - core_mask * core_protect)

    if img.ndim == 3:
        pull_3d = pull[:, :, np.newaxis]
        result = img * (1.0 - pull_3d) + local_bg * pull_3d
    else:
        result = img * (1.0 - pull) + local_bg * pull

    # ── 5. Chroma cleanup in halo zones ──
    if chroma_reduce > 0.01 and img.ndim == 3:
        result_lum = _luminance(result)
        for c in range(3):
            # Difference from luminance = chroma component
            chroma_diff = result[:, :, c] - result_lum
            # Suppress chroma in halo regions
            suppression = halo_mask * chroma_reduce * (1.0 - core_mask * 0.7)
            result[:, :, c] = result[:, :, c] - chroma_diff * suppression

    result = np.clip(result, 0.0, 1.0).astype(np.float32)

    _emit(_progress_cb, "Halo temizliği tamamlandı", step=total, total=total,
          preview=result)

    return {
        "result": result,
        "halo_mask": halo_mask,
        "core_mask": core_mask,
        "n_stars": len(stars),
    }
