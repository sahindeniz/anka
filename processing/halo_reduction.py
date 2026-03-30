"""
Astro Maestro Pro - natural halo reduction pipeline.

Builds a starless/stars workflow internally so halo cleanup can stay
subtle and preserve bright star cores.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import cv2
import numpy as np


def _emit_progress(
    progress_cb: Optional[Callable[..., None]],
    msg: str,
    *,
    step: int,
    total: int,
    preview: Optional[np.ndarray] = None,
) -> None:
    if progress_cb is None:
        return
    try:
        progress_cb(msg, step=step, total=total, preview=preview)
    except TypeError:
        try:
            progress_cb(msg)
        except Exception:
            pass


def _luminance(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img.astype(np.float32)
    return (
        0.2126 * img[:, :, 0]
        + 0.7152 * img[:, :, 1]
        + 0.0722 * img[:, :, 2]
    ).astype(np.float32)


def _core_recovery_mask(stars: np.ndarray) -> np.ndarray:
    lum = _luminance(stars)
    peak = float(np.max(lum)) if lum.size else 0.0
    if peak <= 1e-6:
        return np.zeros_like(lum, dtype=np.float32)

    active = lum[lum > peak * 0.08]
    if active.size == 0:
        return np.zeros_like(lum, dtype=np.float32)

    threshold = float(np.percentile(active, 80))
    mask = np.clip((lum - threshold) / max(peak - threshold, 1e-6), 0.0, 1.0)
    return cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.1)


def reduce_halos(
    image: np.ndarray,
    denoise_strength: float = 0.25,
    halo_strength: float = 0.15,
    chroma_cleanup: float = 0.35,
    core_protect: float = 0.70,
    recompose_opacity: float = 0.90,
    blend_mode: str = "screen",
    _progress_cb: Optional[Callable[..., None]] = None,
    **_kw: Any,
) -> Dict[str, np.ndarray]:
    """
    Reduce star halos while keeping the image natural.

    Returns a dict so the GUI can keep the generated starless/stars layers
    available for later recomposition.
    """
    from ai.star_net import separate_stars
    from gui.recomposition import blend
    from processing.noise_reduction import reduce_noise
    from processing.star_aberration import fix_aberration
    from processing.starsmaller import reduce_stars

    img = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    denoise_strength = float(np.clip(denoise_strength, 0.0, 1.0))
    halo_strength = float(np.clip(halo_strength, 0.0, 1.0))
    chroma_cleanup = float(np.clip(chroma_cleanup, 0.0, 1.0))
    core_protect = float(np.clip(core_protect, 0.0, 1.0))
    recompose_opacity = float(np.clip(recompose_opacity, 0.0, 1.0))
    blend_mode = str(blend_mode or "screen").strip().lower() or "screen"

    total = 5
    _emit_progress(_progress_cb, "Halo pipeline hazirlaniyor", step=1, total=total)

    max_dim = max(img.shape[:2]) if img.ndim >= 2 else 0
    sensitivity = 0.028 if max_dim <= 2200 else 0.024
    max_star_size = 14 if max_dim <= 2200 else 18
    growth_factor = 1.55 if max_dim <= 2200 else 1.75
    inpaint_radius = max(3, min(8, int(round(max_dim / 900.0))))

    _emit_progress(_progress_cb, "Yildiz katmani ayriliyor", step=2, total=total)
    separated = separate_stars(
        img,
        sensitivity=sensitivity,
        min_star_size=1,
        max_star_size=max_star_size,
        growth_factor=growth_factor,
        inpaint_radius=inpaint_radius,
        ai_enhance=True,
    )
    starless = np.clip(separated["starless"], 0.0, 1.0).astype(np.float32)
    stars = np.clip(separated["stars_only"], 0.0, 1.0).astype(np.float32)
    star_mask = np.clip(separated.get("star_mask", 0.0), 0.0, 1.0).astype(np.float32)

    _emit_progress(_progress_cb, "Arka plan hafif denoise ediliyor", step=3, total=total)
    if denoise_strength > 0.01:
        starless = reduce_noise(
            starless,
            method="mastro_noise",
            strength=max(0.12, min(1.0, denoise_strength * 1.35)),
            modulation=0.45 + denoise_strength * 0.35,
        )
        starless = np.clip(starless, 0.0, 1.0).astype(np.float32)

    _emit_progress(_progress_cb, "Halo katmani yumusatiliyor", step=4, total=total)
    processed_stars = stars.copy()
    if halo_strength > 0.01:
        shrink_strength = float(np.clip(0.18 + halo_strength * 1.05, 0.12, 0.95))
        threshold = float(max(0.01, 0.03 - halo_strength * 0.03))
        processed_stars, _ = reduce_stars(
            processed_stars,
            strength=shrink_strength,
            sensitivity=0.55,
            feather=4,
            max_sigma=8,
            min_sigma=1,
            threshold=threshold,
            protect_nebula=False,
        )
        processed_stars = np.clip(processed_stars, 0.0, 1.0).astype(np.float32)

    if chroma_cleanup > 0.01 and processed_stars.ndim == 3:
        processed_stars = fix_aberration(
            processed_stars,
            method="chromatic",
            chromatic_strength=chroma_cleanup,
            coma_strength=0.0,
            roundness_strength=0.0,
            spike_strength=0.0,
            sensitivity=0.40,
            protect_nebula=False,
        )
        processed_stars = np.clip(processed_stars, 0.0, 1.0).astype(np.float32)

    if core_protect > 0.0:
        core_mask = _core_recovery_mask(stars)
        if processed_stars.ndim == 3:
            core_mask = core_mask[:, :, np.newaxis]
        recovery = core_mask * core_protect
        processed_stars = (
            processed_stars * (1.0 - recovery) + stars * recovery
        ).astype(np.float32)
        np.clip(processed_stars, 0.0, 1.0, out=processed_stars)

    result = blend(starless, processed_stars, mode=blend_mode, opacity=recompose_opacity)
    result = np.clip(result, 0.0, 1.0).astype(np.float32)

    _emit_progress(
        _progress_cb,
        "Halo temizligi tamamlandi",
        step=5,
        total=total,
        preview=result,
    )
    return {
        "result": result,
        "starless": starless.astype(np.float32),
        "stars": processed_stars.astype(np.float32),
        "original_stars": stars.astype(np.float32),
        "star_mask": star_mask,
    }
