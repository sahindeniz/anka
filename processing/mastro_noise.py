"""
Astro Maestro Pro - Mastro Noise
Self-contained denoise engine with no Siril dependency.
"""
from __future__ import annotations

from typing import Callable, Optional

import cv2
import numpy as np


_CPU_MAX_PX = 3072


def _emit_progress(progress_callback: Optional[Callable[[int], None]], value: int) -> None:
    if progress_callback is not None:
        progress_callback(int(max(0, min(100, value))))


def _resize_limit(img: np.ndarray, max_dim: int) -> tuple[np.ndarray, tuple[int, int], bool]:
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return img, (w, h), False

    scale = float(max_dim) / float(longest)
    new_w = max(8, int(round(w * scale)))
    new_h = max(8, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, (w, h), True


def _preserve_detail(original: np.ndarray, denoised: np.ndarray, amount: float) -> np.ndarray:
    sigma = 1.1
    if original.ndim == 2:
        orig_low = cv2.GaussianBlur(original, (0, 0), sigmaX=sigma)
        detail = original - orig_low
        restored = denoised + detail * amount
    else:
        restored = np.empty_like(denoised, dtype=np.float32)
        for c in range(original.shape[2]):
            orig_low = cv2.GaussianBlur(original[:, :, c], (0, 0), sigmaX=sigma)
            detail = original[:, :, c] - orig_low
            restored[:, :, c] = denoised[:, :, c] + detail * amount
    return np.clip(restored, 0, 1).astype(np.float32)


def process_denoise(
    img: np.ndarray,
    tile: int = 512,
    overlap: int = 64,
    modulation: float = 1.0,
    strength: Optional[float] = None,
    detail_preserve: Optional[float] = None,
    detail: Optional[float] = None,
    use_gpu: bool = True,
    model_path: Optional[str] = None,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> np.ndarray:
    """
    Mastro Noise - self-contained denoise.

    Keeps the public signature stable, but does not require Siril, syqon,
    external Python modules, or model files.
    """
    del tile, overlap, use_gpu, model_path

    from processing.noisexterminator import noisexterminator

    image = np.clip(np.asarray(img, dtype=np.float32), 0, 1)
    mix = float(np.clip(modulation, 0, 1))
    denoise_strength = float(np.clip(strength, 0, 1)) if strength is not None else 0.22 + 0.58 * mix
    detail_amount = detail_preserve if detail_preserve is not None else detail
    if detail_amount is None:
        detail_amount = max(0.25, 0.78 - 0.20 * mix)
    detail_amount = float(np.clip(detail_amount, 0, 1))

    _emit_progress(progress_callback, 5)
    work, original_size, was_downscaled = _resize_limit(image, _CPU_MAX_PX)

    _emit_progress(progress_callback, 20)
    denoised, _meta = noisexterminator(
        work,
        strength=denoise_strength,
        detail=detail_amount,
    )

    _emit_progress(progress_callback, 72)
    if was_downscaled:
        denoised = cv2.resize(denoised, original_size, interpolation=cv2.INTER_LANCZOS4)

    _emit_progress(progress_callback, 88)
    denoised = _preserve_detail(image, denoised, amount=0.08 + 0.16 * detail_amount)
    result = image * (1.0 - mix) + denoised * mix

    _emit_progress(progress_callback, 100)
    return np.clip(result, 0, 1).astype(np.float32)


def reset_model() -> None:
    """Compatibility shim for old call sites."""
    return None
