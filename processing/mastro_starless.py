"""
Astro Maestro Pro - Mastro Starless
Self-contained star-removal engine with no Siril dependency.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np


def _emit_progress(progress_callback: Optional[Callable[[int], None]], value: int) -> None:
    if progress_callback is not None:
        progress_callback(int(max(0, min(100, value))))


def process_starless(
    img: np.ndarray,
    tile: int = 368,
    overlap: int = 64,
    use_gpu: bool = True,
    model_path: Optional[str] = None,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Mastro Starless - self-contained star separation.

    Keeps the public signature stable, but does not require Siril, syqon,
    external model files, or GPU availability.
    """
    del tile, overlap, use_gpu, model_path

    from ai.star_net import separate_stars

    image = np.clip(np.asarray(img, dtype=np.float32), 0, 1)
    max_dim = max(image.shape[:2])

    sensitivity = 0.028 if max_dim <= 2200 else 0.024
    max_star_size = 14 if max_dim <= 2200 else 18
    growth_factor = 1.55 if max_dim <= 2200 else 1.75
    inpaint_radius = max(3, min(8, int(round(max_dim / 900.0))))

    _emit_progress(progress_callback, 8)
    _emit_progress(progress_callback, 24)
    result = separate_stars(
        image,
        sensitivity=sensitivity,
        min_star_size=1,
        max_star_size=max_star_size,
        growth_factor=growth_factor,
        inpaint_radius=inpaint_radius,
        ai_enhance=True,
    )

    _emit_progress(progress_callback, 86)
    starless = np.clip(result["starless"], 0, 1).astype(np.float32)
    star_mask = result.get("star_mask")
    if star_mask is not None:
        star_mask = np.clip(star_mask, 0, 1).astype(np.float32)

    _emit_progress(progress_callback, 100)
    return starless, star_mask


def reset_model() -> None:
    """Compatibility shim for old call sites."""
    return None
