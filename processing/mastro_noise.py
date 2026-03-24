"""
Astro Maestro Pro — Mastro Noise
NAFNet noise reduction — directly uses Siril's syqon_prism engine.
"""
from __future__ import annotations
import os, sys, numpy as np
from pathlib import Path
from typing import Optional, Callable

_SYQON_DIR = Path(os.environ.get("LOCALAPPDATA", "")) / "siril" / "syqon_prism"
_MODEL_PATH = _SYQON_DIR / "prism_mini.pt"


def _ensure_syqon_path():
    """syqon_prism klasörünü sys.path'e ekle."""
    d = str(_SYQON_DIR)
    if d not in sys.path:
        sys.path.insert(0, d)


def process_denoise(
    img: np.ndarray,
    tile: int = 512,
    overlap: int = 64,
    modulation: float = 1.0,
    use_gpu: bool = True,
    model_path: Optional[str] = None,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> np.ndarray:
    """
    Mastro Noise — NAFNet gürültü azaltma.

    Args:
        img:         float32 RGB [0,1] (H,W,C) veya (H,W) mono
        tile:        Tile boyutu (px)
        overlap:     Tile overlap (px)
        modulation:  0.0=orijinal, 1.0=tam denoise
        use_gpu:     GPU kullan
        model_path:  Özel model yolu (None = prism_mini.pt)
        progress_callback: İlerleme (0-100)

    Returns:
        Denoised float32 [0,1] görüntü
    """
    _ensure_syqon_path()
    from syqon_prism_inference import process_image as _syqon_process

    mp = Path(model_path) if model_path else _MODEL_PATH
    if not mp.exists():
        raise FileNotFoundError(
            f"Mastro Noise model bulunamadı: {mp}\n"
            f"prism_mini.pt dosyasını {_SYQON_DIR} klasörüne koyun.")

    result = _syqon_process(
        img,
        tile=tile,
        overlap=overlap,
        use_amp=True,
        use_gpu=use_gpu,
        modulation=modulation,
        model_path=mp,
        progress_callback=progress_callback,
    )

    return np.clip(result, 0, 1).astype(np.float32)


def reset_model():
    """Model önbelleğini temizle."""
    _ensure_syqon_path()
    try:
        import syqon_prism_inference as _spi
        _spi._MODEL = None
        _spi._DEVICE = None
    except Exception:
        pass
