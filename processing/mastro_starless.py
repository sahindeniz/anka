"""
Astro Maestro Pro — Mastro Starless
NAFNet star removal — directly uses Siril's syqon_starless engine.
"""
from __future__ import annotations
import os, sys, numpy as np
from pathlib import Path
from typing import Optional, Callable, Tuple

_SYQON_DIR = Path(os.environ.get("LOCALAPPDATA", "")) / "siril" / "syqon_starless"
_MODEL_PATH = _SYQON_DIR / "zenith.pt"


def _ensure_syqon_path():
    """syqon_starless klasörünü sys.path'e ekle."""
    d = str(_SYQON_DIR)
    if d not in sys.path:
        sys.path.insert(0, d)


def process_starless(
    img: np.ndarray,
    tile: int = 368,
    overlap: int = 64,
    use_gpu: bool = True,
    model_path: Optional[str] = None,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Mastro Starless — NAFNet yıldız silme.

    Args:
        img:    float32 RGB [0,1]  (H,W,C) veya (H,W) mono
        tile:   Tile boyutu (px)
        overlap: Tile overlap (px)
        use_gpu: GPU kullan
        model_path: Özel model yolu (None = zenith.pt)
        progress_callback: İlerleme (0-100)

    Returns:
        (starless_image, star_mask)  — her ikisi de float32 [0,1]
    """
    _ensure_syqon_path()
    from syqon_starless_inference import process_image as _syqon_process
    from syqon_starless_inference import load_model as _syqon_load

    mp = Path(model_path) if model_path else _MODEL_PATH
    if not mp.exists():
        raise FileNotFoundError(
            f"Mastro Starless model bulunamadı: {mp}\n"
            f"zenith.pt dosyasını {_SYQON_DIR} klasörüne koyun.")

    # Syqon process_image float32 [0,1] HWC bekliyor, aynı bizim formatımız
    starless, star_mask = _syqon_process(
        img,
        tile=tile,
        overlap=overlap,
        generate_mask=True,
        mask_method="subtraction",
        use_amp=True,
        use_gpu=use_gpu,
        model_path=mp,
        progress_callback=progress_callback,
    )

    starless = np.clip(starless, 0, 1).astype(np.float32)
    if star_mask is not None:
        star_mask = np.clip(star_mask, 0, 1).astype(np.float32)

    return starless, star_mask


def reset_model():
    """Model önbelleğini temizle."""
    _ensure_syqon_path()
    try:
        import syqon_starless_inference as _ssi
        _ssi._MODEL = None
        _ssi._DEVICE = None
    except Exception:
        pass
