"""
Astro Maestro Pro — Mastro Noise
NAFNet noise reduction — directly uses Siril's syqon_prism engine.

CPU hızlandırma: Büyük resimleri yarıya küçült → denoise → geri büyüt.
GPU varsa tam çözünürlükte çalışır.
"""
from __future__ import annotations
import os, sys, numpy as np
import cv2
from pathlib import Path
from typing import Optional, Callable

_SYQON_DIR = Path(os.environ.get("LOCALAPPDATA", "")) / "siril" / "syqon_prism"
_MODEL_PATH = _SYQON_DIR / "prism_mini.pt"

# CPU'da max çalışma çözünürlüğü — daha büyük resimler küçültülüp işlenir
_CPU_MAX_PX = 2048


def _ensure_syqon_path():
    """syqon_prism klasörünü sys.path'e ekle."""
    d = str(_SYQON_DIR)
    if d not in sys.path:
        sys.path.insert(0, d)


def _has_gpu():
    """CUDA GPU mevcut mu kontrol et (cache'li)."""
    if not hasattr(_has_gpu, "_val"):
        try:
            import torch
            _has_gpu._val = torch.cuda.is_available()
        except Exception:
            _has_gpu._val = False
    return _has_gpu._val


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

    CPU modunda büyük resimleri otomatik küçültür → denoise → geri büyütür.
    Bu sayede CPU'da 3-5x hızlanma sağlanır, detay kaybı minimal kalır
    çünkü gürültü zaten yüksek frekanslı sinyal.

    GPU varsa tam çözünürlükte çalışır.
    """
    _ensure_syqon_path()
    from syqon_prism_inference import process_image as _syqon_process

    mp = Path(model_path) if model_path else _MODEL_PATH
    if not mp.exists():
        raise FileNotFoundError(
            f"Mastro Noise model bulunamadı: {mp}\n"
            f"prism_mini.pt dosyasını {_SYQON_DIR} klasörüne koyun.")

    gpu_available = _has_gpu() and use_gpu
    h, w = img.shape[:2]
    max_dim = max(h, w)

    # ── CPU hızlandırma: büyük resimleri küçült ─────────────────────
    downscaled = False
    if not gpu_available and max_dim > _CPU_MAX_PX:
        scale = _CPU_MAX_PX / max_dim
        new_w, new_h = int(w * scale), int(h * scale)
        print(f"[Mastro Noise] CPU mode: {w}x{h} → {new_w}x{new_h} "
              f"(scale={scale:.2f})", flush=True)
        small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        downscaled = True
        # Küçük resim için tile/overlap da küçült
        tile = min(tile, 256)
        overlap = min(overlap, 32)
    else:
        small = img

    # ── PyTorch thread sayısını artır ────────────────────────────────
    if not gpu_available:
        try:
            import torch
            n_cpu = max(1, os.cpu_count() or 4)
            torch.set_num_threads(n_cpu)
            torch.set_num_interop_threads(max(1, n_cpu // 2))
        except Exception:
            pass

    # ── Denoise ─────────────────────────────────────────────────────
    result = _syqon_process(
        small,
        tile=tile,
        overlap=overlap,
        use_amp=True,
        use_gpu=use_gpu,
        modulation=modulation,
        model_path=mp,
        progress_callback=progress_callback,
    )

    # ── Geri büyüt ──────────────────────────────────────────────────
    if downscaled:
        # Denoise sonucunu orijinal boyuta büyüt
        result_up = cv2.resize(result, (w, h), interpolation=cv2.INTER_LANCZOS4)
        # Orijinal ile blend: yüksek frekans detayları koru
        # denoise_lowfreq + original_highfreq
        orig_blur = cv2.GaussianBlur(img, (0, 0), sigmaX=1.5)
        result_blur = cv2.GaussianBlur(result_up, (0, 0), sigmaX=1.5)
        # Orijinalin yüksek frekans detayları
        high_freq = img - orig_blur
        # Denoise'un düşük frekansı + orijinal yüksek frekans
        result = result_blur + high_freq * 0.7
        print(f"[Mastro Noise] Upscaled back to {w}x{h} with detail preservation",
              flush=True)

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
