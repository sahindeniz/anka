"""
Astro Maestro Pro — Noise Reduction  (optimised v3)
- Downsample to 1024px max (was 900)
- uint8 işlemler nerede mümkünse
- OpenCV CUDA flag
"""
import cv2
import numpy as np

_MAX_PX = 4096  # çalışma çözünürlüğü — astro detayları korumak için yüksek tut

def reduce_noise(image, strength=0.7, method="bilateral", iterations=1,
                  progress_cb=None, **kw):
    img = np.ascontiguousarray(image, dtype=np.float32)
    np.clip(img, 0, 1, out=img)
    s = float(np.clip(strength, 0, 1))
    n = max(1, min(3, int(iterations)))
    modulation = float(kw.get("modulation", 1.0))
    detail = float(kw.get("detail", 0.5))

    # ── Harici modül dispatch ───────────────────────────────────────
    if method == "mastro_noise":
        try:
            from processing.mastro_noise import process_denoise
            result = process_denoise(img, modulation=modulation,
                                      progress_callback=progress_cb)
            return np.clip(result, 0, 1).astype(np.float32)
        except Exception as e:
            print(f"[NOISE] mastro_noise failed ({e}), fallback to bilateral", flush=True)
            method = "bilateral"

    if method == "silentium":
        try:
            from processing.veralux_silentium import denoise_silentium
            result = denoise_silentium(img, strength=s)
            return np.clip(result, 0, 1).astype(np.float32)
        except Exception as e:
            print(f"[NOISE] silentium failed ({e}), fallback to bilateral", flush=True)
            method = "bilateral"

    if method == "noisexterminator":
        try:
            from processing.noisexterminator import denoise as _nx_denoise
            result = _nx_denoise(img, strength=s, detail_preserve=detail)
            return np.clip(result, 0, 1).astype(np.float32)
        except Exception as e:
            print(f"[NOISE] noisexterminator failed ({e}), fallback to bilateral", flush=True)
            method = "bilateral"

    if method == "graxpert":
        try:
            from processing.graxpert_engine import graxpert_denoise
            result = graxpert_denoise(img, strength=s)
            return np.clip(result, 0, 1).astype(np.float32)
        except Exception as e:
            print(f"[NOISE] graxpert failed ({e}), fallback to bilateral", flush=True)
            method = "bilateral"

    # ── Klasik OpenCV methodlar (hızlı) ─────────────────────────────
    h, w = img.shape[:2]
    scale = min(1.0, _MAX_PX / max(h, w, 1))
    if scale < 0.99:
        sw, sh = max(4, int(w * scale)), max(4, int(h * scale))
        small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    else:
        small = img

    fn = {"gaussian": _gauss, "median": _median, "nlm": _nlm}.get(method, _bilateral)
    for _ in range(n):
        small = fn(small, s)

    if scale < 0.99:
        result = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        result = small

    np.clip(result, 0, 1, out=result)
    return result.astype(np.float32)


def _bilateral(img, s):
    d  = max(3, int(3 + s * 6))
    sc = max(10.0, s * 75.0)
    u8 = (img * 255).astype(np.uint8)
    out = cv2.bilateralFilter(u8, d, sc, sc)
    return out.astype(np.float32) / 255.0

def _gauss(img, s):
    ks = max(3, int(s * 10) | 1)
    u8 = (img * 255).astype(np.uint8)
    out = cv2.GaussianBlur(u8, (ks, ks), 0)
    return out.astype(np.float32) / 255.0

def _median(img, s):
    ks = max(3, min(7, int(s * 6) | 1))
    u8 = (img * 255).astype(np.uint8)
    return cv2.medianBlur(u8, ks).astype(np.float32) / 255.0

def _nlm(img, s):
    h_val = max(3, int(s * 12))
    u8 = (img * 255).astype(np.uint8)
    # Küçük template+search window = çok daha hızlı
    if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
        out = cv2.fastNlMeansDenoising(u8, None, h_val, 5, 15)
    else:
        out = cv2.fastNlMeansDenoisingColored(u8, None, h_val, h_val, 5, 15)
    return out.astype(np.float32) / 255.0
