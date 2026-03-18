"""
Astro Maestro Pro — Noise Reduction  (optimised v3)
- Downsample to 1024px max (was 900)
- uint8 işlemler nerede mümkünse
- OpenCV CUDA flag
"""
import cv2
import numpy as np

_MAX_PX = 768  # çalışma çözünürlüğü

def reduce_noise(image, strength=0.7, method="bilateral", iterations=1, **kw):
    img = np.ascontiguousarray(image, dtype=np.float32)
    np.clip(img, 0, 1, out=img)
    s = float(np.clip(strength, 0, 1))
    n = max(1, min(3, int(iterations)))
    h, w = img.shape[:2]

    scale = min(1.0, _MAX_PX / max(h, w, 1))
    if scale < 0.99:
        sw, sh = max(4, int(w * scale)), max(4, int(h * scale))
        small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    else:
        small = img.copy()

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
