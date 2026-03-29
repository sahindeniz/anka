"""
Astro Maestro Pro — Histogram Stretch  (optimised v3)
Full vectorisation, float32 throughout, no Python loops.
"""
import numpy as np
import cv2

def stretch(image, low=2.0, high=98.0, method="auto_stf", gamma=1.0,
            hs_D=5000.0, hs_b=0.0, hs_SP=0.1, hs_LP=0.0, hs_HP=1.0,
            stat_k=2.5, power_alpha=0.5,
            stf_target=0.25, stf_clip=-2.8, **kwargs):

    img = np.ascontiguousarray(image, dtype=np.float32)
    np.clip(img, 0, 1, out=img)

    if   method == "auto_stf":    out = _auto_stf(img, float(stf_target), float(stf_clip))
    elif method == "hyperbolic":  out = _ghs(img, float(hs_D), float(hs_b), float(hs_SP), float(hs_LP), float(hs_HP))
    elif method == "asinh":       out = _asinh(img, float(low), float(high))
    elif method == "log":         out = _log(img, float(low), float(high))
    elif method == "midtone":     out = _midtone(img, float(low), float(high), float(gamma))
    elif method == "statistical": out = _statistical(img, float(stat_k))
    elif method == "power":       out = np.power(np.clip(img, 1e-9, 1), float(power_alpha), dtype=np.float32)
    else:                         out = _linear(img, float(low), float(high))

    if abs(gamma - 1.0) > 0.01 and method not in ("midtone", "power"):
        np.clip(out, 1e-9, 1, out=out)
        out = np.power(out, 1.0 / gamma, dtype=np.float32)

    np.clip(out, 0, 1, out=out)
    return out.astype(np.float32)


def _auto_stf(img, target=0.25, shadow_clip=-2.8):
    """PixInsight Auto STF — RENK KORUYUCU.

    Linked yontem + gamut koruma + hot pixel temizleme:
    1) Tek c0 ve m_n luminance'tan hesaplanir (tum kanallara ayni)
    2) MTF her kanala ayni parametrelerle uygulanir
    3) Gamut koruma: kanal > 1.0 ise orantili kucultme
    4) Renkli izole outlier temizleme (hot pixel / artefakt)
    """
    def _mtf_1d(data, c0, m_n):
        """Midtone Transfer Function — tek kanal."""
        if abs(1.0 - c0) < 1e-9:
            return data.copy()
        norm = np.subtract(data, c0, dtype=np.float32)
        np.clip(norm, 0, None, out=norm)
        norm /= (1.0 - c0)
        np.clip(norm, 0, 1, out=norm)
        denom = (2 * m_n - 1) * norm - m_n
        mtf = np.where(np.abs(denom) > 1e-9,
                       (m_n - 1) * norm / denom, 0.0).astype(np.float32)
        np.clip(mtf, 0, 1, out=mtf)
        return mtf

    def _clean_hot_pixels(data):
        """Izole renkli hot/cold pixel temizleme — OpenCV ile hızlı.
        Büyük resimlerde downscale ile maske hesaplar."""
        if data.ndim != 3 or data.shape[2] < 3:
            return data

        h, w = data.shape[:2]
        # Büyük resimlerde maske hesabını küçük kopya üzerinde yap
        max_px = 2000
        if max(h, w) > max_px:
            scale = max_px / max(h, w)
            sh, sw = int(h * scale), int(w * scale)
            small = cv2.resize(data, (sw, sh), interpolation=cv2.INTER_AREA)
        else:
            small = data
            sh, sw = h, w

        # OpenCV medianBlur — scipy'den 10x+ hızlı
        # medianBlur uint8 ister veya ksize=3/5 float32 kabul eder
        med = cv2.medianBlur(small, 3)

        eps = 1e-7
        lum = small[:, :, 0] + small[:, :, 1] + small[:, :, 2] + eps
        med_lum = med[:, :, 0] + med[:, :, 1] + med[:, :, 2] + eps

        # Renk fark metrigi
        color_diff = np.float32(0)
        for c in range(3):
            color_diff = color_diff + np.abs(small[:, :, c] / lum - med[:, :, c] / med_lum)

        mask_small = (color_diff > 0.15) & (lum < np.percentile(lum, 99))

        if not np.any(mask_small):
            return data

        # Maskeyi orijinal boyuta büyüt
        if max(h, w) > max_px:
            mask = cv2.resize(mask_small.astype(np.uint8), (w, h),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
            med_full = cv2.medianBlur(data, 3)
        else:
            mask = mask_small
            med_full = med

        cleaned = data.copy()
        for c in range(3):
            cleaned[:, :, c] = np.where(mask, med_full[:, :, c], data[:, :, c])
        return cleaned

    if img.ndim == 2:
        # Mono — klasik tek kanal STF
        med = float(np.median(img))
        diff = np.abs(img - med)
        mad = float(np.median(diff)) * 1.4826
        c0 = max(0.0, med + shadow_clip * mad)
        m_n = max(1e-9, min(1 - 1e-9, (med - c0) / (1.0 - c0)))
        return _mtf_1d(img, c0, m_n)

    # ── 0) Hot pixel temizleme — STF ONCESI (lineer veride) ────────
    img = _clean_hot_pixels(img)

    n_ch = img.shape[2]

    # ── 1) İstatistikleri topla (tek geçiş) ─────────────────────────
    # Linked parametreleri (luminance'tan)
    lum = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
    med = float(np.median(lum))
    mad = float(np.median(np.abs(lum - med))) * 1.4826
    c0 = max(0.0, med + shadow_clip * mad)
    m_n = max(1e-9, min(1 - 1e-9, (med - c0) / (1.0 - c0)))

    # Per-channel parametreleri
    ch_params = []
    for c in range(n_ch):
        ch = img[:, :, c]
        med_ch = float(np.median(ch))
        mad_ch = float(np.median(np.abs(ch - med_ch))) * 1.4826
        c0_ch = max(0.0, med_ch + shadow_clip * mad_ch)
        m_n_ch = max(1e-9, min(1 - 1e-9, (med_ch - c0_ch) / (1.0 - c0_ch)))
        ch_params.append((c0_ch, m_n_ch))

    # ── 2) Tek geçişte blend: %85 linked + %15 unlinked ────────────
    #   Linked ağırlıklı = renk oranlarını korur, kanal kaymasını önler
    alpha = 0.85
    result = np.empty_like(img, dtype=np.float32)
    for c in range(n_ch):
        linked_ch   = _mtf_1d(img[:, :, c], c0, m_n)
        unlinked_ch = _mtf_1d(img[:, :, c], ch_params[c][0], ch_params[c][1])
        result[:, :, c] = alpha * linked_ch + (1.0 - alpha) * unlinked_ch

    # ── 4) Gamut koruma ───────────────────────────────────────────
    max_ch = np.max(result, axis=2)
    overflow = max_ch > 1.0
    if np.any(overflow):
        safe_max = np.where(overflow, max_ch, 1.0)[:, :, np.newaxis]
        result = np.where(overflow[:, :, np.newaxis],
                          result / safe_max, result)

    np.clip(result, 0, 1, out=result)
    return result


def _linear(img, low, high):
    lo = float(np.percentile(img, low))
    hi = float(np.percentile(img, high))
    rng = max(hi - lo, 1e-9)
    out = np.subtract(img, lo, dtype=np.float32)
    out /= rng
    np.clip(out, 0, 1, out=out)
    return out


def _ghs(img, D=5000, b=0.0, SP=0.1, LP=0.0, HP=1.0):
    x  = img.astype(np.float64)
    q  = D * (x - SP)
    q0 = D * (LP - SP)
    q1 = D * (HP - SP)
    if b == 0.0:
        f  = np.log1p(np.abs(q))  * np.sign(q)
        f0 = np.log1p(abs(q0))    * np.sign(q0)
        f1 = np.log1p(abs(q1))    * np.sign(q1)
    else:
        eb = np.exp(b)
        f  = (eb**q  - 1) / (eb - 1)
        f0 = (eb**q0 - 1) / (eb - 1)
        f1 = (eb**q1 - 1) / (eb - 1)
    rng = max(f1 - f0, 1e-9)
    out = ((f - f0) / rng).astype(np.float32)
    np.clip(out, 0, 1, out=out)
    return out


def _asinh(img, low, high):
    lo  = float(np.percentile(img, low))
    hi  = float(np.percentile(img, high))
    rng = max(hi - lo, 1e-9)
    x   = (img - lo) / rng
    out = (np.arcsinh(x * 3) / np.arcsinh(3)).astype(np.float32)
    np.clip(out, 0, 1, out=out)
    return out


def _log(img, low, high):
    lo  = float(np.percentile(img, low))
    hi  = float(np.percentile(img, high))
    rng = max(hi - lo, 1e-9)
    x   = np.clip((img - lo) / rng, 1e-9, 1)
    out = (np.log1p(x * 9) / np.log(10)).astype(np.float32)
    np.clip(out, 0, 1, out=out)
    return out


def _midtone(img, low, high, gamma=1.0):
    lo  = float(np.percentile(img, low))
    hi  = float(np.percentile(img, high))
    rng = max(hi - lo, 1e-9)
    x   = np.clip((img - lo) / rng, 0, 1).astype(np.float32)
    if abs(gamma - 1.0) > 0.01:
        np.clip(x, 1e-9, 1, out=x)
        x = np.power(x, 1.0 / gamma, dtype=np.float32)
    return x


def _statistical(img, k=2.5):
    flat = img.ravel()
    med  = float(np.median(flat))
    mad  = float(np.median(np.abs(flat - med))) * 1.4826
    lo   = max(0.0, med - k * mad)
    hi   = min(1.0, med + k * mad)
    rng  = max(hi - lo, 1e-9)
    out  = np.clip((img - lo) / rng, 0, 1).astype(np.float32)
    return out
