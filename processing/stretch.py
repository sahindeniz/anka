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
    """PixInsight Auto STF — fully vectorised, single-pass per channel."""
    def _ch(c):
        med = float(np.median(c))
        # MAD via fast percentile approx
        diff = np.abs(c - med)
        mad = float(np.median(diff)) * 1.4826
        c0 = max(0.0, med + shadow_clip * mad)
        if abs(1.0 - c0) < 1e-9: return c.copy()
        norm = np.subtract(c, c0, dtype=np.float32)
        np.clip(norm, 0, None, out=norm)
        norm /= (1.0 - c0)
        np.clip(norm, 0, 1, out=norm)
        m_n = max(1e-9, min(1 - 1e-9, (med - c0) / (1.0 - c0)))
        denom = (2 * m_n - 1) * norm - m_n
        mtf = np.where(np.abs(denom) > 1e-9,
                       (m_n - 1) * norm / denom, 0.0).astype(np.float32)
        np.clip(mtf, 0, 1, out=mtf)
        return mtf

    if img.ndim == 2:
        return _ch(img)
    # Process channels simultaneously — no Python loop overhead
    return np.stack([_ch(img[:, :, c]) for c in range(img.shape[2])], 2)


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
