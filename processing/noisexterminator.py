"""
Astro Maestro Pro - NoiseX-style denoise

Official RC Astro documentation highlights three core ideas that are safe to
approximate without copying any proprietary model internals:
1. Successive-approximation iterations.
2. Intensity/color separation.
3. High-frequency / low-frequency separation.

This module implements those principles with classical wavelet + frequency
processing so AstroMaestro can offer an honest, self-contained equivalent.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pywt


_LIFT_SCALE = 12.0


def noisexterminator(
    image,
    strength=0.7,
    detail=0.5,
    iterations=1,
    denoise_color=None,
    denoise_lf=None,
    denoise_lf_color=None,
    hf_lf_scale=5.0,
    linear_hint=None,
    **kw,
):
    del kw
    img = np.ascontiguousarray(image, dtype=np.float32)
    np.clip(img, 0, 1, out=img)

    s = float(np.clip(strength, 0, 1))
    d = float(np.clip(detail, 0, 1))
    n_iter = max(1, min(7, int(round(iterations))))
    color_strength = float(np.clip(s if denoise_color is None else denoise_color, 0, 1))
    low_strength = float(np.clip(s * 0.45 if denoise_lf is None else denoise_lf, 0, 1))
    low_color_strength = float(
        np.clip(color_strength if denoise_lf_color is None else denoise_lf_color, 0, 1)
    )
    hf_sigma = max(1.5, float(hf_lf_scale))

    lifted = _should_use_linear_lift(img, linear_hint)
    work = _forward_lift(img) if lifted else img.copy()

    if work.ndim == 2:
        denoised = _successive_component_denoise(
            work,
            strength=s,
            detail=d,
            iterations=n_iter,
            lf_strength=low_strength,
            hf_sigma=hf_sigma,
        )
        used_color = False
    else:
        denoised = _denoise_color_image(
            work,
            strength=s,
            detail=d,
            iterations=n_iter,
            color_strength=color_strength,
            lf_strength=low_strength,
            lf_color_strength=low_color_strength,
            hf_sigma=hf_sigma,
        )
        used_color = True

    if lifted:
        denoised = _reverse_lift(denoised)

    np.clip(denoised, 0, 1, out=denoised)
    meta = {
        "iterations": n_iter,
        "used_linear_lift": bool(lifted),
        "used_color_separation": bool(used_color),
        "hf_lf_scale": float(hf_sigma),
        "strength": s,
        "color_strength": color_strength,
        "lf_strength": low_strength,
        "lf_color_strength": low_color_strength,
    }
    return denoised.astype(np.float32), meta


def astro_noise_x(
    image,
    strength=0.75,
    detail=0.65,
    iterations=2,
    denoise_color=0.9,
    denoise_lf=0.4,
    denoise_lf_color=0.65,
    hf_lf_scale=6.0,
    linear_hint=None,
    **kw,
):
    """AstroMaestro's RC Astro-inspired NoiseX profile."""
    return noisexterminator(
        image,
        strength=strength,
        detail=detail,
        iterations=iterations,
        denoise_color=denoise_color,
        denoise_lf=denoise_lf,
        denoise_lf_color=denoise_lf_color,
        hf_lf_scale=hf_lf_scale,
        linear_hint=linear_hint,
        **kw,
    )


def _denoise_color_image(
    img: np.ndarray,
    *,
    strength: float,
    detail: float,
    iterations: int,
    color_strength: float,
    lf_strength: float,
    lf_color_strength: float,
    hf_sigma: float,
) -> np.ndarray:
    lum = _luminance(img)
    chroma = img - lum[:, :, None]

    lum_dn = _successive_component_denoise(
        lum,
        strength=strength,
        detail=detail,
        iterations=iterations,
        lf_strength=lf_strength,
        hf_sigma=hf_sigma,
    )

    def _denoise_ch(idx: int) -> np.ndarray:
        return _successive_component_denoise(
            chroma[:, :, idx],
            strength=color_strength,
            detail=detail,
            iterations=iterations,
            lf_strength=lf_color_strength,
            hf_sigma=hf_sigma,
        )

    with ThreadPoolExecutor(max_workers=min(img.shape[2], 3)) as pool:
        chroma_channels = list(pool.map(_denoise_ch, range(img.shape[2])))
    chroma_dn = np.stack(chroma_channels, axis=2)

    merged = lum_dn[:, :, None] + chroma_dn
    return np.clip(merged, 0, 1).astype(np.float32)


def _successive_component_denoise(
    channel: np.ndarray,
    *,
    strength: float,
    detail: float,
    iterations: int,
    lf_strength: float,
    hf_sigma: float,
) -> np.ndarray:
    out = np.asarray(channel, dtype=np.float32).copy()
    step_strength = strength / max(iterations, 1)
    step_lf = lf_strength / max(iterations, 1)

    for _ in range(iterations):
        low = cv2.GaussianBlur(out, (0, 0), hf_sigma, borderType=cv2.BORDER_REFLECT)
        high = out - low

        high_dn = _wavelet_denoise_any(high, step_strength, detail)
        if step_lf > 1e-6:
            low_smoothed = cv2.GaussianBlur(
                low,
                (0, 0),
                max(1.0, hf_sigma * (0.8 + step_lf * 1.2)),
                borderType=cv2.BORDER_REFLECT,
            )
            low = low * (1.0 - step_lf) + low_smoothed * step_lf

        out = low + high_dn

    return out.astype(np.float32)


def _wavelet_denoise_any(channel: np.ndarray, strength: float, detail: float) -> np.ndarray:
    h, w = channel.shape
    max_px = 2048
    scale = min(1.0, max_px / max(h, w, 1))
    if scale < 0.99:
        sw, sh = max(4, int(w * scale)), max(4, int(h * scale))
        ch = cv2.resize(channel, (sw, sh), interpolation=cv2.INTER_AREA)
    else:
        ch = channel.copy()

    wavelet = "db4"
    level = min(4, pywt.dwt_max_level(min(ch.shape), wavelet))
    coeffs = pywt.wavedec2(ch, wavelet, level=level)

    sigma = _estimate_noise(coeffs[-1][0])
    thr = sigma * np.sqrt(2.0 * np.log(max(ch.size, 2))) * float(strength)

    new_coeffs = [coeffs[0]]
    for idx, coeff_triplet in enumerate(coeffs[1:], start=1):
        preserve = 1.0 - detail * (idx - 1) / max(level, 1)
        thr_level = max(thr * preserve, 0.0)
        new_coeffs.append(
            tuple(pywt.threshold(coeff, thr_level, mode="soft") for coeff in coeff_triplet)
        )

    denoised = pywt.waverec2(new_coeffs, wavelet).astype(np.float32)
    denoised = denoised[: ch.shape[0], : ch.shape[1]]

    if scale < 0.99:
        denoised = cv2.resize(denoised, (w, h), interpolation=cv2.INTER_LINEAR)

    return denoised.astype(np.float32)


def _estimate_noise(detail_band: np.ndarray) -> float:
    return float(np.median(np.abs(detail_band - np.median(detail_band)))) / 0.6745


def _should_use_linear_lift(img: np.ndarray, linear_hint) -> bool:
    if linear_hint is not None:
        return bool(linear_hint)

    gray = img if img.ndim == 2 else _luminance(img)
    med = float(np.median(gray))
    bright_fraction = float(np.mean(gray > 0.92))
    return med < 0.18 and bright_fraction < 0.01


def _forward_lift(img: np.ndarray) -> np.ndarray:
    scale = np.float32(_LIFT_SCALE)
    return (np.arcsinh(np.asarray(img, dtype=np.float32) * scale) / np.arcsinh(scale)).astype(
        np.float32
    )


def _reverse_lift(img: np.ndarray) -> np.ndarray:
    scale = np.float32(_LIFT_SCALE)
    return (np.sinh(np.asarray(img, dtype=np.float32) * np.arcsinh(scale)) / scale).astype(
        np.float32
    )


def _luminance(img: np.ndarray) -> np.ndarray:
    return (
        0.2126 * img[:, :, 0]
        + 0.7152 * img[:, :, 1]
        + 0.0722 * img[:, :, 2]
    ).astype(np.float32)


def denoise(image, strength=0.7, detail_preserve=0.5, **kw):
    """Compatibility wrapper used by generic noise dispatch."""
    return noisexterminator(image, strength=strength, detail=detail_preserve, **kw)
