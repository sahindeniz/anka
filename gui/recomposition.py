"""
Astro Mastro Pro — Star Recomposition Dialog
by Deniz

Blends a starless image and a stars-only layer back together with full control:
  • Layer mode (Screen, Add, Lighten, Luminosity, Soft Light, Hard Light)
  • Star opacity (0–100%)
  • Star size scaling (shrink or enlarge stars via morphology)
  • Star colour (hue rotation, saturation boost)
  • Luminosity mask (protect bright nebula regions)
  • Live preview
"""

import numpy as np
import cv2


# ── Blend modes ───────────────────────────────────────────────────────────────
def blend(starless: np.ndarray, stars: np.ndarray,
          mode: str = "screen", opacity: float = 1.0) -> np.ndarray:
    """Blend stars layer onto starless using given mode."""
    s = np.clip(stars,   0, 1).astype(np.float32)
    b = np.clip(starless,0, 1).astype(np.float32)
    o = float(np.clip(opacity, 0, 1))

    if mode == "screen":
        out = 1.0 - (1.0 - b) * (1.0 - s)
    elif mode == "add":
        out = np.clip(b + s, 0, 1)
    elif mode == "lighten":
        out = np.maximum(b, s)
    elif mode == "soft_light":
        out = (1.0 - 2*s) * b**2 + 2*s*b
    elif mode == "hard_light":
        out = np.where(s < 0.5,
                       2*b*s,
                       1.0 - 2*(1-b)*(1-s))
    elif mode == "luminosity":
        # Apply star luminosity to starless colour
        if b.ndim == 3:
            lum_s = 0.2126*s[:,:,0]+0.7152*s[:,:,1]+0.0722*s[:,:,2]
            lum_b = 0.2126*b[:,:,0]+0.7152*b[:,:,1]+0.0722*b[:,:,2]
            lum_s = lum_s[:,:,np.newaxis]
            lum_b = lum_b[:,:,np.newaxis]
            out   = b * np.where(lum_b > 0, np.clip(lum_s/np.maximum(lum_b,1e-6),0,2), 1)
        else:
            out = s
    elif mode == "multiply":
        out = b * s
    elif mode == "overlay":
        out = np.where(b < 0.5, 2*b*s, 1.0-2*(1-b)*(1-s))
    else:
        out = np.clip(b + s, 0, 1)

    out = np.clip(out, 0, 1)
    return (b * (1.0-o) + out * o).astype(np.float32)


# ── Star size adjustment ───────────────────────────────────────────────────────
def adjust_star_size(stars: np.ndarray, factor: float = 1.0) -> np.ndarray:
    """
    Scale star sizes.
    factor < 1.0 → smaller stars (erosion/threshold)
    factor = 1.0 → no change
    factor > 1.0 → larger stars (dilation)
    """
    if abs(factor - 1.0) < 0.02:
        return stars

    img = np.clip(stars, 0, 1).astype(np.float32)
    img8 = (img * 255).astype(np.uint8)

    if factor < 1.0:
        # Shrink: erode then feather
        shrink = int(round((1.0 - factor) * 6))
        if shrink < 1: return stars
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (shrink*2+1, shrink*2+1))
        if img8.ndim == 3:
            result = np.stack([cv2.erode(img8[:,:,c], k, iterations=1)
                               for c in range(img8.shape[2])], axis=2)
        else:
            result = cv2.erode(img8, k, iterations=1)
        # Feather edges
        sigma = max(1.0, shrink * 0.5)
        if result.ndim == 3:
            result = np.stack([cv2.GaussianBlur(result[:,:,c], (0,0), sigma)
                               for c in range(result.shape[2])], axis=2)
        else:
            result = cv2.GaussianBlur(result, (0,0), sigma)
    else:
        # Grow: dilate then feather
        grow = int(round((factor - 1.0) * 6))
        if grow < 1: return stars
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow*2+1, grow*2+1))
        if img8.ndim == 3:
            result = np.stack([cv2.dilate(img8[:,:,c], k, iterations=1)
                               for c in range(img8.shape[2])], axis=2)
        else:
            result = cv2.dilate(img8, k, iterations=1)
        sigma = max(1.0, grow * 0.5)
        if result.ndim == 3:
            result = np.stack([cv2.GaussianBlur(result[:,:,c], (0,0), sigma)
                               for c in range(result.shape[2])], axis=2)
        else:
            result = cv2.GaussianBlur(result, (0,0), sigma)

    return np.clip(result.astype(np.float32) / 255.0, 0, 1)


# ── Star colour adjustment ────────────────────────────────────────────────────
def adjust_star_colour(stars: np.ndarray,
                        hue_shift: float = 0.0,
                        saturation: float = 1.0,
                        brightness: float = 1.0) -> np.ndarray:
    """
    Adjust star layer colour.
    hue_shift   : degrees (-180 to 180)
    saturation  : 0=grey, 1=original, 2=double
    brightness  : 0=black, 1=original, 2=double
    """
    img = np.clip(stars, 0, 1).astype(np.float32)
    if img.ndim == 2:
        return np.clip(img * float(brightness), 0, 1)

    # Brightness
    img = np.clip(img * float(brightness), 0, 1)

    if abs(hue_shift) < 0.5 and abs(saturation - 1.0) < 0.02:
        return img

    # Convert to HSV for hue/sat adjustment
    img8 = (img * 255).astype(np.uint8)
    bgr  = cv2.cvtColor(img8, cv2.COLOR_RGB2BGR)
    hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Hue shift
    hsv[:,:,0] = (hsv[:,:,0] + hue_shift / 2.0) % 180.0
    # Saturation
    hsv[:,:,1] = np.clip(hsv[:,:,1] * float(saturation), 0, 255)

    hsv8  = np.clip(hsv, 0, 255).astype(np.uint8)
    bgr2  = cv2.cvtColor(hsv8, cv2.COLOR_HSV2BGR)
    rgb2  = cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB)
    return rgb2.astype(np.float32) / 255.0


# ── Luminosity mask ───────────────────────────────────────────────────────────
def luminosity_mask(starless: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Build a luminosity protection mask.
    Bright regions (nebula) get less star overlay.
    """
    lum = starless if starless.ndim == 2 else (
        0.2126*starless[:,:,0] + 0.7152*starless[:,:,1] + 0.0722*starless[:,:,2])
    mask = np.clip(1.0 - (lum - threshold) / max(1.0 - threshold, 1e-6), 0, 1)
    return mask.astype(np.float32)


# ── Full recompose pipeline ───────────────────────────────────────────────────
def recompose(starless: np.ndarray, stars: np.ndarray,
              blend_mode:  str   = "screen",
              opacity:     float = 1.0,
              star_size:   float = 1.0,
              hue_shift:   float = 0.0,
              saturation:  float = 1.0,
              brightness:  float = 1.0,
              use_lum_mask:bool  = False,
              lum_threshold:float= 0.5) -> np.ndarray:
    """Full recomposition pipeline."""

    # 1. Adjust star size
    s = adjust_star_size(stars, star_size)

    # 2. Adjust star colour
    s = adjust_star_colour(s, hue_shift, saturation, brightness)

    # 3. Luminosity mask
    if use_lum_mask:
        mask = luminosity_mask(starless, lum_threshold)
        if s.ndim == 3:
            mask = mask[:,:,np.newaxis]
        s = s * mask

    # 4. Blend
    return blend(starless, s, mode=blend_mode, opacity=opacity)
