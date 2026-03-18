"""
Astro Mastro Pro — Nebula & Galaxy Enhancement
AI-guided multi-scale enhancement with structure preservation.

Methods:
  multiscale_lce  — Multi-scale Local Contrast Enhancement (PixInsight LCE)
  hdrgc           — HDR Gamma Compression (PixInsight HDRMTFTransformation)
  structure_amp   — Structure Amplification (guided detail extraction)
  clahe_astro     — CLAHE adapted for astronomical imaging
"""

import cv2
import numpy as np


def enhance_nebula(image, method="multiscale_lce", strength=2.0,
                   blur_kernel=51, blend=1.0, clahe_clip=3.0, **kwargs):
    img = np.clip(image, 0, 1).astype(np.float32)

    if method == "multiscale_lce":
        enhanced = _multiscale_lce(img, float(strength), int(blur_kernel)|1)
    elif method == "hdrgc":
        enhanced = _hdrgc(img, float(strength))
    elif method == "structure_amp":
        enhanced = _structure_amp(img, float(strength))
    elif method == "clahe_astro":
        enhanced = _clahe_astro(img, float(clahe_clip), float(strength))
    else:
        # Legacy gaussian difference
        ks = int(blur_kernel) | 1
        if img.ndim == 2:
            blur = cv2.GaussianBlur(img, (ks,ks), 0)
        else:
            blur = np.stack([cv2.GaussianBlur(img[:,:,c],(ks,ks),0)
                             for c in range(img.shape[2])], axis=2)
        enhanced = np.clip(img + (img - blur) * float(strength), 0, 1)

    result = (1.0 - float(blend)) * img + float(blend) * enhanced
    return np.clip(result, 0, 1).astype(np.float32)


def _multiscale_lce(img, strength, ks):
    """
    Multi-scale LCE: PixInsight-inspired.
    Extracts detail at 3 scales, weights by local significance.
    """
    def process_ch(ch):
        result = ch.copy()
        for scale_ks in [ks//4|1, ks//2|1, ks]:
            scale_ks = max(3, scale_ks)
            blur = cv2.GaussianBlur(ch, (scale_ks, scale_ks), 0)
            detail = ch - blur
            # Noise-adaptive weight
            mad = float(np.median(np.abs(detail))) * 1.4826
            weight = np.tanh(np.abs(detail) / max(mad * 2, 1e-6))
            result += detail * weight * (strength / 3.0)
        return result

    if img.ndim == 2:
        return np.clip(process_ch(img), 0, 1)
    return np.clip(np.stack([process_ch(img[:,:,c])
                              for c in range(img.shape[2])], axis=2), 0, 1)


def _hdrgc(img, strength):
    """
    HDR Gamma Compression — brings out faint nebula while preserving cores.
    PixInsight HDRMTFTransformation equivalent.
    """
    gamma = 1.0 / max(1.0 + strength * 0.5, 0.1)
    compressed = np.power(np.clip(img, 1e-9, 1), gamma)

    # Luminosity-based blend: protect highlights
    lum = img.mean(axis=2, keepdims=True) if img.ndim == 3 else img
    highlight_mask = np.clip(lum * 2, 0, 1)
    return (1 - highlight_mask) * compressed + highlight_mask * img


def _structure_amp(img, strength):
    """
    AI-guided structure amplification.
    Uses gradient coherence to find filaments and amplify them.
    """
    gray = img if img.ndim == 2 else img.mean(axis=2)
    gray8 = (np.clip(gray, 0, 1) * 255).astype(np.uint8)

    # Gradient magnitude and direction
    gx = cv2.Sobel(gray8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray8, cv2.CV_32F, 0, 1, ksize=3)
    gmag = np.sqrt(gx**2 + gy**2) / 255.0

    # Structure map: coherent gradients = filaments
    structure = cv2.GaussianBlur(gmag, (21, 21), 0)
    structure = structure / (structure.max() + 1e-9)

    if img.ndim == 3:
        structure = structure[:, :, np.newaxis]

    enhanced = img + img * structure * strength
    return np.clip(enhanced, 0, 1)


def _clahe_astro(img, clip_limit, strength):
    """CLAHE adapted for 32-bit astronomical images."""
    def process_ch(ch):
        ch16 = (np.clip(ch, 0, 1) * 65535).astype(np.uint16)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        result16 = clahe.apply(ch16)
        result = result16.astype(np.float32) / 65535.0
        return ch + (result - ch) * strength

    if img.ndim == 2:
        return np.clip(process_ch(img), 0, 1)
    return np.clip(np.stack([process_ch(img[:,:,c])
                              for c in range(img.shape[2])], axis=2), 0, 1)
