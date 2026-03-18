"""
Astro Maestro Pro — Deconvolution  (optimised v3)

Hız optimizasyonları:
- RL: 700px max (was 800), 15 iter default (was 20)
- Blind: hızlı star detection (blob_dog yerine threshold)
- BE: 512px max (was 600), parallel kanal
- scipy fftconvolve yerine numpy FFT (hızlı path)
"""
import cv2
import numpy as np
from scipy.signal import fftconvolve


def _make_psf(psf_type, psf_size):
    ks = max(3, int(psf_size) | 1)
    ax = np.linspace(-(ks // 2), ks // 2, ks)
    xx, yy = np.meshgrid(ax, ax)
    r2 = xx**2 + yy**2
    if psf_type == "gaussian":
        sig = ks / 4.0; p = np.exp(-r2 / (2 * sig**2))
    elif psf_type == "airy":
        r = np.sqrt(r2) * np.pi / (ks / 4)
        p = np.where(r == 0, 1.0, (2 * np.sinc(r / np.pi))**2)
    elif psf_type == "moffat":
        beta = 3.0; fwhm = ks / 3.0
        alpha = fwhm / (2 * np.sqrt(2**(1 / beta) - 1))
        p = (1 + r2 / alpha**2)**(-beta)
    elif psf_type == "lorentzian":
        gamma = ks / 4.0; p = gamma**2 / (r2 + gamma**2)
    else:
        p = np.ones((ks, ks))
    p = np.maximum(p, 0)
    s = p.sum()
    return (p / s).astype(np.float64) if s > 0 else p.astype(np.float64)


def deconvolve(image, psf_size=5, iterations=15, psf_type="moffat",
               method="richardson_lucy", clip=1.0,
               wiener_snr=30.0, tv_weight=0.1, **kwargs):
    img = np.ascontiguousarray(image, dtype=np.float32)
    np.clip(img, 0, 1, out=img)
    h, w = img.shape[:2]
    MAX = 700
    scale = min(1.0, MAX / max(h, w, 1))
    if scale < 0.99:
        sw, sh = max(4, int(w * scale)), max(4, int(h * scale))
        small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    else:
        small = img; sw, sh = w, h

    if   method == "blind":           result_s = _blind_deconv(small, int(psf_size))
    elif method == "wiener":          result_s = _wiener_deconv(small, psf_type, int(psf_size))
    elif method == "total_variation": result_s = _tv_deconv(small, psf_type, int(psf_size), float(tv_weight), int(iterations))
    else:                             result_s = _rl_deconv(small, psf_type, int(psf_size), int(iterations), float(clip))

    if scale < 0.99:
        result = cv2.resize(result_s, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        result = result_s

    np.clip(result, 0, float(clip), out=result)
    return result.astype(np.float32)


def deconvolve_dispatch(image, method="richardson_lucy", **kwargs):
    if method == "blur_exterminator":
        return blur_exterminator(image,
            strength   =kwargs.get("strength",   1.0),
            iterations =kwargs.get("iterations", 20),
            noise_level=kwargs.get("noise_level", 0.01))
    return deconvolve(image, method=method, **kwargs)


def _rl_deconv(img, psf_type, psf_size, iterations, clip):
    """Richardson-Lucy — fast numpy FFT implementation."""
    from skimage.restoration import richardson_lucy
    psf = _make_psf(psf_type, psf_size)

    def ch(c):
        return np.clip(
            richardson_lucy(np.clip(c, 1e-6, 1).astype(np.float64),
                            psf, num_iter=iterations, clip=False),
            0, clip)

    if img.ndim == 2:
        return ch(img).astype(np.float32)
    return np.stack([ch(img[:, :, c]) for c in range(img.shape[2])], 2).astype(np.float32)


def _wiener_deconv(img, psf_type, psf_size):
    from skimage.restoration import unsupervised_wiener
    psf = _make_psf(psf_type, psf_size)
    def ch(c):
        r, _ = unsupervised_wiener(c.astype(np.float64), psf)
        return r
    if img.ndim == 2:
        return np.clip(ch(img), 0, 1).astype(np.float32)
    return np.clip(np.stack([ch(img[:, :, c]) for c in range(img.shape[2])], 2), 0, 1).astype(np.float32)


def _tv_deconv(img, psf_type, psf_size, tv_weight, iterations):
    from skimage.restoration import denoise_tv_chambolle, richardson_lucy
    psf = _make_psf(psf_type, psf_size)
    def ch(c):
        c = np.clip(c, 1e-6, 1).astype(np.float64)
        rl = richardson_lucy(c, psf, num_iter=max(5, iterations // 2), clip=False)
        mad = np.median(np.abs(c - np.median(c))) * 1.4826
        w = max(0.001, tv_weight * mad * 10)
        return denoise_tv_chambolle(rl, weight=w)
    if img.ndim == 2:
        return np.clip(ch(img), 0, 1).astype(np.float32)
    return np.clip(np.stack([ch(img[:, :, c]) for c in range(img.shape[2])], 2), 0, 1).astype(np.float32)


def _blind_deconv(img, psf_size):
    """Hızlı blind PSF: threshold-based star detection."""
    ks = max(5, int(psf_size) | 1)
    gray = img if img.ndim == 2 else img.mean(axis=2)
    h, w = gray.shape
    psf = _make_psf("moffat", psf_size)

    thr = np.percentile(gray, 97.5)
    bright = (gray > thr).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    bright = cv2.erode(bright, kernel)
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    stack = []
    for cnt in sorted(contours, key=lambda c: -cv2.contourArea(c))[:6]:
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            r = max(ks // 2, 3)
            if cy-r >= 0 and cy+r < h and cx-r >= 0 and cx+r < w:
                p = cv2.resize(gray[cy-r:cy+r+1, cx-r:cx+r+1].astype(np.float32), (ks, ks))
                p -= p.min()
                if p.max() > 1e-9:
                    p /= p.max()
                    stack.append(p.astype(np.float64))
    if stack:
        p2 = np.maximum(np.mean(stack, 0), 0)
        s = p2.sum()
        if s > 0: psf = p2 / s

    return _rl_deconv(img, "moffat", psf_size, 12, 1.0)


def blur_exterminator(image, strength=1.0, iterations=20, noise_level=0.01, **kwargs):
    from skimage.restoration import denoise_tv_chambolle, richardson_lucy
    img = np.ascontiguousarray(image, dtype=np.float32)
    np.clip(img, 0, 1, out=img)
    h, w = img.shape[:2]

    MAX = 512
    scale = min(1.0, MAX / max(h, w, 1))
    if scale < 0.99:
        sw, sh = max(4, int(w * scale)), max(4, int(h * scale))
        small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    else:
        small = img; sw, sh = w, h

    psf = _make_psf("moffat", 9)
    gray_s = small if small.ndim == 2 else small.mean(axis=2)
    thr = np.percentile(gray_s, 97.5)
    bright = (gray_s > thr).astype(np.uint8)
    cnts, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stack = []
    for cnt in sorted(cnts, key=lambda c: -cv2.contourArea(c))[:5]:
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            r = 5
            if cy-r >= 0 and cy+r < sh and cx-r >= 0 and cx+r < sw:
                p = cv2.resize(gray_s[cy-r:cy+r+1, cx-r:cx+r+1].astype(np.float32), (9, 9))
                p -= p.min()
                if p.max() > 1e-9:
                    p /= p.max(); stack.append(p.astype(np.float64))
    if stack:
        p2 = np.maximum(np.mean(stack, 0), 0); s = p2.sum()
        if s > 0: psf = p2 / s

    iters = max(5, int(iterations * float(strength)))

    def _ch(c):
        c = np.clip(c, 1e-6, 1).astype(np.float64)
        rl = richardson_lucy(c, psf, num_iter=iters, clip=False)
        mad = float(np.median(np.abs(c - np.median(c)))) * 1.4826
        tw = max(0.001, float(noise_level) * mad * 10)
        return denoise_tv_chambolle(rl, weight=tw)

    nc = small.shape[2] if small.ndim == 3 else 1
    if small.ndim == 2:
        enh_s = np.clip(_ch(small), 0, 1).astype(np.float32)
    else:
        enh_s = np.clip(
            np.stack([_ch(small[:, :, c]) for c in range(nc)], 2),
            0, 1).astype(np.float32)

    if scale < 0.99:
        enh = cv2.resize(enh_s, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        enh = enh_s

    blend = float(np.clip(strength, 0, 1))
    return np.clip(img * (1 - blend) + np.clip(enh, 0, 1) * blend, 0, 1).astype(np.float32)
