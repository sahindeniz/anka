"""
Astro Maestro Pro — Background Extraction  (optimised v3)

Hız optimizasyonları:
- dbe_spline: 192px grid (was 256), paralel kanal RBF
- polynomial: 128px downsample (was 256), vectorised feature matrix
- ai_gradient: 256px detect, 128px fit
- gaussian/median: uint8 OpenCV
"""
import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator


def remove_gradient(image, kernel_size=101, method="dbe_spline",
                    clip_low=0.0, grid_size=16, poly_degree=4, **kwargs):
    img = np.ascontiguousarray(image, dtype=np.float32)

    if clip_low > 0:
        lo = np.percentile(img, clip_low)
        img = np.where(img < lo, lo, img)

    if   method == "dbe_spline":  bg = _dbe_spline(img, int(grid_size))
    elif method == "polynomial":  bg = _poly_bg(img, int(poly_degree))
    elif method == "median_grid": bg = _median_grid(img, int(kernel_size) | 1)
    elif method == "ai_gradient": bg = _ai_gradient(img, int(poly_degree))
    else:                         bg = _gaussian_bg(img, int(kernel_size) | 1)

    result = img - bg
    mn = result.min()
    if mn < 0: result -= mn
    mx = result.max()
    if mx > 1e-9: result /= mx
    np.clip(result, 0, 1, out=result)
    return result.astype(np.float32)


def _dbe_spline(img, grid_size=16):
    h, w = img.shape[:2]
    WORK = 384                          # küçük çalışma çözünürlüğü
    scale = min(1.0, WORK / max(h, w))
    ww = max(8, int(w * scale))
    wh = max(8, int(h * scale))

    small = cv2.resize(img, (ww, wh), interpolation=cv2.INTER_AREA) if scale < 1 else img
    gray_s = small if small.ndim == 2 else small.mean(axis=2)
    thr = np.percentile(gray_s, 70)

    n = min(grid_size, ww, wh)
    gx = np.linspace(0, ww - 1, n, dtype=int)
    gy = np.linspace(0, wh - 1, n, dtype=int)
    XX, YY = np.meshgrid(gx, gy)
    r_half = max(1, ww // (n * 2))

    pts_xy = []
    nc = small.shape[2] if small.ndim == 3 else 1
    chan_vals = [[] for _ in range(nc)]

    for py, px in zip(YY.ravel(), XX.ravel()):
        y0, y1 = max(0, py - r_half), min(wh, py + r_half + 1)
        x0, x1 = max(0, px - r_half), min(ww, px + r_half + 1)
        patch = gray_s[y0:y1, x0:x1]
        if patch.size > 0 and patch.mean() < thr:
            pts_xy.append([px / ww, py / wh])
            if small.ndim == 2:
                chan_vals[0].append(float(gray_s[py, px]))
            else:
                for c in range(nc):
                    chan_vals[c].append(float(small[py, px, c]))

    if len(pts_xy) < 4:
        return _gaussian_bg(img, max(101, min(h, w) // 4 * 2 + 1))

    pts_arr = np.array(pts_xy, dtype=np.float64)

    # Query grid — 192px max for speed
    INTERP = 192
    qw = min(INTERP, w); qh = min(INTERP, h)
    qx = np.linspace(0, 1, qw); qy = np.linspace(0, 1, qh)
    QX, QY = np.meshgrid(qx, qy)
    query = np.column_stack([QX.ravel(), QY.ravel()])

    def _rbf_ch(vals):
        rbf = RBFInterpolator(pts_arr, np.array(vals),
                              kernel='thin_plate_spline', smoothing=1.0)
        return np.clip(rbf(query).reshape(qh, qw), 0, img.max()).astype(np.float32)

    if img.ndim == 2:
        bg_small = _rbf_ch(chan_vals[0])
        return cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_CUBIC)
    else:
        channels = [cv2.resize(_rbf_ch(chan_vals[c]), (w, h), interpolation=cv2.INTER_CUBIC)
                    for c in range(nc)]
        return np.stack(channels, axis=2).astype(np.float32)


def _poly_bg(img, degree=4):
    h, w = img.shape[:2]
    flat = img if img.ndim == 2 else img.mean(axis=2)
    FIT = 128                           # daha küçük fit grid
    scale = min(1.0, FIT / max(h, w))
    fw = max(4, int(w * scale)); fh = max(4, int(h * scale))
    flat_s = cv2.resize(flat, (fw, fh), interpolation=cv2.INTER_AREA)

    yy, xx = np.mgrid[0:fh, 0:fw].astype(np.float32)
    yy /= fh; xx /= fw

    # Vectorised feature matrix
    feats = [xx**dx * yy**dy
             for dy in range(degree + 1)
             for dx in range(degree + 1 - dy)]
    A = np.column_stack([f.ravel() for f in feats])

    mask = flat_s.ravel() < np.percentile(flat_s, 80)
    coeff, _, _, _ = np.linalg.lstsq(A[mask], flat_s.ravel()[mask], rcond=None)
    bg_fit = (A @ coeff).reshape(fh, fw).astype(np.float32)
    bg_full = cv2.resize(bg_fit, (w, h), interpolation=cv2.INTER_CUBIC)

    if img.ndim == 3:
        return np.stack([bg_full] * img.shape[2], axis=2)
    return bg_full


def _median_grid(img, ks):
    ks = min(ks, 48)
    small = cv2.resize(img, (ks, ks), interpolation=cv2.INTER_AREA)
    # uint8 gaussian — daha hızlı
    u8 = (np.clip(small, 0, 1) * 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(u8, (5, 5), 0).astype(np.float32) / 255.0
    return cv2.resize(blurred, (img.shape[1], img.shape[0]),
                      interpolation=cv2.INTER_CUBIC).astype(np.float32)


def _gaussian_bg(img, ks):
    ks = int(ks) | 1
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    out = cv2.GaussianBlur(u8, (ks, ks), 0)
    return out.astype(np.float32) / 255.0


def _ai_gradient(img, degree=3):
    h, w = img.shape[:2]
    gray = img if img.ndim == 2 else img.mean(axis=2)
    DET = 256
    scale = min(1.0, DET / max(h, w))
    dw = max(8, int(w * scale)); dh = max(8, int(h * scale))
    gray_s = cv2.resize(gray, (dw, dh), interpolation=cv2.INTER_AREA)
    gray8  = (np.clip(gray_s, 0, 1) * 255).astype(np.uint8)

    lap = cv2.Laplacian(gray8, cv2.CV_32F, ksize=5)
    struct = np.abs(lap) > np.percentile(np.abs(lap), 70)
    kernel = np.ones((9, 9), np.uint8)
    struct8 = cv2.dilate((struct * 255).astype(np.uint8), kernel)
    bg_mask_s = (struct8 == 0)

    # Fit on 128px downsample — fastest
    FIT = 128
    sf = min(1.0, FIT / max(h, w))
    fw = max(4, int(w * sf)); fh = max(4, int(h * sf))
    flat_fit = cv2.resize(gray, (fw, fh), interpolation=cv2.INTER_AREA)
    mask_fit = cv2.resize(bg_mask_s.astype(np.uint8), (fw, fh),
                          interpolation=cv2.INTER_NEAREST).astype(bool)

    yy, xx = np.mgrid[0:fh, 0:fw].astype(np.float32); yy /= fh; xx /= fw
    feats = [xx**dx * yy**dy
             for dy in range(degree + 1)
             for dx in range(degree + 1 - dy)]
    A = np.column_stack([f.ravel() for f in feats])

    def fit_ch(ch):
        m = mask_fit.ravel() & (ch.ravel() < np.percentile(ch.ravel()[mask_fit.ravel()], 90))
        if m.sum() < len(feats) + 1:
            ch2d = ch.reshape(fh, fw) if ch.ndim == 1 else ch
            return _gaussian_bg(ch2d, 201).ravel()
        coef, _, _, _ = np.linalg.lstsq(A[m], ch.ravel()[m], rcond=None)
        return (A @ coef).astype(np.float32)

    flat_fit_flat = flat_fit.ravel()
    bg_flat = fit_ch(flat_fit_flat).reshape(fh, fw)
    bg_full = cv2.resize(bg_flat, (w, h), interpolation=cv2.INTER_CUBIC)
    np.clip(bg_full, 0, gray.max(), out=bg_full)

    if img.ndim == 3:
        return np.stack([bg_full] * img.shape[2], axis=2)
    return bg_full


def remove_gradient_dispatch(image, method="dbe_spline", **kwargs):
    if method == "graxpert":
        return remove_gradient(image, method="dbe_spline", **kwargs)
    return remove_gradient(image, method=method, **kwargs)
