"""
Astro Mastro Pro — Image Stacking (DeepSkyStacker-style)

Calibration pipeline:
  1. Master Bias     = median of bias frames
  2. Master Dark     = kappa-sigma median(darks) - bias
  3. Master Flat     = normalize(median(flats) - dark_flat)
  4. Calibrated Light= (Light - Dark) / Flat
  5. Quality Score   = per-frame FWHM + SNR score → optional frame rejection
  6. Alignment       = ECC / ORB / SURF / SIFT / star-based
  7. Stacking        = Mean/Median/Kappa-Sigma/Winsorized/EntropyWeighted/Maximum

Methods match DSS defaults:
  - Default: Kappa-Sigma 2.0, 5 iterations
  - Auto-rejection of bad frames (quality < threshold)
  - Comet mode: star-aligned + comet-aligned dual stack
"""

import os
import numpy as np
import cv2
from typing import List, Optional, Callable

# ─── Loader ───────────────────────────────────────────────────────────────────
def _load(path: str) -> np.ndarray:
    from core.loader import load_image
    return load_image(path)

def _to_gray(img):
    if img.ndim == 3: return (img.mean(axis=2) * 255).astype(np.uint8)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)

def _to_u8(img):
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)

def _from_u8(img):
    return img.astype(np.float32) / 255.0


# ─── Quality Scoring (DSS-style) ─────────────────────────────────────────────
def score_frame(img: np.ndarray) -> dict:
    """
    Kare kalite skoru. DSS'deki gibi yildizlari tespit et,
    FWHM ve SNR hesapla.
    Returns dict: score, fwhm, snr, star_count, stars
    """
    gray = img if img.ndim == 2 else img.mean(axis=2)
    gray32 = gray.astype(np.float32)

    # Arka plani cikar
    from scipy.ndimage import gaussian_filter
    bg = gaussian_filter(gray32, sigma=20)
    residual = np.clip(gray32 - bg, 0, 1)

    # Yildiz tespiti: blob detection
    gray8 = (np.clip(residual, 0, 1) * 255).astype(np.uint8)
    thresh = np.percentile(gray8, 96)
    _, bw = cv2.threshold(gray8, int(thresh), 255, cv2.THRESH_BINARY)
    kernel = np.ones((3,3), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stars = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 3 <= area <= 500:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                # Yaklasik FWHM: alan'dan daire yaricapi
                fwhm = 2 * np.sqrt(area / np.pi)
                stars.append({"x": cx, "y": cy, "fwhm": fwhm, "area": area})

    n_stars = len(stars)
    if n_stars == 0:
        return {"score": 0.0, "fwhm": 99.0, "snr": 0.0,
                "star_count": 0, "stars": []}

    fwhm_vals = [s["fwhm"] for s in stars]
    mean_fwhm = float(np.median(fwhm_vals))

    # SNR: sinyal / gurultu
    signal = float(np.percentile(gray32, 95))
    noise  = float(np.std(gray32 - gaussian_filter(gray32, sigma=2)) + 1e-9)
    snr    = signal / noise

    # DSS benzeri skor: cok yildiz + dusuk FWHM + yuksek SNR
    score = (min(n_stars, 500) / 500) * 0.4 \
          + (1 / (1 + mean_fwhm / 4)) * 0.4 \
          + min(snr / 30, 1.0) * 0.2

    return {"score": float(score), "fwhm": mean_fwhm,
            "snr": float(snr), "star_count": n_stars, "stars": stars}


# ─── Calibration ──────────────────────────────────────────────────────────────
def _kappa_median(stack: np.ndarray, kappa=2.5, iters=3) -> np.ndarray:
    combined = stack.copy()
    for _ in range(iters):
        med = np.median(combined, axis=0)
        std = combined.std(axis=0) + 1e-9
        mask = np.abs(combined - med[None]) > kappa * std[None]
        combined = np.where(mask, np.nan, combined)
    result = np.nanmedian(combined, axis=0)
    return np.where(np.isnan(result), np.median(stack, axis=0), result).astype(np.float32)


def make_master_bias(bias_paths, progress_cb=None):
    if not bias_paths: return None
    frames = []
    for i, p in enumerate(bias_paths):
        if progress_cb: progress_cb(i+1, len(bias_paths), os.path.basename(p))
        frames.append(_load(p))
    return np.median(np.stack(frames, 0), axis=0).astype(np.float32)


def make_master_dark(dark_paths, master_bias=None, progress_cb=None):
    if not dark_paths: return None
    frames = []
    for i, p in enumerate(dark_paths):
        if progress_cb: progress_cb(i+1, len(dark_paths), os.path.basename(p))
        f = _load(p).astype(np.float32)
        if master_bias is not None: f = np.clip(f - master_bias, 0, 1)
        frames.append(f)
    return _kappa_median(np.stack(frames, 0)).astype(np.float32)


def make_master_dark_flat(dark_flat_paths, progress_cb=None):
    if not dark_flat_paths: return None
    frames = []
    for i, p in enumerate(dark_flat_paths):
        if progress_cb: progress_cb(i+1, len(dark_flat_paths), os.path.basename(p))
        frames.append(_load(p))
    return np.median(np.stack(frames, 0), axis=0).astype(np.float32)


def make_master_flat(flat_paths, dark_flat=None, master_bias=None, progress_cb=None):
    if not flat_paths: return None
    frames = []
    for i, p in enumerate(flat_paths):
        if progress_cb: progress_cb(i+1, len(flat_paths), os.path.basename(p))
        f = _load(p).astype(np.float32)
        if dark_flat is not None:
            f = np.clip(f - dark_flat, 0, 1)
        elif master_bias is not None:
            f = np.clip(f - master_bias, 0, 1)
        # Normalize
        mn = f.mean()
        if mn > 1e-9: f /= mn
        frames.append(f)
    master = np.median(np.stack(frames, 0), axis=0).astype(np.float32)
    mn = master.mean()
    if mn > 1e-9: master /= mn
    return master


def calibrate_light(light, master_dark=None, master_flat=None, master_bias=None):
    img = light.astype(np.float32)
    if master_bias is not None:
        img = np.clip(img - master_bias, 0, 1)
    if master_dark is not None:
        img = np.clip(img - master_dark, 0, 1)
    if master_flat is not None:
        flat_safe = np.where(master_flat < 0.01, 1.0, master_flat)
        img = np.clip(img / flat_safe, 0, 1)
    return img.astype(np.float32)


# ─── Alignment ────────────────────────────────────────────────────────────────
def align_frame(src: np.ndarray, ref: np.ndarray,
                method: str = "ecc_euclidean") -> np.ndarray:
    h, w = ref.shape[:2]
    if method == "none":
        return src

    src_g = _to_gray(src)
    ref_g = _to_gray(ref)

    if method.startswith("ecc"):
        return _align_ecc(src, ref_g, src_g, method, h, w)
    elif method == "orb_homography":
        return _align_orb(src, ref_g, src_g, h, w)
    elif method in ("sift_homography", "surf_homography"):
        return _align_feature(src, ref_g, src_g, method, h, w)
    elif method == "star_match":
        return _align_stars(src, ref, h, w)
    return src


def _align_ecc(src, ref_g, src_g, method, h, w):
    motion_map = {
        "ecc_translation": cv2.MOTION_TRANSLATION,
        "ecc_euclidean":   cv2.MOTION_EUCLIDEAN,
        "ecc_affine":      cv2.MOTION_AFFINE,
        "ecc_homography":  cv2.MOTION_HOMOGRAPHY,
    }
    mtype = motion_map.get(method, cv2.MOTION_EUCLIDEAN)
    if mtype == cv2.MOTION_HOMOGRAPHY:
        warp = np.eye(3, 3, dtype=np.float32)
    else:
        warp = np.eye(2, 3, dtype=np.float32)
    try:
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-6)
        _, warp = cv2.findTransformECC(ref_g, src_g, warp, mtype, criteria)
    except cv2.error:
        return src
    return _apply_warp(src, warp, h, w, mtype)


def _align_orb(src, ref_g, src_g, h, w):
    orb  = cv2.ORB_create(5000)
    kp1, d1 = orb.detectAndCompute(ref_g, None)
    kp2, d2 = orb.detectAndCompute(src_g, None)
    if d1 is None or d2 is None or len(kp1) < 4:
        return src
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(d1, d2, k=2)
    good = [m for m,n in matches if m.distance < 0.75*n.distance]
    if len(good) < 4:
        return src
    src_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None: return src
    return _apply_warp(src, H, h, w, cv2.MOTION_HOMOGRAPHY)


def _align_feature(src, ref_g, src_g, method, h, w):
    try:
        if method == "sift_homography":
            det = cv2.SIFT_create()
            norm = cv2.NORM_L2
        else:
            det = cv2.xfeatures2d.SURF_create(400) if hasattr(cv2,"xfeatures2d") else cv2.SIFT_create()
            norm = cv2.NORM_L2
        kp1, d1 = det.detectAndCompute(ref_g, None)
        kp2, d2 = det.detectAndCompute(src_g, None)
        if d1 is None or len(kp1) < 4: return src
        bf = cv2.BFMatcher(norm, crossCheck=False)
        matches = bf.knnMatch(d1, d2, k=2)
        good = [m for m,n in matches if m.distance < 0.75*n.distance]
        if len(good) < 4: return src
        src_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None: return src
        return _apply_warp(src, H, h, w, cv2.MOTION_HOMOGRAPHY)
    except Exception:
        return _align_orb(src, ref_g, src_g, h, w)


def _align_stars(src, ref, h, w):
    """Parlak yildizlari bul, koordinat eslesme ile hizala."""
    def get_star_centers(img, n=30):
        gray = img if img.ndim == 2 else img.mean(axis=2)
        g8 = (np.clip(gray, 0, 1) * 255).astype(np.uint8)
        thr = np.percentile(g8, 97)
        _, bw = cv2.threshold(g8, int(thr), 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centers = []
        for c in cnts:
            if 4 < cv2.contourArea(c) < 400:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    centers.append((M["m10"]/M["m00"], M["m01"]/M["m00"],
                                    cv2.contourArea(c)))
        centers.sort(key=lambda x: -x[2])
        return np.float32([[c[0],c[1]] for c in centers[:n]])

    ref_pts = get_star_centers(ref)
    src_pts = get_star_centers(src)
    if len(ref_pts) < 3 or len(src_pts) < 3:
        return _align_orb(src, _to_gray(ref), _to_gray(src), h, w)
    n = min(len(ref_pts), len(src_pts), 20)
    H, _ = cv2.findHomography(src_pts[:n], ref_pts[:n], cv2.RANSAC, 3.0)
    if H is None: return src
    return _apply_warp(src, H, h, w, cv2.MOTION_HOMOGRAPHY)


def _apply_warp(src, warp, h, w, mtype):
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    if src.ndim == 2:
        if mtype == cv2.MOTION_HOMOGRAPHY:
            return cv2.warpPerspective(src, warp, (w,h), flags=cv2.INTER_LINEAR)
        return cv2.warpAffine(src, warp, (w,h), flags=flags)
    channels = []
    for c in range(src.shape[2]):
        if mtype == cv2.MOTION_HOMOGRAPHY:
            channels.append(cv2.warpPerspective(src[:,:,c], warp, (w,h),
                                                flags=cv2.INTER_LINEAR))
        else:
            channels.append(cv2.warpAffine(src[:,:,c], warp, (w,h), flags=flags))
    return np.stack(channels, axis=2).astype(np.float32)


# ─── Stacking Methods ─────────────────────────────────────────────────────────
def _mean_stack(stack, **kw):
    return stack.mean(axis=0).astype(np.float32)

def _median_stack(stack, **kw):
    return np.median(stack, axis=0).astype(np.float32)

def _kappa_sigma(stack, kappa=2.0, iterations=5, **kw):
    combined = stack.astype(np.float64).copy()
    for _ in range(iterations):
        med = np.median(combined, axis=0)
        std = np.nanstd(combined, axis=0) + 1e-9
        mask = np.abs(combined - med[None]) > kappa * std[None]
        combined = np.where(mask, np.nan, combined)
    result = np.nanmean(combined, axis=0)
    fallback = np.median(stack, axis=0)
    return np.where(np.isnan(result), fallback, result).astype(np.float32)

def _winsorized(stack, kappa_low=2.0, kappa_high=2.0, iterations=5, **kw):
    combined = stack.astype(np.float64).copy()
    for _ in range(iterations):
        med = np.median(combined, axis=0)
        std = np.nanstd(combined, axis=0) + 1e-9
        lo  = med - kappa_low  * std
        hi  = med + kappa_high * std
        combined = np.clip(combined, lo[None], hi[None])
    return combined.mean(axis=0).astype(np.float32)

def _linear_fit(stack, **kw):
    """Weighted linear fit — frames with lower noise get higher weight."""
    n = stack.shape[0]
    weights = np.array([1.0 / (stack[i].std() + 1e-9) for i in range(n)])
    weights /= weights.sum()
    return (stack * weights[:, None, None, None]).sum(axis=0).astype(np.float32) \
        if stack.ndim == 4 else \
        (stack * weights[:, None, None]).sum(axis=0).astype(np.float32)

def _entropy_weighted(stack, **kw):
    """
    Entropy-weighted stacking: frames with more detail get higher weight.
    DSS'de de benzer bir agirliklandirma var.
    """
    n = stack.shape[0]
    weights = []
    for i in range(n):
        frame = stack[i]
        gray  = frame if frame.ndim == 2 else frame.mean(axis=2)
        hist, _ = np.histogram(gray, bins=256, range=(0,1), density=True)
        hist = hist[hist > 0]
        entropy = float(-np.sum(hist * np.log2(hist + 1e-12)))
        weights.append(entropy)
    weights = np.array(weights, dtype=np.float32)
    weights /= weights.sum() + 1e-9
    if stack.ndim == 4:
        return (stack * weights[:, None, None, None]).sum(axis=0).astype(np.float32)
    return (stack * weights[:, None, None]).sum(axis=0).astype(np.float32)

def _maximum_stack(stack, **kw):
    """Maximum stacking — comet / meteor / satellite trails."""
    return stack.max(axis=0).astype(np.float32)

def _minimum_stack(stack, **kw):
    return stack.min(axis=0).astype(np.float32)

def _sum_stack(stack, **kw):
    """Sum stacking (normalize sonra) — uzun pozlama simülasyonu."""
    s = stack.sum(axis=0).astype(np.float32)
    mx = s.max()
    return s / mx if mx > 0 else s


# ─── Main pipeline ────────────────────────────────────────────────────────────
def stack_lights(
    light_paths: List[str],
    dark_paths=None,
    flat_paths=None,
    dark_flat_paths=None,
    bias_paths=None,
    method: str = "kappa_sigma",
    align_method: str = "ecc_euclidean",
    ref_index: int = 0,
    ref_mode: str = "manual",       # manual | best_quality | median_quality
    kappa: float = 2.0,
    kappa_low: float = 2.0,
    kappa_high: float = 2.0,
    iterations: int = 5,
    quality_reject: bool = True,    # DSS: kotu kareleri reddet
    quality_threshold: float = 0.2, # alt eşik (0-1)
    normalize: bool = True,         # DSS: kareleri normalize et
    weight_mode: str = "none",      # none | snr | fwhm | stars
    progress_cb: Optional[Callable] = None,
) -> dict:

    def _cb(step, msg):
        if progress_cb: progress_cb(step, msg)

    # ── Kalibrasyon kareleri ──────────────────────────────────────────────────
    _cb(1, f"Master Bias hazırlaniyor..." if bias_paths else "Bias atlanıyor")
    master_bias = make_master_bias(bias_paths,
        lambda i,n,f: _cb(1, f"  Bias {i}/{n}: {f}")) if bias_paths else None

    _cb(2, f"Master Dark hazırlaniyor..." if dark_paths else "Dark atlanıyor")
    master_dark = make_master_dark(dark_paths, master_bias,
        lambda i,n,f: _cb(2, f"  Dark {i}/{n}: {f}")) if dark_paths else None

    _cb(3, f"Dark Flat hazırlaniyor..." if dark_flat_paths else "Dark Flat atlanıyor")
    master_dark_flat = make_master_dark_flat(dark_flat_paths,
        lambda i,n,f: _cb(3, f"  DarkFlat {i}/{n}: {f}")) if dark_flat_paths else None

    _cb(4, f"Master Flat hazırlaniyor..." if flat_paths else "Flat atlanıyor")
    master_flat = make_master_flat(flat_paths, master_dark_flat, master_bias,
        lambda i,n,f: _cb(4, f"  Flat {i}/{n}: {f}")) if flat_paths else None

    # ── Light kalibrasyonu ────────────────────────────────────────────────────
    _cb(5, f"Isik kareleri kalibre ediliyor ({len(light_paths)} kare)...")
    calibrated = []
    for i, p in enumerate(light_paths):
        _cb(5, f"  Kalibrasyon {i+1}/{len(light_paths)}: {os.path.basename(p)}")
        raw = _load(p)
        cal = calibrate_light(raw, master_dark, master_flat, master_bias)
        calibrated.append(cal)

    # ── Kalite skorlama ve referans secme ─────────────────────────────────────
    _cb(6, "Kalite skorlanıyor...")
    scores = []
    for i, img in enumerate(calibrated):
        sc = score_frame(img)
        scores.append(sc)
        _cb(6, f"  Kare {i+1}: skor={sc['score']:.3f} "
              f"FWHM={sc['fwhm']:.1f}px "
              f"yildiz={sc['star_count']} "
              f"SNR={sc['snr']:.1f}")

    # Referans kare sec
    if ref_mode == "best_quality":
        ref_idx = int(np.argmax([s["score"] for s in scores]))
        _cb(6, f"  En iyi kare: #{ref_idx+1} (skor={scores[ref_idx]['score']:.3f})")
    elif ref_mode == "median_quality":
        sc_arr = [s["score"] for s in scores]
        med_sc = np.median(sc_arr)
        ref_idx = int(np.argmin([abs(s - med_sc) for s in sc_arr]))
        _cb(6, f"  Medyan kalite kare: #{ref_idx+1}")
    else:
        ref_idx = max(0, min(int(ref_index), len(calibrated)-1))
        _cb(6, f"  Manuel referans: #{ref_idx+1}")

    # Kotu kareleri reddet (DSS stili)
    keep_idx = list(range(len(calibrated)))
    rejected = []
    if quality_reject and len(calibrated) > 3:
        sc_vals = [s["score"] for s in scores]
        max_sc  = max(sc_vals) if sc_vals else 1.0
        auto_thr = max(quality_threshold, max_sc * 0.15)
        keep_idx = [i for i in keep_idx if scores[i]["score"] >= auto_thr]
        rejected = [i for i in range(len(calibrated)) if i not in keep_idx]
        if ref_idx not in keep_idx:
            keep_idx.append(ref_idx)
        if rejected:
            _cb(6, f"  ⚠ Reddedilen kareler: {[i+1 for i in rejected]} "
                  f"(eşik={auto_thr:.3f})")

    used_frames = [calibrated[i] for i in keep_idx]
    used_paths  = [light_paths[i] for i in keep_idx]
    used_scores = [scores[i] for i in keep_idx]
    new_ref     = keep_idx.index(ref_idx) if ref_idx in keep_idx else 0
    reference   = used_frames[new_ref]

    _cb(6, f"  Kullanılacak kare: {len(used_frames)}/{len(calibrated)}")

    # ── Normalize ─────────────────────────────────────────────────────────────
    if normalize:
        _cb(6, "  Normalizasyon uygulanıyor...")
        ref_med = float(np.median(reference))
        for i in range(len(used_frames)):
            f_med = float(np.median(used_frames[i]))
            if f_med > 1e-9:
                used_frames[i] = np.clip(
                    used_frames[i] * (ref_med / f_med), 0, 1).astype(np.float32)

    # ── Hizalama ──────────────────────────────────────────────────────────────
    if align_method != "none":
        _cb(7, f"Hizalanıyor ({align_method}, {len(used_frames)} kare)...")
        aligned = []
        for i, frame in enumerate(used_frames):
            if i == new_ref:
                aligned.append(frame)
            else:
                _cb(7, f"  Hizalama {i+1}/{len(used_frames)}: {os.path.basename(used_paths[i])}")
                try:
                    al = align_frame(frame, reference, method=align_method)
                    aligned.append(al)
                except Exception as e:
                    _cb(7, f"  ⚠ Hizalama basarisiz kare {i+1}: {e}")
                    aligned.append(frame)
    else:
        _cb(7, "Hizalama atlandı")
        aligned = used_frames

    stack = np.stack(aligned, axis=0).astype(np.float32)

    # ── Agirliklandirma ───────────────────────────────────────────────────────
    frame_weights = None
    if weight_mode == "snr":
        snr_vals = np.array([s["snr"] for s in used_scores], dtype=np.float32)
        snr_vals = np.clip(snr_vals, 0.1, None)
        frame_weights = snr_vals / snr_vals.sum()
        _cb(7, f"  SNR agirlandirma: {frame_weights.round(3).tolist()}")
    elif weight_mode == "fwhm":
        fwhm_vals = np.array([max(s["fwhm"], 0.5) for s in used_scores], dtype=np.float32)
        fw = 1.0 / fwhm_vals
        frame_weights = fw / fw.sum()
    elif weight_mode == "stars":
        star_vals = np.array([max(s["star_count"], 1) for s in used_scores], dtype=np.float32)
        frame_weights = star_vals / star_vals.sum()

    # ── Stack ─────────────────────────────────────────────────────────────────
    _cb(8, f"Stack yapılıyor ({method}, {len(aligned)} kare)...")

    if frame_weights is not None and method in ("mean","entropy_weighted"):
        if stack.ndim == 4:
            result = (stack * frame_weights[:, None, None, None]).sum(axis=0)
        else:
            result = (stack * frame_weights[:, None, None]).sum(axis=0)
        result = result.astype(np.float32)
    else:
        stack_fns = {
            "mean":             _mean_stack,
            "median":           _median_stack,
            "kappa_sigma":      lambda s: _kappa_sigma(s, kappa=kappa, iterations=int(iterations)),
            "kappa_sigma_dual": lambda s: _kappa_sigma(s, kappa=kappa, iterations=int(iterations)),
            "winsorized":       lambda s: _winsorized(s, kappa_low=kappa_low,
                                                      kappa_high=kappa_high,
                                                      iterations=int(iterations)),
            "linear_fit":       _linear_fit,
            "entropy_weighted": _entropy_weighted,
            "maximum":          _maximum_stack,
            "minimum":          _minimum_stack,
            "sum":              _sum_stack,
        }
        fn = stack_fns.get(method, _kappa_sigma)
        result = fn(stack)

    result = np.clip(result, 0, 1).astype(np.float32)
    _cb(8, f"✅ Tamamlandi — {len(aligned)} kare birlestirildi")

    return {
        "result":         result,
        "master_dark":    master_dark,
        "master_flat":    master_flat,
        "master_bias":    master_bias,
        "n_lights":       len(aligned),
        "n_rejected":     len(rejected),
        "rejected_frames":rejected,
        "frame_scores":   scores,
        "method":         method,
        "align_method":   align_method,
        "ref_index":      ref_idx,
    }


# ─── 2-Aşamalı Pipeline ────────────────────────────────────────────────────────

def align_frames_only(
    light_paths: List[str],
    dark_paths=None,
    flat_paths=None,
    dark_flat_paths=None,
    bias_paths=None,
    align_method: str = "ecc_euclidean",
    ref_index: int = 0,
    ref_mode: str = "best_quality",
    normalize: bool = True,
    quality_reject: bool = True,
    quality_threshold: float = 0.2,
    progress_cb: Optional[Callable] = None,
) -> List[np.ndarray]:
    """
    Faz 1: Sadece kalibrasyon + kalite eleme + hizalama.
    Hizalanmış np.ndarray listesi döndürür (stacking yapılmaz).
    """
    def _cb(step, msg):
        if progress_cb: progress_cb(step, msg)

    _cb(1, f"Master Bias..." if bias_paths else "Bias atlanıyor")
    master_bias = make_master_bias(bias_paths,
        lambda i,n,f: _cb(1, f"  Bias {i}/{n}: {f}")) if bias_paths else None

    _cb(2, "Master Dark..." if dark_paths else "Dark atlanıyor")
    master_dark = make_master_dark(dark_paths, master_bias,
        lambda i,n,f: _cb(2, f"  Dark {i}/{n}: {f}")) if dark_paths else None

    _cb(3, "Dark Flat..." if dark_flat_paths else "Dark Flat atlanıyor")
    master_dark_flat = make_master_dark_flat(dark_flat_paths,
        lambda i,n,f: _cb(3, f"  DarkFlat {i}/{n}: {f}")) if dark_flat_paths else None

    _cb(4, "Master Flat..." if flat_paths else "Flat atlanıyor")
    master_flat = make_master_flat(flat_paths, master_dark_flat, master_bias,
        lambda i,n,f: _cb(4, f"  Flat {i}/{n}: {f}")) if flat_paths else None

    # Kalibrasyon
    _cb(5, f"Kalibrasyon ({len(light_paths)} kare)...")
    calibrated = []
    for i, p in enumerate(light_paths):
        _cb(5, f"  {i+1}/{len(light_paths)}: {os.path.basename(p)}")
        raw = _load(p)
        cal = calibrate_light(raw, master_dark, master_flat, master_bias)
        calibrated.append(cal)

    # Kalite skorlama
    _cb(6, "Kalite skorlanıyor...")
    scores = [score_frame(img) for img in calibrated]
    for i, sc in enumerate(scores):
        _cb(6, f"  #{i+1}: skor={sc['score']:.3f} yıldız={sc['star_count']}")

    # Referans seçimi
    if ref_mode == "best_quality":
        ref_idx = int(np.argmax([s["score"] for s in scores]))
    elif ref_mode == "median_quality":
        sc_arr = [s["score"] for s in scores]
        ref_idx = int(np.argmin([abs(s - np.median(sc_arr)) for s in sc_arr]))
    else:
        ref_idx = max(0, min(int(ref_index), len(calibrated)-1))

    _cb(6, f"  Referans kare: #{ref_idx+1}")

    # Kalite eleme
    keep_idx = list(range(len(calibrated)))
    if quality_reject and len(calibrated) > 3:
        sc_vals = [s["score"] for s in scores]
        max_sc  = max(sc_vals) if sc_vals else 1.0
        auto_thr = max(quality_threshold, max_sc * 0.15)
        keep_idx = [i for i in keep_idx if scores[i]["score"] >= auto_thr]
        if ref_idx not in keep_idx:
            keep_idx.append(ref_idx)
        rejected = [i+1 for i in range(len(calibrated)) if i not in keep_idx]
        if rejected:
            _cb(6, f"  Reddedilen: {rejected}")

    used = [calibrated[i] for i in keep_idx]
    used_paths = [light_paths[i] for i in keep_idx]
    new_ref = keep_idx.index(ref_idx) if ref_idx in keep_idx else 0
    reference = used[new_ref]

    # Normalize
    if normalize:
        ref_med = float(np.median(reference))
        for i in range(len(used)):
            f_med = float(np.median(used[i]))
            if f_med > 1e-9:
                used[i] = np.clip(used[i] * (ref_med / f_med), 0, 1).astype(np.float32)

    # Hizalama
    if align_method == "none":
        _cb(7, "Hizalama atlandı")
        return used

    _cb(7, f"Hizalanıyor ({align_method}, {len(used)} kare)...")
    aligned = []
    for i, frame in enumerate(used):
        if i == new_ref:
            aligned.append(frame)
        else:
            _cb(7, f"  {i+1}/{len(used)}: {os.path.basename(used_paths[i])}")
            try:
                al = align_frame(frame, reference, method=align_method)
                aligned.append(al)
            except Exception as e:
                _cb(7, f"  ⚠ #{i+1} hizalama başarısız: {e}")
                aligned.append(frame)

    _cb(7, f"✅ Hizalama tamamlandı — {len(aligned)} kare")
    return aligned


def stack_aligned(
    aligned_frames: List[np.ndarray],
    method: str = "kappa_sigma",
    kappa: float = 2.0,
    kappa_low: float = 2.0,
    kappa_high: float = 2.0,
    iterations: int = 5,
    quality_reject: bool = False,
    quality_threshold: float = 0.2,
    weight_mode: str = "none",
    progress_cb: Optional[Callable] = None,
) -> dict:
    """
    Faz 2: Önceden hizalanmış karelerden stacking.
    align_frames_only çıktısını alır.
    """
    def _cb(step, msg):
        if progress_cb: progress_cb(step, msg)

    if not aligned_frames:
        return {"result": None, "n_lights": 0}

    _cb(1, f"Stack ({method}, {len(aligned_frames)} kare)...")
    stack = np.stack(aligned_frames, axis=0).astype(np.float32)

    # Ağırlıklandırma
    frame_weights = None
    if weight_mode in ("snr","fwhm","stars"):
        _cb(1, "Kare kalitesi ölçülüyor (ağırlıklandırma için)...")
        w_scores = [score_frame(f) for f in aligned_frames]
        if weight_mode == "snr":
            w_vals = np.array([max(s["snr"],0.1) for s in w_scores], dtype=np.float32)
        elif weight_mode == "fwhm":
            w_vals = 1.0 / np.array([max(s["fwhm"],0.5) for s in w_scores], dtype=np.float32)
        else:
            w_vals = np.array([max(s["star_count"],1) for s in w_scores], dtype=np.float32)
        frame_weights = w_vals / w_vals.sum()

    stack_fns = {
        "mean":             _mean_stack,
        "median":           _median_stack,
        "kappa_sigma":      lambda s: _kappa_sigma(s, kappa=kappa, iterations=int(iterations)),
        "kappa_sigma_dual": lambda s: _kappa_sigma(s, kappa=kappa, iterations=int(iterations)),
        "winsorized":       lambda s: _winsorized(s, kappa_low=kappa_low,
                                                   kappa_high=kappa_high,
                                                   iterations=int(iterations)),
        "linear_fit":       _linear_fit,
        "entropy_weighted": _entropy_weighted,
        "maximum":          _maximum_stack,
        "minimum":          _minimum_stack,
        "sum":              _sum_stack,
    }

    if frame_weights is not None and method in ("mean","entropy_weighted"):
        if stack.ndim == 4:
            result = (stack * frame_weights[:, None, None, None]).sum(axis=0)
        else:
            result = (stack * frame_weights[:, None, None]).sum(axis=0)
        result = result.astype(np.float32)
    else:
        fn = stack_fns.get(method, _kappa_sigma)
        result = fn(stack)

    result = np.clip(result, 0, 1).astype(np.float32)
    _cb(2, f"✅ Stack tamamlandı — {len(aligned_frames)} kare birleştirildi")

    return {
        "result":          result,
        "n_lights":        len(aligned_frames),
        "n_rejected":      0,
        "rejected_frames": [],
        "frame_scores":    [],
        "method":          method,
        "align_method":    "pre-aligned",
        "ref_index":       0,
    }
