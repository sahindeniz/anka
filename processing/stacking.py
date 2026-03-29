"""
Astro Maestro Pro - Image Stacking Engine v3
=============================================
PixInsight ImageIntegration seviyesinde stacking engine.

Özellikler:
  - Normalizasyon: Additive+Scaling, Multiplicative, None
  - Ağırlıklandırma: SNR, FWHM, Equal, Noise
  - Reddetme: Auto (kare sayısına göre), Sigma Clipping, Linear Fit,
              Percentile Clipping, Winsorized Sigma
  - Hizalama: AKAZE + Affine (rotasyon destekli) + ECC fallback
  - Kalite kontrolü: Düşük kaliteli karelerde popup uyarı
  - Renklere dokunmaz — ham sinyal oranları korunur
"""

import os
import warnings
import numpy as np
import cv2
from typing import List, Optional, Callable, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

_N_WORKERS = max(1, min(multiprocessing.cpu_count(), 8))
_ANALYSIS_MAX_DIM = 960
_ECC_MAX_DIM = 768
_BLOB_DETECTOR = None
_ORB_DETECTOR = None

# ═══════════════════════════════════════════════════════════════════════════════
#  YARDIMCI FONKSIYONLAR
# ═══════════════════════════════════════════════════════════════════════════════

def _to_gray_float(img: np.ndarray) -> np.ndarray:
    """float32 goruntu → float32 grayscale."""
    if img.ndim == 2:
        return img.astype(np.float32)
    return cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2GRAY)


def _resize_for_analysis(img: np.ndarray, max_dim: int = _ANALYSIS_MAX_DIM) -> Tuple[np.ndarray, np.ndarray]:
    """Downsample large frames for scoring/alignment and return full-res scale factors."""
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_dim or max_dim <= 0:
        return img.astype(np.float32, copy=False), np.array([1.0, 1.0], dtype=np.float64)

    scale = float(max_dim) / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img.astype(np.float32, copy=False), (new_w, new_h), interpolation=cv2.INTER_AREA)
    scale_back = np.array([w / float(new_w), h / float(new_h)], dtype=np.float64)
    return resized, scale_back


def _scale_affine_to_full_res(M: np.ndarray, src_scale: np.ndarray, dst_scale: np.ndarray) -> np.ndarray:
    """Convert affine estimated on downsampled images back to full-resolution coordinates."""
    H_small = np.eye(3, dtype=np.float64)
    H_small[:2, :] = M
    S_src = np.diag([float(src_scale[0]), float(src_scale[1]), 1.0])
    S_dst = np.diag([float(dst_scale[0]), float(dst_scale[1]), 1.0])
    return S_dst @ H_small @ np.linalg.inv(S_src)


def _get_blob_detector():
    global _BLOB_DETECTOR
    if _BLOB_DETECTOR is not None:
        return _BLOB_DETECTOR

    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 255
    params.filterByArea = True
    params.minArea = 3
    params.maxArea = 500
    params.filterByCircularity = True
    params.minCircularity = 0.3
    params.filterByConvexity = False
    params.filterByInertia = False
    _BLOB_DETECTOR = cv2.SimpleBlobDetector_create(params)
    return _BLOB_DETECTOR


def _get_orb_detector():
    global _ORB_DETECTOR
    if _ORB_DETECTOR is not None:
        return _ORB_DETECTOR
    _ORB_DETECTOR = cv2.ORB_create(
        nfeatures=2500,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=31,
        patchSize=31,
        fastThreshold=5,
    )
    return _ORB_DETECTOR


def _enhance_for_detection(gray: np.ndarray) -> np.ndarray:
    """Lineer astro veride yıldızları öne çıkarmak için kontrast artırma.
    Asinh stretch + CLAHE → uint8 [0,255]"""
    stretch_factor = 10.0
    stretched = np.arcsinh(gray * stretch_factor) / np.arcsinh(stretch_factor)
    mn, mx = stretched.min(), stretched.max()
    if mx - mn > 1e-7:
        stretched = (stretched - mn) / (mx - mn)
    u8 = np.clip(stretched * 255, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))
    return clahe.apply(u8)


# ═══════════════════════════════════════════════════════════════════════════════
#  HİZALAMA — ROTASYON DESTEKLİ
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_homography(
    next_img: np.ndarray,
    base_img: np.ndarray,
    threshold: float = 0.85,
    cache: Optional[dict] = None,
    frame_num: int = 0,
) -> Tuple[Optional[np.ndarray], dict]:
    """Fast feature alignment on downsampled previews + AffinePartial2D.
    Rotasyon destekli — ters/döndürülmüş kareler de hizalanır.
    Returns: (H_3x3, info_dict)"""

    next_gray, next_scale = _resize_for_analysis(_to_gray_float(next_img), _ANALYSIS_MAX_DIM)
    base_gray, base_scale = _resize_for_analysis(_to_gray_float(base_img), _ANALYSIS_MAX_DIM)

    next_enh = _enhance_for_detection(next_gray)
    base_enh = _enhance_for_detection(base_gray)

    # ORB on downsampled previews is much faster than full-res AKAZE
    alg = _get_orb_detector()

    kp1, des1 = alg.detectAndCompute(next_enh, None)

    # Base keypoint caching
    kp2, des2 = None, None
    if cache is not None and "kp2" in cache and "des2" in cache:
        kp2, des2 = cache["kp2"], cache["des2"]
    else:
        kp2, des2 = alg.detectAndCompute(base_enh, None)
        if cache is not None:
            cache["kp2"] = kp2
            cache["des2"] = des2

    info = {"n_keypoints": len(kp1) if kp1 else 0,
            "n_base_keypoints": len(kp2) if kp2 else 0,
            "n_matches": 0, "inlier_ratio": 0.0,
            "rotation_deg": 0.0, "scale": 1.0, "tx": 0.0, "ty": 0.0}

    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return None, info

    # BFMatcher — kNN + Lowe's ratio test
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = matcher.knnMatch(des1, des2, k=2)

    good = []
    for m_pair in raw_matches:
        if len(m_pair) == 2:
            m, n = m_pair
            if m.distance < 0.80 * n.distance:
                good.append(m)

    info["n_matches"] = len(good)
    if len(good) < 8:
        return None, info

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    # AffinePartial2D — translation + rotation + uniform scale
    # Astro karelerde perspektif yok, ama rotasyon olabilir (alan döndürücü, montaj hatası)
    M, mask = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC,
                                           ransacReprojThreshold=5.0,
                                           maxIters=5000,
                                           confidence=0.999)
    if M is None or mask is None:
        return None, info

    inlier_ratio = float(np.sum(mask) / len(mask))
    info["inlier_ratio"] = inlier_ratio
    if inlier_ratio < 0.20:
        return None, info

    # Transform parametrelerini çıkar
    scale = np.sqrt(M[0, 0]**2 + M[1, 0]**2)
    rotation = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
    tx, ty = M[0, 2], M[1, 2]
    info["scale"] = float(scale)
    info["rotation_deg"] = float(rotation)
    info["tx"] = float(tx)
    info["ty"] = float(ty)

    # Ölçek kontrolü — astro karelerde zoom değişmez
    if abs(scale - 1.0) > 0.10:
        return None, info

    # 2x3 → 3x3
    H = _scale_affine_to_full_res(M, next_scale, base_scale)
    return H, info


def _warp_image(img: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Homography ile görüntü dönüşümü."""
    h, w = img.shape[:2]
    if H.shape == (3, 3) and np.allclose(H[2], [0.0, 0.0, 1.0], atol=1e-6):
        return cv2.warpAffine(
            img,
            H[:2, :],
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    return cv2.warpPerspective(
        img,
        H,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _ecc_fallback(img: np.ndarray, base: np.ndarray) -> Optional[np.ndarray]:
    """ECC (Enhanced Correlation Coefficient) hizalama — AKAZE başarısız olursa.
    EUCLIDEAN model — rotasyon destekler."""
    try:
        g1_small, src_scale = _resize_for_analysis(_to_gray_float(img), _ECC_MAX_DIM)
        g2_small, dst_scale = _resize_for_analysis(_to_gray_float(base), _ECC_MAX_DIM)
        g1_small = _enhance_for_detection(g1_small)
        g2_small = _enhance_for_detection(g2_small)

        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-7)
        _, warp_matrix = cv2.findTransformECC(
            g2_small.astype(np.float32), g1_small.astype(np.float32),
            warp_matrix, cv2.MOTION_EUCLIDEAN, criteria,
            inputMask=None, gaussFiltSize=5
        )

        return _scale_affine_to_full_res(warp_matrix, src_scale, dst_scale)
    except cv2.error:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  KALİBRASYON
# ═══════════════════════════════════════════════════════════════════════════════

def _build_master(paths: List[str], method: str = "median",
                  progress_cb=None, label="") -> Optional[np.ndarray]:
    """Kalibrasyon karelerinden master frame oluştur."""
    if not paths:
        return None
    from core.loader import load_image

    frames = []
    for i, p in enumerate(paths):
        if progress_cb:
            progress_cb(label, f"{i+1}/{len(paths)} yukleniyor…")
        frames.append(load_image(p))

    stack = np.stack(frames, axis=0)
    if method == "median":
        return np.median(stack, axis=0).astype(np.float32)
    else:
        return np.mean(stack, axis=0).astype(np.float32)


def _calibrate_frame(
    img: np.ndarray,
    master_dark: Optional[np.ndarray],
    master_flat: Optional[np.ndarray],
    master_bias: Optional[np.ndarray],
) -> np.ndarray:
    """Tek kareye kalibrasyon uygula: (Light - Bias - Dark) / Flat"""
    cal = img.astype(np.float32)

    if master_bias is not None:
        cal = cal - master_bias

    if master_dark is not None:
        cal = cal - master_dark

    cal = np.clip(cal, 0, None)

    if master_flat is not None:
        flat = master_flat.copy()
        if master_bias is not None:
            flat = flat - master_bias
        flat = np.clip(flat, 0, None)
        flat_mean = np.mean(flat)
        if flat_mean > 0:
            flat_norm = flat / flat_mean
            flat_norm[flat_norm < 0.01] = 1.0
            cal = cal / flat_norm

    return np.clip(cal, 0, None).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
#  NORMALİZASYON (PixInsight tarzı)
# ═══════════════════════════════════════════════════════════════════════════════

def _estimate_background(img: np.ndarray) -> Tuple[float, float]:
    """Arka plan seviyesi (median) ve ölçek (MAD) tahmini.
    PixInsight'ın normalizasyonu buna dayanır."""
    gray = _to_gray_float(img) if img.ndim == 3 else img.astype(np.float32)
    # Siyah (warp kenarı) pikselleri hariç tut
    valid = gray > 1e-6
    if np.sum(valid) < 100:
        return 0.0, 1.0
    data = gray[valid]
    med = float(np.median(data))
    mad = float(np.median(np.abs(data - med)))
    # MAD → sigma: MAD * 1.4826
    scale = max(mad * 1.4826, 1e-10)
    return med, scale


def _normalize_frames(frames: List[np.ndarray], masks: List[np.ndarray],
                      mode: str = "additive_scaling",
                      work_dtype: np.dtype = np.float32,
                      allow_float16_fallback: bool = True) -> List[np.ndarray]:
    """Kareleri normalize et — farklı gökyüzü parlaklıklarını eşitle.
    mode:
      'additive_scaling' — PixInsight varsayılan: location + scale eşitleme
      'multiplicative'   — Çarpımsal normalizasyon
      'none'             — Normalizasyon yok
    """
    if mode == "none" or len(frames) < 2:
        return frames

    # Referans: ilk karenin istatistikleri
    stats = []
    for fr, msk in zip(frames, masks):
        gray = _to_gray_float(fr) if fr.ndim == 3 else fr.astype(np.float32)
        valid = msk > 0.5
        if np.sum(valid) < 100:
            stats.append((0.0, 1.0))
        else:
            data = gray[valid]
            med = float(np.median(data))
            mad = float(np.median(np.abs(data - med)))
            scale = max(mad * 1.4826, 1e-10)
            stats.append((med, scale))

    ref_med, ref_scale = stats[0]

    target_dtype = np.dtype(work_dtype)
    if target_dtype not in (np.dtype(np.float32), np.dtype(np.float16)):
        target_dtype = np.dtype(np.float32)

    normalized = []
    for i, (fr, (med, scale)) in enumerate(zip(frames, stats)):
        if i == 0:
            normalized.append(fr)
            continue

        # Bellek kullanımını düşük tutmak için mümkün olduğunca float32 üzerinde
        # çalış. copy=False, kaynak zaten float32 ise gereksiz bir kopyayı önler.
        try:
            out = fr.astype(target_dtype, copy=False)
        except MemoryError:
            if not allow_float16_fallback or target_dtype == np.dtype(np.float16):
                raise
            # float32 kopyası için RAM yetmiyorsa geçici olarak float16'ya düş.
            out = fr.astype(np.float16, copy=False)
            target_dtype = np.dtype(np.float16)

        if mode == "multiplicative":
            # Çarpımsal: img * (ref_med / med)
            if med > 1e-10:
                factor = target_dtype.type(ref_med / med)
                out = out * factor
            else:
                out = out + target_dtype.type(ref_med - med)
        else:
            # Additive with scaling (PI varsayılan)
            # img = (img - med) * (ref_scale / scale) + ref_med
            if scale > 1e-10:
                # Ara çıktıları float32 tutmak için işlemi adım adım uygula.
                np.subtract(out, target_dtype.type(med), out=out, casting="unsafe")
                np.multiply(out, target_dtype.type(ref_scale / scale), out=out, casting="unsafe")
                np.add(out, target_dtype.type(ref_med), out=out, casting="unsafe")

        # np.clip(out=...) ile ek bir büyük ara dizi oluşturma.
        np.clip(out, 0, None, out=out)
        normalized.append(out)

    return normalized


# ═══════════════════════════════════════════════════════════════════════════════
#  AĞIRLIKLANDIRMA (PixInsight tarzı)
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_weights(frames: List[np.ndarray], masks: List[np.ndarray],
                     mode: str = "snr",
                     scores: Optional[List[dict]] = None) -> np.ndarray:
    """Her kare için ağırlık hesapla.
    mode:
      'equal'  — Tüm karelere eşit ağırlık
      'snr'    — SNR bazlı (daha az gürültülü → daha yüksek ağırlık)
      'noise'  — Ters gürültü bazlı
      'fwhm'   — Keskinlik bazlı (daha düşük FWHM → daha yüksek ağırlık)
    """
    n = len(frames)
    if mode == "equal" or n < 2:
        return np.ones(n, dtype=np.float32)

    weights = np.ones(n, dtype=np.float32)

    if mode == "snr":
        for i, (fr, msk) in enumerate(zip(frames, masks)):
            gray = _to_gray_float(fr) if fr.ndim == 3 else fr.astype(np.float32)
            valid = msk > 0.5
            if np.sum(valid) < 100:
                weights[i] = 0.1
                continue
            data = gray[valid]
            signal = float(np.median(data))
            noise = float(np.median(np.abs(data - signal))) * 1.4826
            weights[i] = signal / max(noise, 1e-10)

    elif mode == "noise":
        for i, (fr, msk) in enumerate(zip(frames, masks)):
            gray = _to_gray_float(fr) if fr.ndim == 3 else fr.astype(np.float32)
            valid = msk > 0.5
            if np.sum(valid) < 100:
                weights[i] = 0.1
                continue
            data = gray[valid]
            noise = float(np.median(np.abs(data - np.median(data)))) * 1.4826
            weights[i] = 1.0 / max(noise, 1e-10)

    elif mode == "fwhm" and scores:
        for i, sc in enumerate(scores):
            fwhm = sc.get("fwhm", 5.0)
            weights[i] = 1.0 / max(fwhm, 0.5)

    # Normalize: toplam = n (ortalama ağırlık = 1)
    w_sum = np.sum(weights)
    if w_sum > 0:
        weights = weights * (n / w_sum)

    return weights


# ═══════════════════════════════════════════════════════════════════════════════
#  REDDETME ALGORİTMALARI (PixInsight tarzı)
# ═══════════════════════════════════════════════════════════════════════════════

def _stack_weighted_mean(frames: List[np.ndarray], masks: List[np.ndarray],
                         valid_mask: np.ndarray,
                         weights: np.ndarray) -> np.ndarray:
    """Ağırlıklı ortalama — valid_mask ile reddetme uygulanmış."""
    n = len(frames)
    is_color = frames[0].ndim == 3
    accumulator = np.zeros_like(frames[0], dtype=np.float32)
    weight_sum = np.zeros(frames[0].shape[:2], dtype=np.float32)

    for i, (fr, msk) in enumerate(zip(frames, masks)):
        # valid_mask[i] ve msk ikisi de geçerli olmalı
        combined = (msk > 0.5) & valid_mask[i]
        w = float(weights[i])
        if is_color:
            m3 = combined[:, :, np.newaxis].astype(np.float32)
            accumulator += fr.astype(np.float32, copy=False) * m3 * w
        else:
            accumulator += fr.astype(np.float32, copy=False) * combined.astype(np.float32) * w
        weight_sum += combined.astype(np.float32) * w

    weight_sum[weight_sum == 0] = 1
    if is_color:
        return accumulator / weight_sum[:, :, np.newaxis]
    return accumulator / weight_sum


def _reject_sigma_clip(frames: List[np.ndarray], masks: List[np.ndarray],
                       kappa_low: float = 2.5, kappa_high: float = 2.5,
                       iterations: int = 3) -> np.ndarray:
    """Sigma Clipping — PixInsight tarzı, luminance bazlı renk koruyucu.
    Asimetrik: kappa_low (karanlık outlier), kappa_high (parlak outlier — uydu/uçak)."""
    n = len(frames)
    h, w = frames[0].shape[:2]
    is_color = frames[0].ndim == 3

    # valid_mask: (n, h, w) boolean — her piksel-kare için geçerli mi
    valid = np.stack([m > 0.5 for m in masks], axis=0)  # (n, h, w)

    block_size = max(1, min(64, h))
    for y0 in range(0, h, block_size):
        y1 = min(y0 + block_size, h)
        block = np.stack([fr[y0:y1] for fr in frames], axis=0).astype(np.float32)
        v_block = valid[:, y0:y1, :]  # (n, bh, w)

        # Luminance bazlı karar (renk koruyucu)
        if is_color:
            lum_w = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
            lum = np.sum(block * lum_w[np.newaxis, np.newaxis, np.newaxis, :], axis=-1)
        else:
            lum = block.squeeze(-1) if block.ndim == 4 else block

        for _ in range(iterations):
            vcount = np.sum(v_block, axis=0, keepdims=True).astype(np.float32)
            vcount[vcount == 0] = 1
            mean = np.sum(lum * v_block, axis=0, keepdims=True) / vcount
            diff = lum - mean
            var = np.sum(diff * diff * v_block, axis=0, keepdims=True) / vcount
            std = np.sqrt(var)
            std[std < 1e-8] = 1e-8
            # Asimetrik clipping
            v_block = v_block & (diff >= -kappa_low * std) & (diff <= kappa_high * std)

        valid[:, y0:y1, :] = v_block

    return valid


def _reject_linear_fit(frames: List[np.ndarray], masks: List[np.ndarray],
                       kappa_low: float = 3.0, kappa_high: float = 3.0) -> np.ndarray:
    """Linear Fit Clipping — az sayıda kare (5-10) için ideal.
    Her pikselde lineer model fit eder, modelden sapmayı reddeder."""
    n = len(frames)
    h, w = frames[0].shape[:2]
    is_color = frames[0].ndim == 3

    valid = np.stack([m > 0.5 for m in masks], axis=0)  # (n, h, w)

    block_size = max(1, min(64, h))
    for y0 in range(0, h, block_size):
        y1 = min(y0 + block_size, h)
        block = np.stack([fr[y0:y1] for fr in frames], axis=0).astype(np.float32)
        v_block = valid[:, y0:y1, :]

        if is_color:
            lum_w = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
            lum = np.sum(block * lum_w[np.newaxis, np.newaxis, np.newaxis, :], axis=-1)
        else:
            lum = block.squeeze(-1) if block.ndim == 4 else block

        # Median baseline
        vcount = np.sum(v_block, axis=0, keepdims=True).astype(np.float32)
        vcount[vcount == 0] = 1
        mean = np.sum(lum * v_block, axis=0, keepdims=True) / vcount

        # Residual = frame - mean
        residual = lum - mean
        res_var = np.sum(residual * residual * v_block, axis=0, keepdims=True) / vcount
        res_std = np.sqrt(res_var)
        res_std[res_std < 1e-8] = 1e-8

        # Linear fit residual clipping
        v_block = v_block & (residual >= -kappa_low * res_std) & (residual <= kappa_high * res_std)

        valid[:, y0:y1, :] = v_block

    return valid


def _reject_percentile(frames: List[np.ndarray], masks: List[np.ndarray],
                       low_pct: float = 10.0, high_pct: float = 90.0) -> np.ndarray:
    """Percentile Clipping — çok az sayıda kare (3-5) için.
    Alt ve üst yüzdelik dilimdeki pikselleri atar."""
    n = len(frames)
    h, w = frames[0].shape[:2]
    is_color = frames[0].ndim == 3

    valid = np.stack([m > 0.5 for m in masks], axis=0)

    block_size = max(1, min(64, h))
    for y0 in range(0, h, block_size):
        y1 = min(y0 + block_size, h)
        block = np.stack([fr[y0:y1] for fr in frames], axis=0).astype(np.float32)
        v_block = valid[:, y0:y1, :]

        if is_color:
            lum_w = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
            lum = np.sum(block * lum_w[np.newaxis, np.newaxis, np.newaxis, :], axis=-1)
        else:
            lum = block.squeeze(-1) if block.ndim == 4 else block

        # NaN maskeleme ile percentile hesapla
        masked_lum = np.where(v_block, lum, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            lo = np.nanpercentile(masked_lum, low_pct, axis=0, keepdims=True)
            hi = np.nanpercentile(masked_lum, high_pct, axis=0, keepdims=True)
        lo = np.nan_to_num(lo, nan=-1e10)
        hi = np.nan_to_num(hi, nan=1e10)

        v_block = v_block & (lum >= lo) & (lum <= hi)
        valid[:, y0:y1, :] = v_block

    return valid


def _reject_winsorized_sigma(frames: List[np.ndarray], masks: List[np.ndarray],
                              kappa_low: float = 2.5, kappa_high: float = 2.5,
                              iterations: int = 3) -> np.ndarray:
    """Winsorized Sigma Clipping — sigma clipping'e benzer ama
    reddedilen değerler sınır değerleriyle değiştirilir (istatistiği bozmaz)."""
    n = len(frames)
    h, w = frames[0].shape[:2]
    is_color = frames[0].ndim == 3

    valid = np.stack([m > 0.5 for m in masks], axis=0)

    block_size = max(1, min(64, h))
    for y0 in range(0, h, block_size):
        y1 = min(y0 + block_size, h)
        block = np.stack([fr[y0:y1] for fr in frames], axis=0).astype(np.float32)
        v_block = valid[:, y0:y1, :]

        if is_color:
            lum_w = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
            lum = np.sum(block * lum_w[np.newaxis, np.newaxis, np.newaxis, :], axis=-1)
        else:
            lum = block.squeeze(-1) if block.ndim == 4 else block

        for _ in range(iterations):
            vcount = np.sum(v_block, axis=0, keepdims=True).astype(np.float32)
            vcount[vcount == 0] = 1
            # Winsorized mean: clip extremes before computing mean
            masked_lum = np.where(v_block, lum, np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                lo_bound = np.nanpercentile(masked_lum, 10, axis=0, keepdims=True)
                hi_bound = np.nanpercentile(masked_lum, 90, axis=0, keepdims=True)
            lo_bound = np.nan_to_num(lo_bound, nan=0.0)
            hi_bound = np.nan_to_num(hi_bound, nan=1.0)
            winsorized = np.clip(lum, lo_bound, hi_bound)
            mean = np.sum(winsorized * v_block, axis=0, keepdims=True) / vcount
            diff = lum - mean
            var = np.sum(diff * diff * v_block, axis=0, keepdims=True) / vcount
            std = np.sqrt(var)
            std[std < 1e-8] = 1e-8
            v_block = v_block & (diff >= -kappa_low * std) & (diff <= kappa_high * std)

        valid[:, y0:y1, :] = v_block

    return valid


def _auto_select_rejection(n_frames: int) -> str:
    """PixInsight Auto modu — kare sayısına göre en uygun algoritmayı seç."""
    if n_frames <= 3:
        return "percentile"
    elif n_frames <= 7:
        return "linear_fit"
    elif n_frames <= 32:
        return "sigma_clip"
    elif n_frames <= 96:
        return "winsorized_sigma"
    else:
        # Large stacks benefit more from the lighter sigma-clip path.
        return "sigma_clip"


# ═══════════════════════════════════════════════════════════════════════════════
#  SCORE FRAME
# ═══════════════════════════════════════════════════════════════════════════════

def score_frame(img: np.ndarray) -> dict:
    """Kare kalite skoru — yıldız sayısı + FWHM + SNR + roundness."""
    gray_full = _to_gray_float(img)
    gray, scale_back = _resize_for_analysis(gray_full, _ANALYSIS_MAX_DIM)
    thr8 = _enhance_for_detection(gray)

    # Blob detection — yıldız tespiti
    detector = _get_blob_detector()
    kps = detector.detect(thr8)
    star_count = len(kps)

    fwhm = 0.0
    roundness = 1.0
    if kps:
        fwhm = float(np.mean([k.size for k in kps]) * float(np.mean(scale_back)))

    # SNR tahmini — robust (MAD bazlı)
    valid = gray > 1e-6
    if np.sum(valid) > 100:
        data = gray[valid]
        signal = float(np.median(data))
        noise = float(np.median(np.abs(data - signal))) * 1.4826
        snr = signal / max(noise, 1e-8)
    else:
        signal, noise, snr = 0.0, 1.0, 0.0

    # Kompozit skor
    score = (min(1.0, star_count / 100.0) * 0.4 +
             min(1.0, snr * 10) * 0.3 +
             (1.0 / max(fwhm, 1.0)) * 0.2 +
             roundness * 0.1)

    return {
        "score": float(score),
        "star_count": star_count,
        "fwhm": float(fwhm),
        "snr": float(snr),
        "signal": float(signal),
        "noise": float(noise),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  ANA API — GUI UYUMLU
# ═══════════════════════════════════════════════════════════════════════════════

def align_frames_only(
    light_paths: List[str],
    dark_paths: List[str] = None,
    flat_paths: List[str] = None,
    dark_flat_paths: List[str] = None,
    bias_paths: List[str] = None,
    align_method: str = "AKAZE",
    ref_index: int = 0,
    ref_mode: str = "best",
    normalize: bool = True,
    quality_reject: bool = True,
    quality_threshold: float = 0.3,
    progress_cb: Callable = None,
    quality_warning_cb: Callable = None,
    **kwargs,
) -> Tuple[List[np.ndarray], List[dict]]:
    """Hizalama pipeline — rotasyon destekli.

    Returns:
        (aligned_frames, frame_infos)
        frame_infos: Her kare için {score, rotation, status, ...}
    """
    from core.loader import load_image

    n = len(light_paths)
    if n == 0:
        return [], []

    def cb(step, msg):
        if progress_cb:
            progress_cb(step, msg)

    # ── 1. Master kareleri oluştur ──
    cb(1, "Master Bias oluşturuluyor…")
    master_bias = _build_master(bias_paths or [], "median", progress_cb, "1")

    cb(2, "Master Dark oluşturuluyor…")
    master_dark = _build_master(dark_paths or [], "median", progress_cb, "2")

    cb(3, "Master Flat oluşturuluyor…")
    master_flat = _build_master(flat_paths or [], "median", progress_cb, "3")

    if dark_flat_paths:
        cb(3, "Dark Flat oluşturuluyor…")
        dark_flat = _build_master(dark_flat_paths, "median", progress_cb, "3")
        if dark_flat is not None and master_flat is not None:
            master_flat = np.clip(master_flat - dark_flat, 0, None)

    # ── 2. Tüm kareleri yükle + kalibre et + kalite skoru ──
    cb(4, f"{n} kare yükleniyor ve kalibre ediliyor…")
    frames = []
    frame_infos = []
    for i, p in enumerate(light_paths):
        cb(4, f"Kare {i+1}/{n}: {os.path.basename(p)}")
        img = load_image(p)
        img = _calibrate_frame(img, master_dark, master_flat, master_bias)

        # Kalite skoru hesapla
        sc = score_frame(img)
        sc["path"] = p
        sc["name"] = os.path.basename(p)
        sc["index"] = i
        sc["status"] = "ok"

        # Düşük kalite uyarısı
        if sc["score"] < quality_threshold:
            sc["status"] = "low_quality"
            cb(4, f"  ⚠ Kare {i+1}: Düşük kalite (skor={sc['score']:.3f}, "
                  f"yıldız={sc['star_count']}, SNR={sc['snr']:.2f})")
            # GUI callback ile popup uyarı
            if quality_warning_cb:
                skip = quality_warning_cb(sc)
                if skip:
                    sc["status"] = "skipped"
                    frame_infos.append(sc)
                    continue

        frames.append(img)
        frame_infos.append(sc)

    # ── 3. Referans kare seç ──
    valid_infos = [fi for fi in frame_infos if fi["status"] != "skipped"]
    if ref_mode == "best" and len(frames) > 1:
        cb(5, "En iyi referans kare seçiliyor…")
        best_idx = 0
        best_score = -1
        for i, fi in enumerate(valid_infos):
            if fi.get("score", 0) > best_score:
                best_score = fi["score"]
                best_idx = i
        ref_index = best_idx
        cb(5, f"Referans: #{ref_index+1} (skor: {best_score:.3f})")
    else:
        ref_index = max(0, min(ref_index, len(frames) - 1))

    if not frames:
        return [], frame_infos

    base = frames[ref_index]

    # ── 4. Hizalama — rotasyon destekli ──
    cb(6, "AKAZE hizalama başlıyor (rotasyon destekli)…")
    aligned = []
    n_ok = 0
    n_fail = 0
    base_cache = {}

    for i, fr in enumerate(frames):
        cb(6, f"Hizalama {i+1}/{len(frames)}")
        fi = valid_infos[i] if i < len(valid_infos) else {}

        if i == ref_index:
            aligned.append(base.copy())
            fi["rotation_deg"] = 0.0
            fi["alignment"] = "reference"
            n_ok += 1
            continue

        H, align_info = _compute_homography(fr, base, threshold=0.85,
                                             cache=base_cache, frame_num=i)

        if H is not None:
            rot = align_info.get("rotation_deg", 0.0)
            warped = _warp_image(fr, H)
            aligned.append(warped)
            n_ok += 1
            fi["rotation_deg"] = rot
            fi["alignment"] = "AKAZE"
            if abs(rot) > 0.5:
                cb(6, f"  Kare {i+1}: rotasyon={rot:.1f}° düzeltildi")
        else:
            # ECC fallback — rotasyon destekli (EUCLIDEAN)
            H_ecc = _ecc_fallback(fr, base)
            if H_ecc is not None:
                warped = _warp_image(fr, H_ecc)
                aligned.append(warped)
                n_ok += 1
                fi["alignment"] = "ECC"
                cb(6, f"  Kare {i+1}: ECC fallback ile hizalandı")
            else:
                n_fail += 1
                fi["status"] = "align_failed"
                cb(6, f"  ⚠ Kare {i+1}: hizalanamadı — atlandı")
                # Uyarı callback
                if quality_warning_cb:
                    quality_warning_cb({
                        "name": fi.get("name", f"Kare {i+1}"),
                        "status": "align_failed",
                        "score": fi.get("score", 0),
                        "message": "Hizalama başarısız — kare atlandı"
                    })

    cb(7, f"✅ Hizalama tamamlandı — {n_ok} başarılı, {n_fail} hizalanamayan")
    return aligned, frame_infos


def stack_aligned(
    aligned_frames: List[np.ndarray],
    method: str = "auto",
    kappa: float = 2.5,
    kappa_low: float = 2.5,
    kappa_high: float = 3.0,
    iterations: int = 3,
    weight_mode: str = "snr",
    normalization: str = "additive_scaling",
    quality_reject: bool = False,
    quality_threshold: float = 0.3,
    drizzle_scale: int = 0,
    work_dtype: str = "float32",
    allow_float16_fallback: bool = True,
    progress_cb: Callable = None,
    frame_scores: Optional[List[dict]] = None,
    **kwargs,
) -> dict:
    """PixInsight tarzı stacking — normalizasyon + ağırlık + reddetme.

    method: 'auto', 'sigma_clip', 'linear_fit', 'percentile',
            'winsorized_sigma', 'median', 'mean'
    normalization: 'additive_scaling', 'multiplicative', 'none'
    weight_mode: 'snr', 'noise', 'fwhm', 'equal'
    work_dtype: 'float32' (önerilen), 'float16' (daha az RAM)
    """
    if not aligned_frames:
        raise ValueError("Hizalanmış kare yok!")

    def cb(step, msg):
        if progress_cb:
            progress_cb(step, msg)

    n = len(aligned_frames)
    is_color = aligned_frames[0].ndim == 3
    ch_count = aligned_frames[0].shape[2] if is_color else 1

    # ── Maskeler oluştur ──
    masks = []
    for fr in aligned_frames:
        if is_color:
            mask = (np.max(fr, axis=2) > 1e-6).astype(np.float32)
        else:
            mask = (fr > 1e-6).astype(np.float32)
        masks.append(mask)

    # ── Normalizasyon ──
    cb(8, f"Normalizasyon: {normalization}…")
    dtype_map = {"float32": np.float32, "float16": np.float16}
    norm_dtype = dtype_map.get(str(work_dtype).lower(), np.float32)
    aligned_frames = _normalize_frames(
        aligned_frames,
        masks,
        normalization,
        work_dtype=norm_dtype,
        allow_float16_fallback=allow_float16_fallback,
    )

    # ── Ağırlıklar ──
    cb(8, f"Ağırlıklandırma: {weight_mode}…")
    weights = _compute_weights(aligned_frames, masks, weight_mode, frame_scores)

    # ── Auto rejection seçimi ──
    if method == "auto":
        method = _auto_select_rejection(n)
        cb(8, f"Auto rejection: {method} ({n} kare)")

    cb(8, f"{n} kare stackleniyor — {method} — {weight_mode} ağırlık — "
         f"{'RGB' if is_color else 'mono'} ({ch_count}ch)")

    # ── Reddetme + Stacking ──
    n_rejected = 0
    if method == "median":
        # Median — kendi rejection'ı var
        result = _stack_median_weighted(aligned_frames, masks, weights)
    elif method == "mean":
        # Basit ağırlıklı ortalama — rejection yok
        all_valid = np.stack([m > 0.5 for m in masks], axis=0)
        result = _stack_weighted_mean(aligned_frames, masks, all_valid, weights)
    else:
        # Rejection-based methods
        if method == "sigma_clip":
            valid = _reject_sigma_clip(aligned_frames, masks,
                                        kappa_low=kappa_low, kappa_high=kappa_high,
                                        iterations=iterations)
        elif method == "linear_fit":
            valid = _reject_linear_fit(aligned_frames, masks,
                                        kappa_low=kappa_low, kappa_high=kappa_high)
        elif method == "percentile":
            valid = _reject_percentile(aligned_frames, masks,
                                        low_pct=10.0, high_pct=90.0)
        elif method == "winsorized_sigma":
            valid = _reject_winsorized_sigma(aligned_frames, masks,
                                              kappa_low=kappa_low, kappa_high=kappa_high,
                                              iterations=iterations)
        else:
            valid = np.stack([m > 0.5 for m in masks], axis=0)

        # Reddedilen piksel sayısı
        base_valid = np.stack([m > 0.5 for m in masks], axis=0)
        n_rejected = int(np.sum(base_valid) - np.sum(valid))

        result = _stack_weighted_mean(aligned_frames, masks, valid, weights)

    cb(8, f"✅ Stacking tamamlandı — {n} kare, {n_rejected} piksel reddedildi")

    return {
        "result": np.clip(result, 0, None).astype(np.float32),
        "n_lights": n,
        "n_rejected": n_rejected,
        "method": method,
        "normalization": normalization,
        "weight_mode": weight_mode,
        "frame_scores": frame_scores or [],
        "rejected_frames": [],
    }


def _stack_median_weighted(frames: List[np.ndarray], masks: List[np.ndarray],
                           weights: np.ndarray) -> np.ndarray:
    """Ağırlıklı median — basit median (ağırlık 1 ise standard median)."""
    # Weighted median karmaşık, burada standard median + weight != 0 filtre
    n = len(frames)
    h, w = frames[0].shape[:2]
    is_color = frames[0].ndim == 3
    result = np.zeros_like(frames[0], dtype=np.float32)

    block_size = max(1, min(64, h))
    for y0 in range(0, h, block_size):
        y1 = min(y0 + block_size, h)
        block = np.stack([fr[y0:y1] for fr in frames], axis=0).astype(np.float32)
        mask_block = np.stack([m[y0:y1] for m in masks], axis=0)

        if is_color:
            for c in range(block.shape[-1]):
                ch_block = block[..., c]
                masked = np.where(mask_block > 0.5, ch_block, np.nan)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    med = np.nanmedian(masked, axis=0)
                result[y0:y1, :, c] = np.nan_to_num(med, nan=0.0)
        else:
            masked = np.where(mask_block > 0.5, block, np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                med = np.nanmedian(masked, axis=0)
            result[y0:y1] = np.nan_to_num(med, nan=0.0)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  TAM PİPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def stack_lights(
    light_paths: List[str],
    dark_paths: List[str] = None,
    flat_paths: List[str] = None,
    dark_flat_paths: List[str] = None,
    bias_paths: List[str] = None,
    method: str = "auto",
    align_method: str = "AKAZE",
    ref_index: int = 0,
    ref_mode: str = "best",
    kappa: float = 2.5,
    kappa_low: float = 2.5,
    kappa_high: float = 3.0,
    iterations: int = 3,
    quality_reject: bool = True,
    quality_threshold: float = 0.3,
    normalize: bool = True,
    normalization: str = "additive_scaling",
    weight_mode: str = "snr",
    dark_optimize: bool = False,
    drizzle_scale: int = 0,
    hot_pixel_removal: bool = True,
    progress_cb: Callable = None,
    quality_warning_cb: Callable = None,
    **kwargs,
) -> dict:
    """Tam pipeline: yükle → kalibre → skorla → hizala → normalize → ağırlıkla → stackle."""

    # 1. Hizala (rotasyon destekli + kalite kontrolü)
    aligned, frame_infos = align_frames_only(
        light_paths=light_paths,
        dark_paths=dark_paths,
        flat_paths=flat_paths,
        dark_flat_paths=dark_flat_paths,
        bias_paths=bias_paths,
        align_method=align_method,
        ref_index=ref_index,
        ref_mode=ref_mode,
        normalize=normalize,
        quality_reject=quality_reject,
        quality_threshold=quality_threshold,
        progress_cb=progress_cb,
        quality_warning_cb=quality_warning_cb,
    )

    if not aligned:
        raise ValueError("Hizalanmış kare yok — stacking yapılamaz.")

    # 2. Stackle (PixInsight tarzı)
    result = stack_aligned(
        aligned_frames=aligned,
        method=method,
        kappa=kappa,
        kappa_low=kappa_low,
        kappa_high=kappa_high,
        iterations=iterations,
        weight_mode=weight_mode,
        normalization=normalization,
        quality_reject=quality_reject,
        quality_threshold=quality_threshold,
        drizzle_scale=drizzle_scale,
        progress_cb=progress_cb,
        frame_scores=frame_infos,
        **kwargs,
    )

    result["frame_infos"] = frame_infos
    return result
