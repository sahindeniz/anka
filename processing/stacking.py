"""
Astro Maestro Pro - Image Stacking Engine v2
=============================================
wkjarosz/astro-stacker temel alinarak yazildi.
AKAZE feature detection + Homography + RANSAC hizalama.
Temiz, guvenilir, float32 pipeline.

Kaynak: https://github.com/wkjarosz/astro-stacker
"""

import os
import numpy as np
import cv2
from typing import List, Optional, Callable, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

_N_WORKERS = max(1, min(multiprocessing.cpu_count(), 8))

# ═══════════════════════════════════════════════════════════════════════════════
#  YARDIMCI FONKSIYONLAR
# ═══════════════════════════════════════════════════════════════════════════════

def _to_gray_float(img: np.ndarray) -> np.ndarray:
    """float32 goruntu → float32 grayscale.  Asla uint8'e donusturulmez!"""
    if img.ndim == 2:
        return img.astype(np.float32)
    return cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2GRAY)


def _enhance_for_detection(gray: np.ndarray) -> np.ndarray:
    """Lineer astro veride yildizlari one cikarmak icin agresif kontrast artirma.
    1) Asinh stretch (astro standart — cok parlak yildizlari ezmeden faint detayi cikarir)
    2) CLAHE (lokal kontrast artirma — feature detector icin ideal)
    Sonuc: uint8 [0,255] — AKAZE/SIFT feature detection icin optimize."""
    # Asinh stretch — lineer → logaritmik benzeri
    stretch_factor = 10.0
    stretched = np.arcsinh(gray * stretch_factor) / np.arcsinh(stretch_factor)
    # [0,1] normalize
    mn, mx = stretched.min(), stretched.max()
    if mx - mn > 1e-7:
        stretched = (stretched - mn) / (mx - mn)
    # uint8 + CLAHE
    u8 = np.clip(stretched * 255, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))
    return clahe.apply(u8)


def _compute_homography(
    next_img: np.ndarray,
    base_img: np.ndarray,
    threshold: float = 0.85,
    cache: Optional[dict] = None,
    frame_num: int = 0,
) -> Optional[np.ndarray]:
    """AKAZE feature detection + BFMatcher + RANSAC homography.
    astro-stacker yaklasimi — ORB yerine AKAZE (daha robust)."""

    next_gray = _to_gray_float(next_img)
    base_gray = _to_gray_float(base_img)

    # Asinh + CLAHE kontrast artirma — lineer veride feature detection icin kritik
    next_enh = _enhance_for_detection(next_gray)
    base_enh = _enhance_for_detection(base_gray)

    # AKAZE — binary descriptor, hizli ve robust
    alg = cv2.AKAZE_create(
        threshold=0.0005,  # daha dusuk esik = daha fazla keypoint
    )

    kp1, des1 = alg.detectAndCompute(next_enh, None)

    # Base keypoint caching — bir kez hesapla
    kp2, des2 = None, None
    if cache is not None and "kp2" in cache and "des2" in cache:
        kp2, des2 = cache["kp2"], cache["des2"]
    else:
        kp2, des2 = alg.detectAndCompute(base_enh, None)
        if cache is not None:
            cache["kp2"] = kp2
            cache["des2"] = des2

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None

    # BFMatcher — kNN ile ratio test (Lowe's ratio)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = matcher.knnMatch(des1, des2, k=2)

    # Lowe's ratio test — daha robust esleme
    good = []
    for m_pair in raw_matches:
        if len(m_pair) == 2:
            m, n = m_pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

    if len(good) < 6:
        return None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    # Affine (translation + rotation + scale) — astro karelerde perspektif yok
    M, mask = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC,
                                           ransacReprojThreshold=5.0)
    if M is None or mask is None:
        return None
    inlier_ratio = np.sum(mask) / len(mask)
    if inlier_ratio < 0.25:
        return None

    # 2x3 → 3x3
    H = np.eye(3, dtype=np.float64)
    H[:2, :] = M
    return H


def _warp_image(img: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Homography ile goruntu donusumu."""
    h, w = img.shape[:2]
    return cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)



def _warp_mask(shape: tuple, H: np.ndarray) -> np.ndarray:
    """Dondurulen karenin gecerli pixel maskesini olustur."""
    h, w = shape[:2]
    mask = np.ones((h, w), dtype=np.float32)
    return cv2.warpPerspective(mask, H, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)


# ═══════════════════════════════════════════════════════════════════════════════
#  KALIBRASYON
# ═══════════════════════════════════════════════════════════════════════════════

def _build_master(paths: List[str], method: str = "median",
                  progress_cb=None, label="") -> Optional[np.ndarray]:
    """Kalibrasyon karelerinden master frame olustur."""
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
        # Flat normalizasyonu: mean(flat) / flat
        flat = master_flat.copy()
        if master_bias is not None:
            flat = flat - master_bias
        flat = np.clip(flat, 0, None)
        flat_mean = np.mean(flat)
        if flat_mean > 0:
            flat_norm = flat / flat_mean
            flat_norm[flat_norm < 0.01] = 1.0  # sifira bolunme engelle
            cal = cal / flat_norm

    return np.clip(cal, 0, None).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
#  STACKING YONTEMLERI
# ═══════════════════════════════════════════════════════════════════════════════

def _stack_mean(frames: List[np.ndarray], masks: List[np.ndarray]) -> np.ndarray:
    """Maskeli ortalama stacking — her pikselde katki yapan karelerin ortalamasini al."""
    accumulator = np.zeros_like(frames[0], dtype=np.float64)
    weight = np.zeros(frames[0].shape[:2], dtype=np.float64)

    for fr, msk in zip(frames, masks):
        if fr.ndim == 3:
            m3 = msk[:, :, np.newaxis] if msk.ndim == 2 else msk
            accumulator += fr.astype(np.float64) * m3
        else:
            accumulator += fr.astype(np.float64) * msk
        weight += msk.astype(np.float64)

    weight[weight == 0] = 1
    if accumulator.ndim == 3:
        return (accumulator / weight[:, :, np.newaxis]).astype(np.float32)
    return (accumulator / weight).astype(np.float32)


def _stack_median(frames: List[np.ndarray], masks: List[np.ndarray]) -> np.ndarray:
    """Maskeli median stacking — siyah kenar piksellerini dislar."""
    n = len(frames)
    h, w = frames[0].shape[:2]
    is_color = frames[0].ndim == 3
    result = np.zeros_like(frames[0], dtype=np.float32)

    # Satir bloklari halinde isle — tum kareleri belleğe almadan
    block_size = max(1, min(64, h))
    for y0 in range(0, h, block_size):
        y1 = min(y0 + block_size, h)
        block = np.stack([fr[y0:y1] for fr in frames], axis=0).astype(np.float32)
        mask_block = np.stack([m[y0:y1] for m in masks], axis=0)  # (n, bh, w)

        if is_color:
            # Her piksel icin gecerli karelerin median'ini al
            for c in range(block.shape[-1]):
                ch_block = block[..., c]  # (n, bh, w)
                # NaN maskeleme — gecersiz pikselleri NaN yap, nanmedian ile hesapla
                masked = np.where(mask_block > 0.5, ch_block, np.nan)
                with np.errstate(all='ignore'):
                    med = np.nanmedian(masked, axis=0)
                # Tum kareler gecersiz → 0
                med = np.nan_to_num(med, nan=0.0)
                result[y0:y1, :, c] = med
        else:
            masked = np.where(mask_block > 0.5, block, np.nan)
            with np.errstate(all='ignore'):
                med = np.nanmedian(masked, axis=0)
            result[y0:y1] = np.nan_to_num(med, nan=0.0)

    return result


def _stack_kappa_sigma(
    frames: List[np.ndarray],
    masks: List[np.ndarray],
    kappa: float = 2.5,
    iterations: int = 3,
) -> np.ndarray:
    """Maskeli kappa-sigma clipping — RENK KORUYUCU.
    Renkli goruntuler icin: luminance bazli red/kabul karari.
    Ayni kare-piksel icin TUM kanallar birlikte kabul/red edilir
    → renk oranları korunur, renk kayması olmaz."""
    n = len(frames)
    h, w = frames[0].shape[:2]
    is_color = frames[0].ndim == 3
    result = np.zeros_like(frames[0], dtype=np.float32)

    block_size = max(1, min(32, h))
    for y0 in range(0, h, block_size):
        y1 = min(y0 + block_size, h)
        block = np.stack([fr[y0:y1] for fr in frames], axis=0).astype(np.float32)
        mask_block = np.stack([m[y0:y1] for m in masks], axis=0)  # (n, bh, w)

        if is_color:
            # ── RENK KORUYUCU: Luminance bazli sigma clipping ──
            # Karar luminance uzerinden verilir, tum kanallara AYNI uygulanir
            # Boylece R/G/B icin farkli kareler reddedilmez → renk orani bozulmaz
            lum_weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
            lum_block = np.sum(block * lum_weights[np.newaxis, np.newaxis, np.newaxis, :],
                               axis=-1)  # (n, bh, w)

            # valid maske: piksel bazli (kanal bazli DEGiL!)
            valid_px = mask_block > 0.5  # (n, bh, w)

            for _ in range(iterations):
                vcount_px = np.sum(valid_px, axis=0, keepdims=True).astype(np.float32)
                vcount_px[vcount_px == 0] = 1
                lum_mean = np.sum(lum_block * valid_px, axis=0, keepdims=True) / vcount_px
                lum_diff = lum_block - lum_mean
                lum_var = np.sum(lum_diff * lum_diff * valid_px, axis=0, keepdims=True) / vcount_px
                lum_std = np.sqrt(lum_var)
                lum_std[lum_std < 1e-8] = 1e-8
                valid_px = valid_px & (np.abs(lum_diff) <= kappa * lum_std)

            # Ayni valid maskesini tum kanallara genislet
            valid_3ch = valid_px[:, :, :, np.newaxis]  # (n, bh, w, 1) — broadcast
            vcount_3ch = np.sum(valid_3ch, axis=0, keepdims=True).astype(np.float32)
            vcount_3ch[vcount_3ch == 0] = 1
            result[y0:y1] = (np.sum(block * valid_3ch, axis=0, keepdims=True) / vcount_3ch).squeeze(0)

        else:
            # Mono goruntu — klasik per-pixel clipping
            valid = mask_block > 0.5

            for _ in range(iterations):
                vcount = np.sum(valid, axis=0, keepdims=True).astype(np.float32)
                vcount[vcount == 0] = 1
                mean = np.sum(block * valid, axis=0, keepdims=True) / vcount
                diff = block - mean
                var = np.sum(diff * diff * valid, axis=0, keepdims=True) / vcount
                std = np.sqrt(var)
                std[std < 1e-8] = 1e-8
                valid = valid & (np.abs(diff) <= kappa * std)

            vcount = np.sum(valid, axis=0, keepdims=True).astype(np.float32)
            vcount[vcount == 0] = 1
            result[y0:y1] = (np.sum(block * valid, axis=0, keepdims=True) / vcount).squeeze(0)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  SCORE FRAME
# ═══════════════════════════════════════════════════════════════════════════════

def score_frame(img: np.ndarray) -> dict:
    """Basit kare kalite skoru — yildiz sayisi + FWHM tahmini."""
    gray = _to_gray_float(img)
    # Kontrast artirma — yildiz tespiti icin
    thr8 = _enhance_for_detection(gray)

    # Blob detection
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

    detector = cv2.SimpleBlobDetector_create(params)
    kps = detector.detect(thr8)
    star_count = len(kps)

    fwhm = 0.0
    if kps:
        fwhm = np.mean([k.size for k in kps])

    # SNR tahmini — sinyal / gurultu
    signal = np.mean(gray)
    noise = np.std(gray)
    snr = signal / max(noise, 1e-8)

    score = min(1.0, star_count / 100.0) * 0.5 + min(1.0, snr * 10) * 0.3 + (1.0 / max(fwhm, 1.0)) * 0.2

    return {
        "score": float(score),
        "star_count": star_count,
        "fwhm": float(fwhm),
        "snr": float(snr),
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
    ref_mode: str = "first",
    normalize: bool = True,
    quality_reject: bool = False,
    quality_threshold: float = 0.3,
    progress_cb: Callable = None,
    **kwargs,
) -> List[np.ndarray]:
    """Hizalama — hizalanmis frame listesi dondurur."""
    from core.loader import load_image

    n = len(light_paths)
    if n == 0:
        return []

    def cb(step, msg):
        if progress_cb:
            progress_cb(step, msg)

    # ── 1. Master kareleri olustur ──
    cb(1, "Master Bias olusturuluyor…")
    master_bias = _build_master(bias_paths or [], "median", progress_cb, "1")

    cb(2, "Master Dark olusturuluyor…")
    master_dark = _build_master(dark_paths or [], "median", progress_cb, "2")

    cb(3, "Master Flat olusturuluyor…")
    master_flat = _build_master(flat_paths or [], "median", progress_cb, "3")

    # dark_flat varsa flat'tan cikar
    if dark_flat_paths:
        cb(3, "Dark Flat olusturuluyor…")
        dark_flat = _build_master(dark_flat_paths, "median", progress_cb, "3")
        if dark_flat is not None and master_flat is not None:
            master_flat = np.clip(master_flat - dark_flat, 0, None)

    # ── 2. Tum kareleri yukle + kalibre et ──
    cb(4, f"{n} kare yukleniyor ve kalibre ediliyor…")
    frames = []
    for i, p in enumerate(light_paths):
        cb(4, f"Kare {i+1}/{n}: {os.path.basename(p)}")
        img = load_image(p)
        _sh = img.shape
        _info = f"{'RGB' if img.ndim==3 else 'MONO'} {_sh}"
        cb(4, f"Kare {i+1}/{n}: {os.path.basename(p)} — {_info}")
        img = _calibrate_frame(img, master_dark, master_flat, master_bias)
        frames.append(img)

    # ── 3. Referans kare sec ──
    if ref_mode == "best" and len(frames) > 1:
        cb(5, "En iyi referans kare seciliyor…")
        best_idx = 0
        best_score = -1
        for i, fr in enumerate(frames):
            sc = score_frame(fr)
            if sc["score"] > best_score:
                best_score = sc["score"]
                best_idx = i
        ref_index = best_idx
        cb(5, f"Referans: #{ref_index+1} (skor: {best_score:.3f})")
    else:
        ref_index = max(0, min(ref_index, len(frames) - 1))

    base = frames[ref_index]

    # ── 4. AKAZE Hizalama — her kare dogrudan referansa hizalanir ──
    cb(6, "AKAZE hizalama basliyor…")
    aligned = []
    n_ok = 0
    n_fail = 0

    # Referans kare keypoint cache — bir kez hesapla, tum kareler icin kullan
    base_cache = {}

    for i, fr in enumerate(frames):
        cb(6, f"Hizalama {i+1}/{n}")
        if i == ref_index:
            aligned.append(base.copy())
            n_ok += 1
            continue

        H = _compute_homography(fr, base, threshold=0.85,
                                cache=base_cache, frame_num=i)

        # AKAZE basarili — transform parametrelerini kontrol et
        if H is not None:
            tx, ty = H[0, 2], H[1, 2]
            scale = np.sqrt(H[0, 0]**2 + H[1, 0]**2)
            # Olcek ~1.0 olmali (astro karelerde zoom degismez)
            if abs(scale - 1.0) > 0.05:
                cb(6, f"  ⚠ Kare {i+1}: scale={scale:.3f} — AKAZE yanlis eslesme")
                H = None

        if H is not None:
            warped = _warp_image(fr, H)
            aligned.append(warped)
            n_ok += 1
        else:
            # ECC fallback
            H_ecc = _ecc_fallback(fr, base)
            if H_ecc is not None:
                warped = _warp_image(fr, H_ecc)
                aligned.append(warped)
                n_ok += 1
                cb(6, f"  Kare {i+1}: ECC fallback ile hizalandi")
            else:
                n_fail += 1
                cb(6, f"  ⚠ Kare {i+1}: hizalanamadi — atlandi")

    cb(7, f"✅ Hizalama tamamlandi — {n_ok} basarili, {n_fail} hizalanamayan")
    return aligned


def _ecc_fallback(img: np.ndarray, base: np.ndarray) -> Optional[np.ndarray]:
    """ECC (Enhanced Correlation Coefficient) hizalama — AKAZE basarisiz olursa.
    Asinh + CLAHE ile normalize eder — farkli parlakliktaki seanslar icin."""
    try:
        # Ayni enhance pipeline'i kullan — parlaklık farklarini giderir
        g1 = _enhance_for_detection(_to_gray_float(img))
        g2 = _enhance_for_detection(_to_gray_float(base))

        # Kucultulmus goruntu ile ECC — hiz icin
        scale = 0.25
        h, w = g1.shape[:2]
        small1 = cv2.resize(g1, (int(w * scale), int(h * scale)))
        small2 = cv2.resize(g2, (int(w * scale), int(h * scale)))

        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
        _, warp_matrix = cv2.findTransformECC(
            small2.astype(np.float32), small1.astype(np.float32),
            warp_matrix, cv2.MOTION_EUCLIDEAN, criteria,
            inputMask=None, gaussFiltSize=5
        )

        # Kucuk olcekteki transformu tam olcege cevir
        warp_matrix[0, 2] /= scale
        warp_matrix[1, 2] /= scale

        H = np.eye(3, dtype=np.float64)
        H[:2, :] = warp_matrix
        return H
    except cv2.error:
        return None


def stack_aligned(
    aligned_frames: List[np.ndarray],
    method: str = "kappa_sigma",
    kappa: float = 2.5,
    kappa_low: float = 2.5,
    kappa_high: float = 2.5,
    iterations: int = 3,
    weight_mode: str = "equal",
    quality_reject: bool = False,
    quality_threshold: float = 0.3,
    drizzle_scale: int = 0,
    progress_cb: Callable = None,
    **kwargs,
) -> dict:
    """Onceden hizalanmis kareleri stackle."""
    if not aligned_frames:
        raise ValueError("Hizalanmis kare yok!")

    def cb(step, msg):
        if progress_cb:
            progress_cb(step, msg)

    n = len(aligned_frames)
    is_color = aligned_frames[0].ndim == 3
    ch_count = aligned_frames[0].shape[2] if is_color else 1
    cb(8, f"{n} kare stackleniyor — metot: {method} — "
         f"{'RGB renkli' if is_color else 'mono'} ({ch_count} kanal)")

    # Maskeleri olustur (siyah kenarlar icin)
    # Renkli goruntulerde: herhangi bir kanal > 0 ise piksel gecerli
    masks = []
    for fr in aligned_frames:
        if is_color:
            # Tum kanallar sifir olan pikselleri maskele
            # Herhangi bir kanalda sinyal varsa gecerli say
            mask = (np.max(fr, axis=2) > 1e-6).astype(np.float32)
        else:
            mask = (fr > 1e-6).astype(np.float32)
        masks.append(mask)

    # Stack
    if method == "median":
        result = _stack_median(aligned_frames, masks)
    elif method in ("kappa_sigma", "median_kappa_sigma"):
        result = _stack_kappa_sigma(aligned_frames, masks,
                                    kappa=kappa, iterations=iterations)
    else:
        # Varsayilan: maskeli ortalama (mean)
        result = _stack_mean(aligned_frames, masks)

    cb(8, f"✅ Stacking tamamlandi — {n} kare birlestirildi")

    # Arka plan nötralizasyonu stacking'de YAPILMAZ — renk oranlarını bozar.
    # Kullanıcı bunu ayrı "BG Siyah" panelinden manuel olarak uygular.

    return {
        "result": result,
        "n_lights": n,
        "n_rejected": 0,
        "method": method,
        "frame_scores": [],
        "rejected_frames": [],
    }


def _neutralize_background(img: np.ndarray) -> np.ndarray:
    """
    Arka plan renk sapmasını düzelt.
    Her kanaldan arka plan seviyesini çıkarır → arka plan siyah,
    sinyal renkleri (nebula/yıldız) aynen korunur.
    """
    if img is None or img.size == 0:
        return img
    if img.ndim != 3 or img.shape[2] < 3:
        return img

    result = img.astype(np.float32).copy()

    # En karanlık %5 pikselleri saf arka plan olarak al (çok tutucu)
    gray = np.mean(result, axis=2)
    threshold = np.percentile(gray, 5)
    bg_mask = gray <= threshold

    if bg_mask.sum() < 50:
        return result

    # Her kanalın arka plan seviyesi
    bg_levels = np.array([
        float(np.median(result[:, :, ch][bg_mask]))
        for ch in range(3)
    ])

    # Çok düşükse müdahale etme
    if bg_levels.max() < 0.003:
        return result

    # Her kanaldan kendi arka plan seviyesini çıkar
    # Sinyal renklerini KORUR:
    #   nebula(R=0.5) - bg(R=0.3) = 0.2
    #   nebula(G=0.2) - bg(G=0.1) = 0.1
    #   → R > G oranı korunur
    for ch in range(3):
        result[:, :, ch] -= bg_levels[ch]

    # Negatif değerleri sıfırla — normalize YAPMA (renk oranları bozulur)
    return np.clip(result, 0, 1).astype(np.float32)


def stack_lights(
    light_paths: List[str],
    dark_paths: List[str] = None,
    flat_paths: List[str] = None,
    dark_flat_paths: List[str] = None,
    bias_paths: List[str] = None,
    method: str = "kappa_sigma",
    align_method: str = "AKAZE",
    ref_index: int = 0,
    ref_mode: str = "first",
    kappa: float = 2.5,
    kappa_low: float = 2.5,
    kappa_high: float = 2.5,
    iterations: int = 3,
    quality_reject: bool = False,
    quality_threshold: float = 0.3,
    normalize: bool = True,
    weight_mode: str = "equal",
    dark_optimize: bool = False,
    drizzle_scale: int = 0,
    hot_pixel_removal: bool = True,
    progress_cb: Callable = None,
    **kwargs,
) -> dict:
    """Tam pipeline: yukle → kalibre → hizala → stackle."""

    # 1. Hizala
    aligned = align_frames_only(
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
    )

    if not aligned:
        raise ValueError("Hizalanmis kare yok — stacking yapilamaz.")

    # 2. Stackle
    result = stack_aligned(
        aligned_frames=aligned,
        method=method,
        kappa=kappa,
        kappa_low=kappa_low,
        kappa_high=kappa_high,
        iterations=iterations,
        weight_mode=weight_mode,
        quality_reject=quality_reject,
        quality_threshold=quality_threshold,
        drizzle_scale=drizzle_scale,
        progress_cb=progress_cb,
    )

    return result
