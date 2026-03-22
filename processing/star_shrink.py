"""
Astro Maestro Pro — Star Shrink + Full Astro Pipeline
Yıldız küçültme: Morfolojik yıldız tespiti + Gauss profil küçültme.
Galaksi/nebula korumalı. Halo artefaktı olmayan yumuşak geçiş.
Full Process: stretch + BG extract + color enhance + star shrink + sharpen + denoise.
"""
import cv2
import numpy as np


def _detect_stars(gray_f32, sigma_threshold=2.0, min_area=3, max_area_ratio=0.002):
    """
    Yıldız tespiti — nokta kaynakları (point sources) bul, genişletilmiş
    nesneleri (galaksi, nebula) KORU.

    Returns: star_labels, star_mask (bool), num_stars
    """
    h, w = gray_f32.shape
    max_area = int(h * w * max_area_ratio)  # max yıldız alanı (orantılı)

    # Adaptif eşik — mean + sigma * std
    mean_val = np.mean(gray_f32)
    std_val = np.std(gray_f32)
    thresh = mean_val + sigma_threshold * std_val

    binary = (gray_f32 > thresh).astype(np.uint8) * 255

    # Morfolojik temizlik — gürültü noktaları sil
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    # Connected components ile bölge analizi
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)

    star_mask = np.zeros((h, w), dtype=bool)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]

        # Çok küçük → gürültü, atla
        if area < min_area:
            continue

        # Çok büyük → galaksi/nebula, KORU (atla)
        if area > max_area:
            continue

        # Aspect ratio kontrolü — yıldızlar yuvarlaktır
        aspect = max(bw, bh) / max(min(bw, bh), 1)
        if aspect > 3.0:
            continue  # çok uzun → uydu izi veya nebula kolları

        # Compactness kontrolü — yıldızlar kompakttır
        bbox_area = max(bw * bh, 1)
        compactness = area / bbox_area
        if compactness < 0.25:
            continue  # boşluklu bölge → nebula

        # Bu bir yıldız
        star_mask[labels == i] = True

    return star_mask


def star_shrink(image, shrink_factor=1.0, halo_fill_ratio=0.3,
                noise_level=5.0, star_density_threshold=2.0, **kw):
    """
    Yıldızları küçültür — morfolojik yıldız tespiti + Gauss profil küçültme.
    Galaksi ve nebula korunur. Halo artefaktı olmaz.

    Parameters
    ----------
    image : ndarray float32 [0,1], shape (H,W) veya (H,W,3)
    shrink_factor : float  (0.1–3.0)  küçültme gücü
    halo_fill_ratio : float (0–1)     yıldız parlaklık azaltma (0=tamamen, 1=değişiklik yok)
    noise_level : float (0–50)        arka plan gürültü seviyesi
    star_density_threshold : float     yıldız tespiti sigma eşiği

    Returns
    -------
    result : ndarray float32 [0,1]
    """
    img = np.ascontiguousarray(image, dtype=np.float32)
    np.clip(img, 0, 1, out=img)

    is_color = img.ndim == 3

    # Grayscale luminance
    if is_color:
        # Weighted luminance (daha doğru)
        gray = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
    else:
        gray = img.copy()

    # ── 1. Yıldız tespiti (galaksi/nebula korumalı) ─────────────────────
    star_mask = _detect_stars(gray, sigma_threshold=star_density_threshold)

    if not np.any(star_mask):
        # Yıldız bulunamadı — orijinali döndür
        return img.copy()

    # ── 2. Yıldız küçültme maskesi oluştur ───────────────────────────────
    star_u8 = star_mask.astype(np.uint8) * 255

    # Orijinal yıldız maskesini genişlet → "tam yıldız bölgesi"
    expand_size = max(3, int(5 * shrink_factor)) | 1
    kernel_expand = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand_size, expand_size))
    expanded_mask = cv2.dilate(star_u8, kernel_expand, iterations=1)

    # Küçültülmüş çekirdek maskesi — orijinalden erode
    erode_size = max(3, int(3 + 4 * shrink_factor)) | 1
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_size, erode_size))
    eroded_core = cv2.erode(star_u8, kernel_erode, iterations=1)

    # "Silinecek bölge" = expanded - eroded_core
    # Bu bölgede yıldız parlaklığı azaltılacak
    reduce_zone = (expanded_mask > 0) & (eroded_core == 0)

    # ── 3. Arka plan tahmini (yıldız olmayan bölgelerden) ────────────────
    # Yıldız bölgelerini maskele, medyan filtre ile arka planı tahmin et
    bg_estimate = cv2.medianBlur(
        (np.clip(img, 0, 1) * 255).astype(np.uint8),
        max(3, int(11 * shrink_factor)) | 1
    ).astype(np.float32) / 255.0

    # ── 4. Yumuşak geçiş maskesi (Gauss blend) ──────────────────────────
    # reduce_zone'u float'a çevir ve Gauss blur ile yumuşat
    reduce_float = reduce_zone.astype(np.float32)
    blur_size = max(3, int(7 * shrink_factor)) | 1
    alpha = cv2.GaussianBlur(reduce_float, (blur_size, blur_size), 0)
    # Parlaklık azaltma oranı
    reduction = 1.0 - (1.0 - halo_fill_ratio) * alpha

    # ── 5. Sonucu oluştur ────────────────────────────────────────────────
    if is_color:
        reduction_3d = np.expand_dims(reduction, axis=2)
        # Yıldız bölgesinde: orijinal × azaltma + arka plan × (1 - azaltma_oranı)
        result = img * reduction_3d + bg_estimate * (1.0 - reduction_3d) * alpha[:, :, np.newaxis]
        # Tam koruma bölgelerinde (alpha=0) orijinal kalır
        result = np.where(alpha[:, :, np.newaxis] > 0.001,
                          result, img)
    else:
        result = img * reduction + bg_estimate * (1.0 - reduction) * alpha
        result = np.where(alpha > 0.001, result, img)

    # ── 6. Arka plan gürültü ─────────────────────────────────────────────
    if noise_level > 0:
        noise_sigma = noise_level / 255.0
        noise = np.random.normal(0, noise_sigma, result.shape).astype(np.float32)
        # Sadece yıldız azaltılan bölgelere gürültü ekle
        if is_color:
            noise_mask = alpha[:, :, np.newaxis] > 0.01
        else:
            noise_mask = alpha > 0.01
        result = np.where(noise_mask, result + noise, result)

    np.clip(result, 0, 1, out=result)
    return result.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
#  FULL ASTRO PROCESS PIPELINE
#  Ham/lineer astro görüntüyü tam işlenmiş hale getirir.
#  Sıra: BG Remove → Stretch → Color Enhance → Star Shrink → Sharpen → Denoise
# ═══════════════════════════════════════════════════════════════════════════

def _bg_extract(img, grid_size=8):
    """Arka plan gradyan çıkarma — sigma-clipped medyan grid ile."""
    h, w = img.shape[:2]
    gs = max(2, grid_size)
    bh, bw = max(1, h // gs), max(1, w // gs)

    def _sample_grid(channel):
        samples = np.zeros((gs, gs), dtype=np.float32)
        for iy in range(gs):
            for ix in range(gs):
                y0, y1 = iy * bh, min((iy + 1) * bh, h)
                x0, x1 = ix * bw, min((ix + 1) * bw, w)
                patch = channel[y0:y1, x0:x1]
                # Sigma-clipped median: yıldız/parlak objeleri dışla
                med = np.median(patch)
                mad = np.median(np.abs(patch - med)) * 1.4826
                clipped = patch[(patch < med + 2.5 * mad)]
                samples[iy, ix] = np.median(clipped) if clipped.size > 0 else med
        return cv2.resize(samples, (w, h), interpolation=cv2.INTER_CUBIC)

    if img.ndim == 3:
        bg = np.stack([_sample_grid(img[:, :, c]) for c in range(3)], axis=2)
    else:
        bg = _sample_grid(img)

    result = img - bg
    mn = np.percentile(result, 0.5)
    result = result - mn
    return np.clip(result, 0, None).astype(np.float32)


def _auto_stretch(img, target_bg=0.20, clip_sigma=-2.8):
    """PixInsight-tarzı Auto STF stretch."""
    def _mtf_ch(c):
        med = float(np.median(c))
        diff = np.abs(c - med)
        mad = float(np.median(diff)) * 1.4826
        c0 = max(0.0, med + clip_sigma * mad)
        if abs(1.0 - c0) < 1e-9:
            return c.copy()
        norm = np.clip((c - c0) / max(1.0 - c0, 1e-9), 0, 1).astype(np.float32)
        m_n = max(1e-9, min(1 - 1e-9, (med - c0) / max(1.0 - c0, 1e-9)))
        denom = (2 * m_n - 1) * norm - m_n
        safe = np.abs(denom) > 1e-9
        mtf = np.where(safe, (m_n - 1) * norm / denom, 0.5).astype(np.float32)
        return np.clip(mtf, 0, 1)

    if img.ndim == 2:
        return _mtf_ch(img)
    return np.stack([_mtf_ch(img[:, :, c]) for c in range(img.shape[2])], axis=2)


def _color_boost(img, saturation=1.4, vibrance=0.3):
    """Renk doygunluğu ve canlılık artırma — Lab uzayında."""
    if img.ndim != 3:
        return img
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    a, b = lab[:, :, 1], lab[:, :, 2]

    chroma = np.sqrt((a - 128) ** 2 + (b - 128) ** 2)
    max_c = max(chroma.max(), 1)
    vib_factor = 1.0 + vibrance * (1.0 - chroma / max_c)
    total_factor = saturation * vib_factor

    lab[:, :, 1] = np.clip(128 + (a - 128) * total_factor, 0, 255)
    lab[:, :, 2] = np.clip(128 + (b - 128) * total_factor, 0, 255)

    result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    return result.astype(np.float32) / 255.0


def _sharpen_detail(img, amount=0.7, radius=1.5):
    """Unsharp mask ile detay keskinleştirme."""
    blurred = cv2.GaussianBlur(img, (0, 0), radius)
    sharpened = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 1).astype(np.float32)


def _denoise_light(img, strength=5):
    """Hafif gürültü azaltma — bilateral filter."""
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    denoised = cv2.bilateralFilter(u8, 7, strength, strength)
    return denoised.astype(np.float32) / 255.0


def _local_contrast(img, clip_limit=2.0, tile_size=8):
    """CLAHE ile lokal kontrast artırma."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    if img.ndim == 3:
        u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return result.astype(np.float32) / 255.0
    else:
        u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        result = clahe.apply(u8)
        return result.astype(np.float32) / 255.0


def full_astro_process(image,
                       # Stretch
                       stretch_strength=0.25,
                       # BG
                       bg_extract=True,
                       bg_grid=8,
                       # Color
                       saturation=1.4,
                       vibrance=0.3,
                       # Star shrink
                       do_star_shrink=True,
                       shrink_factor=1.0,
                       halo_fill_ratio=0.3,
                       star_density_threshold=2.0,
                       # Sharpen
                       sharpen_amount=0.7,
                       sharpen_radius=1.5,
                       # Local contrast
                       local_contrast=1.5,
                       # Denoise
                       denoise_strength=5,
                       **kw):
    """
    Tam astro-görüntü işleme pipeline.
    Ham/lineer görüntüyü tek adımda profesyonel sonuca dönüştürür.
    """
    img = np.ascontiguousarray(image, dtype=np.float32)
    np.clip(img, 0, 1, out=img)

    # 1. BG Extract
    if bg_extract:
        img = _bg_extract(img, grid_size=int(bg_grid))
        mx = img.max()
        if mx > 0:
            img /= mx
        np.clip(img, 0, 1, out=img)

    # 2. Stretch
    img = _auto_stretch(img, target_bg=float(stretch_strength))

    # 3. Local contrast
    if local_contrast > 0.1:
        img = _local_contrast(img, clip_limit=float(local_contrast))

    # 4. Color boost
    if img.ndim == 3 and (saturation != 1.0 or vibrance > 0):
        img = _color_boost(img, saturation=float(saturation),
                           vibrance=float(vibrance))

    # 5. Star shrink
    if do_star_shrink:
        img = star_shrink(img,
                          shrink_factor=float(shrink_factor),
                          halo_fill_ratio=float(halo_fill_ratio),
                          noise_level=0,
                          star_density_threshold=float(star_density_threshold))

    # 6. Sharpen
    if sharpen_amount > 0.01:
        img = _sharpen_detail(img, amount=float(sharpen_amount),
                              radius=float(sharpen_radius))

    # 7. Denoise
    if denoise_strength > 0:
        img = _denoise_light(img, strength=int(denoise_strength))

    np.clip(img, 0, 1, out=img)
    return img.astype(np.float32)
