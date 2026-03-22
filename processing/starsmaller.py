"""
Astro Maestro Pro — StarSmaller  (v18 — Sil ve geri çiz)
========================================================
Her yıldız için:
  1. Tüm yıldızı (çekirdek + halo) maskele
  2. Inpaint ile tamamen sil → temiz arka plan/nebula
  3. Orijinalden sadece küçük merkezi geri kopyala
  → Yıldız merkezleri berrak kalır
  → Halo yok olur, arka plan/nebula çevreden doldurulur
  → Siyah halka imkansız (inpaint çevreden sürekli geçiş yapar)
"""
import cv2
import numpy as np


def reduce_stars(image, strength=0.9, sensitivity=0.5, feather=3,
                 max_sigma=6, min_sigma=1, threshold=0.03,
                 protect_nebula=True, **kw):
    img = np.ascontiguousarray(image, dtype=np.float32)
    np.clip(img, 0, 1, out=img)
    h, w = img.shape[:2]
    is_color = img.ndim == 3

    gray = img.mean(axis=2) if is_color else img.copy()
    s = float(np.clip(strength, 0, 1))
    shrink = max(0.15, 1.0 - s * 0.80)

    # ── 1. Yıldız maskesi ──
    core_mask = _fast_star_mask(gray, float(sensitivity), float(threshold),
                                int(max_sigma), int(min_sigma))

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (core_mask > 0).astype(np.uint8), connectivity=8)

    max_star_area = max(500, h * w * 0.001)

    # ── 2. Tüm yıldız maskesi (silme) + küçük yıldız maskesi (geri koyma) ──
    erase_mask = np.zeros((h, w), np.uint8)    # inpaint ile silinecek
    restore_mask = np.zeros((h, w), np.float32)  # geri konulacak (yumuşak)

    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 2 or area > max_star_area:
            continue
        bw_i = stats[i, cv2.CC_STAT_WIDTH]
        bh_i = stats[i, cv2.CC_STAT_HEIGHT]
        if bw_i > 50 or bh_i > 50:
            continue

        cx, cy = centroids[i]
        icx, icy = int(round(cx)), int(round(cy))
        core_r = max(1.0, np.sqrt(area / np.pi))

        # Gerçek yıldız yarıçapı
        real_r = _measure_star_radius(gray, icx, icy, core_r, h, w)
        new_r = max(0.8, real_r * shrink)

        # Silme maskesi: tüm yıldız
        cv2.circle(erase_mask, (icx, icy), int(real_r) + 2, 255, -1)

        # Geri koyma maskesi: küçük merkez (yumuşak kenarlı)
        pad = int(new_r * 3) + 2
        y0, y1 = max(0, icy - pad), min(h, icy + pad + 1)
        x0, x1 = max(0, icx - pad), min(w, icx + pad + 1)
        rh, rw = y1 - y0, x1 - x0
        if rh < 2 or rw < 2:
            continue

        yy, xx = np.mgrid[0:rh, 0:rw]
        dist = np.sqrt((xx - (icx - x0))**2 + (yy - (icy - y0))**2).astype(np.float32)

        # Yumuşak Gauss maskesi: 1 merkezde, 0 kenarda
        restore_local = np.exp(-0.5 * (dist / max(new_r * 0.8, 0.5))**2)
        restore_local[dist > new_r * 2] = 0

        # Max ile birleştir (üst üste binen yıldızlar için)
        restore_mask[y0:y1, x0:x1] = np.maximum(
            restore_mask[y0:y1, x0:x1], restore_local)

    if erase_mask.sum() == 0:
        feat = max(3, int(feather) * 2 + 1) | 1
        mask_f = cv2.GaussianBlur(core_mask.astype(np.float32),
                                  (feat, feat), feat * 0.4)
        return img.copy(), mask_f

    # ── 3. Inpaint: tüm yıldızları sil ──
    img_u16 = (img * 65535).clip(0, 65535).astype(np.uint16)

    if is_color:
        starless_u16 = np.zeros_like(img_u16)
        for c in range(3):
            starless_u16[:,:,c] = cv2.inpaint(img_u16[:,:,c], erase_mask,
                                               5, cv2.INPAINT_TELEA)
        starless = starless_u16.astype(np.float32) / 65535.0
    else:
        starless_u16 = cv2.inpaint(img_u16, erase_mask, 5, cv2.INPAINT_TELEA)
        starless = starless_u16.astype(np.float32) / 65535.0

    # ── 4. Sonuç: starless + küçük yıldızlar ──
    # erase bölgesinde: starless × (1-restore) + original × restore
    # erase dışında: original (dokunma)
    erase_f = (erase_mask > 0).astype(np.float32)
    # Feather
    feat = max(3, int(feather) * 2 + 1) | 1
    erase_smooth = cv2.GaussianBlur(erase_f, (feat, feat), feat * 0.3)
    erase_smooth = np.clip(erase_smooth, 0, 1)

    # Restore maskesi: erase bölgesi içinde küçük yıldızları geri koy
    np.clip(restore_mask, 0, 1, out=restore_mask)

    if is_color:
        e3 = erase_smooth[:,:,np.newaxis]
        r3 = restore_mask[:,:,np.newaxis]
        # Erase bölgesinde: starless + orijinal yıldız merkezi
        blended_erase = starless * (1 - r3) + img * r3
        # Erase dışı dokunma
        result = img * (1 - e3) + blended_erase * e3
    else:
        blended_erase = starless * (1 - restore_mask) + img * restore_mask
        result = img * (1 - erase_smooth) + blended_erase * erase_smooth

    np.clip(result, 0, 1, out=result)
    mask_f = cv2.GaussianBlur(erase_f, (feat, feat), feat * 0.4)

    return result.astype(np.float32), mask_f


def _measure_star_radius(gray, cx, cy, core_r, h, w):
    """
    Yıldızın gerçek yarıçapını ölç.
    Profil düzleştiğinde (türev ~0) yıldız bitti demektir.
    Nebula/galaksi üzerindeki yıldızlarda erken durur.
    """
    max_r = min(int(core_r * 6), 30, cx, w-cx-1, cy, h-cy-1)
    if max_r < 3:
        return core_r * 1.5

    step = max(1, int(core_r * 0.3))
    prev_val = None

    for r in range(int(core_r), max_r, step):
        vals = []
        for angle in range(0, 360, 30):
            rad = angle * np.pi / 180
            px = int(cx + r * np.cos(rad))
            py = int(cy + r * np.sin(rad))
            if 0 <= px < w and 0 <= py < h:
                vals.append(float(gray[py, px]))
        if not vals:
            continue

        ring_val = float(np.median(vals))

        if prev_val is not None:
            # Profil türevi: ne kadar azalıyor?
            drop = prev_val - ring_val
            # Azalma çok yavaşladıysa → yıldız bitti, nebula/galaksi başladı
            if abs(drop) < 0.005:
                return float(r)

        prev_val = ring_val

    return min(core_r * 2.5, max_r)


def _fast_star_mask(gray, sensitivity, threshold, max_sigma, min_sigma):
    """DoG-based star detection."""
    g8 = (np.clip(gray, 0, 1) * 255).astype(np.uint8)
    h, w = g8.shape
    mask = np.zeros((h, w), np.uint8)

    for sigma in range(min_sigma, max_sigma + 1, max(1, (max_sigma - min_sigma) // 3)):
        ks1 = max(3, sigma * 2 + 1) | 1
        ks2 = max(3, sigma * 4 + 1) | 1
        g1 = cv2.GaussianBlur(g8, (ks1, ks1), sigma)
        g2 = cv2.GaussianBlur(g8, (ks2, ks2), sigma * 2)
        dog = cv2.subtract(g1, g2)
        thr_val = max(1, int(threshold * 255 * (1.1 - sensitivity)))
        _, bw = cv2.threshold(dog, thr_val, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_or(mask, bw)

    return mask
