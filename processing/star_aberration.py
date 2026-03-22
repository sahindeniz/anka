"""
Astro Maestro Pro — Star Aberration Remover
============================================
Optik aberasyonları düzeltir:
  • Kromatik aberasyon (renk saçağı — mor/yeşil/kırmızı fringe)
  • Koma (kenar yıldızları kuyruklu)
  • Astigmatizm (yıldızlar uzamış)
  • Difraksiyon spike temizleme

Yöntem: Yıldız tespiti → her yıldız için aberasyon analizi → düzeltme.
"""
import cv2
import numpy as np


def fix_aberration(image, method="auto",
                   chromatic_strength=0.8,
                   coma_strength=0.7,
                   roundness_strength=0.6,
                   spike_strength=0.0,
                   sensitivity=0.5,
                   protect_nebula=True,
                   **kw):
    """
    Yıldız aberasyonlarını düzeltir.

    Parameters
    ----------
    method : str
        "auto"      — tüm aberasyonları otomatik tespit ve düzelt
        "chromatic"  — sadece kromatik aberasyon
        "coma"       — sadece koma düzeltme
        "roundness"  — sadece yıldız yuvarlaklığı (astigmatizm)
        "spike"      — difraksiyon spike temizleme
    chromatic_strength : float 0-1
        Kromatik aberasyon düzeltme gücü
    coma_strength : float 0-1
        Koma düzeltme gücü
    roundness_strength : float 0-1
        Yıldız yuvarlaklığı düzeltme gücü
    spike_strength : float 0-1
        Difraksiyon spike temizleme gücü
    sensitivity : float 0-1
        Yıldız tespiti hassasiyeti
    protect_nebula : bool
        Nebula/galaksi bölgelerini koru
    """
    img = np.ascontiguousarray(image, dtype=np.float32)
    np.clip(img, 0, 1, out=img)
    if img.ndim != 3:
        return img

    result = img.copy()

    # Yıldız tespiti
    stars, star_mask = _detect_stars_detailed(img, sensitivity)
    if len(stars) < 3:
        return img

    # Nebula koruma maskesi
    nebula_mask = None
    if protect_nebula:
        nebula_mask = _detect_nebula_mask(img)

    if method == "auto" or method == "chromatic":
        if chromatic_strength > 0.01:
            result = _fix_chromatic(result, stars, star_mask,
                                    chromatic_strength, nebula_mask)

    if method == "auto" or method == "coma":
        if coma_strength > 0.01:
            result = _fix_coma(result, stars, coma_strength, nebula_mask)

    if method == "auto" or method == "roundness":
        if roundness_strength > 0.01:
            result = _fix_roundness(result, stars, roundness_strength, nebula_mask)

    if method == "auto" or method == "spike":
        if spike_strength > 0.01:
            result = _fix_spikes(result, stars, star_mask,
                                 spike_strength, nebula_mask)

    np.clip(result, 0, 1, out=result)
    return result.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Yıldız tespiti (detaylı — centroid, yarıçap, elongasyon, açı)
# ─────────────────────────────────────────────────────────────────────────────
def _detect_stars_detailed(img, sensitivity=0.5):
    """
    Yıldızları tespit et ve her biri için detaylı bilgi döndür.
    Returns: list of dict, star_mask
    """
    h, w = img.shape[:2]
    gray = (0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1]
            + 0.0722 * img[:, :, 2])

    # DoG ile yıldız tespiti
    g8 = (np.clip(gray, 0, 1) * 255).astype(np.uint8)
    g1 = cv2.GaussianBlur(g8, (3, 3), 1.0)
    g2 = cv2.GaussianBlur(g8, (9, 9), 3.0)
    dog = cv2.subtract(g1, g2)
    thr = max(1, int(15 * (1.1 - sensitivity)))
    _, bw = cv2.threshold(dog, thr, 255, cv2.THRESH_BINARY)

    # Morfolojik temizlik
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kern)
    bw = cv2.dilate(bw, kern)

    # Connected components
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, 8)

    max_area = int(h * w * 0.003)
    stars = []

    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 3 or area > max_area:
            continue

        cx, cy = centroids[i]
        sx = stats[i, cv2.CC_STAT_LEFT]
        sy = stats[i, cv2.CC_STAT_TOP]
        sw = stats[i, cv2.CC_STAT_WIDTH]
        sh = stats[i, cv2.CC_STAT_HEIGHT]

        # Aspect ratio — çok uzun nesneleri atla (uydu izi vb.)
        aspect = max(sw, sh) / max(min(sw, sh), 1)
        if aspect > 5:
            continue

        # Yıldız yarıçapı
        radius = max(1.5, np.sqrt(area / np.pi))

        # Elongasyon ve açı (moments'tan)
        region = (labels == i).astype(np.uint8)
        M = cv2.moments(region)
        if M["m00"] < 1:
            continue

        # İkinci dereceden momentler → elongasyon
        mu20 = M["mu20"] / M["m00"]
        mu02 = M["mu02"] / M["m00"]
        mu11 = M["mu11"] / M["m00"]

        # Eigenvalue'lar (elde major/minor axis)
        delta = np.sqrt(4 * mu11**2 + (mu20 - mu02)**2)
        lam1 = (mu20 + mu02 + delta) / 2
        lam2 = max((mu20 + mu02 - delta) / 2, 0.01)
        elongation = np.sqrt(lam1 / lam2) if lam2 > 0 else 1.0
        angle = 0.5 * np.degrees(np.arctan2(2 * mu11, mu20 - mu02))

        # Renk bilgisi (centroid civarı)
        ci = int(np.clip(round(cy), 0, h - 1))
        cj = int(np.clip(round(cx), 0, w - 1))
        r_pad = max(2, int(radius))
        y0 = max(0, ci - r_pad)
        y1 = min(h, ci + r_pad + 1)
        x0 = max(0, cj - r_pad)
        x1 = min(w, cj + r_pad + 1)
        patch = img[y0:y1, x0:x1]
        color = patch.mean(axis=(0, 1)) if patch.size > 0 else img[ci, cj]

        # Merkeze olan mesafe (koma analizi için)
        dist_center = np.sqrt((cx - w / 2)**2 + (cy - h / 2)**2)
        dist_norm = dist_center / (np.sqrt(w**2 + h**2) / 2)

        stars.append({
            "cx": float(cx), "cy": float(cy),
            "radius": float(radius),
            "area": int(area),
            "elongation": float(elongation),
            "angle": float(angle),
            "color": color.copy(),
            "aspect": float(aspect),
            "dist_center": float(dist_norm),
            "label_id": i,
        })

    star_mask = (bw > 0).astype(np.uint8)
    return stars, star_mask


def _detect_nebula_mask(img):
    """Nebula/galaksi bölgelerini tespit et (koruma için)."""
    h, w = img.shape[:2]
    gray = img.mean(axis=2)

    # Büyük yapıları bul (yıldız değil)
    blurred = cv2.GaussianBlur(gray, (31, 31), 10)
    med = float(np.median(blurred))
    std = float(np.std(blurred))

    # Arka plandan belirgin ama çok parlak olmayan bölgeler
    nebula = ((blurred > med + 0.5 * std) &
              (blurred < med + 5 * std)).astype(np.uint8) * 255

    # Genişlet — koruma alanını büyüt
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    nebula = cv2.dilate(nebula, kern)
    return cv2.GaussianBlur(nebula.astype(np.float32) / 255.0,
                            (15, 15), 5)


# ─────────────────────────────────────────────────────────────────────────────
#  Kromatik Aberasyon Düzeltme
# ─────────────────────────────────────────────────────────────────────────────
def _fix_chromatic(img, stars, star_mask, strength, nebula_mask):
    """
    Kromatik aberasyon: R ve B kanalları G'ye göre hafif kaymış olur.
    Her yıldız etrafında R ve B kanallarını G kanalına hizala.
    """
    h, w = img.shape[:2]
    result = img.copy()

    # Global kromatik kayma tahmini — parlak yıldızlardan
    bright_stars = sorted(stars, key=lambda s: -s["area"])[:min(50, len(stars))]

    # Her yıldız için R-G ve B-G renk farkı haritası
    for star in bright_stars:
        cx, cy = star["cx"], star["cy"]
        r = max(3, int(star["radius"] * 2.5))

        y0 = max(0, int(cy) - r)
        y1 = min(h, int(cy) + r + 1)
        x0 = max(0, int(cx) - r)
        x1 = min(w, int(cx) + r + 1)

        rh, rw = y1 - y0, x1 - x0
        if rh < 3 or rw < 3:
            continue

        # Nebula koruma
        if nebula_mask is not None:
            if nebula_mask[int(cy), int(cx)] > 0.5:
                continue

        patch = result[y0:y1, x0:x1].copy()
        g_ch = patch[:, :, 1]  # Green kanal (referans)

        # Radyal mesafe maskesi
        yy, xx = np.mgrid[0:rh, 0:rw]
        lcx, lcy = cx - x0, cy - y0
        dist = np.sqrt((xx - lcx)**2 + (yy - lcy)**2).astype(np.float32)
        star_profile = np.exp(-0.5 * (dist / max(star["radius"] * 0.8, 1))**2)

        # R ve B kanalını G'ye yaklaştır (fringe kaldır)
        for ch in [0, 2]:  # R, B
            diff = patch[:, :, ch] - g_ch
            # Sadece renk sapmasını düzelt (parlak kısımlarda)
            correction = diff * star_profile * strength * 0.7
            result[y0:y1, x0:x1, ch] -= correction

    np.clip(result, 0, 1, out=result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Koma Düzeltme
# ─────────────────────────────────────────────────────────────────────────────
def _fix_coma(img, stars, strength, nebula_mask):
    """
    Koma: kenar yıldızları merkeze doğru kuyruk oluşturur.
    Her yıldızı simetrik hale getir — merkeze doğru olan fazlalığı azalt.
    """
    h, w = img.shape[:2]
    result = img.copy()
    img_center_x, img_center_y = w / 2.0, h / 2.0

    for star in stars:
        # Sadece uzamış yıldızlara uygula
        if star["elongation"] < 1.3:
            continue

        cx, cy = star["cx"], star["cy"]
        r = max(4, int(star["radius"] * 3))

        y0 = max(0, int(cy) - r)
        y1 = min(h, int(cy) + r + 1)
        x0 = max(0, int(cx) - r)
        x1 = min(w, int(cx) + r + 1)

        rh, rw = y1 - y0, x1 - x0
        if rh < 4 or rw < 4:
            continue

        if nebula_mask is not None:
            if nebula_mask[int(cy), int(cx)] > 0.5:
                continue

        patch = result[y0:y1, x0:x1].copy()
        lcx, lcy = cx - x0, cy - y0

        # Koma yönü: yıldızdan görüntü merkezine doğru
        dx = img_center_x - cx
        dy = img_center_y - cy
        dist_to_center = np.sqrt(dx**2 + dy**2) + 1e-9

        # Kenar yıldızlarında daha güçlü düzeltme
        edge_factor = min(1.0, star["dist_center"] * 1.5)
        elong_factor = min(1.0, (star["elongation"] - 1.0) / 1.5)
        s = strength * edge_factor * elong_factor

        if s < 0.05:
            continue

        # Radyal simetrik yapma: yıldızı dairesel Gauss ile yeniden çiz
        yy, xx = np.mgrid[0:rh, 0:rw]
        dist = np.sqrt((xx - lcx)**2 + (yy - lcy)**2).astype(np.float32)
        sym_sigma = max(star["radius"] * 0.6, 1.0)
        sym_profile = np.exp(-0.5 * (dist / sym_sigma)**2)

        # Mevcut profil ile simetrik profil arasını blend
        for ch in range(3):
            ch_data = patch[:, :, ch]
            # Arka plan seviyesi
            bg_level = float(np.percentile(ch_data, 15))
            # Yıldız ışığı
            star_light = np.clip(ch_data - bg_level, 0, 1)
            peak = float(star_light.max())
            if peak < 0.02:
                continue
            # Simetrik versiyon
            sym_light = sym_profile * peak
            # Blend
            blended = star_light * (1 - s) + sym_light * s
            result[y0:y1, x0:x1, ch] = bg_level + blended

    np.clip(result, 0, 1, out=result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Yıldız Yuvarlaklığı (Astigmatizm Düzeltme)
# ─────────────────────────────────────────────────────────────────────────────
def _fix_roundness(img, stars, strength, nebula_mask):
    """
    Astigmatizm/tracking hatası: yıldızlar eliptik olur.
    Her yıldızı dairesel hale getir.
    """
    h, w = img.shape[:2]
    result = img.copy()

    for star in stars:
        if star["elongation"] < 1.15:
            continue

        cx, cy = star["cx"], star["cy"]
        r = max(4, int(star["radius"] * 2.5))

        y0 = max(0, int(cy) - r)
        y1 = min(h, int(cy) + r + 1)
        x0 = max(0, int(cx) - r)
        x1 = min(w, int(cx) + r + 1)

        rh, rw = y1 - y0, x1 - x0
        if rh < 4 or rw < 4:
            continue

        if nebula_mask is not None:
            if nebula_mask[int(cy), int(cx)] > 0.5:
                continue

        elong_factor = min(1.0, (star["elongation"] - 1.0) / 2.0)
        s = strength * elong_factor

        if s < 0.05:
            continue

        patch = result[y0:y1, x0:x1].copy()
        lcx, lcy = cx - x0, cy - y0

        # Simetrik dairesel profil
        yy, xx = np.mgrid[0:rh, 0:rw]
        dist = np.sqrt((xx - lcx)**2 + (yy - lcy)**2).astype(np.float32)
        circ_sigma = max(star["radius"] * 0.5, 0.8)
        circ_profile = np.exp(-0.5 * (dist / circ_sigma)**2)

        for ch in range(3):
            ch_data = patch[:, :, ch]
            bg_level = float(np.percentile(ch_data, 15))
            star_light = np.clip(ch_data - bg_level, 0, 1)
            peak = float(star_light.max())
            if peak < 0.02:
                continue
            circ_light = circ_profile * peak
            blended = star_light * (1 - s) + circ_light * s
            result[y0:y1, x0:x1, ch] = bg_level + blended

    np.clip(result, 0, 1, out=result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Difraksiyon Spike Temizleme
# ─────────────────────────────────────────────────────────────────────────────
def _fix_spikes(img, stars, star_mask, strength, nebula_mask):
    """
    Difraksiyon spike'ları: parlak yıldızlardan çıkan çizgisel ışık.
    Spike yönünü tespit et, o bölgeyi arka planla doldur.
    """
    h, w = img.shape[:2]
    result = img.copy()

    # Sadece parlak yıldızlarda spike olur
    bright = [s for s in stars if s["area"] > 20]
    bright.sort(key=lambda s: -s["area"])
    bright = bright[:30]

    for star in bright:
        cx, cy = star["cx"], star["cy"]
        r = max(8, int(star["radius"] * 4))

        y0 = max(0, int(cy) - r)
        y1 = min(h, int(cy) + r + 1)
        x0 = max(0, int(cx) - r)
        x1 = min(w, int(cx) + r + 1)

        rh, rw = y1 - y0, x1 - x0
        if rh < 8 or rw < 8:
            continue

        if nebula_mask is not None:
            if nebula_mask[int(cy), int(cx)] > 0.5:
                continue

        patch = result[y0:y1, x0:x1].copy()
        lcx, lcy = cx - x0, cy - y0

        # Radyal profil — dairesel simetrik
        yy, xx = np.mgrid[0:rh, 0:rw]
        dist = np.sqrt((xx - lcx)**2 + (yy - lcy)**2).astype(np.float32)
        circ_sigma = max(star["radius"] * 0.7, 1.5)

        # Spike tespiti: dairesel profilden sapan pikselller
        gray_patch = patch.mean(axis=2)
        bg_level = float(np.percentile(gray_patch, 25))
        star_light = np.clip(gray_patch - bg_level, 0, 1)

        # Beklenen dairesel profil
        expected = np.exp(-0.5 * (dist / circ_sigma)**2) * float(star_light.max())

        # Spike = gerçek ışık dairesel profilden fazla ve merkezden uzak
        excess = np.clip(star_light - expected, 0, 1)
        spike_zone = (excess > 0.02) & (dist > star["radius"] * 1.2)

        if not spike_zone.any():
            continue

        # Spike bölgesini arka planla doldur
        spike_f = cv2.GaussianBlur(spike_zone.astype(np.float32), (5, 5), 1.5)
        spike_blend = spike_f * strength

        for ch in range(3):
            ch_data = result[y0:y1, x0:x1, ch]
            bg_ch = float(np.percentile(patch[:, :, ch], 25))
            result[y0:y1, x0:x1, ch] = (ch_data * (1 - spike_blend)
                                         + bg_ch * spike_blend)

    np.clip(result, 0, 1, out=result)
    return result
