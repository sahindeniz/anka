"""
Astro Maestro Pro — Color Calibration
=======================================
Best-practice renk kalibrasyonu:

Referans: "Average Spiral Galaxy" / G2V yildiz (Gunes tipi, ~5800K)
Kaynak  : PixInsight SPCC / Gaia DR3 fotometrik standartlari
          G2V yildiz renk sicakligi RGB oranlari: R:G:B = 1.000 : 0.940 : 0.880

Metodlar:
  spcc_g2v      — G2V yildiz referansi (en fotometrik dogru)
  avg_spiral    — Average Spiral Galaxy referansi (PixInsight default)
  pcc           — Yildiz tabanlı PCC (blob tespiti)
  photometric   — Basit ortalam fotometri
  white_balance — Yuzdelik beyaz nokta
  ai_neutral    — AI arka plan notralizasyonu
"""

import cv2
import numpy as np


# ── Referans RGB agirliклari ──────────────────────────────────────────────────
# Gaia DR3 / PixInsight SPCC standartlarindan alinmistir.
# Kaynak: PixInsight SPCC dokumantasyonu + Siril SPCC uygulamasi

# G2V yildiz (Gunes tipi, ~5778K) — en fotometrik dogru beyaz referans
# RGB: (1.000, 0.955, 0.880) — hafif sicak beyaz
G2V_RGB = np.array([1.000, 0.955, 0.880], dtype=np.float64)

# Average Spiral Galaxy — PixInsight PCC/SPCC varsayilan referansi
# Sarmansi galaksinin toplu yildiz isigi ~5500K renk sicakligi
# RGB: (1.000, 0.965, 0.895)
AVG_SPIRAL_RGB = np.array([1.000, 0.965, 0.895], dtype=np.float64)

# A0V yildiz (Vega, 9600K) — Mavi-beyaz, dar band icin
A0V_RGB = np.array([0.820, 0.900, 1.000], dtype=np.float64)

# G0V yildiz (Procyon, 6100K)
G0V_RGB = np.array([1.000, 0.972, 0.920], dtype=np.float64)


def calibrate_color(image, method="spcc_g2v",
                    neutral_percentile=50.0,
                    white_reference="g2v",
                    _progress_cb=None,
                    **kwargs):
    if image.ndim != 3:
        return image
    img = image.astype(np.float32)

    if   method == "pcc_solve":   return _pcc_platesolve(img, _progress_cb, **kwargs)
    elif method == "spcc_g2v":    return _spcc(img, G2V_RGB)
    elif method == "avg_spiral":  return _spcc(img, AVG_SPIRAL_RGB)
    elif method == "pcc":         return _pcc_fast(img)
    elif method == "photometric": return _photometric(img)
    elif method == "white_balance":return _white_balance(img)
    elif method == "ai_neutral":  return _ai_neutral(img, float(neutral_percentile))
    elif method == "vectra":      return img   # Vectra kendi pipeline'inda
    elif method == "alchemy":     return img   # Alchemy kendi pipeline'inda
    return img


# ── SPCC — G2V / Average Spiral Galaxy referanslı ────────────────────────────
def _spcc(img: np.ndarray, reference_rgb: np.ndarray) -> np.ndarray:
    """
    Spectrophotometric Color Calibration.

    Adimlar:
    1. Arkaplan notralizasyonu (koyu bolgeler)
    2. Yildiz renk olcumu (blob tespiti, hizli)
    3. Referans RGB (G2V / Avg Spiral) ile karsilastirma
    4. Per-channel kazanc uygula

    reference_rgb: [R_ref, G_ref, B_ref] — hedef beyaz noktasi
    """
    h, w = img.shape[:2]

    # ── 1. Arka plan tespiti ──────────────────────────────────────────────────
    gray = img.mean(axis=2)
    bg_percentile = np.percentile(gray, 25)
    bg_mask = gray < bg_percentile
    if not np.any(bg_mask):
        bg_mask = gray <= bg_percentile

    # Arka plan notralizasyonu
    bg = np.median(img[bg_mask], axis=0).astype(np.float64) + 1e-9

    # ── 2. Yildiz tespiti (kucuk resimde hizli) ───────────────────────────────
    SCALE = 512
    scale = min(1.0, SCALE / max(h, w))
    sw, sh = max(4,int(w*scale)), max(4,int(h*scale))
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    gray_s = np.nan_to_num(small.mean(axis=2), nan=0.0)

    # Parlak yildizlari tespit et
    gray_u8 = np.clip(gray_s * 255, 0, 255).astype(np.uint8)
    threshold = max(1, int(np.percentile(gray_u8, 97)))
    _, bw = cv2.threshold(gray_u8, threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3,3), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    star_colors = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 2 <= area <= 200:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                r = max(2, int(np.sqrt(area/np.pi)) + 1)
                y0,y1 = max(0,cy-r), min(sh,cy+r+1)
                x0,x1 = max(0,cx-r), min(sw,cx+r+1)
                patch = small[y0:y1, x0:x1]
                if patch.size > 0:
                    star_colors.append(patch.mean(axis=(0, 1)).tolist())

    if len(star_colors) < 5:
        # Yetersiz yildiz — sadece background neutralize et
        img_bg = img - bg[None,None,:] * 0.7
        img_bg = np.clip(img_bg, 0, 1)
        # Referans oranina gore scale
        ref_norm = reference_rgb / reference_rgb.max()
        means = img_bg.mean(axis=(0, 1)).astype(np.float64) + 1e-9
        gains = (means.mean() * ref_norm) / means
        gains = np.clip(gains / gains.max(), 0.5, 2.0)
        return np.clip(img_bg * gains[None,None,:], 0, 1).astype(np.float32)

    # ── 3. Yildizlarin ortalama rengi ────────────────────────────────────────
    sc_arr = np.array(star_colors)  # (N, 3)

    # Outlier'lari kaldir (kappa-sigma)
    mean_sc = sc_arr.mean(axis=0)
    std_sc  = sc_arr.std(axis=0) + 1e-9
    mask    = np.all(np.abs(sc_arr - mean_sc) < 2.5 * std_sc, axis=1)
    if mask.sum() >= 3:
        sc_arr = sc_arr[mask]

    star_rgb = np.median(sc_arr, axis=0) + 1e-9  # [R_star, G_star, B_star]

    # ── 4. Kazanc hesapla ─────────────────────────────────────────────────────
    # Hedef: yildiz rengi reference_rgb ile eslesin
    # gain[c] = (reference_rgb[c] / reference_rgb.max()) / (star_rgb[c] / star_rgb.max())
    ref_norm  = reference_rgb / reference_rgb[1]   # G kanalini normalize et
    star_norm = star_rgb / star_rgb[1]

    gains = ref_norm / star_norm
    gains = np.clip(gains, 0.5, 2.5)

    # ── 5. Uygula ─────────────────────────────────────────────────────────────
    result = img.astype(np.float64)

    # Arka plan cikar (hafif)
    result -= bg[None,None,:] * 0.3
    result  = np.clip(result, 0, 1)

    # Kazanc uygula (vectorized)
    result *= gains[None, None, :]

    # Yeniden normalize (parlaklik koru)
    orig_mean = img.mean()
    new_mean  = result.mean()
    if new_mean > 1e-9:
        result *= orig_mean / new_mean

    return np.clip(result, 0, 1).astype(np.float32)


# ── Photometric ───────────────────────────────────────────────────────────────
def _photometric(img):
    r,g,b = img[:,:,0],img[:,:,1],img[:,:,2]
    mr,mg,mb = np.mean(r)+1e-9, np.mean(g)+1e-9, np.mean(b)+1e-9
    ref = (mr+mg+mb)/3
    return np.clip(np.stack([r*(ref/mr),g*(ref/mg),b*(ref/mb)],2),0,1).astype(np.float32)


# ── White Balance ─────────────────────────────────────────────────────────────
def _white_balance(img):
    p = np.percentile(img, 99, axis=(0, 1)) + 1e-9
    return np.clip(img / p[None, None, :], 0, 1).astype(np.float32)


# ── PCC Plate Solve — Katalog tabanlı gerçek fotometrik kalibrasyon ───────────
def _pcc_platesolve(img, progress_cb=None, **kwargs):
    """
    Plate solve sonucu + Gaia/APASS katalog ile gerçek PCC.

    Adımlar:
    1. Görüntüdeki yıldızları tespit et (piksel koordinatları)
    2. Plate solve WCS ile piksel→RA/Dec dönüşümü
    3. VizieR/Gaia DR3'ten referans yıldız renk indekslerini çek
    4. Görüntüdeki yıldız renkleri ile katalog renklerini eşleştir
    5. Per-kanal kazanç hesapla ve uygula
    """
    def cb(msg):
        if progress_cb:
            progress_cb(str(msg))

    ra_center  = float(kwargs.get("solve_ra", 0))
    dec_center = float(kwargs.get("solve_dec", 0))
    scale_aspp = float(kwargs.get("solve_scale", 1.8))
    rotation   = float(kwargs.get("solve_rotation", 0.0))
    limit_mag  = float(kwargs.get("catalog_limit_mag", 16.0))

    h, w = img.shape[:2]
    cb("[1/5] Yıldız tespiti...")

    # ── 1. Görüntüdeki yıldızları tespit et ──
    stars_px = _detect_stars_for_pcc(img)
    cb(f"[1/5] {len(stars_px)} yıldız tespit edildi")

    if len(stars_px) < 10:
        cb("Yetersiz yıldız — G2V referansına geçiliyor")
        return _spcc(img, G2V_RGB)

    # ── 2. Piksel → RA/Dec dönüşümü (basit WCS: TAN projeksiyon) ──
    cb("[2/5] WCS dönüşümü...")
    stars_radec = _pixel_to_radec(
        stars_px, w, h, ra_center, dec_center, scale_aspp, rotation)

    # ── 3. Katalogdan referans yıldız renklerini çek ──
    fov_deg = max(w, h) * scale_aspp / 3600.0
    cb(
        "[3/5] Katalog sorgusu: Gaia DR3/APASS "
        f"(yarıçap ~{min(max(0.2, fov_deg * 0.5), 1.5):.2f}°, "
        f"limit mag {limit_mag:.2f})"
    )
    catalog_stars, catalog_meta = _query_gaia_colors(
        ra_center, dec_center, fov_deg, limit_mag=limit_mag
    )

    if catalog_stars is None or len(catalog_stars) < 5:
        source_name = catalog_meta.get("source_name", "Katalog")
        err = catalog_meta.get("error")
        if err:
            cb(f"[3/5] {source_name} sorgusu başarısız: {err}")
        cb("Katalog sorgusu başarısız — G2V referansına geçiliyor")
        return _spcc(img, G2V_RGB)
    cb(
        f"[3/5] {catalog_meta.get('source_name', 'Katalog')} "
        f"kaynağından {len(catalog_stars)} yıldız alındı"
    )

    # Debug: RA/Dec aralıklarını karşılaştır
    img_ras  = [rd[0] for rd in stars_radec]
    img_decs = [rd[1] for rd in stars_radec]
    cat_ras  = [s[0] for s in catalog_stars]
    cat_decs = [s[1] for s in catalog_stars]
    cb(f"  Img RA:  {min(img_ras):.3f}-{max(img_ras):.3f}  "
       f"Dec: {min(img_decs):.3f}-{max(img_decs):.3f}")
    cb(f"  Cat RA:  {min(cat_ras):.3f}-{max(cat_ras):.3f}  "
       f"Dec: {min(cat_decs):.3f}-{max(cat_decs):.3f}")

    # ── 4. Eşleştir: görüntü yıldızları ↔ katalog yıldızları ──
    cb("[4/5] Yıldız eşleştirmesi...")
    matched, match_stats = _match_stars(
        stars_px, stars_radec, catalog_stars, img, scale_aspp
    )
    cb(
        f"[4/5] {len(matched)} yıldız eşleştirildi "
        f"(eşleşmedi: {match_stats['excluded_no_catalog']}, "
        f"sönük: {match_stats['excluded_mag']}, "
        f"tekrar: {match_stats['excluded_duplicate']})"
    )

    if len(matched) < 5:
        cb("Yetersiz eşleşme — G2V referansına geçiliyor")
        return _spcc(img, G2V_RGB)

    # ── 5. Kazanç hesapla ve uygula ──
    cb("[5/5] Renk kalibrasyonu uygulanıyor...")
    result, cal_stats = _apply_catalog_calibration(img, matched)

    fit_rg = cal_stats.get("fit_rg")
    fit_bg = cal_stats.get("fit_bg")
    if fit_rg or fit_bg:
        cb("  PCC Linear Fits")
    if fit_rg:
        cb(
            "  Image R/G = "
            f"{fit_rg['intercept']:.6f} + {fit_rg['slope']:.6f} * "
            f"Catalog R/G (sigma: {fit_rg['sigma']:.6f})"
        )
    if fit_bg:
        cb(
            "  Image B/G = "
            f"{fit_bg['intercept']:.6f} + {fit_bg['slope']:.6f} * "
            f"Catalog B/G (sigma: {fit_bg['sigma']:.6f})"
        )

    cb(
        f"  Çözüm {cal_stats['used_star_count']} yıldız ile bulundu "
        f"(outlier hariç tutulan: {cal_stats['excluded_outliers']})"
    )
    cb("  White balance factors:")
    gains = cal_stats["gains"]
    for idx, gain in enumerate(gains):
        cb(f"  K{idx}: {gain:.3f}")
    cb("  Background reference:")
    bg = cal_stats["background"]
    for idx, value in enumerate(bg):
        cb(f"  B{idx}: {value:+.5e}")

    quality_warning = cal_stats.get("quality_warning")
    if quality_warning:
        cb(f"  {quality_warning}")

    cb(f"[5/5] Tamamlandı ({cal_stats['used_star_count']} yıldız ile kalibre edildi)")
    return result


def _detect_stars_for_pcc(img):
    """Görüntüdeki yıldızları tespit et. [(x, y, brightness, r, g, b), ...]"""
    h, w = img.shape[:2]
    # Küçük resimde çalış
    MAX_DIM = 1024
    scale = min(1.0, MAX_DIM / max(h, w))
    sw, sh = max(4, int(w * scale)), max(4, int(h * scale))
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    gray = small.mean(axis=2)

    # Yıldız tespiti: parlak noktalar
    threshold = np.percentile(gray, 95)
    bw = (gray > threshold).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
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
                r = max(2, int(np.sqrt(area / np.pi)) + 1)
                y0, y1 = max(0, int(cy) - r), min(sh, int(cy) + r + 1)
                x0, x1 = max(0, int(cx) - r), min(sw, int(cx) + r + 1)
                patch = small[y0:y1, x0:x1]
                if patch.size > 0:
                    rgb = patch.mean(axis=(0, 1)).tolist()
                    brightness = sum(rgb) / 3.0
                    # Orijinal koordinatlara dönüştür
                    ox = cx / scale
                    oy = cy / scale
                    stars.append((ox, oy, brightness, rgb[0], rgb[1], rgb[2]))

    # Parlaklığa göre sırala, en parlak 200 yıldız
    stars.sort(key=lambda s: -s[2])
    return stars[:200]


def _pixel_to_radec(stars_px, w, h, ra0, dec0, scale_aspp, rotation_deg):
    """Basit TAN projeksiyon ile piksel → RA/Dec."""
    cx, cy = w / 2.0, h / 2.0
    scale_deg = scale_aspp / 3600.0
    rot_rad = np.radians(rotation_deg)
    cos_r, sin_r = np.cos(rot_rad), np.sin(rot_rad)
    dec0_rad = np.radians(dec0)
    cos_dec = np.cos(dec0_rad)

    result = []
    for (px, py, *rest) in stars_px:
        # Piksel offset (merkeze göre)
        dx = (px - cx) * scale_deg
        dy = (cy - py) * scale_deg  # y ters
        # Rotasyon uygula
        xi = cos_r * dx - sin_r * dy
        eta = sin_r * dx + cos_r * dy
        # TAN → RA/Dec
        ra = ra0 + xi / (cos_dec if cos_dec > 0.01 else 0.01)
        dec = dec0 + eta
        result.append((ra, dec))
    return result


def _query_gaia_colors(ra_deg, dec_deg, fov_deg, limit_mag=16.0):
    """
    VizieR üzerinden Gaia DR3 yıldız renklerini çek.
    Gaia DR3 native kolon adları: RA_ICRS, DE_ICRS, Gmag, BP-RP
    """
    radius = min(max(0.2, fov_deg * 0.5), 1.5)
    meta = {
        "source_name": "Gaia DR3",
        "radius_deg": float(radius),
        "limit_mag": float(limit_mag),
        "error": "",
    }

    try:
        from astroquery.vizier import Vizier
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        coord = SkyCoord(ra=ra_deg, dec=dec_deg, unit="deg", frame="icrs")

        # Gaia DR3 native kolonları + mesafe sıralaması
        v = Vizier(columns=["RA_ICRS", "DE_ICRS", "Gmag", "BP-RP", "+_r"],
                   row_limit=2000,
                   column_filters={"Gmag": f"<{float(limit_mag):.2f}"})
        v.ROW_LIMIT = 2000
        tables = v.query_region(coord, radius=radius * u.deg,
                                catalog="I/355/gaiadr3")

        if tables and len(tables) > 0:
            t = tables[0]
            stars = []
            # Kolon adlarını tespit et (VizieR bazen farklı isimler döner)
            ra_col = None
            dec_col = None
            for col_name in ["RA_ICRS", "RAJ2000", "_RAJ2000", "ra"]:
                if col_name in t.colnames:
                    ra_col = col_name
                    break
            for col_name in ["DE_ICRS", "DEJ2000", "_DEJ2000", "dec"]:
                if col_name in t.colnames:
                    dec_col = col_name
                    break
            if ra_col is None or dec_col is None:
                # Kolon bulunamadı — APASS'a düş
                tables = None
            else:
                bp_rp_col = None
                for col_name in ["BP-RP", "BPmag-RPmag", "bp_rp"]:
                    if col_name in t.colnames:
                        bp_rp_col = col_name
                        break
                gmag_col = None
                for col_name in ["Gmag", "phot_g_mean_mag", "gmag"]:
                    if col_name in t.colnames:
                        gmag_col = col_name
                        break

                if bp_rp_col and gmag_col:
                    for row in t:
                        try:
                            # Masked/NaN değerleri güvenli kontrol
                            _ra_val = row[ra_col]
                            _dec_val = row[dec_col]
                            _bp_rp_val = row[bp_rp_col]
                            _mag_val = row[gmag_col]
                            # astropy masked column kontrolü
                            if hasattr(_bp_rp_val, 'mask') or hasattr(_mag_val, 'mask'):
                                continue
                            ra = float(_ra_val)
                            dec = float(_dec_val)
                            bp_rp = float(_bp_rp_val)
                            mag = float(_mag_val)
                            if not np.isfinite(bp_rp) or not np.isfinite(mag):
                                continue
                            if mag > float(limit_mag):
                                continue
                            bv = 0.0981 + 0.9287 * bp_rp - 0.0736 * bp_rp**2
                            rgb = _bv_to_rgb(bv)
                            stars.append((ra, dec, mag, rgb))
                        except (ValueError, KeyError):
                            continue
                    if len(stars) >= 5:
                        meta["source_name"] = "Gaia DR3"
                        return stars, meta

        # APASS DR9 alternatifi
        meta["source_name"] = "APASS DR9"
        v2 = Vizier(columns=["RAJ2000", "DEJ2000", "Vmag", "B-V"],
                    row_limit=2000)
        tables = v2.query_region(coord, radius=radius * u.deg,
                                 catalog="II/336/apass9")
        if not tables or len(tables) == 0:
            return None, meta
        t = tables[0]
        stars = []
        for row in t:
            try:
                ra = float(row["RAJ2000"])
                dec = float(row["DEJ2000"])
                bv = float(row["B-V"])
                mag = float(row["Vmag"])
                if not np.isfinite(bv) or not np.isfinite(mag):
                    continue
                if mag > float(limit_mag):
                    continue
                rgb = _bv_to_rgb(bv)
                stars.append((ra, dec, mag, rgb))
            except (ValueError, KeyError):
                continue
        return (stars, meta) if len(stars) >= 5 else (None, meta)

    except ImportError:
        meta["error"] = "astroquery yüklü değil"
        return None, meta
    except Exception as exc:
        meta["error"] = str(exc)
        return None, meta


def _bv_to_rgb(bv):
    """
    B-V renk indeksinden RGB dönüşümü.
    Ballesteros (2012) formülüne dayalı, Gaia DR3 ile kalibre.
    """
    bv = float(np.clip(bv, -0.4, 2.0))
    # Renk sıcaklığı (Ballesteros 2012)
    temp = 4600.0 * (1.0 / (0.92 * bv + 1.7) + 1.0 / (0.92 * bv + 0.62))
    # Planck yaklaşımı ile RGB
    return _temp_to_rgb(temp)


def _temp_to_rgb(temp_k):
    """Renk sıcaklığı → normalize RGB (Tanner Helland algoritması)."""
    t = np.clip(temp_k, 1000, 40000) / 100.0

    # R
    if t <= 66:
        r = 1.0
    else:
        r = np.clip(1.292 * ((t - 60) ** -0.1332), 0, 1)
    # G
    if t <= 66:
        g = np.clip(0.3901 * np.log(t) - 0.6318, 0, 1)
    else:
        g = np.clip(1.130 * ((t - 60) ** -0.0755), 0, 1)
    # B
    if t >= 66:
        b = 1.0
    elif t <= 19:
        b = 0.0
    else:
        b = np.clip(0.5432 * np.log(t - 10) - 1.1963, 0, 1)

    mx = max(r, g, b, 1e-9)
    return (r / mx, g / mx, b / mx)


def _match_stars(stars_px, stars_radec, catalog_stars, img, scale_aspp):
    """
    Görüntü yıldızlarını katalog yıldızlarıyla eşleştir.
    Dönüş: [(img_rgb, catalog_rgb), ...]
    """
    stats = {
        "candidate_count": len(stars_px),
        "matched_count": 0,
        "excluded_no_catalog": 0,
        "excluded_mag": 0,
        "excluded_duplicate": 0,
        "match_radius_arcsec": float(scale_aspp * 30.0),
    }
    if not catalog_stars or not stars_radec:
        return [], stats

    cat_ra  = np.array([s[0] for s in catalog_stars])
    cat_dec = np.array([s[1] for s in catalog_stars])
    cat_mag = np.array([s[2] for s in catalog_stars])

    match_radius_deg = scale_aspp * 30.0 / 3600.0  # 30 piksel tolerans (~54 arcsec)
    matched = []
    used_catalog = set()

    for i, (px_data, (ra, dec)) in enumerate(zip(stars_px, stars_radec)):
        x, y, brt, r, g, b = px_data
        # En yakın katalog yıldızını bul
        d_ra = (cat_ra - ra) * np.cos(np.radians(dec))
        d_dec = cat_dec - dec
        dist = np.sqrt(d_ra**2 + d_dec**2)
        idx = np.argmin(dist)

        if dist[idx] >= match_radius_deg:
            stats["excluded_no_catalog"] += 1
            continue

        idx_int = int(idx)
        if idx_int in used_catalog:
            stats["excluded_duplicate"] += 1
            continue

        if cat_mag[idx] >= 14:
            stats["excluded_mag"] += 1
            continue

        used_catalog.add(idx_int)
        cat_rgb = catalog_stars[idx][3]
        matched.append(((r, g, b), cat_rgb))

    stats["matched_count"] = len(matched)
    return matched, stats


def _fit_color_relation(img_rgbs, cat_rgbs, channel_idx):
    img_g = img_rgbs[:, 1] + 1e-9
    cat_g = cat_rgbs[:, 1] + 1e-9
    x = cat_rgbs[:, channel_idx] / cat_g
    y = img_rgbs[:, channel_idx] / img_g
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return None
    x = x[valid]
    y = y[valid]
    try:
        slope, intercept = np.polyfit(x, y, 1)
    except Exception:
        return None
    residual = y - (intercept + slope * x)
    return {
        "intercept": float(intercept),
        "slope": float(slope),
        "sigma": float(np.std(residual)),
    }


def _build_quality_warning(bg, fit_rg, fit_bg):
    sigmas = [
        item["sigma"]
        for item in (fit_rg, fit_bg)
        if item and np.isfinite(item.get("sigma", np.nan))
    ]
    max_sigma = max(sigmas) if sigmas else 0.0
    bg = np.asarray(bg, dtype=np.float64)
    bg_spread = float(np.std(bg) / max(np.mean(bg), 1e-9))
    if max_sigma > 0.12 or bg_spread > 0.65:
        return (
            "Fotometrik çözüm biraz belirsiz görünüyor; "
            "önce gradient düzeltmeyi deneyin"
        )
    return ""


def _apply_catalog_calibration(img, matched):
    """
    Eşleştirilmiş yıldız renklerinden per-kanal kazanç hesapla.
    """
    # ── Arka plan notralizasyonu ──
    gray = img.mean(axis=2)
    bg_percentile = np.percentile(gray, 25)
    bg_mask = gray < bg_percentile
    if not np.any(bg_mask):
        bg_mask = gray <= bg_percentile
    bg = np.median(img[bg_mask], axis=0).astype(np.float64) + 1e-9

    # ── Kazanç hesapla ──
    img_rgbs = np.array([m[0] for m in matched])  # (N, 3)
    cat_rgbs = np.array([m[1] for m in matched])  # (N, 3)

    # Outlier temizleme (kappa-sigma)
    ratios = cat_rgbs / (img_rgbs + 1e-9)  # (N, 3)
    med_ratio = np.median(ratios, axis=0)
    std_ratio = np.std(ratios, axis=0) + 1e-9
    good = np.all(np.abs(ratios - med_ratio) < 2.0 * std_ratio, axis=1)
    excluded_outliers = 0
    used_img_rgbs = img_rgbs
    used_cat_rgbs = cat_rgbs
    if good.sum() >= 5:
        excluded_outliers = int(len(ratios) - int(good.sum()))
        ratios = ratios[good]
        used_img_rgbs = img_rgbs[good]
        used_cat_rgbs = cat_rgbs[good]

    # Medyan oran = kazanç
    gains = np.median(ratios, axis=0)  # (3,)
    # G kanalına normalize et
    gains = gains / (gains[1] + 1e-9)
    gains = np.clip(gains, 0.3, 3.0)

    # ── Uygula ──
    result = img.astype(np.float64)

    # Arka plan çıkar
    result -= bg[None, None, :] * 0.5
    result = np.clip(result, 0, 1)

    # Kazanç uygula
    result *= gains[None, None, :]

    # Parlaklık koru
    orig_mean = float(img.mean())
    new_mean = float(result.mean())
    if new_mean > 1e-9:
        result *= orig_mean / new_mean

    fit_rg = _fit_color_relation(used_img_rgbs, used_cat_rgbs, 0)
    fit_bg = _fit_color_relation(used_img_rgbs, used_cat_rgbs, 2)
    stats = {
        "background": bg.astype(np.float64),
        "gains": gains.astype(np.float64),
        "input_star_count": int(len(matched)),
        "used_star_count": int(len(used_img_rgbs)),
        "excluded_outliers": int(excluded_outliers),
        "fit_rg": fit_rg,
        "fit_bg": fit_bg,
        "quality_warning": _build_quality_warning(bg, fit_rg, fit_bg),
    }

    return np.clip(result, 0, 1).astype(np.float32), stats


# ── PCC (yildiz tabanlı, hizli) ───────────────────────────────────────────────
def _pcc_fast(img):
    """G2V referansli yildiz tabanlı kalibrasyon."""
    return _spcc(img, G2V_RGB)


# ── AI Neutral ────────────────────────────────────────────────────────────────
def _ai_neutral(img, neutral_percentile=50.0):
    gray = img.mean(axis=2)
    med  = float(np.median(gray))
    bs   = 32
    mask = np.zeros(gray.shape, dtype=bool)
    h,w  = gray.shape
    for y in range(0,h,bs):
        for x in range(0,w,bs):
            p = gray[y:y+bs, x:x+bs]
            if p.size>0 and p.mean()<med and p.std()<med*0.5:
                mask[y:y+bs, x:x+bs] = True
    if mask.sum()<100:
        mask = gray < np.percentile(gray,30)
    r_bg = np.percentile(img[:,:,0][mask],neutral_percentile)+1e-9
    g_bg = np.percentile(img[:,:,1][mask],neutral_percentile)+1e-9
    b_bg = np.percentile(img[:,:,2][mask],neutral_percentile)+1e-9
    ref  = (r_bg+g_bg+b_bg)/3
    return np.clip(np.stack([
        img[:,:,0]*(ref/r_bg), img[:,:,1]*(ref/g_bg), img[:,:,2]*(ref/b_bg)
    ],2),0,1).astype(np.float32)
