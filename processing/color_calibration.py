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
                    **kwargs):
    if image.ndim != 3:
        return image
    img = image.astype(np.float32)

    if   method == "spcc_g2v":    return _spcc(img, G2V_RGB)
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

    # Arka plan notralizasyonu
    bg = np.array([
        float(np.median(img[:,:,c][bg_mask])) for c in range(3)
    ]) + 1e-9

    # ── 2. Yildiz tespiti (kucuk resimde hizli) ───────────────────────────────
    SCALE = 512
    scale = min(1.0, SCALE / max(h, w))
    sw, sh = max(4,int(w*scale)), max(4,int(h*scale))
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    gray_s = small.mean(axis=2)

    # Parlak yildizlari tespit et
    threshold = np.percentile(gray_s, 97)
    _, bw = cv2.threshold((gray_s*255).astype(np.uint8), int(threshold*255), 255, cv2.THRESH_BINARY)
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
                    star_colors.append([float(patch[:,:,c].mean()) for c in range(3)])

    if len(star_colors) < 5:
        # Yetersiz yildiz — sadece background neutralize et
        img_bg = img - bg[None,None,:] * 0.7
        img_bg = np.clip(img_bg, 0, 1)
        # Referans oranina gore scale
        ref_norm = reference_rgb / reference_rgb.max()
        means = np.array([float(img_bg[:,:,c].mean()) for c in range(3)]) + 1e-9
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

    # Kazanc uygula
    for c in range(3):
        result[:,:,c] *= gains[c]

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
    out = img.copy()
    for c in range(3):
        p = np.percentile(out[:,:,c],99)+1e-9
        out[:,:,c] /= p
    return np.clip(out,0,1).astype(np.float32)


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
