# Galaxy Enhance — CLAHE + Vibrance + Sharpen + Gamma
# ====================================================
# Bu script hem bagimsiz hem AstroMastroPro icinde calisir.
#
# AstroMastroPro'da kullanim:
#   Script Editoru → listeden "Galaxy Enhance (Grok)" sec → Calistir → Uygula
#
# Bagimsiz kullanim:
#   python grok.py resim.jpg
#   python grok.py resim.fits

import cv2
import numpy as np
import os
import sys

# ── Parametreler ──────────────────────────────────────────────
CLAHE_CLIP    = 3.0     # CLAHE clip limit (1-10)
CLAHE_GRID    = 8       # CLAHE tile boyutu
SAT_BOOST     = 1.35    # Saturasyon carpani
BRIGHT_BOOST  = 1.08    # Parlaklik carpani
GAMMA         = 0.95    # Gamma (<1 = acik, >1 = koyu)
SHARPEN_POWER = 1.0     # Keskinlik gucu


def enhance_galaxy(img_float):
    """
    Galaksi resmini gelistir.

    Parameters:
        img_float: numpy float32 [0,1], shape (H,W,3) veya (H,W)
    Returns:
        numpy float32 [0,1] — islenenmis resim
    """
    img8 = (np.clip(img_float, 0, 1) * 255).astype(np.uint8)

    # 1. CLAHE — Lokal kontrast
    if img8.ndim == 3:
        lab = cv2.cvtColor(img8, cv2.COLOR_RGB2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP,
                                tileGridSize=(CLAHE_GRID, CLAHE_GRID))
        l_ch = clahe.apply(l_ch)
        lab = cv2.merge((l_ch, a_ch, b_ch))
        img8 = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    else:
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP,
                                tileGridSize=(CLAHE_GRID, CLAHE_GRID))
        img8 = clahe.apply(img8)

    # 2. Renk Canlilik
    if img8.ndim == 3:
        hsv = cv2.cvtColor(img8, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        s = np.clip(s.astype(np.float32) * SAT_BOOST, 0, 255).astype(np.uint8)
        v = np.clip(v.astype(np.float32) * BRIGHT_BOOST, 0, 255).astype(np.uint8)
        hsv = cv2.merge([h, s, v])
        img8 = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # 3. Keskinlestirme (Unsharp Mask)
    if SHARPEN_POWER > 0:
        blur = cv2.GaussianBlur(img8, (0, 0), 3)
        img8 = cv2.addWeighted(img8, 1.0 + SHARPEN_POWER,
                               blur, -SHARPEN_POWER, 0)
        img8 = np.clip(img8, 0, 255).astype(np.uint8)

    # 4. Gamma
    if abs(GAMMA - 1.0) > 0.01:
        inv_gamma = 1.0 / GAMMA
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in range(256)]).astype(np.uint8)
        img8 = cv2.LUT(img8, table)

    return np.clip(img8.astype(np.float32) / 255.0, 0, 1)


# ── AstroMastroPro Script API uyumlulugu ──────────────────────
# Eger 'image' degiskeni tanimli ise, AstroMastroPro icinde calisiyoruz
try:
    _img = image  # noqa: F821 — AstroMastroPro tarafindan saglanir
    result = enhance_galaxy(_img)
    print(f"Galaxy Enhance tamamlandi! {result.shape}")
except NameError:
    pass  # Bagimsiz calistiriliyor, asagidaki __main__ blogu kullanilacak


# ── Bagimsiz calistirma ──────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanim: python grok.py <resim_yolu>")
        print("Ornek:    python grok.py m83Aligned.jpg")
        sys.exit(1)

    input_path = sys.argv[1]
    if not os.path.isfile(input_path):
        print(f"HATA: Dosya bulunamadi: {input_path}")
        sys.exit(1)

    # Resmi oku (BGR → RGB → float32)
    raw = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        print(f"HATA: Resim okunamadi: {input_path}")
        sys.exit(1)

    # BGR → RGB
    if raw.ndim == 3:
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

    # float32 [0,1] normalize
    if raw.dtype == np.uint8:
        img_f = raw.astype(np.float32) / 255.0
    elif raw.dtype == np.uint16:
        img_f = raw.astype(np.float32) / 65535.0
    else:
        img_f = raw.astype(np.float32)
        mx = img_f.max()
        if mx > 1.0:
            img_f /= mx

    # Islemi uygula
    enhanced = enhance_galaxy(img_f)

    # Kaydet (RGB → BGR → uint8/uint16)
    name, ext = os.path.splitext(input_path)
    output_path = f"{name}_enhanced{ext}"

    out8 = (np.clip(enhanced, 0, 1) * 255).astype(np.uint8)
    if out8.ndim == 3:
        out8 = cv2.cvtColor(out8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, out8)
    print(f"Kaydedildi: {output_path}")
    print(f"Boyut: {enhanced.shape}, min={enhanced.min():.3f}, max={enhanced.max():.3f}")
