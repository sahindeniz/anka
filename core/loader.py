"""
Universal image loader — FITS, PNG, TIFF, JPEG, BMP, CR2/NEF raw, XISF, etc.
Returns a normalised float32 numpy array, always 2-D (grayscale) or 3-D (H,W,3) RGB.
"""
import os, numpy as np


def _fix_hot_pixels_bayer(raw, sigma=5.0):
    """Raw Bayer veride hot/dead pixel temizleme — debayer'dan ÖNCE çağrılır.
    Her piksel kendi Bayer kanalındaki komşularıyla (2px uzak) karşılaştırılır."""
    import cv2
    h, w = raw.shape
    out = raw.copy()
    # Her 2x2 Bayer pozisyonu için ayrı ayrı işle
    for dy in range(2):
        for dx in range(2):
            ch = raw[dy::2, dx::2].astype(np.float32)
            # Aynı kanal komşuları ile median (5x5 → gerçek 3x3 komşu)
            med = cv2.medianBlur(ch, 5)
            diff = np.abs(ch - med)
            std = max(float(np.std(ch)), 1e-6)
            # Medyandan sigma*std kadar sapan pikseller → hot/dead
            bad = diff > (sigma * std)
            ch_fixed = np.where(bad, med, ch)
            out[dy::2, dx::2] = ch_fixed.astype(raw.dtype)
    return out

# All RAW extensions supported by rawpy / libraw
_RAW_EXTS = {
    ".cr2", ".cr3", ".nef", ".arw", ".dng", ".orf", ".rw2", ".raf",
    ".pef", ".srw", ".x3f", ".mrw", ".3fr", ".mef", ".mos", ".nrw",
    ".rwl", ".sr2", ".erf", ".kdc", ".dcr", ".raw", ".iiq", ".rwz",
}

# Standard file filter string for QFileDialog (use everywhere)
FILE_FILTER = (
    "All Supported ("
    "*.fits *.fit *.fts *.xisf "
    "*.png *.tiff *.tif *.jpg *.jpeg *.bmp *.gif *.webp *.pbm *.pgm *.ppm *.pnm *.hdr *.exr *.sr "
    "*.cr2 *.cr3 *.nef *.arw *.dng *.orf *.rw2 *.raf *.pef *.srw *.x3f *.mrw *.3fr *.mef "
    "*.mos *.nrw *.rwl *.sr2 *.erf *.kdc *.dcr *.raw *.iiq *.rwz"
    ");;"
    "FITS (*.fits *.fit *.fts);;"
    "XISF (*.xisf);;"
    "PNG (*.png);;"
    "TIFF (*.tiff *.tif);;"
    "JPEG (*.jpg *.jpeg);;"
    "BMP (*.bmp);;"
    "WebP (*.webp);;"
    "GIF (*.gif);;"
    "HDR / EXR (*.hdr *.exr);;"
    "RAW (*.cr2 *.cr3 *.nef *.arw *.dng *.orf *.rw2 *.raf *.pef *.srw *.x3f *.raw *.iiq);;"
    "All Files (*)"
)


def load_image(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    orig_dtype = None   # orijinal veri tipi (normalize icin)

    # ── FITS ──────────────────────────────────────────────────────────────
    if ext in (".fits", ".fit", ".fts"):
        from astropy.io import fits
        import cv2
        with fits.open(path, ignore_missing_simple=True) as hdul:
            data = None
            header = None
            for hdu in hdul:
                if hdu.data is not None and hdu.data.ndim >= 2:
                    data = hdu.data
                    header = hdu.header
                    break
        if data is None:
            raise ValueError("FITS: goruntu verisi bulunamadi.")
        orig_dtype = data.dtype
        img = np.array(data, dtype=np.float32)
        if img.ndim == 3:
            if img.shape[0] in (1,3,4):
                img = np.moveaxis(img, 0, -1)
            if img.shape[2] == 1:
                img = img[:,:,0]

        # Bayer CFA tespiti ve debayer
        if img.ndim == 2 and header is not None:
            bayer_pat = None
            # FITS header'dan Bayer pattern bul
            for key in ("BAYERPAT", "COLORTYP", "CFA-PAT", "CFA_PAT",
                         "MOSAIC", "DEBESSION"):
                val = str(header.get(key, "")).strip().upper()
                if val in ("RGGB", "BGGR", "GRBG", "GBRG"):
                    bayer_pat = val
                    break
            # Header'da yoksa otomatik tespit — SADECE ham integer veri için
            # float32 = zaten işlenmiş veri, debayer uygulanmaz
            if (bayer_pat is None
                    and orig_dtype in (np.uint8, np.uint16, np.uint32)
                    and img.shape[0] % 2 == 0 and img.shape[1] % 2 == 0):
                # 2x2 blok varyansı kontrol et — Bayer desende yüksek olur
                h2, w2 = img.shape[0] // 2, img.shape[1] // 2
                block = img[:h2*2, :w2*2].reshape(h2, 2, w2, 2)
                ch_means = block.mean(axis=(0, 2))  # 2x2 kanal ortalamaları
                ch_range = ch_means.max() - ch_means.min()
                img_range = img.max() - img.min()
                if img_range > 1e-6 and ch_range / img_range > 0.05:
                    bayer_pat = "RGGB"  # en yaygın DSLR/CMOS deseni

            if bayer_pat is not None:
                # Normalize → uint16
                mx = img.max()
                if mx > 0:
                    img_u16 = (img / mx * 65535).astype(np.uint16)
                else:
                    img_u16 = img.astype(np.uint16)

                # 1) Hot/dead pixel temizle — debayer'dan ÖNCE (kritik!)
                img_u16 = _fix_hot_pixels_bayer(img_u16, sigma=5.0)

                # 2) Debayer — EA (Edge-Aware)
                bayer_map_ea = {
                    "RGGB": cv2.COLOR_BAYER_RG2RGB_EA,
                    "BGGR": cv2.COLOR_BAYER_BG2RGB_EA,
                    "GRBG": cv2.COLOR_BAYER_GR2RGB_EA,
                    "GBRG": cv2.COLOR_BAYER_GB2RGB_EA,
                }
                bayer_map_std = {
                    "RGGB": cv2.COLOR_BAYER_RG2RGB,
                    "BGGR": cv2.COLOR_BAYER_BG2RGB,
                    "GRBG": cv2.COLOR_BAYER_GR2RGB,
                    "GBRG": cv2.COLOR_BAYER_GB2RGB,
                }
                try:
                    code = bayer_map_ea.get(bayer_pat, cv2.COLOR_BAYER_RG2RGB_EA)
                    img = cv2.cvtColor(img_u16, code).astype(np.float32)
                except cv2.error:
                    code = bayer_map_std.get(bayer_pat, cv2.COLOR_BAYER_RG2RGB)
                    img = cv2.cvtColor(img_u16, code).astype(np.float32)
                if mx > 0:
                    img = img / 65535.0 * mx

    # ── XISF (PixInsight) ────────────────────────────────────────────────
    elif ext == ".xisf":
        try:
            from xisf import XISF
            xisf = XISF(path)
            raw_data = xisf.read_image(0)
            orig_dtype = raw_data.dtype
            img = raw_data.astype(np.float32)
            if img.ndim == 3 and img.shape[0] in (1, 3, 4):
                img = np.moveaxis(img, 0, -1)
            if img.ndim == 3 and img.shape[2] == 1:
                img = img[:, :, 0]
        except ImportError:
            raise ImportError("XISF format icin: pip install xisf")

    # ── RAW (rawpy) ───────────────────────────────────────────────────────
    elif ext in _RAW_EXTS:
        try:
            import rawpy
            with rawpy.imread(path) as raw:
                rgb = raw.postprocess(use_camera_wb=True, output_bps=16)
            orig_dtype = np.uint16   # rawpy 16-bit cikti
            img = rgb.astype(np.float32)
        except ImportError:
            raise ImportError("RAW format icin: pip install rawpy")

    # ── Standard bitmap (PNG/TIFF/JPEG/BMP/WebP/GIF/HDR/EXR …) ──────────
    else:
        import cv2
        if ext in (".hdr", ".exr"):
            data = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        else:
            data = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if data is None:
            raise ValueError(f"Goruntu okunamadi: {path}")
        orig_dtype = data.dtype
        if data.ndim == 2:
            img = data
        elif data.shape[2] == 4:
            img = cv2.cvtColor(data, cv2.COLOR_BGRA2RGB)
        else:
            img = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)

    # ── Normalise to [0,1] — veri tipine gore mutlak olcekle ─────────────
    # Mutlak olcekleme: dark frame 0.02 ise 0.02 olarak kalir.
    # Min-max KULLANILMAZ — kalibrasyon kareleri bozulur.
    if orig_dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    elif orig_dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    elif orig_dtype == np.uint32:
        img = img.astype(np.float32) / 4294967295.0
    elif orig_dtype in (np.float32, np.float64):
        img = img.astype(np.float32)
        # Float: FITS/XISF genelde [0,1] veya [0,65535] olabilir
        mx = img.max()
        if mx > 1.5:
            img /= mx
    else:
        # Bilinmeyen dtype — guvenli normalize
        img = img.astype(np.float32)
        mx = img.max()
        if mx > 1.0:
            img /= mx

    return np.clip(img, 0, None).astype(np.float32)


def save_image(path: str, img: np.ndarray):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".fits", ".fit", ".fts"):
        from astropy.io import fits
        img_f32 = np.clip(img, 0, 1).astype(np.float32)
        if img_f32.ndim == 3:
            fits_data = np.transpose(img_f32, (2, 0, 1))
        else:
            fits_data = img_f32
        hdr = fits.Header()
        hdr["SOFTWARE"] = "AstroMastroPro"
        hdr["BITPIX"] = -32
        hdu = fits.PrimaryHDU(data=fits_data, header=hdr)
        hdu.writeto(path, overwrite=True)
    elif ext in (".tif", ".tiff"):
        img16 = (np.clip(img, 0, 1) * 65535).astype(np.uint16)
        try:
            import tifffile
            tifffile.imwrite(path, img16,
                             photometric="rgb" if img.ndim == 3 else "minisblack")
        except ImportError:
            import cv2
            if img.ndim == 3:
                img16 = cv2.cvtColor(img16, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, img16)
    else:
        import cv2
        out = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        if out.ndim == 3:
            out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, out)
