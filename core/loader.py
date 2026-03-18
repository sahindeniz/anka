"""
Universal image loader — FITS, PNG, TIFF, JPEG, BMP, CR2/NEF raw, etc.
Returns a normalised float32 numpy array, always 2-D (grayscale) or 3-D (H,W,3) RGB.
"""
import os, numpy as np

def load_image(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()

    # ── FITS ──────────────────────────────────────────────────────────────
    if ext in (".fits", ".fit", ".fts"):
        from astropy.io import fits
        with fits.open(path) as hdul:
            data = None
            for hdu in hdul:
                if hdu.data is not None and hdu.data.ndim >= 2:
                    data = hdu.data; break
        if data is None:
            raise ValueError("FITS dosyasında görüntü verisi bulunamadı.")
        img = np.array(data, dtype=np.float32)
        if img.ndim == 3:
            # (C,H,W) → (H,W,C)
            if img.shape[0] in (1,3,4):
                img = np.moveaxis(img, 0, -1)
            if img.shape[2] == 1:
                img = img[:,:,0]

    # ── RAW (rawpy) ───────────────────────────────────────────────────────
    elif ext in (".cr2",".cr3",".nef",".arw",".dng",".orf",".rw2",".raf"):
        try:
            import rawpy
            with rawpy.imread(path) as raw:
                rgb = raw.postprocess(use_camera_wb=True, output_bps=16)
            img = rgb.astype(np.float32)
        except ImportError:
            raise ImportError("RAW format için: pip install rawpy")

    # ── Standard bitmap (PNG/TIFF/JPEG/BMP …) ────────────────────────────
    else:
        import cv2
        data = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if data is None:
            raise ValueError(f"Görüntü okunamadı: {path}")
        if data.ndim == 2:
            img = data.astype(np.float32)
        elif data.shape[2] == 4:
            img = cv2.cvtColor(data, cv2.COLOR_BGRA2RGB).astype(np.float32)
        else:
            img = cv2.cvtColor(data, cv2.COLOR_BGR2RGB).astype(np.float32)

    # ── Normalise to [0,1] ────────────────────────────────────────────────
    mn, mx = img.min(), img.max()
    if mx > mn:
        img = (img - mn) / (mx - mn)
    else:
        img = np.zeros_like(img)
    return img.astype(np.float32)


def save_image(path: str, img: np.ndarray):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".fits", ".fit", ".fts"):
        from astropy.io import fits
        fits.writeto(path, img.astype(np.float32), overwrite=True)
    else:
        import cv2
        out = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        if out.ndim == 3:
            out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, out)
