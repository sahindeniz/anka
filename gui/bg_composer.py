"""
Astro Maestro Pro — Background Composer
gui/backgrounds/ klasorundeki resimleri yukleyip
arka plan kompozisyonu olusturur.
Resim yuklendiginde kalkar.
"""
import numpy as np, os, glob

_BG_DIR = os.path.join(os.path.dirname(__file__), "backgrounds")
_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")
_CACHE_FILE = os.path.join(_CACHE_DIR, "bg_comp_v4.npy")


def _resize_fill(img, tw, th):
    """Aspect-ratio koruyarak resize + center crop."""
    import cv2
    h, w = img.shape[:2]
    scale = max(tw / w, th / h)
    nw, nh = int(w * scale + 0.5), int(h * scale + 0.5)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    y0 = (nh - th) // 2
    x0 = (nw - tw) // 2
    return resized[y0:y0+th, x0:x0+tw]


def generate_composite_background(width=1920, height=1080, use_cache=True):
    """backgrounds/ klasorundeki resimleri fade-blend ile birlestir."""
    if use_cache and os.path.exists(_CACHE_FILE):
        try:
            cached = np.load(_CACHE_FILE)
            if cached.shape == (height, width, 3):
                return cached
        except Exception:
            pass

    # Resimleri yukle
    import cv2
    exts = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(_BG_DIR, ext)))
    paths = sorted(set(paths))

    W, H = width, height

    if not paths:
        # Bos — koyu arka plan
        canvas = np.zeros((H, W, 3), dtype=np.float32)
        canvas[:] = 0.02
        return canvas

    images = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            images.append(img)

    if not images:
        canvas = np.zeros((H, W, 3), dtype=np.float32)
        canvas[:] = 0.02
        return canvas

    n = len(images)
    dim = 0.55  # hafif koyulastir — arka plan

    if n == 1:
        canvas = _resize_fill(images[0], W, H) * dim

    elif n == 2:
        # 2 resim: sol ve sag, ortada yumusak gecis
        imgs = [_resize_fill(img, W, H) for img in images]
        x_norm = np.linspace(0, 1, W, dtype=np.float32)
        # Sigmoid gecis — ortada yumusak blend
        blend = 1.0 / (1.0 + np.exp(-12.0 * (x_norm - 0.5)))
        w_left = 1.0 - blend
        w_right = blend

        canvas = np.zeros((H, W, 3), dtype=np.float32)
        for ch in range(3):
            canvas[:, :, ch] = (
                imgs[0][:, :, ch] * dim * w_left[np.newaxis, :] +
                imgs[1][:, :, ch] * dim * w_right[np.newaxis, :]
            )

    else:
        # 3+ resim: esit paylasim
        imgs = [_resize_fill(img, W, H) for img in images[:4]]
        m = len(imgs)
        canvas = np.zeros((H, W, 3), dtype=np.float32)
        x_norm = np.linspace(0, 1, W, dtype=np.float32)
        centers = np.linspace(0, 1, m)
        sigma = 1.0 / m
        weights = np.zeros((m, W), dtype=np.float32)
        for i in range(m):
            weights[i] = np.exp(-((x_norm - centers[i]) / sigma) ** 2)
        total = weights.sum(axis=0, keepdims=True) + 1e-8
        weights /= total
        for i in range(m):
            for ch in range(3):
                canvas[:, :, ch] += imgs[i][:, :, ch] * dim * weights[i][np.newaxis, :]

    # Vignette
    vy, vx = np.ogrid[0:H, 0:W]
    vdist = np.sqrt(((vx - W/2) / (W*0.60))**2 + ((vy - H/2) / (H*0.58))**2)
    vignette = np.clip(1.0 - 0.30 * vdist**1.5, 0, 1).astype(np.float32)
    canvas *= vignette[:, :, np.newaxis]

    canvas = np.clip(canvas, 0, 1).astype(np.float32)

    # Cache
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        np.save(_CACHE_FILE, canvas)
    except Exception:
        pass

    return canvas


def generate_welcome_overlay(bg: np.ndarray) -> np.ndarray:
    """Baslik yazisi overlay."""
    try:
        import cv2
        out = bg.copy()
        h, w = out.shape[:2]

        title = "Astro Maestro Pro"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = w / 850.0
        thick = max(2, int(scale * 2.5))
        (tw, th), _ = cv2.getTextSize(title, font, scale, thick)
        tx = (w - tw) // 2
        ty = int(h * 0.91)

        # Kutu
        pad = 18
        ov = out.copy()
        cv2.rectangle(ov, (tx-pad, ty-th-pad), (tx+tw+pad, ty+pad+30),
                       (0.01, 0.01, 0.03), -1)
        out = cv2.addWeighted(ov, 0.5, out, 0.5, 0)

        cv2.putText(out, title, (tx+2, ty+2), font, scale,
                    (0, 0, 0), thick+2, cv2.LINE_AA)
        cv2.putText(out, title, (tx, ty), font, scale,
                    (0.48, 0.70, 0.95), thick, cv2.LINE_AA)

        sub = "Drag & Drop or Open File to Start"
        ss = scale * 0.36
        st = max(1, int(ss * 2))
        (sw2, sh2), _ = cv2.getTextSize(sub, font, ss, st)
        sx = (w - sw2) // 2
        sy = ty + int(th * 2.2)
        cv2.putText(out, sub, (sx+1, sy+1), font, ss,
                    (0, 0, 0), st+1, cv2.LINE_AA)
        cv2.putText(out, sub, (sx, sy), font, ss,
                    (0.32, 0.42, 0.52), st, cv2.LINE_AA)

        return np.clip(out, 0, 1).astype(np.float32)
    except ImportError:
        return bg


def clear_cache():
    try:
        if os.path.exists(_CACHE_FILE):
            os.remove(_CACHE_FILE)
    except Exception:
        pass
