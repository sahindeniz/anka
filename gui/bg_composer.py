"""
Astro Maestro Pro - Background Composer
Varsayilan olarak gui/backgrounds icindeki uygun galaksi gorselini kullanir.
"""
import glob
import os
import numpy as np

_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")
_CACHE_FILE = os.path.join(_CACHE_DIR, "bg_comp_v5.npy")
_BG_DIR = os.path.join(os.path.dirname(__file__), "backgrounds")


def _resize_fill(img, tw, th):
    """Aspect-ratio koruyarak resize + center crop."""
    import cv2

    h, w = img.shape[:2]
    scale = max(tw / w, th / h)
    nw, nh = int(w * scale + 0.5), int(h * scale + 0.5)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    y0 = (nh - th) // 2
    x0 = (nw - tw) // 2
    return resized[y0:y0 + th, x0:x0 + tw]


def _list_background_paths():
    exts = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(_BG_DIR, ext)))
    return sorted(set(paths))


def find_default_background_path():
    """backgrounds klasorunden varsayilan resmi sec."""
    paths = _list_background_paths()
    if not paths:
        return ""

    priorities = (
        ("spiral", "galaxy"),
        ("galaxy",),
        ("nebula",),
    )
    lowered = [(p, os.path.basename(p).lower()) for p in paths]
    for keys in priorities:
        for path, name in lowered:
            if all(k in name for k in keys):
                return path
    return paths[0]


def generate_spiral_galaxy_background(width=1920, height=1080):
    """Sentetik spiral galaksi arka plani."""
    w, h = width, height
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xx = (xx - w * 0.52) / (w * 0.44)
    yy = (yy - h * 0.50) / (h * 0.44)

    xr = xx * 0.92 + yy * 0.18
    yr = -xx * 0.12 + yy * 1.05
    radius = np.sqrt(xr * xr + yr * yr) + 1e-6
    theta = np.arctan2(yr, xr)

    core = np.exp(-(radius / 0.16) ** 2)
    halo = np.exp(-(radius / 0.62) ** 2)
    spiral_phase = theta * 2.0 - radius * 11.5
    arms = np.clip(np.cos(spiral_phase), 0.0, 1.0) ** 2.4
    arm_falloff = np.exp(-(radius / 0.80) ** 2)
    galaxy = np.clip(core * 1.6 + arms * arm_falloff * 0.95 + halo * 0.22, 0.0, 1.0)

    dust = np.clip(np.sin(theta * 5.0 + radius * 18.0) * 0.5 + 0.5, 0.0, 1.0)
    dust = 1.0 - dust * np.exp(-(radius / 0.55) ** 2) * 0.22
    galaxy *= dust

    bg = np.zeros((h, w, 3), dtype=np.float32)
    bg[:, :, 0] = 0.015 + galaxy * 0.20 + halo * 0.03
    bg[:, :, 1] = 0.020 + galaxy * 0.28 + halo * 0.04
    bg[:, :, 2] = 0.045 + galaxy * 0.46 + halo * 0.07

    warm_core = np.clip(core * 1.2, 0.0, 1.0)
    bg[:, :, 0] += warm_core * 0.42
    bg[:, :, 1] += warm_core * 0.30
    bg[:, :, 2] += warm_core * 0.12

    rng = np.random.default_rng(42)
    stars = rng.random((420, 3))
    for sx, sy, sb in stars:
        px = int(sx * (w - 1))
        py = int(sy * (h - 1))
        rad = 1 if sb < 0.7 else 2
        val = 0.28 + sb * 0.55
        x0, x1 = max(0, px - rad), min(w, px + rad + 1)
        y0, y1 = max(0, py - rad), min(h, py + rad + 1)
        patch_y, patch_x = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        dist = np.sqrt((patch_x - px) ** 2 + (patch_y - py) ** 2)
        glow = np.clip(1.0 - dist / (rad + 0.8), 0.0, 1.0) * val
        bg[y0:y1, x0:x1, 0] += glow * 0.85
        bg[y0:y1, x0:x1, 1] += glow * 0.90
        bg[y0:y1, x0:x1, 2] += glow

    vignette = np.clip(1.0 - (radius ** 1.35) * 0.52, 0.12, 1.0)
    bg *= vignette[:, :, np.newaxis]
    return np.clip(bg, 0, 1).astype(np.float32)


def generate_composite_background(width=1920, height=1080, use_cache=True):
    """Varsayilan arka planini backgrounds klasorunden uret."""
    if use_cache and os.path.exists(_CACHE_FILE):
        try:
            cached = np.load(_CACHE_FILE)
            if cached.shape == (height, width, 3):
                return cached
        except Exception:
            pass

    default_path = find_default_background_path()
    if default_path:
        try:
            import cv2
            img = cv2.imread(default_path, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                canvas = _resize_fill(img, width, height) * 0.55
                yy, xx = np.ogrid[0:height, 0:width]
                vdist = np.sqrt(((xx - width / 2) / (width * 0.60)) ** 2 + ((yy - height / 2) / (height * 0.58)) ** 2)
                vignette = np.clip(1.0 - 0.30 * vdist ** 1.5, 0, 1).astype(np.float32)
                canvas *= vignette[:, :, np.newaxis]
                canvas = np.clip(canvas, 0, 1).astype(np.float32)
            else:
                canvas = generate_spiral_galaxy_background(width, height)
        except Exception:
            canvas = generate_spiral_galaxy_background(width, height)
    else:
        canvas = generate_spiral_galaxy_background(width, height)

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

        pad = 18
        ov = out.copy()
        cv2.rectangle(ov, (tx - pad, ty - th - pad), (tx + tw + pad, ty + pad + 30), (0.01, 0.01, 0.03), -1)
        out = cv2.addWeighted(ov, 0.5, out, 0.5, 0)

        cv2.putText(out, title, (tx + 2, ty + 2), font, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.putText(out, title, (tx, ty), font, scale, (0.62, 0.78, 0.98), thick, cv2.LINE_AA)

        sub = "Drag & Drop or Open File to Start"
        ss = scale * 0.36
        st = max(1, int(ss * 2))
        (sw2, sh2), _ = cv2.getTextSize(sub, font, ss, st)
        sx = (w - sw2) // 2
        sy = ty + int(th * 2.2)
        cv2.putText(out, sub, (sx + 1, sy + 1), font, ss, (0, 0, 0), st + 1, cv2.LINE_AA)
        cv2.putText(out, sub, (sx, sy), font, ss, (0.42, 0.50, 0.62), st, cv2.LINE_AA)

        return np.clip(out, 0, 1).astype(np.float32)
    except ImportError:
        return bg


def clear_cache():
    try:
        if os.path.exists(_CACHE_FILE):
            os.remove(_CACHE_FILE)
    except Exception:
        pass
