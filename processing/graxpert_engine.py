"""
Astro Maestro Pro — GraXpert-Style Gradient Extraction Engine
=============================================================
GraXpert (github.com/Steffenhir/GraXpert) algoritmalarinin Python uyarlamasi.

Desteklenen interpolasyon yontemleri:
  1. RBF (Radial Basis Function) — gaussian, multiquadric, thin_plate, cubic, linear
  2. Splines — scipy bisplrep/bisplev 2D bikubik spline
  3. Kriging — Ordinary Kriging (sferik variogram)
  4. Polynomial — 2D polinom (derece 1-6)
  5. AI Grid — Otomatik grid + sigma-clipped medyan (AI model olmadan)

Duzeltme tipi:
  - Subtraction:  result = image - background + mean
  - Division:     result = image / background * mean

Grid Selection:
  - Otomatik grid noktasi secimi (en karanlik ceyregi bul)
  - Sigma-clipped medyan ile outlier filtreleme
  - Tolerans tabanli kotu nokta reddi

Referans: GraXpert background_extraction.py, background_grid_selection.py, radialbasisinterpolation.py
"""

import numpy as np
import cv2
from typing import Optional, Tuple, List


# ═══════════════════════════════════════════════════════════════════════════════
#  GRID POINT SELECTION (GraXpert background_grid_selection.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _sigma_clipped_median(patch: np.ndarray, sigma: float = 2.5) -> float:
    """Sigma-clipped medyan — parlak yildizlari disla."""
    med = np.median(patch)
    mad = np.median(np.abs(patch - med)) * 1.4826
    if mad < 1e-9:
        return float(med)
    mask = patch < (med + sigma * mad)
    clipped = patch[mask]
    return float(np.median(clipped)) if clipped.size > 0 else float(med)


def _find_darkest_quadrant(image: np.ndarray, cy: int, cx: int,
                           half_size: int = 25) -> Tuple[int, int]:
    """
    GraXpert grid_utils.find_darkest_quadrant:
    5 aday konum (merkez + 4 kose) icinden en karanlik olan ceyregi sec.
    """
    h, w = image.shape[:2]
    hs = half_size
    offsets = [(0, 0), (-hs//2, -hs//2), (-hs//2, hs//2),
               (hs//2, -hs//2), (hs//2, hs//2)]

    best_val = 1e9
    best_pos = (cy, cx)

    for dy, dx in offsets:
        ny, nx = cy + dy, cx + dx
        y0 = max(0, ny - hs)
        y1 = min(h, ny + hs)
        x0 = max(0, nx - hs)
        x1 = min(w, nx + hs)
        if y1 - y0 < 4 or x1 - x0 < 4:
            continue
        patch = image[y0:y1, x0:x1]
        if image.ndim == 3:
            patch = patch.mean(axis=2)
        val = _sigma_clipped_median(patch.ravel())
        if val < best_val:
            best_val = val
            best_pos = (ny, nx)

    return best_pos


def select_grid_points(image: np.ndarray,
                       num_pts_per_row: int = 8,
                       sample_size: int = 25,
                       tolerance: float = 1.0) -> np.ndarray:
    """
    GraXpert background_grid_selection.background_grid_selection:
    Otomatik arka plan ornekleme noktasi secimi.

    1. Uniform grid olustur
    2. Her noktayi en karanlik ceyrege kaydir
    3. Tolerans filtresi: global_median + tolerance * MAD ustunu reddet

    Parameters
    ----------
    image : ndarray float32 [0,1], (H,W) or (H,W,3)
    num_pts_per_row : int — grid satir basina nokta sayisi
    sample_size : int — ornekleme pencere yarisi (piksel)
    tolerance : float — outlier reddi: daha yuksek = daha gevşek

    Returns
    -------
    points : ndarray (N, 2) — [y, x] koordinatlari
    """
    h, w = image.shape[:2]

    # Luminance
    if image.ndim == 3:
        lum = 0.2126 * image[:, :, 0] + 0.7152 * image[:, :, 1] + 0.0722 * image[:, :, 2]
    else:
        lum = image.copy()

    dist = max(w, h) / max(num_pts_per_row, 2)
    margin = int(dist * 0.5)

    # Uniform grid
    ys = np.arange(margin, h - margin + 1, dist).astype(int)
    xs = np.arange(margin, w - margin + 1, dist).astype(int)

    # Her noktayi en karanlik ceyrege kaydir
    raw_points = []
    medians = []
    for cy in ys:
        for cx in xs:
            by, bx = _find_darkest_quadrant(lum, cy, cx, sample_size)
            # Sinir kontrol
            by = max(sample_size, min(h - sample_size - 1, by))
            bx = max(sample_size, min(w - sample_size - 1, bx))
            # Bu noktadaki medyan
            patch = lum[by - sample_size:by + sample_size,
                        bx - sample_size:bx + sample_size]
            med = _sigma_clipped_median(patch.ravel())
            raw_points.append((by, bx))
            medians.append(med)

    if not raw_points:
        return np.empty((0, 2), dtype=np.float64)

    medians = np.array(medians, dtype=np.float64)
    global_med = np.median(medians)
    mad = np.median(np.abs(medians - global_med)) * 1.4826

    # Tolerans filtresi
    threshold = global_med + tolerance * max(mad, 1e-6)
    points = []
    for (py, px), m in zip(raw_points, medians):
        if m <= threshold:
            points.append([py, px])

    return np.array(points, dtype=np.float64) if points else np.empty((0, 2), dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════════════
#  SAMPLE POINT STATISTICS (GraXpert calc_mode_dataset)
# ═══════════════════════════════════════════════════════════════════════════════

def _calc_sample_values(image: np.ndarray, points: np.ndarray,
                        sample_size: int = 25) -> np.ndarray:
    """
    Her ornekleme noktasi icin sigma-clipped medyan hesapla.
    Her kanal icin ayri.

    Returns: (N, C) — C=1 (mono) veya C=3 (RGB)
    """
    h, w = image.shape[:2]
    n_channels = image.shape[2] if image.ndim == 3 else 1

    values = np.zeros((len(points), n_channels), dtype=np.float64)

    for i, (py, px) in enumerate(points):
        py, px = int(py), int(px)
        y0 = max(0, py - sample_size)
        y1 = min(h, py + sample_size)
        x0 = max(0, px - sample_size)
        x1 = min(w, px + sample_size)

        if n_channels == 1:
            patch = image[y0:y1, x0:x1].ravel()
            values[i, 0] = _sigma_clipped_median(patch)
        else:
            for c in range(n_channels):
                patch = image[y0:y1, x0:x1, c].ravel()
                values[i, c] = _sigma_clipped_median(patch)

    return values


# ═══════════════════════════════════════════════════════════════════════════════
#  INTERPOLATION METHODS
# ═══════════════════════════════════════════════════════════════════════════════

# ── RBF (Radial Basis Function) ──────────────────────────────────────────────

def _rbf_kernel(r: np.ndarray, kernel: str = "thin_plate") -> np.ndarray:
    """RBF kernel fonksiyonlari."""
    r = np.maximum(r, 1e-12)
    if kernel == "thin_plate":
        return r ** 2 * np.log(r)
    elif kernel == "gaussian":
        return np.exp(-(r ** 2))
    elif kernel == "multiquadric":
        return np.sqrt(1 + r ** 2)
    elif kernel == "inverse":
        return 1.0 / np.sqrt(1 + r ** 2)
    elif kernel == "cubic":
        return r ** 3
    elif kernel == "linear":
        return r
    elif kernel == "quintic":
        return r ** 5
    else:  # thin_plate default
        return r ** 2 * np.log(r)


def _interpolate_rbf(points: np.ndarray, values: np.ndarray,
                     h: int, w: int,
                     kernel: str = "thin_plate",
                     smoothing: float = 0.0) -> np.ndarray:
    """
    GraXpert radialbasisinterpolation.RadialBasisInterpolation:
    Augmented RBF interpolasyonu — polinom augmentasyon ile.

    Parameters
    ----------
    points : (N, 2) — [y, x]
    values : (N,) — kanal degerleri
    h, w : hedef boyut
    kernel : RBF kernel tipi
    smoothing : 0-1 arasi yumusatma

    Returns
    -------
    bg : (h, w) float32 — tahmini arka plan
    """
    N = len(points)
    if N < 3:
        return np.full((h, w), np.median(values), dtype=np.float32)

    # Normalize coordinates to [0, 1]
    pts = points.copy()
    pts[:, 0] /= max(h, 1)
    pts[:, 1] /= max(w, 1)

    # Mesafe matrisi
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_matrix = np.sqrt((diff ** 2).sum(axis=2))

    # Kernel matrisi + smoothing
    K = _rbf_kernel(dist_matrix, kernel)
    if smoothing > 0:
        K += smoothing * np.eye(N)

    # Augmented system (polinom augmentasyon: [1, y, x])
    P = np.hstack([np.ones((N, 1)), pts])  # (N, 3)
    # [K  P] [w]   [v]
    # [P' 0] [c] = [0]
    top = np.hstack([K, P])
    bottom = np.hstack([P.T, np.zeros((3, 3))])
    A = np.vstack([top, bottom])
    rhs = np.concatenate([values, np.zeros(3)])

    try:
        coeffs = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        coeffs = np.linalg.lstsq(A, rhs, rcond=None)[0]

    w_rbf = coeffs[:N]
    c_poly = coeffs[N:]

    # Evaluate on grid (downscaled for speed)
    scale = max(1, max(h, w) // 512)
    gh, gw = h // scale, w // scale
    gy = np.linspace(0, 1, gh)
    gx = np.linspace(0, 1, gw)
    GY, GX = np.meshgrid(gy, gx, indexing='ij')
    grid_pts = np.stack([GY.ravel(), GX.ravel()], axis=1)  # (gh*gw, 2)

    # Distances from grid to sample points
    diffs = grid_pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (M, N, 2)
    dists = np.sqrt((diffs ** 2).sum(axis=2))  # (M, N)

    Kg = _rbf_kernel(dists, kernel)  # (M, N)
    bg_flat = Kg @ w_rbf + c_poly[0] + grid_pts[:, 0] * c_poly[1] + grid_pts[:, 1] * c_poly[2]
    bg_small = bg_flat.reshape(gh, gw).astype(np.float32)

    # Upscale to original size
    bg = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_CUBIC)
    return bg


# ── Spline Interpolation ────────────────────────────────────────────────────

def _interpolate_spline(points: np.ndarray, values: np.ndarray,
                        h: int, w: int,
                        smoothing: float = 0.0,
                        order: int = 3) -> np.ndarray:
    """
    GraXpert spline interpolasyonu — scipy bisplrep/bisplev.
    """
    try:
        from scipy.interpolate import bisplrep, bisplev
    except ImportError:
        # Fallback: polynomial
        return _interpolate_polynomial(points, values, h, w, degree=3)

    N = len(points)
    if N < (order + 1) ** 2:
        return np.full((h, w), np.median(values), dtype=np.float32)

    y_pts = points[:, 0]
    x_pts = points[:, 1]

    # Smoothing factor
    s = smoothing * N if smoothing > 0 else N * 0.5

    try:
        tck = bisplrep(x_pts, y_pts, values, kx=order, ky=order, s=s)
    except Exception:
        return _interpolate_polynomial(points, values, h, w, degree=min(3, order))

    # Evaluate on grid
    scale = max(1, max(h, w) // 512)
    gh, gw = h // scale, w // scale
    gy = np.linspace(0, h - 1, gh)
    gx = np.linspace(0, w - 1, gw)

    try:
        bg_small = bisplev(gx, gy, tck).T.astype(np.float32)
    except Exception:
        return _interpolate_polynomial(points, values, h, w, degree=3)

    bg = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_CUBIC)
    return bg


# ── Kriging (Ordinary Kriging) ───────────────────────────────────────────────

def _interpolate_kriging(points: np.ndarray, values: np.ndarray,
                         h: int, w: int,
                         smoothing: float = 0.0) -> np.ndarray:
    """
    GraXpert Kriging — Ordinary Kriging, sferik variogram.
    PyKrige mevcut degilse RBF'ye fallback.
    """
    try:
        from pykrige.ok import OrdinaryKriging
    except ImportError:
        # Fallback to RBF
        return _interpolate_rbf(points, values, h, w,
                                kernel="thin_plate", smoothing=smoothing)

    N = len(points)
    if N < 3:
        return np.full((h, w), np.median(values), dtype=np.float32)

    # Downscale evaluation grid
    scale = max(1, max(h, w) // 256)
    gh, gw = h // scale, w // scale

    try:
        ok = OrdinaryKriging(
            points[:, 1],  # x
            points[:, 0],  # y
            values,
            variogram_model="spherical",
            verbose=False,
            enable_plotting=False,
            nlags=min(20, N // 2),
        )
        gx = np.linspace(0, w - 1, gw)
        gy = np.linspace(0, h - 1, gh)
        z, ss = ok.execute("grid", gx, gy)
        bg_small = z.astype(np.float32)
    except Exception:
        return _interpolate_rbf(points, values, h, w,
                                kernel="thin_plate", smoothing=smoothing)

    bg = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_CUBIC)
    return bg


# ── Polynomial Interpolation ────────────────────────────────────────────────

def _interpolate_polynomial(points: np.ndarray, values: np.ndarray,
                            h: int, w: int,
                            degree: int = 4) -> np.ndarray:
    """
    2D polinom interpolasyonu — degree dereceli (1-6).
    Vandermonde matrisi ile least-squares fit.
    """
    N = len(points)
    n_terms = (degree + 1) * (degree + 2) // 2
    if N < n_terms:
        degree = max(1, int((-3 + (9 + 8 * (N - 1)) ** 0.5) / 2))
        n_terms = (degree + 1) * (degree + 2) // 2

    y_pts = points[:, 0] / max(h, 1)
    x_pts = points[:, 1] / max(w, 1)

    # Vandermonde matrisi
    V = []
    for p in range(degree + 1):
        for q in range(degree + 1 - p):
            V.append((y_pts ** p) * (x_pts ** q))
    V = np.array(V).T  # (N, n_terms)

    try:
        coeffs, _, _, _ = np.linalg.lstsq(V, values, rcond=None)
    except Exception:
        return np.full((h, w), np.median(values), dtype=np.float32)

    # Evaluate on grid
    scale = max(1, max(h, w) // 512)
    gh, gw = h // scale, w // scale
    gy = np.linspace(0, 1, gh)
    gx = np.linspace(0, 1, gw)
    GY, GX = np.meshgrid(gy, gx, indexing='ij')

    bg_small = np.zeros((gh, gw), dtype=np.float64)
    idx = 0
    for p in range(degree + 1):
        for q in range(degree + 1 - p):
            bg_small += coeffs[idx] * (GY ** p) * (GX ** q)
            idx += 1

    bg = cv2.resize(bg_small.astype(np.float32), (w, h),
                    interpolation=cv2.INTER_CUBIC)
    return bg


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN EXTRACTION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_background(image: np.ndarray,
                       method: str = "rbf",
                       correction: str = "subtraction",
                       smoothing: float = 0.5,
                       grid_pts_per_row: int = 8,
                       sample_size: int = 25,
                       tolerance: float = 1.0,
                       rbf_kernel: str = "thin_plate",
                       spline_order: int = 3,
                       poly_degree: int = 4,
                       points: Optional[np.ndarray] = None,
                       progress_cb=None,
                       **kw) -> dict:
    """
    GraXpert-style arka plan cikarma ana fonksiyonu.

    Parameters
    ----------
    image : ndarray float32 [0,1], (H,W) or (H,W,3)
    method : str — "rbf", "spline", "kriging", "polynomial", "ai_grid"
    correction : str — "subtraction" or "division"
    smoothing : float 0-1
    grid_pts_per_row : int — otomatik grid noktasi sayisi
    sample_size : int — ornekleme pencere boyutu
    tolerance : float — grid nokta filtresi
    rbf_kernel : str — RBF kernel tipi
    spline_order : int — spline derecesi
    poly_degree : int — polinom derecesi
    points : ndarray (N,2) — manuel nokta koordinatlari (None=otomatik)
    progress_cb : callable(step, total, msg) — ilerleme callback

    Returns
    -------
    dict:
      "result" : corrected image (H,W) or (H,W,3)
      "background" : estimated background
      "points" : selected grid points
    """
    img = np.ascontiguousarray(image, dtype=np.float32)
    np.clip(img, 0, 1, out=img)
    h, w = img.shape[:2]
    n_channels = img.shape[2] if img.ndim == 3 else 1

    def _prog(step, total, msg):
        if progress_cb:
            try:
                progress_cb(step, total, msg)
            except Exception:
                pass

    _prog(1, 6, "Grid noktalarini seciyor...")

    # ── 1. Grid Point Selection ─────────────────────────────────────────
    if points is None:
        points = select_grid_points(img, num_pts_per_row=grid_pts_per_row,
                                    sample_size=sample_size,
                                    tolerance=tolerance)

    if len(points) < 3:
        # Yetersiz nokta — fallback: uniform grid (filtresiz)
        ys = np.linspace(sample_size, h - sample_size, grid_pts_per_row).astype(int)
        xs = np.linspace(sample_size, w - sample_size, grid_pts_per_row).astype(int)
        points = np.array([[y, x] for y in ys for x in xs], dtype=np.float64)

    _prog(2, 6, f"{len(points)} nokta secildi. Ornekleme degerleri hesaplaniyor...")

    # ── 2. Sample Values ────────────────────────────────────────────────
    sample_values = _calc_sample_values(img, points, sample_size=sample_size)

    _prog(3, 6, f"Arka plan interpolasyonu ({method})...")

    # ── 3. Interpolation (per channel) ──────────────────────────────────
    bg = np.zeros_like(img)

    for c in range(n_channels):
        vals = sample_values[:, c]

        if method == "rbf":
            bg_ch = _interpolate_rbf(points, vals, h, w,
                                     kernel=rbf_kernel, smoothing=smoothing)
        elif method == "spline":
            bg_ch = _interpolate_spline(points, vals, h, w,
                                        smoothing=smoothing, order=spline_order)
        elif method == "kriging":
            bg_ch = _interpolate_kriging(points, vals, h, w,
                                         smoothing=smoothing)
        elif method == "polynomial":
            bg_ch = _interpolate_polynomial(points, vals, h, w,
                                            degree=poly_degree)
        else:  # ai_grid — simple median grid + smooth
            bg_ch = _interpolate_rbf(points, vals, h, w,
                                     kernel="thin_plate", smoothing=smoothing)

        if n_channels == 1:
            bg = bg_ch
        else:
            bg[:, :, c] = bg_ch

        _prog(3 + c, 6, f"Kanal {c+1}/{n_channels} tamamlandi")

    _prog(5, 6, "Duzeltme uygulanıyor...")

    # ── 4. Gaussian smoothing (smoothing > 0) ──────────────────────────
    if smoothing > 0:
        ksize = max(3, int(smoothing * 51)) | 1
        if bg.ndim == 2:
            bg = cv2.GaussianBlur(bg, (ksize, ksize), 0)
        else:
            for c in range(n_channels):
                bg[:, :, c] = cv2.GaussianBlur(bg[:, :, c], (ksize, ksize), 0)

    # ── 5. Correction ──────────────────────────────────────────────────
    if n_channels == 1:
        bg_mean = float(np.mean(bg))
    else:
        bg_mean = np.mean(bg, axis=(0, 1))  # per-channel mean

    if correction.lower() == "division":
        # Division: result = image / background * mean
        safe_bg = np.maximum(bg, 1e-6)
        result = img / safe_bg * bg_mean
    else:
        # Subtraction: result = image - background + mean
        result = img - bg + bg_mean

    # Pedestal: en dusuk degeri 0'a ayarla
    mn = result.min()
    if mn < 0:
        result -= mn

    # Normalize [0, 1]
    mx = result.max()
    if mx > 1.0:
        result /= mx

    np.clip(result, 0, 1, out=result)

    _prog(6, 6, "Tamamlandi!")

    return {
        "result": result.astype(np.float32),
        "background": np.clip(bg, 0, 1).astype(np.float32),
        "points": points,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  CONVENIENCE WRAPPER (for _run_key dispatch)
# ═══════════════════════════════════════════════════════════════════════════════

def graxpert_extract(image: np.ndarray,
                     method: str = "rbf",
                     correction: str = "subtraction",
                     smoothing: float = 0.5,
                     grid_pts_per_row: int = 8,
                     sample_size: int = 25,
                     tolerance: float = 1.0,
                     rbf_kernel: str = "thin_plate",
                     spline_order: int = 3,
                     poly_degree: int = 4,
                     keep_background: bool = False,
                     **kw) -> np.ndarray:
    """
    Tek-fonksiyon wrapper — ProcessPanel dispatch icin.
    Returns corrected image (or background if keep_background=True).
    """
    result = extract_background(
        image,
        method=method,
        correction=correction,
        smoothing=smoothing,
        grid_pts_per_row=grid_pts_per_row,
        sample_size=sample_size,
        tolerance=tolerance,
        rbf_kernel=rbf_kernel,
        spline_order=spline_order,
        poly_degree=poly_degree,
        progress_cb=kw.get("_progress_cb"),
    )

    if keep_background:
        return result["background"]
    return result["result"]
