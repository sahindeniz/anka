"""
Astro Maestro Pro - StarSmaller

Natural star reduction by:
1. masking the full star profile (core + halo),
2. inpainting the emptied area from the surrounding background,
3. restoring only a smaller soft core from the original star.
"""
import cv2
import numpy as np


def reduce_stars(
    image,
    strength=0.55,
    sensitivity=0.48,
    feather=4,
    max_sigma=7,
    min_sigma=1,
    threshold=0.025,
    protect_nebula=True,
    **kw,
):
    del protect_nebula, kw

    img = np.ascontiguousarray(image, dtype=np.float32)
    np.clip(img, 0, 1, out=img)
    h, w = img.shape[:2]
    is_color = img.ndim == 3

    gray = img.mean(axis=2) if is_color else img.copy()
    s = float(np.clip(strength, 0, 1))
    shrink = max(0.15, 1.0 - s * 0.80)

    # 1. Detect star cores.
    core_mask = _fast_star_mask(
        gray,
        float(sensitivity),
        float(threshold),
        int(max_sigma),
        int(min_sigma),
    )

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (core_mask > 0).astype(np.uint8),
        connectivity=8,
    )
    max_star_area = max(500, h * w * 0.001)

    # 2. Build an erase mask and collect star geometry for reconstruction.
    erase_mask = np.zeros((h, w), np.uint8)
    star_specs = []

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

        real_r = _measure_star_radius(gray, icx, icy, core_r, h, w)
        new_r = max(0.85, real_r * shrink)
        erase_r = int(np.ceil(real_r * (1.10 + 0.12 * s))) + 2
        cv2.circle(erase_mask, (icx, icy), erase_r, 255, -1)
        star_specs.append((icx, icy, float(real_r), float(new_r)))

    if erase_mask.sum() == 0:
        feat = max(3, int(feather) * 2 + 1) | 1
        mask_f = cv2.GaussianBlur(core_mask.astype(np.float32), (feat, feat), feat * 0.4)
        return img.copy(), mask_f

    # 3. Remove the full star profile by inpainting from the surroundings.
    img_u16 = (img * 65535).clip(0, 65535).astype(np.uint16)
    if is_color:
        starless_u16 = np.zeros_like(img_u16)
        for c in range(3):
            starless_u16[:, :, c] = cv2.inpaint(
                img_u16[:, :, c],
                erase_mask,
                5,
                cv2.INPAINT_TELEA,
            )
        starless = starless_u16.astype(np.float32) / 65535.0
    else:
        starless_u16 = cv2.inpaint(img_u16, erase_mask, 5, cv2.INPAINT_TELEA)
        starless = starless_u16.astype(np.float32) / 65535.0

    # 4. Rebuild only a smaller synthetic PSF from the extracted star layer.
    star_layer = np.clip(img - starless, 0.0, 1.0).astype(np.float32)
    restored = np.zeros_like(img, dtype=np.float32)

    for icx, icy, real_r, new_r in star_specs:
        pad = int(max(real_r * 3.0, new_r * 5.0)) + 4
        y0, y1 = max(0, icy - pad), min(h, icy + pad + 1)
        x0, x1 = max(0, icx - pad), min(w, icx + pad + 1)
        rh, rw = y1 - y0, x1 - x0
        if rh < 3 or rw < 3:
            continue

        yy, xx = np.mgrid[0:rh, 0:rw]
        dist = np.sqrt((xx - (icx - x0)) ** 2 + (yy - (icy - y0)) ** 2).astype(np.float32)
        patch = star_layer[y0:y1, x0:x1]
        lum_patch = _luminance(patch)
        core_win = dist <= max(1.2, min(real_r * 0.65, 2.6))
        if not np.any(core_win):
            continue

        peak_lum = float(np.percentile(lum_patch[core_win], 97))
        if peak_lum <= 1e-5:
            continue

        core_sigma = _estimate_core_sigma(lum_patch, dist, real_r)
        new_sigma = max(0.55, core_sigma * max(0.35, shrink))
        cutoff = max(new_r * (2.2 + 0.15 * (1.0 - s)), new_sigma * 5.2, 2.4)
        profile = _moffat_profile(dist, new_sigma, cutoff=cutoff)

        if is_color:
            peak_rgb = np.percentile(patch[core_win], 97, axis=0).astype(np.float32)
            peak_rgb = np.clip(peak_rgb, 0.0, 1.0)
            synth = profile[:, :, np.newaxis] * peak_rgb[np.newaxis, np.newaxis, :]
            restored[y0:y1, x0:x1] += synth.astype(np.float32)
        else:
            peak_gray = float(np.percentile(patch[core_win], 97))
            restored[y0:y1, x0:x1] += (profile * peak_gray).astype(np.float32)

    result = starless + restored
    np.clip(result, 0, 1, out=result)
    erase_f = (erase_mask > 0).astype(np.float32)
    feat = max(3, int(feather) * 2 + 1) | 1
    mask_f = cv2.GaussianBlur(erase_f, (feat, feat), feat * 0.4)
    return result.astype(np.float32), mask_f


def _measure_star_radius(gray, cx, cy, core_r, h, w):
    """
    Estimate the real star radius by sampling rings outward and stopping
    when the radial profile flattens into the local background.
    """
    max_r = min(int(core_r * 6), 30, cx, w - cx - 1, cy, h - cy - 1)
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
            drop = prev_val - ring_val
            if abs(drop) < 0.005:
                return float(r)
        prev_val = ring_val

    return min(core_r * 2.5, max_r)


def _luminance(img):
    if img.ndim == 2:
        return img.astype(np.float32, copy=False)
    return (
        0.2126 * img[:, :, 0]
        + 0.7152 * img[:, :, 1]
        + 0.0722 * img[:, :, 2]
    ).astype(np.float32, copy=False)


def _estimate_core_sigma(lum_patch, dist, real_r):
    radius = max(1.2, min(real_r * 0.95, 4.5))
    core = dist <= radius
    weights = lum_patch[core].astype(np.float32)
    if weights.size == 0:
        return max(0.8, real_r * 0.4)
    total = float(np.sum(weights))
    if total <= 1e-6:
        return max(0.8, real_r * 0.4)
    sigma = np.sqrt(float(np.sum(weights * (dist[core] ** 2)) / (2.0 * total)))
    return float(np.clip(sigma, 0.55, max(0.85, real_r * 0.80)))


def _smoothstep(edge0, edge1, x):
    width = max(float(edge1) - float(edge0), 1e-6)
    t = np.clip((x - float(edge0)) / width, 0.0, 1.0).astype(np.float32)
    return t * t * (3.0 - 2.0 * t)


def _moffat_profile(dist, sigma, cutoff, beta=3.6):
    sigma = max(float(sigma), 0.35)
    fwhm = max(0.95, 2.355 * sigma)
    alpha = fwhm / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))
    profile = (1.0 + (dist / max(alpha, 1e-6)) ** 2) ** (-beta)
    fade = 1.0 - _smoothstep(cutoff * 0.72, cutoff, dist)
    profile = np.clip(profile * fade, 0.0, 1.0)
    return profile.astype(np.float32)


def _fast_star_mask(gray, sensitivity, threshold, max_sigma, min_sigma):
    """DoG-based star detection."""
    g8 = (np.clip(gray, 0, 1) * 255).astype(np.uint8)
    h, w = g8.shape
    mask = np.zeros((h, w), np.uint8)

    step = max(1, (max_sigma - min_sigma) // 3)
    for sigma in range(min_sigma, max_sigma + 1, step):
        ks1 = max(3, sigma * 2 + 1) | 1
        ks2 = max(3, sigma * 4 + 1) | 1
        g1 = cv2.GaussianBlur(g8, (ks1, ks1), sigma)
        g2 = cv2.GaussianBlur(g8, (ks2, ks2), sigma * 2)
        dog = cv2.subtract(g1, g2)
        thr_val = max(1, int(threshold * 255 * (1.1 - sensitivity)))
        _, bw = cv2.threshold(dog, thr_val, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_or(mask, bw)

    return mask
