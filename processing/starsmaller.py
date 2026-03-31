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
    del kw

    img = np.ascontiguousarray(image, dtype=np.float32)
    np.clip(img, 0, 1, out=img)
    h, w = img.shape[:2]
    is_color = img.ndim == 3

    gray = img.mean(axis=2) if is_color else img.copy()
    s = float(np.clip(strength, 0, 1))
    shrink = max(0.15, 1.0 - s * 0.80)
    protect_structures = bool(protect_nebula)
    global_bg, global_noise = _estimate_background_stats(gray)

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
        structure_level = (
            _extended_structure_level(gray, icx, icy, real_r, global_bg, global_noise)
            if protect_structures else 0.0
        )
        if structure_level >= 0.95:
            continue

        local_shrink = 1.0 - (1.0 - shrink) * (1.0 - 0.85 * structure_level)
        local_shrink = float(np.clip(local_shrink, shrink, 1.0))
        new_r = max(0.85, real_r * local_shrink)
        erase_scale = (1.10 + 0.12 * s) * (1.0 - 0.30 * structure_level)
        erase_r = int(np.ceil(real_r * max(0.92, erase_scale))) + 2
        cv2.circle(erase_mask, (icx, icy), erase_r, 255, -1)
        star_specs.append((icx, icy, float(real_r), float(new_r), float(structure_level)))

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

    # 4. Rebuild only a smaller synthetic PSF from the star excess above local background.
    restored = np.zeros_like(img, dtype=np.float32)

    for icx, icy, real_r, new_r, structure_level in star_specs:
        pad = int(max(real_r * 3.0, new_r * 5.0)) + 4
        y0, y1 = max(0, icy - pad), min(h, icy + pad + 1)
        x0, x1 = max(0, icx - pad), min(w, icx + pad + 1)
        rh, rw = y1 - y0, x1 - x0
        if rh < 3 or rw < 3:
            continue

        yy, xx = np.mgrid[0:rh, 0:rw]
        dist = np.sqrt((xx - (icx - x0)) ** 2 + (yy - (icy - y0)) ** 2).astype(np.float32)
        patch = img[y0:y1, x0:x1]
        starless_patch = starless[y0:y1, x0:x1]
        lum_patch = _luminance(np.clip(patch - starless_patch, 0.0, 1.0))
        core_win = dist <= max(1.2, min(real_r * 0.65, 2.6))
        if not np.any(core_win):
            continue

        peak_lum = float(np.percentile(lum_patch[core_win], 97))
        if peak_lum <= 1e-5:
            continue

        core_sigma = _estimate_core_sigma(lum_patch, dist, real_r)
        local_shrink = new_r / max(real_r, 1e-6)
        new_sigma = max(0.52, core_sigma * max(0.28, local_shrink))
        wing_start = max(new_r * (1.10 + 0.35 * s), new_sigma * 1.8, 1.4)
        wing_stop = max(
            wing_start + 0.8,
            min(float(pad - 1), max(real_r * (1.35 - 0.15 * structure_level), new_sigma * 3.4, 2.2)),
        )
        cutoff = max(wing_stop, new_sigma * 2.8, 2.0)
        profile = _moffat_profile(dist, new_sigma, cutoff=cutoff)
        wing_strength = 0.90 - 0.20 * structure_level
        profile *= np.clip(
            1.0 - wing_strength * _smoothstep(wing_start, wing_stop, dist),
            0.0,
            1.0,
        )

        if is_color:
            annulus = _make_annulus_mask(
                dist,
                inner=max(real_r * 1.35, new_r * 1.75, 2.4),
                outer=min(float(pad - 1), max(real_r * 2.8, new_r * 3.0, 4.2)),
            )
            if np.any(annulus):
                local_bg_rgb = np.median(starless_patch[annulus], axis=0).astype(np.float32)
            else:
                local_bg_rgb = np.median(starless_patch.reshape(-1, 3), axis=0).astype(np.float32)
            peak_rgb = np.percentile(patch[core_win], 97, axis=0).astype(np.float32) - local_bg_rgb
            peak_rgb = np.clip(peak_rgb, 0.0, 1.0)
            if float(np.max(peak_rgb)) <= 1e-5:
                continue
            synth = profile[:, :, np.newaxis] * peak_rgb[np.newaxis, np.newaxis, :]
            restored[y0:y1, x0:x1] += synth.astype(np.float32)
        else:
            annulus = _make_annulus_mask(
                dist,
                inner=max(real_r * 1.35, new_r * 1.75, 2.4),
                outer=min(float(pad - 1), max(real_r * 2.8, new_r * 3.0, 4.2)),
            )
            local_bg = float(np.median(starless_patch[annulus])) if np.any(annulus) else float(np.median(starless_patch))
            peak_gray = float(np.percentile(patch[core_win], 97) - local_bg)
            if peak_gray <= 1e-5:
                continue
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
    peak = float(gray[cy, cx])

    outer_vals = []
    outer_start = max(int(core_r * 2.8), int(core_r) + 2)
    for r in range(outer_start, max_r + 1, step):
        vals = []
        for angle in range(0, 360, 30):
            rad = angle * np.pi / 180
            px = int(cx + r * np.cos(rad))
            py = int(cy + r * np.sin(rad))
            if 0 <= px < w and 0 <= py < h:
                vals.append(float(gray[py, px]))
        if vals:
            outer_vals.extend(vals)

    local_bg = float(np.median(outer_vals)) if outer_vals else float(np.median(gray[max(0, cy-2):min(h, cy+3), max(0, cx-2):min(w, cx+3)]))
    peak_excess = max(peak - local_bg, 1e-5)
    frac_stop = 0.08
    abs_stop = max(0.0035, peak_excess * frac_stop)
    last_signal_r = max(core_r * 1.2, 1.5)

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
        excess = ring_val - local_bg
        if excess > abs_stop:
            last_signal_r = float(r)
            continue
        if r > core_r * 1.25:
            break

    return float(min(max(last_signal_r, core_r * 1.35), max_r))


def _estimate_background_stats(gray):
    low = gray[gray <= np.percentile(gray, 55)]
    if low.size == 0:
        low = gray.reshape(-1)
    med = float(np.median(low))
    mad = float(np.median(np.abs(low - med))) * 1.4826
    return med, max(mad, 1e-4)


def _make_annulus_mask(dist, inner, outer):
    inner = max(float(inner), 0.0)
    outer = max(float(outer), inner + 0.5)
    return (dist >= inner) & (dist <= outer)


def _extended_structure_level(gray, cx, cy, real_r, global_bg, global_noise):
    h, w = gray.shape
    outer = max(real_r * 4.2, 8.0)
    pad = int(np.ceil(outer)) + 2
    y0, y1 = max(0, cy - pad), min(h, cy + pad + 1)
    x0, x1 = max(0, cx - pad), min(w, cx + pad + 1)
    if y1 - y0 < 5 or x1 - x0 < 5:
        return 0.0

    yy, xx = np.mgrid[y0:y1, x0:x1]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(np.float32)
    annulus = _make_annulus_mask(dist, max(real_r * 2.4, 4.0), outer)
    if not np.any(annulus):
        return 0.0

    vals = gray[y0:y1, x0:x1][annulus]
    local_med = float(np.median(vals))
    local_p90 = float(np.percentile(vals, 90))
    elevation = max(0.0, local_med - global_bg) / max(global_noise * 6.0, 1e-6)
    texture = max(0.0, local_p90 - local_med) / max(global_noise * 4.0, 1e-6)
    smoothness = np.clip(1.0 - texture / 1.8, 0.0, 1.0)
    return float(np.clip((elevation - 0.5) / 2.5, 0.0, 1.0) * smoothness)


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
