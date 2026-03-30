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
    strength=0.9,
    sensitivity=0.5,
    feather=3,
    max_sigma=6,
    min_sigma=1,
    threshold=0.03,
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

    # 2. Build a full erase mask and a smaller soft restore mask.
    erase_mask = np.zeros((h, w), np.uint8)
    restore_mask = np.zeros((h, w), np.float32)

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
        new_r = max(0.8, real_r * shrink)

        cv2.circle(erase_mask, (icx, icy), int(real_r) + 2, 255, -1)

        pad = int(new_r * 3) + 2
        y0, y1 = max(0, icy - pad), min(h, icy + pad + 1)
        x0, x1 = max(0, icx - pad), min(w, icx + pad + 1)
        rh, rw = y1 - y0, x1 - x0
        if rh < 2 or rw < 2:
            continue

        yy, xx = np.mgrid[0:rh, 0:rw]
        dist = np.sqrt((xx - (icx - x0)) ** 2 + (yy - (icy - y0)) ** 2).astype(np.float32)

        restore_local = np.exp(-0.5 * (dist / max(new_r * 0.65, 0.45)) ** 2)
        restore_local[dist > new_r * 1.5] = 0

        restore_mask[y0:y1, x0:x1] = np.maximum(
            restore_mask[y0:y1, x0:x1],
            restore_local,
        )

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

    # 4. Fill the emptied zone fully from the background, then return
    #    only a smaller soft center from the original star.
    erase_f = (erase_mask > 0).astype(np.float32)
    feat = max(3, int(feather) * 2 + 1) | 1
    edge_softness = max(1.0, float(feather))
    replace_weight = cv2.distanceTransform(
        (erase_mask > 0).astype(np.uint8),
        cv2.DIST_L2,
        3,
    ).astype(np.float32)
    replace_weight = np.clip(replace_weight / edge_softness, 0, 1)

    np.clip(restore_mask, 0, 1, out=restore_mask)

    if is_color:
        w3 = replace_weight[:, :, np.newaxis]
        r3 = restore_mask[:, :, np.newaxis]
        filled_bg = img * (1 - w3) + starless * w3
        result = filled_bg * (1 - r3) + img * r3
    else:
        filled_bg = img * (1 - replace_weight) + starless * replace_weight
        result = filled_bg * (1 - restore_mask) + img * restore_mask

    np.clip(result, 0, 1, out=result)
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
