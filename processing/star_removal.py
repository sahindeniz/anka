import cv2
import numpy as np
from skimage.feature import blob_log


def remove_stars(image, max_sigma=6, min_sigma=1, threshold=0.03,
                 fill_method="local_median", **kwargs):
    """
    Yıldız kaldırma.
    fill_method: local_median | global_median | inpaint
    """
    gray = image if image.ndim == 2 else image.mean(axis=2)
    stars = blob_log(gray,
                     min_sigma=float(min_sigma),
                     max_sigma=float(max_sigma),
                     threshold=float(threshold))

    result = image.copy()
    h, w = image.shape[:2]

    # inpaint yöntemi için maske oluştur
    if fill_method == "inpaint":
        mask = np.zeros((h, w), dtype=np.uint8)
        for y, x, r in stars:
            r = max(1, int(r * 2.83))
            y0, y1 = max(0, int(y) - r), min(h, int(y) + r)
            x0, x1 = max(0, int(x) - r), min(w, int(x) + r)
            mask[y0:y1, x0:x1] = 255

        img8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        if image.ndim == 2:
            inpainted = cv2.inpaint(img8, mask, 3, cv2.INPAINT_TELEA)
            return inpainted.astype(np.float32) / 255.0
        else:
            # RGB için her kanalı ayrı inpaint et
            channels = []
            for c in range(image.shape[2]):
                ch = cv2.inpaint(img8[:, :, c], mask, 3, cv2.INPAINT_TELEA)
                channels.append(ch)
            return (np.stack(channels, axis=2).astype(np.float32) / 255.0)

    # local_median / global_median
    for y, x, r in stars:
        r = max(1, int(r * 2.83))
        y0, y1 = max(0, int(y) - r), min(h, int(y) + r)
        x0, x1 = max(0, int(x) - r), min(w, int(x) + r)

        if fill_method == "local_median":
            pad = r * 2
            ry0, ry1 = max(0, int(y) - pad), min(h, int(y) + pad)
            rx0, rx1 = max(0, int(x) - pad), min(w, int(x) + pad)
            patch = image[ry0:ry1, rx0:rx1]
            fill = float(np.median(patch)) if patch.size else float(np.median(image))
        else:  # global_median
            fill = float(np.median(image))

        result[y0:y1, x0:x1] = fill

    return result.astype(np.float32)
