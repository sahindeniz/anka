import cv2
import numpy as np


def detect_galaxies(image, percentile=95, blur_size=31, **kwargs):
    """
    Galaksi tespiti — koordinatları döndürür VE görüntü üzerine overlay çizer.
    Dönen görüntü: orijinal + kırmızı daire overlay.
    """
    ks = int(blur_size) | 1
    gray = image if image.ndim == 2 else image.mean(axis=2)

    # Arka planı çıkar
    blurred = cv2.GaussianBlur(gray, (ks, ks), 0)
    mask = gray - blurred
    threshold = np.percentile(mask, float(percentile))

    # Bağlı bileşenleri bul
    binary = (mask > threshold).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                               np.ones((5, 5), np.uint8))
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8)

    # Görüntü üzerine overlay çiz
    if image.ndim == 2:
        overlay = np.stack([image, image, image], axis=2).copy()
    else:
        overlay = image.copy()

    overlay_u8 = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)

    min_area = 20
    galaxy_count = 0
    for i in range(1, num_labels):  # 0 = arka plan
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        cx, cy = int(centroids[i][0]), int(centroids[i][1])
        w_box = stats[i, cv2.CC_STAT_WIDTH]
        h_box = stats[i, cv2.CC_STAT_HEIGHT]
        radius = max(8, int(np.sqrt(area / np.pi) * 1.5))
        cv2.circle(overlay_u8, (cx, cy), radius, (220, 60, 60), 1)
        cv2.circle(overlay_u8, (cx, cy), 2, (220, 60, 60), -1)
        galaxy_count += 1

    result = overlay_u8.astype(np.float32) / 255.0
    # galaxy_count bilgisini result'a attribute olarak ekleyemeyiz,
    # ama görüntüde overlay zaten var
    return result
