"""
Astro Mastro Pro — WCS Annotator
=================================
Plate solve sonucu üzerine NGC/IC/star katalog overlay'i çizer.
astroquery ile online katalog sorgusu yapar (opsiyonel).
"""

import numpy as np
import cv2


def annotate_image(
    image: np.ndarray,
    result: dict,
    show_grid: bool = True,
    show_catalog: bool = False,
    catalog_radius_deg: float = 1.0,
    font_scale: float = 0.5,
) -> np.ndarray:
    """
    Plate solve sonucunu görüntü üzerine çizer.

    Parametreler
    ------------
    image             : float32 [0,1] numpy array
    result            : solve_image() dönüş dict'i
    show_grid         : RA/Dec ızgara çiz
    show_catalog      : Messier/NGC nesneleri göster (astroquery gerekir)
    catalog_radius_deg: Katalog arama yarıçapı
    font_scale        : Etiket boyutu

    Dönüş
    ------
    float32 [0,1] annotated image
    """
    if result.get("ra") is None:
        return image

    img = np.clip(image, 0, 1).astype(np.float32)
    overlay = img.copy()
    if overlay.ndim == 2:
        overlay = np.stack([overlay, overlay, overlay], axis=2)

    disp = (overlay * 255).astype(np.uint8)
    h, w = disp.shape[:2]

    ra_center  = result["ra"]
    dec_center = result["dec"]
    scale_arcsec = result.get("scale_arcsec", 1.0)
    rotation   = result.get("rotation_deg", 0.0)

    # ── RA/Dec ızgarası ───────────────────────────────────────────────────
    if show_grid:
        disp = _draw_wcs_grid(disp, ra_center, dec_center,
                              scale_arcsec, rotation, w, h)

    # ── Merkez işareti ────────────────────────────────────────────────────
    cx, cy = w // 2, h // 2
    color = (0, 200, 255)
    cv2.line(disp, (cx - 30, cy), (cx + 30, cy), color, 1, cv2.LINE_AA)
    cv2.line(disp, (cx, cy - 30), (cx, cy + 30), color, 1, cv2.LINE_AA)
    cv2.circle(disp, (cx, cy), 12, color, 1, cv2.LINE_AA)

    # ── Bilgi kutusu ──────────────────────────────────────────────────────
    from ai.astap_bridge import _deg_to_hms_str, _deg_to_dms_str
    ra_str   = _deg_to_hms_str(ra_center)
    dec_str  = _deg_to_dms_str(dec_center)
    scale_s  = f"{scale_arcsec:.3f}\"/px" if isinstance(scale_arcsec, float) else "?"
    rot_s    = f"{rotation:.1f}°"          if isinstance(rotation, float)   else "?"

    info_lines = [
        f"RA:  {ra_str}",
        f"Dec: {dec_str}",
        f"Scale: {scale_s}   Rot: {rot_s}",
    ]
    _draw_info_box(disp, info_lines, x=10, y=10, font_scale=font_scale)

    # ── Katalog nesneleri ──────────────────────────────────────────────────
    if show_catalog:
        try:
            objects = _query_catalog(ra_center, dec_center,
                                     catalog_radius_deg)
            if objects:
                disp = _draw_catalog_objects(disp, objects,
                                              ra_center, dec_center,
                                              scale_arcsec, rotation,
                                              w, h, font_scale)
        except Exception:
            pass  # Katalog sorgusu opsiyonel

    return np.clip(disp.astype(np.float32) / 255.0, 0, 1).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  İç fonksiyonlar
# ─────────────────────────────────────────────────────────────────────────────

def _draw_wcs_grid(disp, ra_c, dec_c, scale_arcsec, rotation, w, h):
    """Basit RA/Dec ızgarası çizer (yaklaşık, WCS fit olmaksızın)."""
    if scale_arcsec <= 0:
        return disp

    # Alan genişliği tahmini
    fov_w_deg = (w * scale_arcsec) / 3600.0
    fov_h_deg = (h * scale_arcsec) / 3600.0

    # Izgara aralığı — FOV'a göre akıllı seç
    for step in [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 5.0, 10.0]:
        if fov_w_deg / step < 12:
            grid_step = step
            break
    else:
        grid_step = 15.0

    cos_dec = max(0.01, abs(np.cos(np.radians(dec_c))))
    ra_step  = grid_step / cos_dec
    dec_step = grid_step

    # Dec çizgileri (yatay)
    dec_start = np.floor((dec_c - fov_h_deg / 2) / dec_step) * dec_step
    dec_end   = dec_c + fov_h_deg / 2
    d = dec_start
    while d <= dec_end + dec_step:
        y_frac = 0.5 - (d - dec_c) / fov_h_deg
        y_px   = int(y_frac * h)
        if 0 <= y_px < h:
            cv2.line(disp, (0, y_px), (w, y_px), (50, 80, 120), 1, cv2.LINE_AA)
            label = f"{d:+.2f}°"
            cv2.putText(disp, label, (4, max(12, y_px - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 130, 180), 1, cv2.LINE_AA)
        d += dec_step

    # RA çizgileri (dikey)
    ra_start = np.floor((ra_c - fov_w_deg / 2 / cos_dec) / ra_step) * ra_step
    ra_end   = ra_c + fov_w_deg / 2 / cos_dec
    r = ra_start
    while r <= ra_end + ra_step:
        x_frac = 0.5 + (r - ra_c) * cos_dec / fov_w_deg
        x_px   = int(x_frac * w)
        if 0 <= x_px < w:
            cv2.line(disp, (x_px, 0), (x_px, h), (50, 80, 120), 1, cv2.LINE_AA)
            # RA → saat:dakika etiket
            hours  = (r % 360) / 15.0
            h_int  = int(hours)
            m_int  = int((hours - h_int) * 60)
            label  = f"{h_int}h{m_int:02d}m"
            cv2.putText(disp, label, (max(2, x_px + 2), h - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 130, 180), 1, cv2.LINE_AA)
        r += ra_step

    return disp


def _draw_info_box(disp, lines, x=10, y=10, font_scale=0.45):
    """Sol üstte bilgi kutusu çizer."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    fw   = 1
    pad  = 6
    lh   = int(18 * font_scale / 0.45)

    # Kutu genişliği
    max_w = max(cv2.getTextSize(l, font, font_scale, fw)[0][0] for l in lines)
    box_w = max_w + pad * 2
    box_h = lh * len(lines) + pad * 2

    # Yarı-saydam arka plan
    sub = disp[y:y+box_h, x:x+box_w]
    bg  = np.zeros_like(sub)
    cv2.addWeighted(sub, 0.4, bg, 0.6, 0, sub)
    disp[y:y+box_h, x:x+box_w] = sub

    # Çerçeve
    cv2.rectangle(disp, (x, y), (x+box_w, y+box_h), (80, 160, 220), 1)

    # Metin
    for i, line in enumerate(lines):
        ty = y + pad + lh * (i + 1) - 2
        cv2.putText(disp, line, (x + pad, ty),
                    font, font_scale, (180, 220, 255), fw, cv2.LINE_AA)


def _query_catalog(ra_deg, dec_deg, radius_deg):
    """astroquery ile Simbad/VizieR'den Messier/NGC sorgular."""
    try:
        from astroquery.simbad import Simbad
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        simbad = Simbad()
        simbad.add_votable_fields("otype", "flux(V)", "size_bibcode")
        coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
        result = simbad.query_region(coord, radius=radius_deg * u.deg)

        if result is None:
            return []

        objects = []
        for row in result:
            try:
                name  = str(row["MAIN_ID"])
                ra_r  = float(row["RA"].split()[0]) * 15  # h → deg
                dec_r = float(row["DEC"].split()[0])
                otype = str(row.get("OTYPE", ""))
                objects.append({
                    "name": name, "ra": ra_r, "dec": dec_r, "type": otype
                })
            except Exception:
                continue
        return objects[:50]  # max 50 nesne
    except Exception:
        return []


def _draw_catalog_objects(disp, objects,
                           ra_c, dec_c, scale_arcsec,
                           rotation, w, h, font_scale):
    """Katalog nesnelerini görüntü üzerine çizer."""
    if scale_arcsec <= 0:
        return disp

    fov_w = w * scale_arcsec / 3600.0
    fov_h = h * scale_arcsec / 3600.0
    cos_dec = max(0.01, abs(np.cos(np.radians(dec_c))))

    for obj in objects:
        dra  = (obj["ra"]  - ra_c) * cos_dec
        ddec = (obj["dec"] - dec_c)
        x_px = int(w/2 + dra  / fov_w * w)
        y_px = int(h/2 - ddec / fov_h * h)

        if not (0 <= x_px < w and 0 <= y_px < h):
            continue

        # Farklı nesne tiplerine göre renk
        otype = obj.get("type", "").lower()
        if "galaxy" in otype or "gx" in otype:
            color = (180, 100, 255)
        elif "nebul" in otype or "neb" in otype or "hii" in otype:
            color = (80, 200, 255)
        elif "cluster" in otype or "cl" in otype:
            color = (255, 200, 80)
        else:
            color = (150, 255, 150)

        cv2.circle(disp, (x_px, y_px), 12, color, 1, cv2.LINE_AA)
        name = obj["name"].strip().replace("  ", " ")
        cv2.putText(disp, name, (x_px + 14, y_px + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8,
                    color, 1, cv2.LINE_AA)

    return disp
