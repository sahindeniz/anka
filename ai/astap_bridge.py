"""
Astro Mastro Pro — ASTAP Plate Solving Bridge  (v2 — düzeltilmiş)
==================================================================
ASTAP (Astrometric STAcking Program) komut satırı ile plate solving yapar.
https://www.hnsky.org/astap.htm

ASTAP'ın gerçek davranışı:
  • Exit code 0  → başarılı
  • Exit code 1  → çözüm bulunamadı (crash değil, normal)
  • Exit code ≥2 → hata (exe bulunamadı vb.)
  • .ini dosyası HER ZAMAN oluşur (başarısız da olsa)
  • SOLUTION=0 → başarısız,  SOLUTION=1 → başarılı
  • RA= derece (saat DEĞİL), SPDT= SPD = dec+90 (derece)
  • CDELT1= deg/px (neg olabilir) → |CDELT1|*3600 = arcsec/px
"""

import os
import sys
import subprocess
import tempfile
import shutil
import glob
import time
import numpy as np


def solve_image(
    image,
    astap_exe: str,
    db_path: str = "",
    catalog_id: str = "",
    search_radius: float = 30.0,
    downsample: int = 0,
    min_stars: int = 10,
    timeout: int = 120,
    ra_hint=None,
    dec_hint=None,
    fov_hint=None,
    progress_cb=None,
) -> dict:
    """Görüntüyü ASTAP ile plate solve eder."""

    def cb(msg):
        if progress_cb:
            progress_cb(str(msg))

    astap_exe = str(astap_exe).strip()
    if not astap_exe or not os.path.isfile(astap_exe):
        return _err(
            f"ASTAP bulunamadı:\n  {astap_exe}\n\n"
            "Settings → ASTAP sekmesinden yolu ayarlayın veya\n"
            "'🔍 ASTAP Bul' butonunu kullanın."
        )

    # CLI sürümü varsa onu tercih et (GUI exe pencere açabilir)
    astap_exe = _prefer_cli_exe(astap_exe)

    img = np.clip(image, 0, 1).astype(np.float32)
    h, w = img.shape[:2]
    cb(f"[1/4] Hazirlanıyor ({w}x{h})...")

    tmpdir = tempfile.mkdtemp(prefix="amp_astap_")

    try:
        fits_in = os.path.join(tmpdir, "solve_input.fits")
        t_start = time.time()

        _save_mono_fits(img, fits_in)
        # Upscale bilgisi
        if max(h, w) < 2000:
            scale = 2000.0 / max(h, w)
            new_w, new_h = int(w * scale + 0.5), int(h * scale + 0.5)
            cb(f"[1/4] FITS kaydedildi (upscale {w}x{h} -> {new_w}x{new_h})")
        else:
            cb("[1/4] FITS kaydedildi")

        # ── RA/Dec ipucu varsa radius'u otomatik küçült ──
        effective_radius = search_radius
        if ra_hint is not None and dec_hint is not None and search_radius >= 90:
            effective_radius = 30.0
            cb(f"  ℹ RA/Dec ipucu verildi, radius {search_radius}° → {effective_radius}° olarak küçültüldü")

        # ── FOV retry stratejisi: verilen FOV başarısızsa 2x ve 4x dene ──
        fov_attempts = [fov_hint]
        if fov_hint and float(fov_hint) > 0:
            fov_attempts += [float(fov_hint) * 2, float(fov_hint) * 4]
        else:
            fov_attempts = [None]  # sadece auto

        proc = None
        for attempt_i, fov_try in enumerate(fov_attempts):
            cmd = _build_cmd(
                astap_exe, fits_in, db_path, catalog_id,
                effective_radius, downsample, min_stars,
                ra_hint, dec_hint, fov_try
            )
            cmd_str = " ".join(
                f'"{c}"' if " " in str(c) else str(c) for c in cmd)
            attempt_label = (f" [deneme {attempt_i+1}/{len(fov_attempts)},"
                             f" FOV={fov_try:.1f}°]") if fov_try else ""
            cb(f"[2/4] Komut{attempt_label}:\n      {cmd_str}")

            try:
                # Her denemeye tam timeout ver (bölme)
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=max(60, timeout),
                    cwd=tmpdir,
                )
            except subprocess.TimeoutExpired:
                cb(f"      ⏱ Zaman aşımı (deneme {attempt_i+1})")
                continue
            except FileNotFoundError:
                return _err(f"ASTAP başlatılamadı:\n{astap_exe}")

            if proc.returncode == 0:
                break  # Çözüm bulundu
            elif attempt_i < len(fov_attempts) - 1:
                cb(f"      Çözüm bulunamadı (FOV={fov_try}°), farklı FOV deneniyor…")

        if proc is None:
            return _err(
                f"ASTAP {timeout} saniyede tamamlanamadı.\n\n"
                "Öneriler:\n"
                "• Timeout değerini artırın (örn. 300s)\n"
                "• Search radius'u küçültün\n"
                "• RA/Dec ipucu girin"
            )

        solve_time = time.time() - t_start
        cb(f"[3/4] Çıkış kodu: {proc.returncode}  ({solve_time:.1f}s)")

        if proc.stdout and proc.stdout.strip():
            cb(f"      stdout: {proc.stdout.strip()[:200]}")
        if proc.stderr and proc.stderr.strip():
            cb(f"      stderr: {proc.stderr.strip()[:200]}")

        cb("[4/4] Sonuç aranıyor…")

        # INI dosyası — girişin yanına yazılır (solve_input.ini)
        ini_path = os.path.splitext(fits_in)[0] + ".ini"

        if not os.path.isfile(ini_path):
            inis = glob.glob(os.path.join(tmpdir, "*.ini"))
            if inis:
                ini_path = sorted(inis, key=os.path.getmtime)[-1]

        if not ini_path or not os.path.isfile(ini_path):
            hints = _build_hint_message(proc, db_path)
            return _err(
                f"ASTAP tamamlandı (kod={proc.returncode}) ama .ini dosyası oluşmadı.\n\n"
                f"{hints}"
            )

        result = _parse_ini(ini_path)
        result["solve_time_s"] = round(solve_time, 2)
        result["astap_exit_code"] = proc.returncode

        solution = result.get("solution", "")
        ra  = result.get("ra")
        dec = result.get("dec")

        if solution == "0" or ra is None or dec is None:
            warning     = result.get("warning", "")
            stars_found = result.get("star_count", 0)
            cb(f"  Bulunan yıldız: {stars_found}")
            if warning:
                cb(f"  ASTAP uyarısı: {warning}")

            reason_parts = []
            if stars_found and int(stars_found) < min_stars:
                reason_parts.append(
                    f"Bulunan yıldız ({stars_found}) < istenen ({min_stars})")
            if warning:
                reason_parts.append(f"ASTAP: {warning}")
            reason = ("\n• ".join(reason_parts) + "\n\n") if reason_parts else ""

            tips = []
            if search_radius < 180:
                tips.append(
                    f"• Search radius'u artırın ({search_radius:.0f}° → {min(180, search_radius*2):.0f}°)")
            if min_stars > 5:
                tips.append(
                    f"• Min Stars'ı azaltın ({min_stars} → {max(5, min_stars//2)})")
            tips.append("• RA/Dec ipucu girin")
            tips.append("• FOV ipucu girin")
            tips.append("• Katalog yolunu kontrol edin")
            if search_radius >= 180:
                tips.append("• Görüntü kalitesini kontrol edin (çok karanlık/parlak olabilir)")
                tips.append("• Downsample=0 (auto) deneyin")

            return _err(
                f"Plate solve başarısız.\n\n"
                f"{'• ' + reason if reason else ''}"
                f"Öneriler:\n" + "\n".join(tips),
                extra=result,
            )

        cb(f"✅ Çözüldü! RA={ra:.4f}°  Dec={dec:.4f}°  "
           f"scale={result.get('scale_arcsec', '?')}\"/px  ({solve_time:.1f}s)")
        return result

    finally:
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass


def _save_mono_fits(img, path: str):
    """float32 [0,1] → mono 16-bit FITS.
    ASTAP 16-bit uint ile daha iyi calisir.
    Kucuk goruntuler (<2000px) 2x upscale edilir — 'small image' uyarisini onler.
    """
    import cv2
    from astropy.io import fits as _fits

    if img.ndim == 3:
        mono = (0.2126 * img[:, :, 0]
                + 0.7152 * img[:, :, 1]
                + 0.0722 * img[:, :, 2]).astype(np.float32)
    else:
        mono = img.astype(np.float32)

    h, w = mono.shape
    # Kucuk goruntuler icin upscale — ASTAP minimum ~2000px ister
    if max(h, w) < 2000:
        scale = 2000.0 / max(h, w)
        if scale > 1.0:
            new_w = int(w * scale + 0.5)
            new_h = int(h * scale + 0.5)
            mono = cv2.resize(mono, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # 16-bit uint — astropy BZERO/BSCALE'i otomatik yonetir
    mono16 = np.clip(mono * 65535.0, 0, 65535).astype(np.uint16)

    hdu = _fits.PrimaryHDU(mono16)
    # NOT: BZERO/BSCALE elle ayarlamayın — astropy uint16 için
    # otomatik BZERO=32768 ekler. Elle eklersek çift uygulanır.
    hdu.writeto(path, overwrite=True)


def _build_cmd(exe, fits_in, db_path, catalog_id, radius, downsample, min_stars,
               ra_hint, dec_hint, fov_hint):
    """
    ASTAP CLI parametreleri (CLI-2025 referans):
      -f    : giriş dosyası
      -d    : veritabanı KLASÖR yolu
      -D    : veritabanı kısaltması (d80, d50, g17 vb.)
      -r    : arama yarıçapı (derece)
      -s    : maksimum yıldız sayısı (varsayılan 500)
      -m    : minimum yıldız boyutu arcsec (varsayılan 1.5)
      -z    : downsample (0=auto, 1=yok, 2=2x, 4=4x)
      -ra   : RA saat cinsinden (H.HHHH)
      -spd  : SPD = Dec + 90 (derece)
      -fov  : görüş alanı çapı (derece, 0=auto)
    """
    cmd = [exe, "-f", fits_in]

    # ── Database — -d klasör yolu ──
    if db_path:
        db_path_str = str(db_path).strip().strip('"').strip("'")

        # .pkg uzantılı ama KLASÖR olabilir (ASTAP paket formatı)
        if os.path.isdir(db_path_str):
            cmd += ["-d", db_path_str]
        elif os.path.isfile(db_path_str):
            # Gerçek dosya ise klasörünü al
            cmd += ["-d", os.path.dirname(db_path_str)]
        else:
            # Belki kisaltma verilmistir (d80 gibi) — -D ile gonder
            cmd += ["-D", db_path_str]

    cmd += ["-r", f"{float(radius):.1f}"]

    # -s max yildiz sayisi (varsayilan 500, dusuk deger cozumu zorlastirir)
    cmd += ["-s", "500"]

    # ── Downsample — ASTAP: 0=auto, 1=yok, 2=2x, 4=4x ──
    if downsample and int(downsample) > 1:
        cmd += ["-z", str(int(downsample))]
    else:
        # -z 0 = auto-detect (ASTAP kendi karar verir)
        cmd += ["-z", "0"]

    if ra_hint is not None:
        ra_val = float(ra_hint)
        if ra_val != 0.0 or dec_hint is not None:
            hours = ra_val / 15.0
            cmd += ["-ra", f"{hours:.6f}"]

    if dec_hint is not None:
        spd = float(dec_hint) + 90.0
        cmd += ["-spd", f"{spd:.4f}"]

    if fov_hint is not None and float(fov_hint) > 0.0:
        cmd += ["-fov", f"{float(fov_hint):.3f}"]
    else:
        # FOV bilinmiyorsa auto-detect (0)
        cmd += ["-fov", "0"]

    # Daha yüksek tolerance — varsayılan 0.007 çok katı
    cmd += ["-t", "0.02"]

    # Progress log — çıktıyı zenginleştir
    cmd += ["-progress"]

    return cmd


def _parse_ini(path: str) -> dict:
    """
    ASTAP .ini dosyasını parse eder.

    Örnek ASTAP çıktısı:
        [astrometry]
        SOLUTION=1
        RA=188.456789        ← DERECE (saat değil!)
        SPDT=134.567890      ← SPD = dec+90
        CDELT1=-0.000411     ← deg/px
        CROTA2=178.32
        NSTARS=342
    """
    result = {"fits_header": {}, "ra": None, "dec": None}

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception as e:
        result["parse_error"] = str(e)
        return result

    for line in content.splitlines():
        line = line.strip()
        if line.startswith("[") or not line or "=" not in line:
            continue

        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()

        result["fits_header"][key] = val
        kl = key.upper()

        try:
            if kl == "SOLUTION":
                result["solution"] = val

            elif kl == "RA":
                # ASTAP RA'yı DERECE olarak yazar (HMS değil)
                result["ra"] = float(val)

            elif kl == "SPDT":
                # SPD = Dec + 90 → Dec = SPD - 90
                result["dec"] = float(val) - 90.0

            elif kl == "DEC":
                # Bazı versiyonlar direkt DEC yazar
                result["dec"] = float(val)

            elif kl in ("CRVAL1",):
                if result.get("ra") is None:
                    result["ra"] = float(val)

            elif kl in ("CRVAL2",):
                if result.get("dec") is None:
                    result["dec"] = float(val)

            elif kl == "CDELT1":
                result["scale_arcsec"] = abs(float(val)) * 3600.0

            elif kl == "CROTA2":
                result["rotation_deg"] = float(val)

            elif kl in ("NSTARS", "STAR_COUNT"):
                result["star_count"] = int(float(val))

            elif kl == "WARNING" and val:
                result["warning"] = val

            elif kl == "ERROR" and val:
                result["astap_error"] = val

        except (ValueError, TypeError):
            pass

    return result


def _err(msg: str, extra: dict = None) -> dict:
    result = {"ra": None, "dec": None, "error": str(msg), "fits_header": {}}
    if extra:
        result.update({k: v for k, v in extra.items()
                        if k not in ("ra", "dec", "error")})
    return result


def _build_hint_message(proc, db_path: str) -> str:
    parts = []
    if not db_path or (not os.path.isdir(db_path) and not os.path.isfile(db_path)):
        parts.append(
            "⚠ Katalog bulunamadı.\n"
            "  d80_star_database.pkg dosyasını ASTAP klasörüne koyun\n"
            "  veya Settings → ASTAP → Katalog klasörünü ayarlayın"
        )
    combined = ((proc.stdout or "") + " " + (proc.stderr or "")).lower()
    if "catalog" in combined or "cat" in combined:
        parts.append("Katalog bulunamadı — katalog yolunu kontrol edin.")
    if "permission" in combined:
        parts.append("İzin hatası — ASTAP'ın dosya yazma izni var mı?")
    if proc.returncode >= 2:
        parts.append(f"ASTAP hata kodu {proc.returncode} — exe bozuk olabilir.")
    parts.append(
        "Genel öneriler:\n"
        "• Search radius'u artırın\n"
        "• Min Stars'ı azaltın\n"
        "• RA/Dec/FOV ipucu girin"
    )
    return "\n\n".join(parts)


def _prefer_cli_exe(exe_path: str) -> str:
    """astap.exe yerine astap_cli.exe varsa onu döndür (pencere açmaz)."""
    d = os.path.dirname(exe_path)
    cli = os.path.join(d, "astap_cli.exe")
    if os.path.isfile(cli):
        return cli
    return exe_path


def find_astap_exe() -> str:
    """ASTAP'ı yaygın konumlarda arar."""
    candidates = []
    if sys.platform == "win32":
        roots = [
            os.environ.get("PROGRAMFILES",      r"C:\Program Files"),
            os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"),
            os.path.expanduser("~\\Desktop"),
            os.path.expanduser("~\\Downloads"),
            os.path.expanduser("~"),
            r"C:\astap", r"D:\astap", r"C:\Program Files\astap",
        ]
        names   = ["astap_cli.exe", "astap.exe", "ASTAP.exe"]
        subdirs = ["", "astap", "ASTAP", "astap_cli", "astap-cli"]
        for root in roots:
            if not root: continue
            for sub in subdirs:
                for name in names:
                    candidates.append(os.path.join(root, sub, name))
    else:
        for d in ["/usr/local/bin", "/usr/bin",
                  os.path.expanduser("~/bin"),
                  os.path.expanduser("~/astap"),
                  "/opt/astap", "/opt/local/bin"]:
            for name in ["astap", "ASTAP"]:
                candidates.append(os.path.join(d, name))
    for p in candidates:
        if os.path.isfile(p):
            return p
    return ""


def astap_available(settings: dict) -> bool:
    exe = settings.get("astap_exe", "").strip()
    return bool(exe) and os.path.isfile(exe)


def format_result(result: dict) -> str:
    """Plate solve sonucunu okunabilir metin olarak biçimlendirir."""
    if result.get("error"):
        return f"❌  Plate Solve Başarısız\n\n{result['error']}"
    ra  = result.get("ra")
    dec = result.get("dec")
    if ra is None or dec is None:
        return "❌  RA/Dec bilgisi alınamadı."

    scale  = result.get("scale_arcsec", "?")
    rot    = result.get("rotation_deg", "?")
    tsolve = result.get("solve_time_s", "?")
    stars  = result.get("star_count", "?")

    lines = [
        "✅  PLATE SOLVE BAŞARILI",
        "",
        f"   RA   :  {_deg_to_hms_str(ra)}",
        f"            ({ra:.6f}°)",
        f"   Dec  :  {_deg_to_dms_str(dec)}",
        f"            ({dec:+.6f}°)",
        f"   Ölçek:  {f'{scale:.3f} arcsec/px' if isinstance(scale, float) else '?'}",
        f"   Dönme:  {f'{rot:.2f}°' if isinstance(rot, float) else '?'}",
        f"   Süre :  {f'{tsolve:.1f}s' if isinstance(tsolve, float) else '?'}",
        f"   Yıldız: {stars}",
    ]
    if result.get("warning"):
        lines += ["", f"⚠  {result['warning']}"]
    return "\n".join(lines)


def _deg_to_hms_str(deg: float) -> str:
    deg = float(deg) % 360
    total = deg / 15.0 * 3600
    h  = int(total // 3600)
    m  = int((total % 3600) // 60)
    s  = total % 60
    return f"{h:02d}h {m:02d}m {s:06.3f}s"


def _deg_to_dms_str(deg: float) -> str:
    sign = "+" if float(deg) >= 0 else "-"
    deg  = abs(float(deg))
    d    = int(deg)
    m    = int((deg - d) * 60)
    s    = ((deg - d) * 60 - m) * 60
    return f"{sign}{d:02d}° {m:02d}' {s:05.2f}\""
