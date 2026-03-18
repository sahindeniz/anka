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
import re
import numpy as np


def solve_image(
    image,
    astap_exe: str,
    db_path: str = "",
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

    img = np.clip(image, 0, 1).astype(np.float32)
    h, w = img.shape[:2]
    cb(f"[1/4] Hazırlanıyor ({w}×{h})…")

    tmpdir = tempfile.mkdtemp(prefix="amp_astap_")

    try:
        fits_in = os.path.join(tmpdir, "solve_input.fits")
        t_start = time.time()

        _save_mono_fits(img, fits_in)
        cb("[1/4] FITS kaydedildi")

        cmd = _build_cmd(
            astap_exe, fits_in, db_path,
            search_radius, downsample, min_stars,
            ra_hint, dec_hint, fov_hint
        )
        # Tam komutu log'a yaz (debug için)
        cmd_str = " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd)
        cb(f"[2/4] Komut:\n      {cmd_str}")

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmpdir,
            )
        except subprocess.TimeoutExpired:
            return _err(
                f"ASTAP {timeout} saniyede tamamlanamadı.\n\n"
                "Öneriler:\n"
                "• Timeout değerini artırın (örn. 300s)\n"
                "• Search radius'u küçültün\n"
                "• RA/Dec ipucu girin"
            )
        except FileNotFoundError:
            return _err(f"ASTAP başlatılamadı:\n{astap_exe}")

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

            return _err(
                f"Plate solve başarısız.\n\n"
                f"{'• ' + reason if reason else ''}"
                f"Öneriler:\n"
                f"• Search radius'u artırın ({search_radius}° → {min(180, search_radius*2):.0f}°)\n"
                f"• Min Stars'ı azaltın ({min_stars} → {max(5, min_stars//2)})\n"
                f"• RA/Dec ipucu girin\n"
                f"• FOV ipucu girin\n"
                f"• Katalog yolunu kontrol edin",
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
    """float32 [0,1] → mono float32 FITS."""
    from astropy.io import fits as _fits
    if img.ndim == 3:
        mono = (0.2126 * img[:, :, 0]
                + 0.7152 * img[:, :, 1]
                + 0.0722 * img[:, :, 2]).astype(np.float32)
    else:
        mono = img.astype(np.float32)
    hdu = _fits.PrimaryHDU(mono)
    hdu.header["SIMPLE"]  = True
    hdu.header["BITPIX"]  = -32
    hdu.header["NAXIS"]   = 2
    hdu.header["NAXIS1"]  = mono.shape[1]
    hdu.header["NAXIS2"]  = mono.shape[0]
    hdu.writeto(path, overwrite=True)


def _build_cmd(exe, fits_in, db_path, radius, downsample, min_stars,
               ra_hint, dec_hint, fov_hint):
    """
    ASTAP parametreleri:
      -d    : katalog yolu — .pkg için TAM DOSYA YOLU, klasör için klasör yolu
      -ra   : RA SAAT cinsinden (H.HHHH)
      -spd  : SPD = Dec + 90 (derece)
    """
    cmd = [exe, "-f", fits_in]

    if db_path:
        if os.path.isfile(db_path):
            # Tam dosya yolu (.pkg veya diğer) — ASTAP doğrudan kabul eder
            cmd += ["-d", db_path]
        elif os.path.isdir(db_path):
            # Klasör yolu — içinde katalog dosyalarını arar
            cmd += ["-d", db_path]

    cmd += ["-r", f"{float(radius):.1f}"]
    cmd += ["-m", str(int(min_stars))]

    if downsample and int(downsample) > 0:
        cmd += ["-z", str(int(downsample))]

    if ra_hint is not None and float(ra_hint) != 0.0:
        hours = float(ra_hint) / 15.0
        cmd += ["-ra", f"{hours:.6f}"]

    if dec_hint is not None and float(dec_hint) != 0.0:
        spd = float(dec_hint) + 90.0
        cmd += ["-spd", f"{spd:.4f}"]

    if fov_hint is not None and float(fov_hint) > 0.0:
        cmd += ["-fov", f"{float(fov_hint):.3f}"]

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
        names   = ["astap.exe", "ASTAP.exe"]
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
