"""
Astro Mastro Pro — GraXpert Bridge (v5)

GraXpert 3.x yok sayabilir -output parametresini.
Strateji:
  1. Input'u EXE'nin yanindaki bir alt klasore koy
  2. Output'u orada ara
  3. Bulamazsan input'un bulundugu klasoru, cwd'yi ve tmpdir'i tara
"""

import os, sys, subprocess, tempfile, shutil, glob, time
import numpy as np
import cv2


def run_graxpert(image: np.ndarray,
                 exe_path: str,
                 command: str = "background-extraction",
                 smoothing: float = 0.5,
                 ai_version: str = "latest",
                 correction: str = "Subtraction",
                 denoise_strength: float = 0.8,
                 progress_cb=None) -> dict:

    def cb(msg):
        if progress_cb: progress_cb(str(msg))

    exe_path = str(exe_path).strip()
    if not exe_path or not os.path.isfile(exe_path):
        raise FileNotFoundError(
            f"GraXpert bulunamadi:\n  {exe_path}\n"
            "Settings → GraXpert'ten yolu ayarlayin.")

    img    = np.clip(image, 0, 1).astype(np.float32)
    h, w   = img.shape[:2]
    is_rgb = (img.ndim == 3)
    exe_dir = os.path.dirname(exe_path)

    cb(f"[1/5] Hazirlaniyor ({w}x{h})...")

    # GraXpert'in output yazacagi yeri bulmak icin:
    # Input'u exe'nin yanina koy — GraXpert genelde buraya yazar
    work_dir = os.path.join(exe_dir, "_amp_work")
    os.makedirs(work_dir, exist_ok=True)
    inp_path = os.path.join(work_dir, "input.tif")

    try:
        _save_tiff16(img, inp_path, is_rgb)
        cb("[1/5] TIFF kaydedildi")

        # Butun arama dizinlerini hazirla
        search_dirs = [work_dir, exe_dir,
                       os.getcwd(),
                       os.path.expanduser("~")]

        # Calistirmadan once snapshot al
        before_files = _snapshot(search_dirs)

        cmd = [
            exe_path, inp_path,
            "-cli",
            "-cmd",    command,
            "-output", work_dir,
        ]
        if command == "background-extraction":
            corr = correction if correction in ("Subtraction","Division") else "Subtraction"
            cmd += ["-correction", corr,
                    "-smoothing",  str(round(float(smoothing), 2))]
        elif command == "denoising":
            cmd += ["-strength", str(round(float(denoise_strength), 2))]
        if ai_version and ai_version.strip().lower() not in ("latest",""):
            cmd += ["-ai_version", ai_version.strip()]

        cb(f"[2/5] Calistiriliyor: {' '.join(cmd)}")

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True, text=True,
                timeout=600,
                cwd=exe_dir,   # exe'nin yaninda calistir
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("GraXpert 10 dakikada bitmedi.")
        except FileNotFoundError:
            raise FileNotFoundError(f"GraXpert baslatılamadi:\n{exe_path}")

        cb(f"[3/5] Cikis kodu: {proc.returncode}")
        if proc.stderr: cb(f"       {proc.stderr[:300]}")

        if proc.returncode > 1:
            raise RuntimeError(
                f"GraXpert hata verdi (kod {proc.returncode}):\n"
                f"{(proc.stderr or proc.stdout or '')[:800]}")

        # Kisa bekleme — bazi versiyonlar yazmayi geciktirebilir
        time.sleep(0.5)

        cb("[4/5] Cikti aranıyor...")
        after_files  = _snapshot(search_dirs)
        new_files    = sorted(after_files - before_files,
                              key=lambda p: os.path.getmtime(p), reverse=True)
        cb(f"       Yeni dosyalar: {[os.path.basename(f) for f in new_files]}")

        # work_dir'deki TUM dosyalari logla
        wd_contents = os.listdir(work_dir)
        cb(f"       work_dir icerigi: {wd_contents}")

        result_path = _pick_result(new_files, work_dir, inp_path, search_dirs)

        if result_path is None:
            # Son care: work_dir + exe_dir'de input disinda her TIF/FITS
            for sd in [work_dir, exe_dir]:
                for fname in os.listdir(sd):
                    fp = os.path.join(sd, fname)
                    if fp == inp_path: continue
                    if fname.lower().endswith((".tif",".tiff",".fits",".fit")):
                        result_path = fp
                        cb(f"       Son care: {fname}")
                        break
                if result_path: break

        if result_path is None:
            all_new = "\n  ".join(
                f"{f} ({os.path.getsize(f)} bytes)" for f in new_files
            ) or "(yok)"
            raise RuntimeError(
                f"GraXpert tamamlandi ama cikti bulunamadi.\n\n"
                f"work_dir: {work_dir}\n"
                f"Yeni dosyalar:\n  {all_new}\n\n"
                f"work_dir icerigi: {wd_contents}\n\n"
                f"stderr:\n{proc.stderr[:600]}")

        cb(f"[4/5] Yukleniyor: {os.path.basename(result_path)}")
        result_img = _load_result(result_path, is_rgb, h, w)

        bg_model = None
        for f in new_files:
            if f != result_path and "background" in os.path.basename(f).lower():
                try: bg_model = _load_result(f, is_rgb, h, w)
                except Exception: pass
                break

        cb("[5/5] Tamam!")
        key = "denoised" if command == "denoising" else "background_removed"
        return {
            key: result_img,
            "background_removed": result_img,
            "background_model":   bg_model,
            "exe_used":           exe_path,
        }

    finally:
        # work_dir'i temizle (input + output)
        shutil.rmtree(work_dir, ignore_errors=True)


def _snapshot(dirs):
    """Verilen klasorlerdeki tum TIF/FITS dosyalarini kaydet."""
    found = set()
    for d in dirs:
        if not os.path.isdir(d): continue
        for ext in ("*.tif","*.tiff","*.fits","*.fit","*.png"):
            found.update(glob.glob(os.path.join(d, ext)))
    return found


def _pick_result(new_files, work_dir, inp_path, search_dirs):
    """En iyi sonuc dosyasini sec."""
    # 1. Yeni, background olmayan dosya
    for f in new_files:
        bn = os.path.basename(f).lower()
        if "background" not in bn and "bg_" not in bn and f != inp_path:
            return f
    # 2. Herhangi yeni dosya
    for f in new_files:
        if f != inp_path:
            return f
    # 3. Bilinen isimler
    for d in search_dirs:
        for name in ["result.tif","result.fits","output.tif","output.fits",
                     "input_GraXpert.tif","input_GraXpert.fits"]:
            p = os.path.join(d, name)
            if os.path.isfile(p) and p != inp_path:
                return p
    return None


def _save_tiff16(img, path, is_rgb):
    img16 = (np.clip(img, 0, 1) * 65535).astype(np.uint16)
    try:
        import tifffile
        ph = "rgb" if is_rgb else "minisblack"
        tifffile.imwrite(path, img16, photometric=ph, compression=None)
        return
    except ImportError:
        pass
    params = [cv2.IMWRITE_TIFF_COMPRESSION, 1]
    if is_rgb:
        img16 = cv2.cvtColor(img16, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img16, params)


def _load_result(path, want_rgb, h, w):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".fits",".fit"):
        from astropy.io import fits
        data = fits.getdata(path).astype(np.float32)
        if data.ndim == 3 and data.shape[0] in (1,3):
            data = np.moveaxis(data, 0, -1)
        if data.max() > 1.0: data /= data.max()
        out = data
    else:
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if raw is None: raise RuntimeError(f"Okunamadi: {path}")
        if raw.dtype == np.uint16:   out = raw.astype(np.float32) / 65535.0
        elif raw.dtype == np.uint8:  out = raw.astype(np.float32) / 255.0
        else:
            out = raw.astype(np.float32)
            if out.max() > 1.0: out /= out.max()
        if out.ndim == 3:
            if out.shape[2] == 3:   out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            elif out.shape[2] >= 4: out = cv2.cvtColor(out, cv2.COLOR_BGRA2RGB)
    if want_rgb and out.ndim == 2:   out = np.stack([out]*3, 2)
    if not want_rgb and out.ndim == 3: out = out.mean(axis=2)
    if out.shape[:2] != (h, w):
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(out, 0, 1).astype(np.float32)


def find_graxpert_exe() -> str:
    candidates = []
    if sys.platform == "win32":
        roots = [
            os.environ.get("PROGRAMFILES",      r"C:\Program Files"),
            os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"),
            os.path.expanduser("~\\Desktop"),
            os.path.expanduser("~\\Downloads"),
            os.path.expanduser("~"),
            r"C:\GraXpert", r"D:\GraXpert", r"D:\a\GraXpert",
        ]
        names   = ["GraXpert-cli.exe","GraXpert.exe","graxpert.exe"]
        subdirs = ["","GraXpert","GraXpert2","GraXpert3","graxpert"]
        for root in roots:
            for sub in subdirs:
                for name in names:
                    candidates.append(os.path.join(root, sub, name))
    else:
        for d in ["/usr/local/bin","/usr/bin",
                  os.path.expanduser("~/bin"),
                  os.path.expanduser("~/GraXpert")]:
            for name in ["GraXpert-cli","GraXpert","graxpert"]:
                candidates.append(os.path.join(d, name))
    for p in candidates:
        if os.path.isfile(p): return p
    return ""
