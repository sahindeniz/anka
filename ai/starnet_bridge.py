"""
Astro Mastro Pro — StarNet++ Bridge  (v2 — production ready)

Supports:
  • StarNet++ v1  (starnet++.exe)  — CLI: exe input.tif output.tif [stride]
  • StarNet2      (StarNet2.exe)   — CLI: exe input.tif output.tif [stride]
  • StarNet2 RGB  (separate R/G/B channels automatically)

The bridge:
  1. Saves input as 16-bit TIFF (temp file)
  2. Calls StarNet++ executable
  3. Reads output TIFF
  4. Returns starless, stars_only, star_mask as float32 [0,1]
"""

import os, sys, subprocess, tempfile, shutil
import numpy as np
import cv2


def run_starnet(image: np.ndarray,
                exe_path: str,
                stride: int = 256,
                use_gpu: bool = False,
                progress_cb=None) -> dict:
    """
    Run StarNet++ on image. Returns dict with starless/stars_only/star_mask.
    progress_cb: callable(str) — called with status messages
    """

    def cb(msg):
        if progress_cb:
            progress_cb(msg)

    # ── Validate exe ──────────────────────────────────────────────────────
    exe_path = str(exe_path).strip()
    if not exe_path:
        raise FileNotFoundError(
            "StarNet++ executable path is empty.\n"
            "Go to Settings → StarNet++ and set the path to starnet++.exe or StarNet2.exe")
    if not os.path.isfile(exe_path):
        raise FileNotFoundError(
            f"StarNet++ executable not found:\n  {exe_path}\n\n"
            "Check the path in Settings → StarNet++.")

    img = np.clip(image, 0, 1).astype(np.float32)
    h, w = img.shape[:2]
    is_rgb = (img.ndim == 3)
    exe_name = os.path.basename(exe_path).lower()

    cb(f"[1/4] Preparing image ({w}×{h}, {'RGB' if is_rgb else 'Gray'})…")

    tmpdir = tempfile.mkdtemp(prefix="amp_sn_")
    try:
        # ── Save 16-bit TIFF ──────────────────────────────────────────────
        inp  = os.path.join(tmpdir, "input.tif")
        outp = os.path.join(tmpdir, "starless.tif")
        _save_tiff16(img, inp, is_rgb)
        cb(f"[1/4] Saved temp TIFF: {os.path.basename(inp)}")

        # ── Detect StarNet version and build command ───────────────────────
        is_v2 = any(x in exe_name for x in ["starnet2", "starnet_v2", "starnet-v2"])
        cmd = _build_cmd(exe_path, inp, outp, stride, use_gpu, is_v2)
        cb(f"[2/4] Running: {' '.join(os.path.basename(c) for c in cmd)}")

        # ── Execute ───────────────────────────────────────────────────────
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=900,        # 15 min
                cwd=os.path.dirname(exe_path),  # run from exe dir (DLL lookup)
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("StarNet++ timed out after 15 minutes.")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Could not start StarNet++:\n{exe_path}\n\n"
                "Make sure the path is correct and the file is executable.")

        # ── Check return code ─────────────────────────────────────────────
        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            stdout = (proc.stdout or "").strip()
            detail = stderr or stdout or "(no output)"
            raise RuntimeError(
                f"StarNet++ exited with code {proc.returncode}.\n\n"
                f"Output:\n{detail[:1000]}")

        cb("[3/4] StarNet++ finished — loading result…")

        # ── Load output ───────────────────────────────────────────────────
        if not os.path.isfile(outp):
            # StarNet2 sometimes writes a different name — search tmpdir
            tifs = [f for f in os.listdir(tmpdir)
                    if f.lower().endswith(".tif") and f != "input.tif"]
            if tifs:
                outp = os.path.join(tmpdir, tifs[0])
            else:
                raise RuntimeError(
                    "StarNet++ finished but no output file was created.\n"
                    f"Expected: {outp}\n\n"
                    "Try a different stride value or check StarNet++ manually.")

        starless = _load_tiff(outp, is_rgb, h, w)

        cb("[4/4] Computing star mask…")
        stars_only = np.clip(img - starless, 0, 1).astype(np.float32)
        # Star mask: where difference > small threshold
        diff = np.abs(img - starless)
        if diff.ndim == 3:
            star_mask = (diff.max(axis=2) > 0.02).astype(np.float32)
        else:
            star_mask = (diff > 0.02).astype(np.float32)

        cb(f"Done ✅ — StarNet++ completed successfully")
        return {
            'starless':   starless,
            'stars_only': stars_only,
            'star_mask':  star_mask,
            'exe_used':   exe_path,
        }

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _save_tiff16(img: np.ndarray, path: str, is_rgb: bool):
    """Save float32 [0,1] image as 16-bit TIFF."""
    img16 = (np.clip(img, 0, 1) * 65535).astype(np.uint16)
    if is_rgb:
        # OpenCV needs BGR
        bgr = cv2.cvtColor(img16, cv2.COLOR_RGB2BGR)
        ok = cv2.imwrite(path, bgr)
    else:
        ok = cv2.imwrite(path, img16)
    if not ok:
        raise RuntimeError(f"Failed to write temp TIFF: {path}")


def _build_cmd(exe: str, inp: str, out: str,
               stride: int, use_gpu: bool, is_v2: bool) -> list:
    """Build StarNet++ command line."""
    cmd = [exe, inp, out, str(int(stride))]
    if is_v2 and use_gpu:
        cmd.append("--use-GPU")
    return cmd


def _load_tiff(path: str, want_rgb: bool, h: int, w: int) -> np.ndarray:
    """Load StarNet++ output TIFF and normalise to float32 [0,1]."""
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise RuntimeError(f"Could not read StarNet++ output:\n{path}")

    # Normalise
    if raw.dtype == np.uint16:
        out = raw.astype(np.float32) / 65535.0
    elif raw.dtype == np.uint8:
        out = raw.astype(np.float32) / 255.0
    else:
        out = raw.astype(np.float32)
        mx = out.max()
        if mx > 1.0:
            out /= mx

    # Channel conversion
    if out.ndim == 3:
        if out.shape[2] == 3:
            out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        elif out.shape[2] == 4:
            out = cv2.cvtColor(out, cv2.COLOR_BGRA2RGB)
    elif out.ndim == 2 and want_rgb:
        out = np.stack([out] * 3, axis=2)

    if not want_rgb and out.ndim == 3:
        out = out.mean(axis=2)

    # Resize if shape mismatch
    if out.shape[:2] != (h, w):
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)

    return np.clip(out, 0, 1).astype(np.float32)


def find_starnet_exe() -> str:
    """Auto-detect StarNet++ executable. Returns '' if not found."""
    candidates = []
    if sys.platform == "win32":
        roots = [
            os.environ.get("PROGRAMFILES",      r"C:\Program Files"),
            os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"),
            os.path.expanduser("~\\Desktop"),
            os.path.expanduser("~\\Downloads"),
            os.path.expanduser("~"),
        ]
        names = ["starnet++.exe","StarNet2.exe","starnet2.exe",
                 "StarNet++.exe","starnet_v2.exe","StarNetv2.exe"]
        subdirs = ["","StarNet","StarNet++","StarNet2","starnet","starnet2"]
        for root in roots:
            for sub in subdirs:
                for name in names:
                    candidates.append(os.path.join(root, sub, name))
    else:
        for d in ["/usr/local/bin","/usr/bin",
                  os.path.expanduser("~/bin"),
                  os.path.expanduser("~/StarNet"),
                  os.path.expanduser("~/starnet++"),
                  os.path.expanduser("~/starnet2")]:
            for name in ["starnet++","StarNet2","starnet2","starnet_v2"]:
                candidates.append(os.path.join(d, name))
    for p in candidates:
        if os.path.isfile(p):
            return p
    return ""
