# processing/auto_pipeline.py
"""
Auto-pipeline: load_settings / run_auto_pipeline stubs.
LastProcessPanel tarafından kullanılır.
"""
import json
import os

_SETTINGS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "settings.json")

_DEFAULTS = {
    "pipeline_enabled": True,
    "stretch_type": "arcsinh",
    "arcsinh_factor": 4.0,
    "color_saturation_boost": 1.30,
    "sharpen_amount": 1.15,
}


def load_settings() -> dict:
    """Load pipeline settings from settings.json, falling back to defaults."""
    try:
        with open(_SETTINGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        pipeline = data.get("pipeline", {})
        merged = {**_DEFAULTS, **pipeline}
        return merged
    except Exception:
        return dict(_DEFAULTS)


def run_auto_pipeline(input_path: str, *, custom_settings: dict | None = None) -> str:
    """
    Run the auto-processing pipeline on *input_path*.
    Returns the path to the processed output file.
    """
    import cv2
    import numpy as np

    settings = custom_settings or load_settings()
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Görüntü okunamadı: {input_path}")

    # --- Stretch ---
    img_f = img.astype(np.float32)
    if img_f.max() > 1.0:
        img_f /= img_f.max() or 1.0

    stretch_type = settings.get("stretch_type", "arcsinh")
    if stretch_type == "arcsinh":
        factor = float(settings.get("arcsinh_factor", 4.0))
        img_f = np.arcsinh(img_f * factor) / np.arcsinh(np.float32(factor))
    elif stretch_type == "clahe":
        gray = (img_f * 255).astype(np.uint8)
        if gray.ndim == 3:
            lab = cv2.cvtColor(gray, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            gray = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        img_f = gray.astype(np.float32) / 255.0
    # percentile: no-op (already 0-1)

    # --- Color saturation boost ---
    sat_boost = float(settings.get("color_saturation_boost", 1.30))
    if img_f.ndim == 3 and sat_boost != 1.0:
        img_u8 = (img_f * 255).astype(np.uint8)
        hsv = cv2.cvtColor(img_u8, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.float32) * sat_boost, 0, 255).astype(np.uint8)
        img_f = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0

    # --- Sharpen ---
    sharpen_amt = float(settings.get("sharpen_amount", 1.15))
    if sharpen_amt > 0:
        blurred = cv2.GaussianBlur(img_f, (0, 0), 2.0)
        img_f = cv2.addWeighted(img_f, 1.0 + sharpen_amt, blurred, -sharpen_amt, 0)

    img_f = np.clip(img_f, 0, 1)

    # --- Save ---
    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_processed{ext or '.tif'}"
    out = (img_f * 255).astype(np.uint8) if img.dtype == np.uint8 else (img_f * 65535).astype(np.uint16)
    if not cv2.imwrite(output_path, out):
        raise IOError(f"Çıktı yazılamadı: {output_path}")
    return output_path
