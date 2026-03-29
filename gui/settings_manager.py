"""
Astro Mastro Pro — Settings Manager
settings.json okuma/yazma
"""
import json
import os

DEFAULTS = {
    "starnet_exe": "",
    "starnet_stride": 256,
    "starnet_use_gpu": False,
    "graxpert_exe": "",
    "graxpert_smoothing": 0.35,
    "graxpert_ai_version": "latest",
    "graxpert_correction": "Subtraction",
    "graxpert_denoise_strength": 0.8,
    "theme": "dark",
    "font_size": 10,
    "panel_width": 415,
    "show_history": True,
    "canvas_interp": "nearest",
    "default_stretch": "auto_stf",
    "default_bg": "graxpert",
    "num_threads": 4,
    "last_open_dir": "",
    "last_save_dir": "",
    "output_format": "FITS",
    "astap_exe": "",
    "astap_db": "",
    "astap_radius": 30.0,
    "astap_downsample": 0,
    "astap_min_stars": 10,
    "astap_timeout": 120,
    "last_open_file": "",
}

_settings_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "settings.json")
_current = dict(DEFAULTS)


def load():
    global _current
    _current = dict(DEFAULTS)
    if os.path.exists(_settings_path):
        try:
            with open(_settings_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            _current.update(data)
        except Exception:
            pass
    return _current


def save():
    try:
        with open(_settings_path, "w", encoding="utf-8") as f:
            json.dump(_current, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def get(key, default=None):
    return _current.get(key, default)


def set(key, value):
    _current[key] = value


# İlk yükle
load()
