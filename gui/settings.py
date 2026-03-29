"""
Astro Mastro Pro — Settings Manager
Reads/writes settings.json in the project root.
"""
import os, json

_DEFAULT = {
    # StarNet++
    "starnet_exe": "",
    "starnet_stride": 256,
    "starnet_use_gpu": False,

    # GraXpert
    "graxpert_exe": "",
    "graxpert_smoothing": 0.35,
    "graxpert_ai_version": "latest",
    "graxpert_correction": "Subtraction",
    "graxpert_denoise_strength": 0.8,

    # ASTAP Plate Solving
    "astap_exe": "",
    "astap_db": "",
    "astap_radius": 30.0,
    "astap_min_stars": 10,
    "astap_timeout": 120,
    "astap_downsample": 0,

    # Display
    "theme": "dark",
    "font_size": 10,
    "panel_width": 415,
    "show_history": True,
    "canvas_interp": "nearest",

    # Processing defaults
    "default_stretch": "auto_stf",
    "default_bg": "graxpert",
    "num_threads": 4,

    # Paths & recent
    "last_open_dir": "",
    "last_save_dir": "",
    "last_open_file": "",
    "output_format": "FITS",
    "recent_files": [],

    # Update
    "check_updates_on_startup": False,
}

_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "settings.json")


def load():
    s = dict(_DEFAULT)
    try:
        if os.path.exists(_PATH):
            with open(_PATH) as f:
                s.update(json.load(f))
    except Exception:
        pass
    return s


def save(data: dict):
    try:
        merged = dict(_DEFAULT)
        merged.update(data)
        # recent_files listesini koru (merged.update string'e çevirmez ama emin ol)
        if "recent_files" in data and isinstance(data["recent_files"], list):
            merged["recent_files"] = data["recent_files"][:15]
        with open(_PATH, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[Settings] save failed: {e}")


def get(key, default=None):
    return load().get(key, default if default is not None else _DEFAULT.get(key))


def set(key, value):
    s = load()
    s[key] = value
    save(s)
