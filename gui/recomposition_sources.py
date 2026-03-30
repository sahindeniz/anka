"""Source collection helpers for star recomposition pickers."""

from __future__ import annotations

import os
from typing import Any, List, Tuple

import numpy as np


RecompositionSource = Tuple[str, np.ndarray, str]


def collect_recomposition_sources(app: Any) -> List[RecompositionSource]:
    """Collect selectable sources from the app state.

    Order is tuned for recomposition work:
    active viewer, open tabs, original image, filmstrip, then history steps.
    """
    options: List[RecompositionSource] = []
    seen_ids: set[str] = set()

    def add_option(label: str, image: Any, source_id: str) -> None:
        if image is None:
            return
        source_key = str(source_id)
        if source_key in seen_ids:
            return
        seen_ids.add(source_key)
        options.append((label, np.array(image, copy=True), source_key))

    current = getattr(app, "_current", None)
    if current is not None:
        add_option("🖼  Mevcut görüntü (aktif)", current, "current")

    tab_entries = list(getattr(app, "_img_tab_data", []) or [])
    active_tab_index = -1
    tab_bar = getattr(app, "_img_tabs", None)
    if tab_bar is not None and hasattr(tab_bar, "currentIndex"):
        try:
            active_tab_index = int(tab_bar.currentIndex())
        except Exception:
            active_tab_index = -1

    for idx, entry in enumerate(tab_entries):
        image = entry.get("image") if isinstance(entry, dict) else None
        title = ""
        if isinstance(entry, dict):
            title = str(entry.get("title") or entry.get("key") or f"Tab {idx + 1}")
        prefix = "🗂  Sekme"
        if idx == active_tab_index:
            prefix = "🗂  Sekme (aktif)"
        add_option(prefix + f": {title}", image, f"tab_{idx}_{title}")

    original = getattr(app, "_orig", None)
    if original is not None:
        add_option("📌  Orijinal görüntü", original, "original")

    for entry in list(getattr(app, "_filmstrip_data", []) or []):
        if not isinstance(entry, dict):
            continue
        path = entry.get("path")
        image = entry.get("img")
        if path and image is not None:
            add_option(f"📂  {os.path.basename(path)}", image, f"filmstrip:{path}")

    history = list(getattr(app, "_history", []) or [])
    for idx, item in enumerate(history):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        label, hist_img = item
        if idx == 0 and original is not None:
            continue
        add_option(f"📜  Step {idx}: {label}", hist_img, f"history_{idx}")

    return options
