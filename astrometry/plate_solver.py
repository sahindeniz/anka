"""
Astro Mastro Pro — Plate Solver Coordinator
============================================
ASTAP bridge üzerinde yüksek seviyeli plate solving koordinatörü.
GUI'den çağrılır.
"""

import numpy as np
from ai.astap_bridge import (
    solve_image, find_astap_exe, astap_available,
    format_result
)

__all__ = [
    "solve_image", "find_astap_exe", "astap_available",
    "format_result", "PlateSolveWorker",
]


class PlateSolveWorker:
    """
    Plate solve işini arka planda çalıştırmak için wrapper.
    QThread uyumlu — dışarıdan finished/error callback alır.
    """

    def __init__(self, image: np.ndarray, settings: dict,
                 ra_hint=None, dec_hint=None, fov_hint=None,
                 progress_cb=None):
        self.image       = image
        self.settings    = settings
        self.ra_hint     = ra_hint
        self.dec_hint    = dec_hint
        self.fov_hint    = fov_hint
        self.progress_cb = progress_cb

    def run(self) -> dict:
        s = self.settings
        return solve_image(
            image         = self.image,
            astap_exe     = s.get("astap_exe", ""),
            db_path       = s.get("astap_db", ""),
            search_radius = float(s.get("astap_radius", 30.0)),
            downsample    = int(s.get("astap_downsample", 0)),
            min_stars     = int(s.get("astap_min_stars", 10)),
            timeout       = int(s.get("astap_timeout", 120)),
            ra_hint       = self.ra_hint,
            dec_hint      = self.dec_hint,
            fov_hint      = self.fov_hint,
            progress_cb   = self.progress_cb,
        )
