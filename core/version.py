"""
Astro Maestro Pro — Versiyon Yönetimi
======================================
Merkezi versiyon bilgisi ve GitHub üzerinden güncelleme kontrolü.

Güncelleme akışı:
  1. GitHub API'den en son release bilgisi çekilir
  2. Mevcut versiyon ile karşılaştırılır
  3. Yeni sürüm varsa → UpdateDialog gösterilir
  4. Kullanıcı onaylarsa → ZIP indirilir, çıkarılır, program yeniden başlar
"""

# ─── Versiyon ─────────────────────────────────────────────────────────────────
APP_NAME       = "Astro Maestro Pro"
APP_VERSION    = "1.0.0"          # Mevcut sürüm
APP_AUTHOR     = "Deniz"
APP_BUILD_DATE = "2025-03-15"

# GitHub güncelleme kaynağı
# Boş bırakılırsa güncelleme kontrolü devre dışı
GITHUB_REPO    = ""   # örn: "deniz/astro-maestro-pro"
GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest" if GITHUB_REPO else ""

# ─── Changelog ───────────────────────────────────────────────────────────────
CHANGELOG = """
v1.0.0  (2025-03-15)
─────────────────────
• İlk kararlı sürüm
• VeraLux HyperMetric Stretch (HMS) v6 — Auto Log D, 27 sensör profili
• ASTAP Plate Solving — Siril tarzı tam dialog (D80/G17/H17 katalog desteği)
• GraXpert AI arka plan çıkarma entegrasyonu
• StarNet++ v2 / AI fallback yıldız ayırma
• DeepSkyStacker tarzı Stacking — kalibrasyon, hizalama, kalite skoru
• DSS-style Image Stacking — Bias/Dark/Flat kalibrasyon
• Photoshop-style Histogram Editörü — Levels, Curves, Adjustments
• Tam menü çubuğu (Dosya/Düzenle/Görünüm/İşlem/Araçlar/Yardım)
• Python Script Editörü — canlı sözdizimi vurgulama
• Star Recomposition — blend mode, hue/sat, luminosity mask
• Veralux Silentium, Revela, Vectra, Alchemy, Nox, StarComposer
• WCS overlay — RA/Dec ızgara, katalog annotasyon
• Son Açılanlar menüsü
• Update kontrol sistemi
"""

# ─── Yardımcı fonksiyonlar ────────────────────────────────────────────────────

def parse_version(v: str) -> tuple:
    """'1.2.3' → (1, 2, 3)"""
    try:
        parts = str(v).lstrip("vV").split(".")
        return tuple(int(x) for x in parts[:3])
    except Exception:
        return (0, 0, 0)


def is_newer(remote: str, current: str = APP_VERSION) -> bool:
    """remote sürümü current'tan yeni mi?"""
    return parse_version(remote) > parse_version(current)


def check_for_updates(timeout: int = 8) -> dict:
    """
    GitHub API'den güncel sürüm bilgisini çeker.

    Dönüş:
      {
        "available"  : bool,
        "version"    : str,      # en son sürüm (örn. "1.2.0")
        "url"        : str,      # release sayfası URL
        "download_url": str,     # ZIP indirme linki
        "notes"      : str,      # release notes
        "error"      : str|None,
      }
    """
    if not GITHUB_API_URL:
        return {
            "available": False,
            "version":   APP_VERSION,
            "url":       "",
            "download_url": "",
            "notes":     "",
            "error":     "GitHub deposu ayarlı değil.",
        }

    try:
        import urllib.request
        import json

        req = urllib.request.Request(
            GITHUB_API_URL,
            headers={
                "Accept":     "application/vnd.github+json",
                "User-Agent": f"AstroMaestroPro/{APP_VERSION}",
            }
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        remote_ver   = data.get("tag_name", "").lstrip("vV")
        release_url  = data.get("html_url", "")
        notes        = data.get("body", "")[:2000]

        # ZIP asset'ini bul
        download_url = ""
        for asset in data.get("assets", []):
            name = asset.get("name", "").lower()
            if name.endswith(".zip") and "astro" in name:
                download_url = asset.get("browser_download_url", "")
                break
        # ZIP yoksa kaynak ZIP'i kullan
        if not download_url:
            download_url = data.get("zipball_url", "")

        return {
            "available":    is_newer(remote_ver),
            "version":      remote_ver,
            "url":          release_url,
            "download_url": download_url,
            "notes":        notes,
            "error":        None,
        }

    except Exception as e:
        return {
            "available":    False,
            "version":      APP_VERSION,
            "url":          "",
            "download_url": "",
            "notes":        "",
            "error":        str(e),
        }
