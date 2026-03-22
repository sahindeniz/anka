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
APP_VERSION    = "1.2.0"          # Mevcut sürüm
APP_AUTHOR     = "Deniz"
APP_BUILD_DATE = "2026-03-22"

# GitHub güncelleme kaynağı
GITHUB_REPO    = "sahindeniz/anka"
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
    Önce releases/latest, yoksa tags'e bakar.

    Dönüş:
      {
        "available"  : bool,
        "version"    : str,
        "url"        : str,
        "download_url": str,
        "notes"      : str,
        "error"      : str|None,
      }
    """
    if not GITHUB_REPO:
        return {
            "available": False,
            "version":   APP_VERSION,
            "url":       "",
            "download_url": "",
            "notes":     "",
            "error":     "GitHub deposu ayarlı değil.",
        }

    import urllib.request
    import json

    headers = {
        "Accept":     "application/vnd.github+json",
        "User-Agent": f"AstroMaestroPro/{APP_VERSION}",
    }

    def _fetch(url):
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    # ── 1) Releases/latest ──
    try:
        data = _fetch(GITHUB_API_URL)
        remote_ver   = data.get("tag_name", "").lstrip("vV")
        release_url  = data.get("html_url", "")
        notes        = data.get("body", "") or ""
        notes        = notes[:2000]

        download_url = ""
        for asset in data.get("assets", []):
            name = asset.get("name", "").lower()
            if name.endswith(".zip"):
                download_url = asset.get("browser_download_url", "")
                break
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
    except Exception:
        pass  # Release yok — tags'e bak

    # ── 2) Tags fallback ──
    try:
        tags_url = f"https://api.github.com/repos/{GITHUB_REPO}/tags?per_page=5"
        tags = _fetch(tags_url)
        if tags and len(tags) > 0:
            latest_tag = tags[0].get("name", "").lstrip("vV")
            repo_url = f"https://github.com/{GITHUB_REPO}"
            zip_url  = f"{repo_url}/archive/refs/tags/{tags[0].get('name','')}.zip"
            return {
                "available":    is_newer(latest_tag),
                "version":      latest_tag,
                "url":          f"{repo_url}/releases",
                "download_url": zip_url,
                "notes":        "",
                "error":        None,
            }
    except Exception:
        pass

    # ── 3) Commits fallback (son commit'i kontrol et) ──
    try:
        commits_url = f"https://api.github.com/repos/{GITHUB_REPO}/commits?per_page=1"
        commits = _fetch(commits_url)
        if commits and len(commits) > 0:
            sha = commits[0].get("sha", "")[:7]
            msg = commits[0].get("commit", {}).get("message", "")[:200]
            repo_url = f"https://github.com/{GITHUB_REPO}"
            return {
                "available":    False,
                "version":      f"{APP_VERSION} ({sha})",
                "url":          repo_url,
                "download_url": f"{repo_url}/archive/refs/heads/main.zip",
                "notes":        f"Son commit: {msg}",
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

    return {
        "available": False,
        "version":   APP_VERSION,
        "url":       f"https://github.com/{GITHUB_REPO}",
        "download_url": "",
        "notes":     "",
        "error":     None,
    }
