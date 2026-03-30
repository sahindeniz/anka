"""Windows launcher shortcut helpers for the Astromastro desktop entry."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


SHORTCUT_NAME = "Astromastro"
SHORTCUT_FILENAME = f"{SHORTCUT_NAME}.lnk"
SHORTCUT_DESCRIPTION = SHORTCUT_NAME
LEGACY_SHORTCUTS = ("Astro Maestro Pro.lnk", "AstroMestro.lnk")
ICON_RELATIVE_PATH = Path("gui") / "icons" / "astromestro_space.ico"
LAUNCHER_RELATIVE_PATH = Path("setup_and_run.bat")


def get_app_root(module_file: str | None = None) -> Path:
    base = Path(module_file or __file__).resolve()
    return base.parents[1]


def get_launcher_path(app_root: str | os.PathLike[str] | None = None) -> Path:
    return Path(app_root or get_app_root()).resolve() / LAUNCHER_RELATIVE_PATH


def get_launcher_icon_path(app_root: str | os.PathLike[str] | None = None) -> Path:
    return Path(app_root or get_app_root()).resolve() / ICON_RELATIVE_PATH


def get_desktop_dir() -> Path:
    return Path.home() / "Desktop"


def _ps_quote(text: str) -> str:
    return text.replace("'", "''")


def build_shortcut_command(
    shortcut_path: str | os.PathLike[str],
    target_path: str | os.PathLike[str],
    working_directory: str | os.PathLike[str],
    icon_path: str | os.PathLike[str],
    description: str = SHORTCUT_DESCRIPTION,
) -> list[str]:
    shortcut = _ps_quote(str(Path(shortcut_path)))
    target = _ps_quote(str(Path(target_path)))
    workdir = _ps_quote(str(Path(working_directory)))
    icon = _ps_quote(str(Path(icon_path)))
    desc = _ps_quote(description)
    script = (
        f"$ws = New-Object -ComObject WScript.Shell; "
        f"$sc = $ws.CreateShortcut('{shortcut}'); "
        f"$sc.TargetPath = '{target}'; "
        f"$sc.WorkingDirectory = '{workdir}'; "
        f"$sc.Description = '{desc}'; "
        f"$sc.IconLocation = '{icon},0'; "
        f"$sc.Save()"
    )
    return [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        script,
    ]


def ensure_shortcut_in_dir(
    app_root: str | os.PathLike[str] | None = None,
    target_dir: str | os.PathLike[str] | None = None,
) -> Path | None:
    if os.name != "nt":
        return None

    root = Path(app_root or get_app_root()).resolve()
    target = Path(target_dir or root).expanduser().resolve()
    if not target.is_dir():
        return None

    launcher = get_launcher_path(root)
    icon = get_launcher_icon_path(root)
    if not launcher.is_file():
        raise FileNotFoundError(f"Could not find launcher at {launcher}")
    if not icon.is_file():
        raise FileNotFoundError(f"Could not find icon at {icon}")

    shortcut = target / SHORTCUT_FILENAME
    for legacy_name in LEGACY_SHORTCUTS:
        legacy = target / legacy_name
        if legacy != shortcut and legacy.exists():
            legacy.unlink()

    subprocess.run(
        build_shortcut_command(shortcut, launcher, root, icon),
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return shortcut


def ensure_local_shortcut(
    app_root: str | os.PathLike[str] | None = None,
) -> Path | None:
    root = Path(app_root or get_app_root()).resolve()
    return ensure_shortcut_in_dir(root, root)


def ensure_desktop_shortcut(
    app_root: str | os.PathLike[str] | None = None,
    desktop_dir: str | os.PathLike[str] | None = None,
) -> Path | None:
    return ensure_shortcut_in_dir(app_root, desktop_dir or get_desktop_dir())


def main() -> int:
    created = []
    for creator in (ensure_local_shortcut, ensure_desktop_shortcut):
        try:
            shortcut = creator()
        except Exception:
            shortcut = None
        if shortcut is not None:
            created.append(shortcut)
    if not created:
        return 1
    for shortcut in created:
        print(shortcut)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
