"""Helpers for applying ZIP-based in-place updates."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import zipfile
from pathlib import Path

from core.launcher_shortcut import ensure_desktop_shortcut


_SKIP_COPY_NAMES = {".git", "__pycache__", "settings.json"}


def get_app_root(module_file: str | None = None) -> Path:
    base = Path(module_file or __file__).resolve()
    return base.parents[1]


def find_extracted_app_root(extract_root: str | os.PathLike[str]) -> Path:
    root = Path(extract_root)
    candidates = [root]
    candidates.extend(path for path in root.iterdir() if path.is_dir())

    for candidate in candidates:
        if _looks_like_app_root(candidate):
            return candidate

    for candidate in root.rglob("main.py"):
        app_root = candidate.parent
        if _looks_like_app_root(app_root):
            return app_root

    raise FileNotFoundError(f"Could not locate Astro Maestro Pro files under {root}")


def extract_release_archive(
    zip_path: str | os.PathLike[str],
    base_dir: str | os.PathLike[str] | None = None,
) -> tuple[Path, Path]:
    zip_file = Path(zip_path)
    staging_parent = Path(base_dir or zip_file.resolve().parent)
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix="astro_update_", dir=staging_parent))
    with zipfile.ZipFile(zip_file) as archive:
        archive.extractall(staging_dir)
    return staging_dir, find_extracted_app_root(staging_dir)


def merge_release_tree(source_root: str | os.PathLike[str], target_root: str | os.PathLike[str]) -> None:
    source = Path(source_root)
    target = Path(target_root)

    for src_path in source.rglob("*"):
        rel_path = src_path.relative_to(source)
        if any(part in _SKIP_COPY_NAMES for part in rel_path.parts):
            continue

        dst_path = target / rel_path
        if src_path.is_dir():
            dst_path.mkdir(parents=True, exist_ok=True)
            continue

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)


def write_update_helper(
    source_root: str | os.PathLike[str],
    target_root: str | os.PathLike[str],
    current_pid: int,
    module_root: str | os.PathLike[str] | None = None,
    helper_dir: str | os.PathLike[str] | None = None,
) -> Path:
    helper_parent = Path(helper_dir or Path(source_root).resolve().parent)
    helper_parent.mkdir(parents=True, exist_ok=True)
    helper_root = Path(tempfile.mkdtemp(prefix="astro_update_helper_", dir=helper_parent))
    helper_path = helper_root / "apply_update.py"
    module_path = Path(module_root or get_app_root()).resolve()

    script = textwrap.dedent(
        f"""
        import sys

        sys.path.insert(0, {str(module_path)!r})

        from core.self_update import apply_update_and_restart


        if __name__ == "__main__":
            apply_update_and_restart(
                {str(Path(source_root).resolve())!r},
                {str(Path(target_root).resolve())!r},
                {int(current_pid)},
            )
        """
    ).strip() + "\n"

    helper_path.write_text(script, encoding="utf-8")
    return helper_path


def launch_update_helper(
    source_root: str | os.PathLike[str],
    target_root: str | os.PathLike[str],
    current_pid: int,
    helper_dir: str | os.PathLike[str] | None = None,
) -> Path:
    helper_path = write_update_helper(
        source_root,
        target_root,
        current_pid,
        helper_dir=helper_dir,
    )
    popen_kwargs = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
    }

    if os.name == "nt":
        flags = 0
        flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        flags |= getattr(subprocess, "DETACHED_PROCESS", 0)
        flags |= getattr(subprocess, "CREATE_NO_WINDOW", 0)
        popen_kwargs["creationflags"] = flags
    else:
        popen_kwargs["start_new_session"] = True

    subprocess.Popen([sys.executable, str(helper_path)], **popen_kwargs)
    return helper_path


def apply_update_and_restart(
    source_root: str | os.PathLike[str],
    target_root: str | os.PathLike[str],
    current_pid: int,
    wait_timeout: float = 120.0,
) -> None:
    source = Path(source_root).resolve()
    target = Path(target_root).resolve()
    staging_dir = source.parent

    try:
        if not wait_for_process_exit(current_pid, timeout=wait_timeout):
            raise TimeoutError(f"Timed out while waiting for process {current_pid} to exit")
        merge_release_tree(source, target)
        restart_application(target)
    except Exception as exc:
        log_path = Path(tempfile.gettempdir()) / "AstroMaestroPro_update_error.txt"
        log_path.write_text(
            "\n".join(
                [
                    "Astro Maestro Pro update failed.",
                    f"Source: {source}",
                    f"Target: {target}",
                    f"Error: {exc!r}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        if os.name == "nt" and hasattr(os, "startfile"):
            os.startfile(str(log_path))
        raise
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


def wait_for_process_exit(pid: int, timeout: float = 120.0, poll_interval: float = 0.25) -> bool:
    if pid <= 0:
        return True

    deadline = time.monotonic() + max(timeout, 0.0)
    while time.monotonic() < deadline:
        if not _pid_exists(pid):
            return True
        time.sleep(poll_interval)

    return not _pid_exists(pid)


def restart_application(target_root: str | os.PathLike[str]) -> None:
    target = Path(target_root)
    try:
        ensure_desktop_shortcut(target)
    except Exception:
        pass

    launcher = target / "setup_and_run.bat"
    if os.name == "nt" and launcher.is_file() and hasattr(os, "startfile"):
        os.startfile(str(launcher))
        return

    main_py = target / "main.py"
    if not main_py.is_file():
        raise FileNotFoundError(f"Could not find launcher under {target}")

    popen_kwargs = {"cwd": str(target)}
    if os.name == "nt":
        flags = 0
        flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        flags |= getattr(subprocess, "DETACHED_PROCESS", 0)
        popen_kwargs["creationflags"] = flags
    else:
        popen_kwargs["start_new_session"] = True

    subprocess.Popen([sys.executable, str(main_py)], **popen_kwargs)


def _looks_like_app_root(path: Path) -> bool:
    return path.is_dir() and (path / "main.py").is_file() and (path / "core" / "version.py").is_file()


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True
