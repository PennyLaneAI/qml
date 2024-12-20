from pathlib import Path
import shutil


def copy_any(src: Path, dest: Path, exist_ok: bool = False):
    """Copy `src` to `path`. If `src` is a directory, copy it recursively."""
    if src.is_dir():
        shutil.copytree(src, dest, dirs_exist_ok=exist_ok)
    else:
        shutil.copy2(src, dest)
