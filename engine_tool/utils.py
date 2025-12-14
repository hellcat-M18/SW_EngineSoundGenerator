"""Utility helpers for the engine tool."""
from __future__ import annotations

import re
from pathlib import Path

INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*]')


def slugify(name: str) -> str:
    """Convert the engine name into a filesystem-friendly slug."""
    text = name.strip().lower()
    text = re.sub(r"[^a-z0-9_-]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "engine"


def ensure_path(value: str | Path) -> Path:
    """Return a Path instance."""
    return value if isinstance(value, Path) else Path(value)


def sanitize_filename_component(name: str, fallback: str = "engine") -> str:
    """Remove characters that cannot appear in filenames."""
    text = name.strip()
    if not text:
        text = fallback
    text = INVALID_FILENAME_CHARS.sub("_", text)
    return text or fallback
