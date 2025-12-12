"""Utility helpers for the engine tool."""
from __future__ import annotations

import re
from pathlib import Path


def slugify(name: str) -> str:
    """Convert the engine name into a filesystem-friendly slug."""
    text = name.strip().lower()
    text = re.sub(r"[^a-z0-9_-]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "engine"


def ensure_path(value: str | Path) -> Path:
    """Return a Path instance."""
    return value if isinstance(value, Path) else Path(value)
