"""Render the Stormworks template with dynamic values."""
from __future__ import annotations

from pathlib import Path


def render_template(template_path: Path, engine_name: str, engine_slug: str, output_path: Path) -> None:
    source = template_path.read_text(encoding="utf-8")
    rendered = source.replace("[engine-name]", f"[{engine_name}]")
    rendered = rendered.replace("engine-name", engine_slug)
    output_path.write_text(rendered, encoding="utf-8")
