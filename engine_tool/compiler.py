"""Wrapper around component_mod_compiler.com."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable

from .exceptions import CompilerError


def run_component_compiler(executable: Path, xml_path: Path, ogg_files: Iterable[Path], cwd: Path | None = None) -> tuple[str, str]:
    cmd = [str(executable), str(xml_path), *map(str, ogg_files)]
    try:
        proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        raise CompilerError(exc.stderr or exc.stdout or "component_mod_compiler failed") from exc
    except OSError as exc:
        raise CompilerError(str(exc)) from exc
    return proc.stdout, proc.stderr
