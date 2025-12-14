"""Wrapper around component_mod_compiler.com."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable

from .exceptions import CompilerError


def run_component_compiler(batch_file: Path, cwd: Path | None = None) -> tuple[str, str]:
    cmd = [str(batch_file)]
    try:
        proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True, shell=True)
    except subprocess.CalledProcessError as exc:
        raise CompilerError(exc.stderr or exc.stdout or "component_mod_compiler failed") from exc
    except OSError as exc:
        raise CompilerError(str(exc)) from exc
    return proc.stdout, proc.stderr
