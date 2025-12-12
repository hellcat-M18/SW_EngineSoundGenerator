"""Bootstrapper that ensures a local venv and runs the GUI inside it."""
from __future__ import annotations

import hashlib
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable

try:  # Tkinter is part of the stdlib, but guard in case it's unavailable
    import tkinter as tk
    from tkinter.scrolledtext import ScrolledText
except Exception:  # pragma: no cover - platform-dependent fallback
    tk = None  # type: ignore
    ScrolledText = None  # type: ignore

PACKAGE_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = PACKAGE_DIR.parent
VENV_DIR = WORKSPACE_ROOT / ".venv"
REQUIREMENTS_FILE = WORKSPACE_ROOT / "requirements.txt"
STAMP_FILE = VENV_DIR / ".requirements.sha256"


def _venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _compute_requirements_hash() -> str:
    if not REQUIREMENTS_FILE.exists():
        return ""
    data = REQUIREMENTS_FILE.read_bytes()
    return hashlib.sha256(data).hexdigest()


def _read_stamp() -> str:
    if not STAMP_FILE.exists():
        return ""
    return STAMP_FILE.read_text(encoding="utf-8").strip()


def _write_stamp(value: str) -> None:
    STAMP_FILE.write_text(value, encoding="utf-8")


def _run_with_output(cmd: list[str], emit: Callable[[str], None]) -> None:
    emit("$ " + " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        emit(line.rstrip())
    ret = proc.wait()
    if ret != 0:  # pragma: no cover - handled via CalledProcessError semantics
        raise subprocess.CalledProcessError(ret, cmd)


def ensure_venv(log_callback: Callable[[str], None] | None = None) -> Path:
    def emit(message: str) -> None:
        print(message)
        if log_callback:
            log_callback(message)

    python_path = _venv_python()
    if not python_path.exists():
        emit("[bootstrap] Creating virtual environment…")
        _run_with_output([sys.executable, "-m", "venv", str(VENV_DIR)], emit)
    desired_hash = _compute_requirements_hash()
    if desired_hash and _read_stamp() != desired_hash:
        emit("[bootstrap] Installing/refreshing dependencies…")
        _run_with_output([str(python_path), "-m", "pip", "install", "--upgrade", "pip"], emit)
        _run_with_output([str(python_path), "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)], emit)
        _write_stamp(desired_hash)
    else:
        emit("[bootstrap] Dependencies already up-to-date.")
    return python_path


def _run_setup_window() -> Path:
    if tk is None or os.environ.get("ENGINE_TOOL_NO_BOOTSTRAP_GUI") == "1":
        return ensure_venv()

    result: dict[str, object | None] = {"python_path": None, "error": None}
    messages: queue.Queue[str | None] = queue.Queue()

    def log_from_worker(msg: str) -> None:
        messages.put(msg)

    def worker() -> None:
        try:
            messages.put("[bootstrap] Checking virtual environment…")
            python_path = ensure_venv(log_callback=log_from_worker)
            result["python_path"] = python_path
        except Exception as exc:  # pragma: no cover - best effort safeguard
            result["error"] = exc
        finally:
            messages.put(None)

    threading.Thread(target=worker, daemon=True).start()

    root = tk.Tk()
    root.title("Stormworks Engine Tool - Setup")
    root.geometry("640x360")
    root.resizable(False, False)
    root.protocol("WM_DELETE_WINDOW", lambda: None)  # disable close during setup

    label = tk.Label(root, text="初回セットアップを実行中です。閉じずにお待ちください…")
    label.pack(padx=12, pady=8)

    text_widget = ScrolledText(root, height=15, state="disabled")
    text_widget.pack(fill="both", expand=True, padx=12, pady=(0, 12))

    def append_line(line: str) -> None:
        text_widget.configure(state="normal")
        text_widget.insert("end", line + "\n")
        text_widget.see("end")
        text_widget.configure(state="disabled")

    def poll_queue() -> None:
        try:
            while True:
                item = messages.get_nowait()
                if item is None:
                    root.destroy()
                    return
                append_line(item)
        except queue.Empty:
            pass
        root.after(100, poll_queue)

    root.after(100, poll_queue)
    root.mainloop()

    if result["error"]:
        raise result["error"]  # type: ignore[misc]
    return result["python_path"]  # type: ignore[return-value]


def launch_gui(python_path: Path) -> None:
    print("[bootstrap] Launching GUI inside .venv …")
    subprocess.run([str(python_path), "-m", "engine_tool.gui"], check=True)


def main() -> None:
    python_path = _run_setup_window()
    launch_gui(python_path)


if __name__ == "__main__":
    main()
