"""Audio loading and loop generation."""
from __future__ import annotations

import math
import subprocess
from pathlib import Path

import librosa
import numpy as np

from .exceptions import LoopProcessingError


def load_audio_mono(path: Path) -> tuple[np.ndarray, int]:
    """Load an audio (or video) file as mono float32 samples."""
    try:
        samples, sr = librosa.load(path, sr=None, mono=True)
    except Exception as exc:  # pragma: no cover - third-party error text
        raise LoopProcessingError(f"Failed to read audio: {path}") from exc
    if samples.size < 2:
        raise LoopProcessingError("Audio file too short to process")
    return samples.astype(np.float32, copy=False), int(sr)


def clamp_sample_index(idx: int, total: int) -> int:
    return max(0, min(total - 1, idx))


def find_zero_cross(samples: np.ndarray, target_idx: int, radius: int) -> int:
    radius = max(1, radius)
    total = samples.size
    start = clamp_sample_index(target_idx - radius, total - 1)
    end = clamp_sample_index(target_idx + radius, total - 2)
    best_idx = target_idx
    best_dist = math.inf
    for i in range(start, end + 1):
        s1 = samples[i]
        s2 = samples[i + 1]
        if s1 == 0.0 or s2 == 0.0 or (s1 > 0 and s2 < 0) or (s1 < 0 and s2 > 0):
            dist = abs(i - target_idx)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
    return clamp_sample_index(best_idx, total - 1)


def extract_loop(samples: np.ndarray, sr: int, start_sec: float, end_sec: float, radius: int) -> np.ndarray:
    if end_sec <= start_sec:
        raise LoopProcessingError("End time must be greater than start time")
    start_idx = int(start_sec * sr)
    end_idx = int(end_sec * sr)
    total = samples.size
    start_idx = clamp_sample_index(start_idx, total)
    end_idx = clamp_sample_index(end_idx, total)
    if end_idx <= start_idx:
        raise LoopProcessingError("Invalid indices after clamping")
    zero_start = find_zero_cross(samples, start_idx, radius)
    zero_end = find_zero_cross(samples, end_idx, radius)
    if zero_end <= zero_start:
        raise LoopProcessingError("Zero-cross search collapsed loop window")
    loop = samples[zero_start:zero_end].copy()
    if loop.size < sr // 20:  # <50ms guard
        raise LoopProcessingError("Looped segment is too short")
    return loop


def encode_vorbis(loop_samples: np.ndarray, sr: int, quality: int, output_path: Path, ffmpeg_path: str = "ffmpeg") -> None:
    if loop_samples.ndim != 1:
        raise ValueError("Expected mono samples")
    quality = max(0, min(10, quality))
    pcm_bytes = loop_samples.astype(np.float32, copy=False).tobytes()
    cmd = [
        ffmpeg_path,
        "-y",
        "-f",
        "f32le",
        "-ar",
        str(sr),
        "-ac",
        "1",
        "-i",
        "pipe:0",
        "-c:a",
        "libvorbis",
        "-qscale:a",
        str(quality),
        "-f",
        "ogg",
        "pipe:1",
    ]
    try:
        proc = subprocess.run(
            cmd,
            input=pcm_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = exc.stderr.decode("utf-8", errors="ignore") if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        raise LoopProcessingError(f"FFmpeg Vorbis encode failed: {detail.strip()}") from exc
    output_path.write_bytes(proc.stdout)
