"""Audio loading and loop generation."""
from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from .exceptions import LoopProcessingError


@dataclass(slots=True)
class ZeroCross:
    position: float
    slope: int


def apply_zero_locked_crossfade(loop_samples: np.ndarray, fade_samples: int = 4096) -> np.ndarray:
    """Blend loop start/end using a long crossfade without forcing zeros."""
    if loop_samples.size <= 4:
        return loop_samples
    fade = min(fade_samples, loop_samples.size // 2)
    if fade <= 0:
        return loop_samples
    result = loop_samples.copy()
    start_slice = slice(0, fade)
    end_slice = slice(result.size - fade, result.size)
    start_region = result[start_slice].copy()
    end_region = result[end_slice].copy()
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    blended = end_region * (1.0 - ramp) + start_region * ramp
    result[start_slice] = blended
    result[end_slice] = blended[::-1]
    return result


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


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def find_zero_cross(samples: np.ndarray, target_idx: float, radius: int, preferred_slope: int | None = None) -> ZeroCross:
    """Locate the nearest zero crossing using linear interpolation, honoring slope preference."""
    radius = max(1, radius)
    total = samples.size
    if total < 2:
        return ZeroCross(0.0, 0)
    target_idx = float(target_idx)
    center = clamp_sample_index(int(round(target_idx)), total - 1)
    start = clamp_sample_index(center - radius, total - 1)
    end = clamp_sample_index(center + radius, total - 2)
    best_pos = float(center)
    best_slope = 0
    best_score = math.inf
    fallback_pos = best_pos
    fallback_slope = best_slope
    fallback_score = math.inf
    for i in range(start, end + 1):
        s1 = samples[i]
        s2 = samples[i + 1]
        candidate = None
        slope = 0
        if s1 == 0.0:
            candidate = float(i)
            prev = samples[i - 1] if i > 0 else s1
            nxt = samples[i + 1] if i + 1 < total else s2
            slope = _sign(nxt - prev)
        elif s2 == 0.0:
            candidate = float(i + 1)
            nxt = samples[i + 2] if i + 2 < total else s2
            slope = _sign(nxt - s1)
        elif (s1 > 0 and s2 < 0) or (s1 < 0 and s2 > 0):
            denom = abs(s1) + abs(s2)
            if denom > 0:
                frac = abs(s1) / denom
                candidate = float(i) + frac
                slope = _sign(s2 - s1)
        if candidate is None:
            continue
        dist = abs(candidate - target_idx)
        if preferred_slope not in (None, 0) and slope not in (preferred_slope, 0):
            score = dist
            if score < fallback_score:
                fallback_score = score
                fallback_pos = candidate
                fallback_slope = slope
            continue
        score = dist
        if score < best_score:
            best_score = score
            best_pos = candidate
            best_slope = slope
    if not math.isfinite(best_score):
        if math.isfinite(fallback_score):
            best_pos = fallback_pos
            best_slope = fallback_slope
        else:
            local = samples[start : end + 1]
            offset = int(np.argmin(np.abs(local)))
            best_pos = float(start + offset)
            best_slope = preferred_slope or 0
    best_pos = float(min(max(best_pos, 0.0), total - 1.0))
    if best_slope == 0:
        left = clamp_sample_index(int(math.floor(best_pos)) - 1, total - 1)
        right = clamp_sample_index(left + 1, total - 1)
        best_slope = _sign(samples[right] - samples[left])
    return ZeroCross(best_pos, best_slope)


def _pick_window_size(loop_len: int) -> int:
    """Choose a correlation window based on the tentative loop length."""
    if loop_len <= 0:
        return 0
    if loop_len < 1024:
        return max(64, loop_len // 2)
    return max(64, min(16384, loop_len // 5))


def extract_local_gradient(samples: np.ndarray, zero_pos: float, neighborhood: int) -> np.ndarray | None:
    """Extract normalized gradient around a zero-cross point."""
    total = samples.size
    if neighborhood < 2:
        return None
    start_idx = clamp_sample_index(int(zero_pos) - neighborhood, total - 1)
    end_idx = clamp_sample_index(int(zero_pos) + neighborhood + 1, total - 1)
    if end_idx - start_idx < 3:
        return None
    window = samples[start_idx:end_idx + 1].astype(np.float64, copy=False)
    grad = np.diff(window)
    if np.linalg.norm(grad) < 1e-10:
        return None
    return grad / float(np.linalg.norm(grad))


def build_orientation_vector(samples: np.ndarray, center_pos: float, radius: int) -> np.ndarray | None:
    if radius < 4:
        return None
    total = samples.size
    offsets = np.linspace(-radius, radius, 2 * radius + 1, dtype=np.float64)
    positions = np.clip(center_pos + offsets, 0.0, total - 1.0)
    xp = np.arange(total, dtype=np.float64)
    values = np.interp(positions, xp, samples.astype(np.float64, copy=False))
    values -= float(np.mean(values))
    norm = float(np.linalg.norm(values))
    if norm == 0.0:
        return None
    return values / norm


def find_best_loop_length(samples: np.ndarray, start_idx: int, start_zero_pos: float, approx_len: int, search_radius: int) -> int:
    """Use multi-metric matching with orientation emphasis to pick the best loop length."""
    total = samples.size
    if approx_len <= 0 or start_idx >= total - 2:
        return approx_len
    max_possible = total - start_idx - 1
    approx_len = min(approx_len, max_possible)
    window = _pick_window_size(approx_len)
    window = min(window, max_possible // 2)
    if window < 32:
        return approx_len
    min_len = max(window + 32, approx_len - search_radius, 64)
    max_len = min(max_possible, approx_len + search_radius)
    if max_len <= min_len:
        return int(max_len)
    start_window = samples[start_idx : start_idx + window].astype(np.float64, copy=False)
    window_taper = np.hanning(window)
    tapered_start = start_window * window_taper
    start_grad = np.diff(start_window)
    start_spec = np.abs(np.fft.rfft(tapered_start))
    start_phase = np.unwrap(np.angle(np.fft.rfft(tapered_start)))
    orientation_radius = max(8, min(window // 8, 128))
    start_orientation = build_orientation_vector(samples, start_zero_pos, orientation_radius)
    neighborhood = max(2, min(8, orientation_radius // 4))
    start_local_grad = extract_local_gradient(samples, start_zero_pos, neighborhood)
    best_score = math.inf
    best_len = approx_len
    grad_weight = 0.5
    spec_weight = 0.1
    phase_weight = 0.05
    orientation_weight = 3.0
    local_grad_weight = 2.0
    for loop_len in range(int(min_len), int(max_len) + 1):
        end_start = start_idx + loop_len - window
        end_stop = end_start + window
        if end_start < start_idx or end_stop > total:
            continue
        tail_window = samples[end_start:end_stop].astype(np.float64, copy=False)
        diff = start_window - tail_window
        rms_score = float(np.mean(diff * diff))
        tail_grad = np.diff(tail_window)
        grad_score = float(np.mean((start_grad - tail_grad) ** 2)) if tail_grad.size == start_grad.size else math.inf
        tapered_tail = tail_window * window_taper
        tail_rfft = np.fft.rfft(tapered_tail)
        tail_spec = np.abs(tail_rfft)
        spec_score = float(np.mean((start_spec - tail_spec) ** 2))
        tail_phase = np.unwrap(np.angle(tail_rfft))
        phase_score = float(np.mean((start_phase - tail_phase) ** 2))
        candidate_pos = start_zero_pos + loop_len
        candidate_pos = float(min(max(candidate_pos, 0.0), total - 1.0))
        orientation_score = 0.0
        if start_orientation is not None:
            tail_orientation = build_orientation_vector(samples, candidate_pos, orientation_radius)
            if tail_orientation is None:
                orientation_score = orientation_weight
            else:
                cos_sim = float(np.clip(np.dot(start_orientation, tail_orientation), -1.0, 1.0))
                orientation_score = (1.0 - cos_sim) * orientation_weight
        local_grad_score = 0.0
        if start_local_grad is not None:
            tail_local_grad = extract_local_gradient(samples, candidate_pos, neighborhood)
            if tail_local_grad is None:
                local_grad_score = local_grad_weight
            else:
                grad_cos_sim = float(np.clip(np.dot(start_local_grad, tail_local_grad), -1.0, 1.0))
                local_grad_score = (1.0 - grad_cos_sim) * local_grad_weight
        score = rms_score + grad_weight * grad_score + spec_weight * spec_score + phase_weight * phase_score + orientation_score + local_grad_score
        if score < best_score:
            best_score = score
            best_len = loop_len
    return best_len


def slice_fractional_segment(samples: np.ndarray, start_pos: float, end_pos: float) -> np.ndarray:
    """Slice the waveform between fractional indices using linear interpolation."""
    if end_pos <= start_pos:
        raise LoopProcessingError("Loop end must be greater than start")
    span = end_pos - start_pos
    if span < 2.0:
        raise LoopProcessingError("Looped segment is too short after alignment")
    xp = np.arange(samples.size, dtype=np.float64)
    interior_positions = np.arange(start_pos, end_pos, 1.0, dtype=np.float64)
    if interior_positions.size == 0:
        raise LoopProcessingError("Looped segment collapsed during slicing")
    interior = np.interp(interior_positions, xp, samples)
    tail_sample = float(np.interp(end_pos, xp, samples))
    segment = np.concatenate((interior, np.array([tail_sample], dtype=np.float64))).astype(np.float32, copy=False)
    segment[0] = 0.0
    segment[-1] = 0.0
    return segment


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
    approx_len = end_idx - start_idx
    zero_start = find_zero_cross(samples, float(start_idx), radius)
    corr_anchor = clamp_sample_index(int(round(zero_start.position)), total - 1)
    best_len = find_best_loop_length(samples, corr_anchor, zero_start.position, approx_len, radius)
    corr_end = zero_start.position + best_len
    corr_end = float(min(max(corr_end, 0.0), total - 1))
    zero_end = find_zero_cross(samples, corr_end, radius, preferred_slope=zero_start.slope)
    if zero_end.position <= zero_start.position:
        raise LoopProcessingError("Zero-cross search collapsed loop window")
    loop = slice_fractional_segment(samples, zero_start.position, zero_end.position)
    if loop.size < sr // 20:  # <50ms guard
        raise LoopProcessingError("Looped segment is too short")
    return apply_zero_locked_crossfade(loop)


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
