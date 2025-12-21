"""Audio loading and loop generation."""
from __future__ import annotations

import math
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from .exceptions import LoopProcessingError


@dataclass(slots=True)
class ZeroCross:
    position: float
    slope: int


def _boundary_cost(loop: np.ndarray, start_offset: int, window: int, grad_weight: float = 0.5) -> float:
    """Cost of loop seam if loop starts at start_offset (circular)."""
    n = loop.size
    if n < window * 2 + 4:
        return math.inf
    base = np.arange(window, dtype=np.int64)
    idx_start = (base + start_offset) % n
    idx_end = (base + start_offset - window) % n
    a = loop[idx_start].astype(np.float64, copy=False)
    b = loop[idx_end].astype(np.float64, copy=False)
    diff = a - b
    rms = float(np.mean(diff * diff))
    if grad_weight <= 0:
        return rms
    da = np.diff(a)
    db = np.diff(b)
    dd = da - db
    grad = float(np.mean(dd * dd))
    return rms + grad_weight * grad


def rotate_loop_for_best_boundary(loop_samples: np.ndarray, search: int = 2048) -> np.ndarray:
    """Rotate the loop to the least-audible boundary within a local search range."""
    n = int(loop_samples.size)
    if n < 4096:
        return loop_samples
    search = int(min(max(0, search), n // 2))
    if search <= 0:
        return loop_samples
    window = int(min(1024, max(128, n // 40)))
    if n < window * 2 + 4:
        return loop_samples

    best_shift = 0
    best = math.inf
    # Search small rotations around the current boundary.
    for shift in range(-search, search + 1):
        cost = _boundary_cost(loop_samples, shift, window, grad_weight=0.75)
        if cost < best:
            best = cost
            best_shift = shift
    if best_shift == 0:
        return loop_samples
    return np.roll(loop_samples, -best_shift).astype(np.float32, copy=False)


def apply_circular_declick(loop_samples: np.ndarray, blend_samples: int = 2048, smooth_passes: int = 0) -> np.ndarray:
    """Reduce clicks by smoothing only around the loop boundary.

    This keeps the loop length unchanged and does not force endpoints to zero.
    It blends the head and tail symmetrically, then optionally applies a tiny smoothing filter to
    the boundary region.
    """
    if loop_samples.size <= 64:
        return loop_samples
    blend = int(min(blend_samples, max(16, loop_samples.size // 8)))
    if blend <= 0 or loop_samples.size <= 2 * blend + 8:
        return loop_samples

    result = loop_samples.astype(np.float32, copy=True)
    head = result[:blend].copy()
    tail = result[-blend:].copy()
    # Equal-power crossfade to avoid perceived level dip.
    theta = np.linspace(0.0, math.pi / 2.0, blend, dtype=np.float32)
    a = np.cos(theta)  # 1 -> 0
    b = np.sin(theta)  # 0 -> 1
    # Symmetric blend at both ends.
    blended_head = (a * head) + (b * tail)
    blended_tail = (a * tail) + (b * head)

    # RMS normalize the boundary to avoid level bumps when signals are correlated.
    eps = 1e-12
    target_rms = float(0.5 * (np.sqrt(np.mean(head * head) + eps) + np.sqrt(np.mean(tail * tail) + eps)))
    blended_rms = float(0.5 * (np.sqrt(np.mean(blended_head * blended_head) + eps) + np.sqrt(np.mean(blended_tail * blended_tail) + eps)))
    if blended_rms > eps and target_rms > eps:
        gain = target_rms / blended_rms
        gain = float(np.clip(gain, 0.5, 1.5))
        blended_head *= gain
        blended_tail *= gain

    result[:blend] = blended_head
    result[-blend:] = blended_tail

    if smooth_passes <= 0:
        return result
    # Very small 3-tap smoothing, applied only inside the boundary bands.
    for _ in range(int(smooth_passes)):
        band = np.concatenate([result[-blend:], result[:blend]]).astype(np.float32, copy=False)
        padded = np.concatenate([band[-1:], band, band[:1]])
        smoothed = (0.25 * padded[:-2]) + (0.5 * padded[1:-1]) + (0.25 * padded[2:])
        smoothed = smoothed.astype(np.float32, copy=False)
        result[-blend:] = smoothed[:blend]
        result[:blend] = smoothed[blend:]
    return result


def write_wav_pcm16(samples: np.ndarray, sr: int, output_path: Path) -> None:
    """Write mono PCM16 WAV for debugging/analysis."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(samples.astype(np.float32, copy=False), -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(pcm16.tobytes())


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
    return segment


def _compute_envelope(samples: np.ndarray, frame_size: int) -> np.ndarray:
    """Compute RMS envelope per frame, then interpolate back to sample rate."""
    n = samples.size
    if n < frame_size:
        rms = float(np.sqrt(np.mean(samples * samples) + 1e-12))
        return np.full(n, rms, dtype=np.float32)
    
    # Compute RMS per frame
    num_frames = n // frame_size
    remainder = n % frame_size
    frames = samples[:num_frames * frame_size].reshape(num_frames, frame_size)
    frame_rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12).astype(np.float32)
    
    # Handle remainder
    if remainder > 0:
        last_rms = float(np.sqrt(np.mean(samples[-remainder:] ** 2) + 1e-12))
        frame_rms = np.append(frame_rms, last_rms)
    
    # Interpolate back to sample rate
    frame_centers = np.arange(len(frame_rms)) * frame_size + frame_size // 2
    frame_centers = np.clip(frame_centers, 0, n - 1)
    sample_indices = np.arange(n)
    envelope = np.interp(sample_indices, frame_centers, frame_rms).astype(np.float32)
    
    return envelope


def simple_overlap_add(loop_samples: np.ndarray, sr: int, overlap_sec: float = 0.05) -> np.ndarray:
    """Simple overlap-add crossfade: overlap head and tail, shorten the loop.
    
    Uses a SHORT crossfade (default 50ms) to minimize phase interference issues.
    Short crossfades work well for periodic signals like engine sounds because
    they don't give the out-of-phase portions enough time to cause audible cancellation.
    
    The loop becomes shorter by `overlap_sec` seconds.
    """
    overlap = int(overlap_sec * sr)
    n = loop_samples.size
    
    # Ensure overlap is reasonable (minimum 32 samples, max 1/4 of loop)
    overlap = max(32, min(overlap, n // 4))
    if n <= overlap * 2 + 64:
        return loop_samples
    
    result_len = n - overlap
    result = np.zeros(result_len, dtype=np.float32)
    
    # Copy the middle section as-is
    result[overlap:] = loop_samples[overlap:n - overlap]
    
    # Overlap-add the head and tail
    head = loop_samples[:overlap].astype(np.float32, copy=False)
    tail = loop_samples[n - overlap:].astype(np.float32, copy=False)
    
    # Simple linear crossfade - no power compensation needed for short crossfades
    t = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
    crossfaded = tail * (1.0 - t) + head * t
    
    result[:overlap] = crossfaded
    
    return result


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
    
    # Simple approach: just cut the segment, no fancy zero-cross hunting
    loop = samples[start_idx:end_idx].astype(np.float32, copy=True)
    
    if loop.size < sr // 10:  # <100ms guard
        raise LoopProcessingError("Looped segment is too short")
    
    # Apply overlap-add crossfade (50ms - short to minimize phase interference)
    return simple_overlap_add(loop, sr, overlap_sec=0.05)


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
