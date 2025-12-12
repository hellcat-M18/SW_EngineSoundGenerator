"""High-level orchestration of the build pipeline."""
from __future__ import annotations

import secrets
import shutil
from pathlib import Path
from typing import Callable

from .audio_loop import encode_vorbis, extract_loop, load_audio_mono
from .compiler import run_component_compiler
from .models import EngineJob, PipelineResult, ProgressCallback, StageInput
from .xml_generator import render_template


def process_stage(stage: StageInput, output_path: Path, search_radius: int, ogg_quality: int, ffmpeg_path: str, progress: ProgressCallback | None = None) -> None:
    if progress:
        progress(f"Processing {stage.label}: loading audio…")
    samples, sr = load_audio_mono(stage.source_path)
    loop = extract_loop(samples, sr, stage.start_sec, stage.end_sec, search_radius)
    if progress:
        progress(f"Processing {stage.label}: encoding Vorbis (q={ogg_quality})…")
    encode_vorbis(loop, sr, ogg_quality, output_path, ffmpeg_path=ffmpeg_path)


def _reserve_destination(base_path: Path) -> Path:
    if not base_path.exists():
        return base_path
    counter = 1
    stem = base_path.stem
    suffix = base_path.suffix
    while True:
        candidate = base_path.with_name(f"{stem}_{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def _reserve_directory(base_path: Path) -> Path:
    if not base_path.exists():
        return base_path
    counter = 1
    while True:
        candidate = base_path.parent / f"{base_path.name}_{counter}"
        if not candidate.exists():
            return candidate
        counter += 1


def _unique_name(engine_slug: str, suffix: str) -> str:
    token = secrets.token_hex(4)
    return f"{engine_slug}_{token}_{suffix}"


def _cleanup_files(paths: list[Path]) -> None:
    for path in paths:
        try:
            if path.exists():
                path.unlink()
        except OSError:
            pass


def run_pipeline(job: EngineJob, progress: ProgressCallback | None = None) -> PipelineResult:
    job.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = _reserve_directory(job.output_dir / job.engine_slug)
    run_dir.mkdir(parents=True, exist_ok=True)
    compiler_dir = job.compiler_path.parent
    staged_files: list[Path] = []
    loop_paths: list[tuple[Path, str]] = []
    try:
        for idx, stage in enumerate(job.stages):
            loop_path = compiler_dir / _unique_name(job.engine_slug, f"L{idx + 1}.ogg")
            process_stage(stage, loop_path, job.loop_params.search_radius, job.loop_params.ogg_quality, job.ffmpeg_path, progress)
            loop_paths.append((loop_path, stage.label))
            staged_files.append(loop_path)
            if progress:
                progress(f"Processing {stage.label}: prepared {loop_path.name}")
        xml_path = compiler_dir / _unique_name(job.engine_slug, "modular_engine_crankshaft.xml")
        if progress:
            progress("Rendering XML template…")
        render_template(job.template_path, job.engine_name, job.engine_slug, xml_path)
        staged_files.append(xml_path)
        if progress:
            progress("Invoking component_mod_compiler…")
        preexisting_bins = {p.resolve() for p in compiler_dir.glob("*.bin")}
        xml_arg = Path(xml_path.name)
        loop_args = [Path(p.name) for p, _ in loop_paths]
        run_component_compiler(job.compiler_path, xml_arg, loop_args, cwd=compiler_dir)
        post_bins = {p.resolve() for p in compiler_dir.glob("*.bin")}
        new_bins = sorted(post_bins - preexisting_bins, key=lambda p: p.stat().st_mtime)
        if progress:
            progress("Archiving XML/OGG artifacts…")
        for loop_path, label in loop_paths:
            target_name = f"{job.engine_slug}-{label}.ogg"
            target_path = _reserve_destination(run_dir / target_name)
            shutil.copy2(loop_path, target_path)
        xml_target = _reserve_destination(run_dir / f"{job.engine_slug}_modular_engine_crankshaft.xml")
        shutil.copy2(xml_path, xml_target)
        bin_outputs: list[Path] = []
        for bin_file in new_bins:
            final_path = _reserve_destination(run_dir / bin_file.name)
            shutil.move(str(bin_file), final_path)
            bin_outputs.append(final_path)
            if progress:
                progress(f"Compiler output stored: {final_path.name}")
        if progress and not new_bins:
            progress("Compiler finished, but .bin file was not detected.")
        return PipelineResult(bin_files=bin_outputs)
    finally:
        _cleanup_files(staged_files)
