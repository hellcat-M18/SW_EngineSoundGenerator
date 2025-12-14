"""High-level orchestration of the build pipeline."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable

from .audio_loop import encode_vorbis, extract_loop, load_audio_mono
from .compiler import run_component_compiler
from .models import EngineJob, PipelineResult, ProgressCallback, StageInput
from .utils import sanitize_filename_component
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
    bracket_label = sanitize_filename_component(job.engine_name, fallback=job.engine_slug)
    staged_files: list[Path] = []
    loop_paths: list[Path] = []
    try:
        safe_slug = job.engine_slug.replace("-", "_")
        for idx, stage in enumerate(job.stages):
            loop_path = compiler_dir / f"{safe_slug}_{stage.label}.ogg"
            process_stage(stage, loop_path, job.loop_params.search_radius, job.loop_params.ogg_quality, job.ffmpeg_path, progress)
            loop_paths.append(loop_path)
            staged_files.append(loop_path)
            if progress:
                progress(f"Processing {stage.label}: prepared {loop_path.name}")
        xml_path = compiler_dir / f"[{bracket_label}]modular_engine_crankshaft.xml"
        if progress:
            progress("Rendering XML template…")
        render_template(job.template_path, job.engine_name, job.engine_slug, xml_path)
        staged_files.append(xml_path)
        if progress:
            progress("Invoking component_mod_compiler…")
        preexisting_bins = {p.resolve() for p in compiler_dir.glob("*.bin")}
        bat_path = compiler_dir / f"_compile_{job.engine_slug}.bat"
        bat_content = f"{job.compiler_path.name} {xml_path.name} " + " ".join(p.name for p in loop_paths)
        bat_path.write_text(bat_content, encoding="utf-8")
        staged_files.append(bat_path)
        run_component_compiler(bat_path, cwd=compiler_dir)
        post_bins = {p.resolve() for p in compiler_dir.glob("*.bin")}
        new_bins = sorted(post_bins - preexisting_bins, key=lambda p: p.stat().st_mtime)
        if progress:
            progress("Archiving XML/OGG artifacts…")
        for loop_path in loop_paths:
            target_path = _reserve_destination(run_dir / loop_path.name)
            shutil.copy2(loop_path, target_path)
        xml_target = _reserve_destination(run_dir / xml_path.name)
        shutil.copy2(xml_path, xml_target)
        bin_outputs: list[Path] = []
        bin_base = run_dir / f"[{bracket_label}]modular_engine_crankshaft.bin"
        for idx, bin_file in enumerate(new_bins):
            candidate = bin_base if idx == 0 else bin_base.with_name(f"{bin_base.stem}_{idx}{bin_base.suffix}")
            final_path = _reserve_destination(candidate)
            shutil.move(str(bin_file), final_path)
            bin_outputs.append(final_path)
            if progress:
                progress(f"Compiler output stored: {final_path.name}")
        if progress and not new_bins:
            progress("Compiler finished, but .bin file was not detected.")
        return PipelineResult(bin_files=bin_outputs)
    finally:
        _cleanup_files(staged_files)
