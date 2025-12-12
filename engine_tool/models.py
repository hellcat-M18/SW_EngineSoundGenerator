"""Dataclasses shared across the application."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from .utils import slugify


@dataclass(slots=True)
class LoopParameters:
    search_radius: int
    ogg_quality: int


@dataclass(slots=True)
class StageInput:
    label: str
    source_path: Path
    start_sec: float
    end_sec: float


@dataclass(slots=True)
class EngineJob:
    engine_name: str
    template_path: Path
    compiler_path: Path
    output_dir: Path
    ffmpeg_path: str
    loop_params: LoopParameters
    stages: list[StageInput]
    engine_slug: str = field(init=False)

    def __post_init__(self) -> None:
        self.engine_slug = slugify(self.engine_name)


ProgressCallback = Callable[[str], None]


@dataclass(slots=True)
class PipelineResult:
    bin_files: list[Path]


def iter_stage_labels(count: int) -> Iterable[str]:
    for idx in range(count):
        yield f"L{idx + 1}"
