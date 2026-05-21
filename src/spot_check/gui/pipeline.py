"""Plan + CSV load pipeline (runs off the GUI thread)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spot_check.pipeline import PipelineConfig, run_data_phases
from spot_check.pipeline.load_result import (
    PipelineLoadOK,
    aggregation_applies,
    file_mtime,
    is_acquisition_csv_file,
    resolve_csv_load_layer_mode,
)
from spot_check.pipeline.progress import NullProgressSink, ProgressSink

__all__ = [
    "GuiRefreshContext",
    "PipelineLoadOK",
    "aggregation_applies",
    "file_mtime",
    "is_acquisition_csv_file",
    "pipeline_load_job",
    "resolve_csv_load_layer_mode",
]


@dataclass(frozen=True)
class GuiRefreshContext:
    plan_path: Path | None
    csv_path: Path | None
    xy_tick_use: float
    qa_mode: str
    qa_pass_f: float
    qa_warn_f: float
    layer_assign_mode: str
    aggregate_spots: bool
    spot_weight_mode_run: str
    pipeline_key: tuple[Any, ...]


def pipeline_load_job(
    plan_path: Path | None,
    csv_path: Path | None,
    *,
    layer_assign_mode: str,
    aggregate_spots: bool,
    spot_weight_mode: str,
    auto_align: bool = False,
    heal_partial_fit_axes: bool = False,
    progress: ProgressSink | None = None,
) -> PipelineLoadOK:
    config = PipelineConfig(
        plan_path=plan_path,
        csv_path=csv_path,
        layer_assign_mode=layer_assign_mode,
        aggregate_spots=aggregate_spots,
        spot_weight_mode=spot_weight_mode,
        auto_align=auto_align,
        heal_partial_fit_axes=heal_partial_fit_axes,
    )
    return run_data_phases(config, progress=progress or NullProgressSink())
