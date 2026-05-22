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
    coarse_flat_align: bool = False,
    heal_partial_fit_axes: bool = False,
    fine_align_xy: bool = True,
    fine_align_rotation: bool = True,
    fine_align_scale: bool = True,
    filter_xy_fliers: bool = False,
    filter_xy_flier_sigma: float = 3.0,
    progress: ProgressSink | None = None,
) -> PipelineLoadOK:
    config = PipelineConfig(
        plan_path=plan_path,
        csv_path=csv_path,
        layer_assign_mode=layer_assign_mode,
        aggregate_spots=aggregate_spots,
        spot_weight_mode=spot_weight_mode,
        coarse_flat_align=coarse_flat_align,
        heal_partial_fit_axes=heal_partial_fit_axes,
        fine_align_xy=fine_align_xy,
        fine_align_rotation=fine_align_rotation,
        fine_align_scale=fine_align_scale,
        filter_xy_fliers=filter_xy_fliers,
        filter_xy_flier_sigma=filter_xy_flier_sigma,
    )
    return run_data_phases(config, progress=progress or NullProgressSink())
