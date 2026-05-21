"""Assign phase — layer/spot assignment from acquisition CSV."""

from __future__ import annotations

from pathlib import Path

from spot_check import analysis
from spot_check.analysis.measured import MeasuredAssignResult, assign_measured_from_csv
from spot_check.pipeline.diagnostics import AssignDiagnostics
from spot_check.pipeline.progress import ProgressEvent, ProgressSink
from spot_check.pipeline.types import PHASE_ASSIGN, PipelineConfig, PipelineState


def _capture_assign_diagnostics(
    layer_mode_run: str,
    auto_assign_method: str,
) -> AssignDiagnostics | None:
    if layer_mode_run != "auto":
        return None
    ep_diag = None
    if auto_assign_method == "episodes":
        ep_diag = analysis.last_auto_episode_diagnostics()
    return AssignDiagnostics(
        auto_layer_params=analysis.last_auto_layer_params(),
        episode_diagnostics=ep_diag,
    )


def _report(
    progress: ProgressSink,
    *,
    step: str,
    message: str,
    current: int | None = None,
    total: int | None = None,
) -> None:
    progress.report(
        ProgressEvent(
            phase_id=PHASE_ASSIGN,
            step=step,
            message=message,
            current=current,
            total=total,
        )
    )


def run_assign_phase(
    state: PipelineState,
    config: PipelineConfig,
    progress: ProgressSink,
    *,
    csv_path: Path,
    layer_mode_run: str,
    auto_assign_method: str,
    auto_infer: bool,
    align_before_assign: bool,
) -> MeasuredAssignResult:
    """Assign layers and spots; returns rows before aggregation."""
    _report(
        progress,
        step="assign_start",
        message=f"Assigning spots ({layer_mode_run})…",
    )
    planned = state.planned
    assigned = assign_measured_from_csv(
        csv_path,
        max_points=None,
        planned_xyz=planned if planned else None,
        a_is_x=False,
        layer_mode=layer_mode_run,
        spot_weight_mode=config.spot_weight_mode,
        auto_infer_params=auto_infer and layer_mode_run == "auto",
        auto_assign_method=auto_assign_method,
        heal_partial_fit_axes=config.heal_partial_fit_axes,
        align_detector_xy_before_assign=align_before_assign,
    )
    n_rows = len(assigned.rows)
    if n_rows == 0:
        raise ValueError("No measured rows to plot.")
    _report(
        progress,
        step="assign_done",
        message=f"Assignment complete — {n_rows} row{'s' if n_rows != 1 else ''}.",
        current=n_rows,
        total=n_rows,
    )
    state.assign_result = assigned
    state.layer_mode_run = layer_mode_run
    state.auto_assign_method = auto_assign_method
    state.assign_diagnostics = _capture_assign_diagnostics(
        layer_mode_run, auto_assign_method
    )
    return assigned
