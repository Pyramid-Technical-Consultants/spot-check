"""Assign phase — layer/spot assignment from acquisition CSV."""

from __future__ import annotations

from pathlib import Path

from spot_check import analysis
from spot_check.analysis.measured import (
    MeasuredAssignResult,
    assign_measured_from_csv,
    finalize_measured_assign_coverage,
)
from spot_check.pipeline.diagnostics import AssignDiagnostics
from spot_check.pipeline.progress import ProgressSink, report_phase_progress
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


def run_assign_phase(
    state: PipelineState,
    config: PipelineConfig,
    progress: ProgressSink,
    *,
    csv_path: Path,
    layer_mode_run: str,
    auto_assign_method: str,
    auto_infer: bool,
) -> MeasuredAssignResult:
    """Assign layers and spots; returns rows before aggregation."""
    report_phase_progress(
        progress,
        PHASE_ASSIGN,
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
        coarse_flat_transform=state.coarse_flat_align,
        skip_column_probe=state.csv_columns_validated,
        preloaded_auto_columns=state.auto_fit_columns,
    )
    if planned:
        assigned = finalize_measured_assign_coverage(assigned, planned_xyz=list(planned))
    n_rows = len(assigned.rows)
    if n_rows == 0:
        raise ValueError("No measured rows to plot.")
    report_phase_progress(
        progress,
        PHASE_ASSIGN,
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
