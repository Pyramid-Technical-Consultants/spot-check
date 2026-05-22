"""Filter phase — validate CSV columns and optional XY σ flier removal after assign."""

from __future__ import annotations

from spot_check.analysis.measured import (
    MeasuredAssignResult,
    _probe_csv_columns_for_measured_weights,
    filter_assigned_xy_fliers,
    normalize_measured_spot_weight_mode,
)
from spot_check.pipeline.progress import ProgressSink, report_phase_progress
from spot_check.pipeline.types import PHASE_FILTER, PipelineConfig, PipelineState


def run_filter_phase(
    state: PipelineState,
    config: PipelineConfig,
    progress: ProgressSink,
    *,
    layer_mode_run: str,
) -> PipelineState:
    """Probe acquisition CSV columns and validate weight mode."""
    csv_path = config.csv_path
    if csv_path is None:
        return state

    swm = normalize_measured_spot_weight_mode(config.spot_weight_mode)
    report_phase_progress(
        progress,
        PHASE_FILTER,
        step="probe_columns",
        message=f"Validating CSV columns ({layer_mode_run})…",
    )
    _probe_csv_columns_for_measured_weights(csv_path, spot_weight_mode=swm)
    state.csv_columns_validated = True
    report_phase_progress(
        progress,
        PHASE_FILTER,
        step="probe_done",
        message="Column validation complete — reading and filtering rows…",
    )
    return state


def apply_xy_flier_filter_if_enabled(
    assigned: MeasuredAssignResult,
    state: PipelineState,
    config: PipelineConfig,
    progress: ProgressSink,
) -> MeasuredAssignResult:
    """Remove assigned rows whose XY offset vs layer-NN plan exceeds ``n_sigma`` fit σ."""
    if not config.filter_xy_fliers or not state.planned:
        return assigned
    n_before = len(assigned.rows)
    if n_before == 0:
        return assigned
    filtered = filter_assigned_xy_fliers(
        assigned,
        state.planned,
        n_sigma=float(config.filter_xy_flier_sigma),
    )
    n_after = len(filtered.rows)
    if n_after == 0:
        raise ValueError(
            f"XY σ flier filter removed all {n_before} assigned row"
            f"{'s' if n_before != 1 else ''} "
            f"(limit {float(config.filter_xy_flier_sigma):g}σ)."
        )
    n_drop = n_before - n_after
    if n_drop > 0:
        report_phase_progress(
            progress,
            PHASE_FILTER,
            step="xy_flier_filter",
            message=(
                f"Removed {n_drop} XY flier row{'s' if n_drop != 1 else ''} "
                f"(>{float(config.filter_xy_flier_sigma):g}σ vs plan)…"
            ),
            current=n_after,
            total=n_before,
        )
    return filtered
