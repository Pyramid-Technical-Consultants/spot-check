"""Load phase — read plan file and open acquisition CSV path."""

from __future__ import annotations

from spot_check.pipeline.progress import ProgressEvent, ProgressSink
from spot_check.pipeline.types import PHASE_LOAD, PipelineConfig, PipelineState
from spot_check.plan import plan_label_from_path, planned_spot_xyz_and_counts_from_plan


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
            phase_id=PHASE_LOAD,
            step=step,
            message=message,
            current=current,
            total=total,
        )
    )


def run_load_phase(
    state: PipelineState,
    config: PipelineConfig,
    progress: ProgressSink,
) -> PipelineState:
    """Load plan spots and record acquisition CSV metadata."""
    plan_path = config.plan_path
    csv_path = config.csv_path

    if plan_path is not None:
        _report(
            progress,
            step="plan_start",
            message=f"Reading plan: {plan_path.name}…",
        )
        label = plan_label_from_path(plan_path)
        planned, plan_fwhm_xy, plan_mu, n_plan_kept, n_plan_raw = (
            planned_spot_xyz_and_counts_from_plan(plan_path)
        )
        _report(
            progress,
            step="plan_done",
            message=f"Plan loaded — {n_plan_kept} spots kept ({n_plan_raw} raw).",
        )
        state.label = label
        state.planned = planned
        state.plan_fwhm_xy = plan_fwhm_xy
        state.plan_mu = plan_mu
        state.n_plan_kept = n_plan_kept
        state.n_plan_raw = n_plan_raw

    if csv_path is not None:
        _report(
            progress,
            step="csv_start",
            message=f"Opening acquisition CSV: {csv_path.name}…",
        )
        state.csv_display_name = csv_path.name

    return state
