"""Load phase — read plan file and open acquisition CSV path."""

from __future__ import annotations

from spot_check.pipeline.progress import ProgressSink, report_phase_progress
from spot_check.pipeline.types import PHASE_LOAD, PipelineConfig, PipelineState
from spot_check.plan import load_plan_from_path


def run_load_phase(
    state: PipelineState,
    config: PipelineConfig,
    progress: ProgressSink,
) -> PipelineState:
    """Load plan spots and record acquisition CSV metadata."""
    plan_path = config.plan_path
    csv_path = config.csv_path

    if plan_path is not None:
        report_phase_progress(
            progress,
            PHASE_LOAD,
            step="plan_start",
            message=f"Reading plan: {plan_path.name}…",
        )
        label, planned, plan_fwhm_xy, plan_mu, n_plan_kept, n_plan_raw = load_plan_from_path(
            plan_path
        )
        report_phase_progress(
            progress,
            PHASE_LOAD,
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
        report_phase_progress(
            progress,
            PHASE_LOAD,
            step="csv_start",
            message=f"Opening acquisition CSV: {csv_path.name}…",
        )
        state.csv_display_name = csv_path.name

    return state
