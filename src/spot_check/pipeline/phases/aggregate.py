"""Aggregate phase — collapse assigned rows by spot id."""

from __future__ import annotations

from spot_check.analysis.measured import (
    MeasuredAssignResult,
    aggregate_measured_assign_result,
)
from spot_check.pipeline.progress import ProgressSink, report_phase_progress
from spot_check.pipeline.types import PHASE_AGGREGATE, PipelineState


def run_aggregate_phase(
    state: PipelineState,
    progress: ProgressSink,
    *,
    assigned: MeasuredAssignResult,
    aggregate_spots: bool,
) -> list[tuple[float, ...]]:
    """Optionally collapse assigned rows to one weighted-mean row per spot."""
    n_in = len(assigned.rows)
    if aggregate_spots:
        report_phase_progress(
            progress,
            PHASE_AGGREGATE,
            step="aggregate_start",
            message=f"Aggregating {n_in} assigned row{'s' if n_in != 1 else ''}…",
            current=0,
            total=max(n_in, 1),
        )
    rows = aggregate_measured_assign_result(assigned, aggregate_spots=aggregate_spots)
    n_out = len(rows)
    if aggregate_spots:
        report_phase_progress(
            progress,
            PHASE_AGGREGATE,
            step="aggregate_done",
            message=f"Aggregated to {n_out} spot row{'s' if n_out != 1 else ''}.",
            current=n_out,
            total=n_out,
        )
    state.measured_unaligned = rows
    state.aggregate_run = bool(aggregate_spots)
    return rows
