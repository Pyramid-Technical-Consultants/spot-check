"""Aggregate phase — collapse assigned rows by spot id."""

from __future__ import annotations

from spot_check.analysis.measured import (
    MeasuredAssignResult,
    aggregate_measured_assign_result,
)
from spot_check.pipeline.progress import ProgressEvent, ProgressSink
from spot_check.pipeline.types import PHASE_AGGREGATE, PipelineState


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
            phase_id=PHASE_AGGREGATE,
            step=step,
            message=message,
            current=current,
            total=total,
        )
    )


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
        _report(
            progress,
            step="aggregate_start",
            message=f"Aggregating {n_in} assigned row{'s' if n_in != 1 else ''}…",
            current=0,
            total=max(n_in, 1),
        )
    rows = aggregate_measured_assign_result(assigned, aggregate_spots=aggregate_spots)
    n_out = len(rows)
    if aggregate_spots:
        _report(
            progress,
            step="aggregate_done",
            message=f"Aggregated to {n_out} spot row{'s' if n_out != 1 else ''}.",
            current=n_out,
            total=n_out,
        )
    state.measured_unaligned = rows
    state.aggregate_run = bool(aggregate_spots)
    return rows
