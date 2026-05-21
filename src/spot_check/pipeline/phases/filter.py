"""Filter phase — validate CSV columns and weight mode before assignment."""

from __future__ import annotations

from spot_check.analysis.measured import (
    _probe_csv_columns_for_measured_weights,
    normalize_measured_spot_weight_mode,
)
from spot_check.pipeline.progress import ProgressEvent, ProgressSink
from spot_check.pipeline.types import PHASE_FILTER, PipelineConfig, PipelineState


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
            phase_id=PHASE_FILTER,
            step=step,
            message=message,
            current=current,
            total=total,
        )
    )


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
    _report(
        progress,
        step="probe_columns",
        message=f"Validating CSV columns ({layer_mode_run})…",
    )
    _probe_csv_columns_for_measured_weights(csv_path, spot_weight_mode=swm)
    _report(
        progress,
        step="probe_done",
        message="Column validation complete — reading and filtering rows…",
    )
    return state
