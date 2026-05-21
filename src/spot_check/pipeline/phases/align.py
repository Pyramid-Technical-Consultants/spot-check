"""Detector alignment phase — pre- or post-assignment rigid XY align."""

from __future__ import annotations

from typing import Any

from spot_check import analysis
from spot_check.pipeline.progress import ProgressEvent, ProgressSink
from spot_check.pipeline.types import PHASE_ALIGN, PipelineConfig, PipelineState


def _report(progress: ProgressSink, *, step: str, message: str) -> None:
    progress.report(
        ProgressEvent(phase_id=PHASE_ALIGN, step=step, message=message)
    )


def run_align_phase(
    state: PipelineState,
    config: PipelineConfig,
    progress: ProgressSink,
    *,
    layer_mode_run: str,
) -> tuple[list[tuple[float, ...]] | None, Any | None, bool]:
    """Align measured rows to plan when enabled; returns aligned rows, info, pre-aligned flag."""
    planned = state.planned
    measured_unaligned = state.measured_unaligned
    if not config.auto_align or not planned or not measured_unaligned:
        return None, None, False

    detector_pre_aligned = bool(
        config.auto_align and layer_mode_run == "auto" and planned and measured_unaligned
    )
    _report(
        progress,
        step="align_start",
        message="Aligning measured detector XY to plan…",
    )

    measured_aligned: list[tuple[float, ...]] | None = None
    align_info: Any | None = None

    if detector_pre_aligned:
        align_info = analysis.last_detector_align_info()
        measured_aligned = list(measured_unaligned)
        _report(
            progress,
            step="align_pre_done",
            message="Pre-assignment alignment applied during CSV processing.",
        )
    else:
        measured_aligned, align_info = analysis.align_measured_to_plan_detector_xy(
            planned,
            measured_unaligned,
            a_is_x=False,
        )
        _report(
            progress,
            step="align_post_done",
            message="Post-assignment detector alignment complete.",
        )

    state.measured_aligned = (
        list(measured_aligned) if measured_aligned is not None else None
    )
    state.align_info = align_info
    state.detector_pre_aligned = detector_pre_aligned
    return measured_aligned, align_info, detector_pre_aligned
