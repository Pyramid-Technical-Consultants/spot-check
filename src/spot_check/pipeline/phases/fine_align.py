"""Fine detector alignment phase — layer-NN ICP + subsampled GN after aggregate."""

from __future__ import annotations

from typing import Any

from spot_check import analysis
from spot_check.pipeline.progress import ProgressSink, report_phase_progress
from spot_check.pipeline.types import PHASE_FINE_ALIGN, PipelineConfig, PipelineState


def run_fine_align_phase(
    state: PipelineState,
    config: PipelineConfig,
    progress: ProgressSink,
) -> tuple[list[tuple[float, ...]] | None, Any | None]:
    """Fine XY / rotation / anisotropic scale on post-aggregate measured rows."""
    state.measured_fine_aligned = None
    state.fine_align_info = None

    planned = state.planned
    base_rows = state.measured_unaligned
    if (
        not planned
        or not base_rows
        or not (
            config.fine_align_xy
            or config.fine_align_rotation
            or config.fine_align_scale
        )
    ):
        return None, None

    report_phase_progress(
        progress, PHASE_FINE_ALIGN, step="fine_start", message="Fine detector alignment…"
    )
    out_rows, info = analysis.fine_align_measured_to_plan(
        planned,
        base_rows,
        a_is_x=False,
        allow_xy=bool(config.fine_align_xy),
        allow_rotation=bool(config.fine_align_rotation),
        allow_scale=bool(config.fine_align_scale),
    )
    if info is None:
        report_phase_progress(
            progress,
            PHASE_FINE_ALIGN,
            step="fine_skip",
            message="Fine alignment skipped (no QA improvement).",
        )
        return None, None

    state.measured_fine_aligned = list(out_rows)
    state.fine_align_info = info
    report_phase_progress(
        progress,
        PHASE_FINE_ALIGN,
        step="fine_done",
        message=(
            f"Fine alignment — QA RMS={info.rms_after_mm:.4g} mm "
            f"(before {info.rms_before_mm:.4g} mm, n={info.n_pairs})."
        ),
    )
    return out_rows, info
