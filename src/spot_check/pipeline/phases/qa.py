"""QA phase — plan vs measured pass/warn/fail tier counts."""

from __future__ import annotations

from spot_check.analysis import plan_qa
from spot_check.pipeline.diagnostics import QAResult
from spot_check.pipeline.progress import ProgressSink, report_phase_progress
from spot_check.pipeline.types import PHASE_QA, PipelineState


def run_qa_phase(
    state: PipelineState,
    progress: ProgressSink,
    *,
    planned: list[tuple[float, float, float]],
    measured: list[tuple[float, ...]],
    qa_mode: str,
    pass_thr: float,
    warn_thr: float,
    plan_mu: object,
    enabled: bool,
) -> QAResult | None:
    """Compute QA tier counts; returns None when QA coloring is disabled."""
    if not enabled or not planned or not measured:
        return None

    report_phase_progress(
        progress, PHASE_QA, step="qa_start", message=f"Computing plan QA ({qa_mode})…"
    )
    try:
        n_pass, n_warn, n_fail = plan_qa.plan_qa_measured_spot_pass_warn_fail(
            planned,
            measured,
            qa_mode=qa_mode,
            pass_thr=float(pass_thr),
            warn_thr=float(warn_thr),
            plan_mu=plan_mu,  # type: ignore[arg-type]
            a_is_x=False,
        )
    except ValueError:
        report_phase_progress(
            progress, PHASE_QA, step="qa_error", message="QA thresholds invalid — skipped."
        )
        return None

    unit = "pp" if qa_mode == "dose" else "mm"
    report_phase_progress(
        progress,
        PHASE_QA,
        step="qa_done",
        message=(
            f"QA complete — {n_pass} pass · {n_warn} warn · {n_fail} fail ({unit})."
        ),
    )
    result = QAResult(
        n_pass=n_pass,
        n_warn=n_warn,
        n_fail=n_fail,
        qa_mode=qa_mode,
        pass_thr=float(pass_thr),
        warn_thr=float(warn_thr),
    )
    state.qa_result = result
    return result
