"""Format pipeline status lines from structured diagnostics."""

from __future__ import annotations

from spot_check.pipeline.diagnostics import AssignDiagnostics, QAResult


def format_qa_counts_label(qa: QAResult | None, *, enabled: bool) -> str:
    if not enabled:
        return "Counts: enable “Color measured spots by plan QA” above."
    if qa is None:
        return "Counts: — (need plan and CSV)"
    unit = "pp" if qa.qa_mode == "dose" else "mm"
    return (
        f"Measured spots: {qa.n_pass} pass · {qa.n_warn} warn · {qa.n_fail} fail "
        f"({unit} thresholds)"
    )


def format_auto_tuning_label(
    diag: AssignDiagnostics | None,
    *,
    assign_method: str,
) -> str:
    if diag is None or diag.auto_layer_params is None:
        return "Tuning: inferred from plan + CSV when Auto is selected."
    auto_p = diag.auto_layer_params
    if assign_method == "plan_sequential":
        return (
            "Deadtime = no A/B position fit; spans aligned to plan count "
            f"(≥{auto_p.min_episode_rows} row(s)/span)."
        )
    return (
        "Inferred: "
        f"Δt {auto_p.episode_gap_s:g} s · XY {auto_p.spot_xy_jump_mm:g} mm · "
        f"weight ≥{auto_p.min_on_spot_weight_na:.3g} nA · "
        f"≥{auto_p.min_episode_rows} row(s)/episode · "
        f"Viterbi {auto_p.viterbi_advance_penalty_mm2:g} mm²"
    )


def append_auto_meas_lines(
    meas_line: str,
    diag: AssignDiagnostics | None,
    *,
    assign_method: str,
    n_meas: int,
    n_plan_kept: int,
) -> str:
    if diag is None or diag.auto_layer_params is None:
        return meas_line
    auto_p = diag.auto_layer_params
    if assign_method == "plan_sequential":
        meas_line += " — plan-seq: one aggregated row per assigned spot"
        if n_plan_kept > 0 and n_meas != n_plan_kept:
            meas_line += f" — WARNING: {n_meas} measured vs {n_plan_kept} plan spots"
        return meas_line
    meas_line += (
        f" — auto Δt≥{auto_p.episode_gap_s:g} s, "
        f"XY>{auto_p.spot_xy_jump_mm:g} mm, "
        f"Viterbi {auto_p.viterbi_advance_penalty_mm2:g} mm²"
    )
    ep = diag.episode_diagnostics
    if ep is not None:
        meas_line += (
            f" — episodes raw={ep.n_raw_episodes}, "
            f"aligned={ep.n_after_align}/{ep.n_plan}"
        )
        if not ep.count_align_ok:
            meas_line += " (spot-count align incomplete)"
    return meas_line
