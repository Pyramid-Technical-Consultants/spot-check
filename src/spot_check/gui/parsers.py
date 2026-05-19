"""Parse user-entered numeric fields from the control panel."""

from __future__ import annotations

from spot_check import analysis
from spot_check import constants as sc_const


def parse_layer_gap_s(raw: str) -> float | None:
    try:
        v = float(str(raw).strip())
        if 0.0 < v < 3600.0:
            return v
    except (ValueError, TypeError):
        pass
    return None


def parse_refill_xy_tol_mm(raw: str) -> float | None:
    try:
        v = float(str(raw).strip())
        if 0.0 < v < 1_000.0:
            return v
    except (ValueError, TypeError):
        pass
    return None


def parse_auto_min_on_spot_weight_na(raw: str) -> float | None:
    try:
        v = float(str(raw).strip())
        if 0.0 <= v <= 1.0e15:
            return v
    except (ValueError, TypeError):
        pass
    return None


def parse_auto_min_episode_rows(raw: str) -> int | None:
    try:
        v = int(float(str(raw).strip()))
        if 1 <= v <= 10_000:
            return v
    except (ValueError, TypeError):
        pass
    return None


def parse_viterbi_penalty_mm2(raw: str) -> float | None:
    try:
        v = float(str(raw).strip())
        if 0.0 <= v <= 1.0e8:
            return v
    except (ValueError, TypeError):
        pass
    return None


def parse_bounds_xy_tick_mm(raw: str) -> float | None:
    try:
        v = float(str(raw).strip())
        if v == 0.0:
            return 0.0
        if 0.05 <= v <= 500.0:
            return v
    except (ValueError, TypeError):
        pass
    return None


def parse_aggregate_even_tail_n(raw: str) -> int | None:
    try:
        v = int(float(str(raw).strip()))
        mx = int(sc_const.AGGREGATE_EVEN_TAIL_MAX)
        if 0 <= v <= mx:
            return v
    except (ValueError, TypeError):
        pass
    return None


def plan_qa_thresholds_input_in_progress(pass_raw: str, warn_raw: str) -> bool:
    """True while the user is mid-edit (e.g. ``0`` or ``0.`` before ``0.5``)."""

    def partial(raw: str) -> bool:
        s = str(raw).strip()
        if not s:
            return True
        if s in ("-", "+", ".", "-.", "+."):
            return True
        return s.endswith(".")

    return partial(pass_raw) or partial(warn_raw)


def parse_plan_qa_thresholds(pass_raw: str, warn_raw: str) -> tuple[float, float] | None:
    if plan_qa_thresholds_input_in_progress(pass_raw, warn_raw):
        return None
    try:
        a = float(str(pass_raw).strip())
        b = float(str(warn_raw).strip())
        if 0.0 < a < b <= 500.0:
            return a, b
    except (ValueError, TypeError):
        pass
    return None


def spot_weight_mode_from_saved(raw: object) -> str:
    try:
        return analysis.normalize_measured_spot_weight_mode(str(raw))
    except (ValueError, TypeError):
        return sc_const.SPOT_WEIGHT_MODE_DEFAULT
