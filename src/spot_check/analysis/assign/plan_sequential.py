"""Plan-order assignment: first burst = plan spot 0, advance +1 after deadtime gaps only.

Does not import any other assignment implementation. Assumes delivery order matches the plan
CSV from index 0 (highest nominal-energy layer, first spot) onward.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from spot_check.analysis.assign.types import (
    AutoAssignResult,
    EpisodeSpan,
    PlanIndexArray,
    PlanSequentialAssignParams,
)
from spot_check.analysis.auto_columns import AutoFitColumns, position_fit_deadtime_mask
from spot_check.constants import (
    AUTO_PLAN_SEQ_ADVANCE_MARGIN_MM2,
    AUTO_PLAN_SEQ_CLUSTER_WINDOW,
)


def _xy_sqdist(x0: float, y0: float, x1: float, y1: float) -> float:
    dx = float(x0) - float(x1)
    dy = float(y0) - float(y1)
    return dx * dx + dy * dy


def _post_gap_centroid(
    cols: AutoFitColumns,
    dead: np.ndarray,
    end_i: int,
    *,
    window: int,
) -> tuple[float, float] | None:
    """Weighted centroid of the on-burst starting at ``end_i`` (do not cross the prior gap)."""
    idxs: list[int] = []
    j = int(end_i)
    win = max(1, int(window))
    while j >= 0 and len(idxs) < win:
        if dead[j]:
            break
        idxs.append(j)
        j -= 1
    if not idxs:
        return None
    ii = np.asarray(idxs, dtype=np.int64)
    ww = np.maximum(cols.weight[ii], 1e-18)
    den = float(ww.sum())
    if den <= 0.0:
        return float(cols.mx[end_i]), float(cols.my[end_i])
    cx = float(np.dot(cols.mx[ii], ww) / den)
    cy = float(np.dot(cols.my[ii], ww) / den)
    return cx, cy


def _post_gap_prefers_next_plan_spot(
    cols: AutoFitColumns,
    dead: np.ndarray,
    plan_xy: np.ndarray,
    plan_i: int,
    end_i: int,
    *,
    window: int,
    margin_mm2: float,
) -> bool:
    """True when post-gap XY is closer to ``plan[plan_i + 1]`` than ``plan[plan_i]``."""
    if plan_i >= int(plan_xy.shape[0]) - 1:
        return False
    cent = _post_gap_centroid(cols, dead, end_i, window=window)
    if cent is None:
        return False
    cx, cy = cent
    cur = plan_xy[plan_i]
    nxt = plan_xy[plan_i + 1]
    d_cur = _xy_sqdist(cx, cy, float(cur[0]), float(cur[1]))
    d_nxt = _xy_sqdist(cx, cy, float(nxt[0]), float(nxt[1]))
    return d_nxt + float(margin_mm2) < d_cur


def assign_plan_indices(
    cols: AutoFitColumns,
    plan_xy: np.ndarray,
    *,
    params: PlanSequentialAssignParams | None = None,
    min_rows_on_spot: int | None = None,
    cluster_window: int | None = None,
    advance_margin_mm2: float | None = None,
    cluster_radius_mm2: float | None = None,  # noqa: ARG001 — unused; kept for callers
) -> PlanIndexArray:
    """Per-row plan delivery index; ``-1`` on deadtime. Always starts at plan spot **0**."""
    if params is None:
        p = PlanSequentialAssignParams(
            min_rows_on_spot=1 if min_rows_on_spot is None else int(min_rows_on_spot),
            cluster_window=int(AUTO_PLAN_SEQ_CLUSTER_WINDOW)
            if cluster_window is None
            else int(cluster_window),
            advance_margin_mm2=float(AUTO_PLAN_SEQ_ADVANCE_MARGIN_MM2)
            if advance_margin_mm2 is None
            else float(advance_margin_mm2),
        )
    else:
        p = params
    n = len(cols)
    m = int(plan_xy.shape[0])
    out = np.full(n, -1, dtype=np.int32)
    if n == 0 or m == 0:
        return out

    dead = position_fit_deadtime_mask(cols)
    plan_i = 0
    in_break = False
    rows_on_spot = 0
    win = max(1, int(p.cluster_window))
    margin = float(p.advance_margin_mm2)

    for ri in range(n):
        if dead[ri]:
            in_break = True
            continue
        if plan_i >= m - 1:
            out[ri] = plan_i
            rows_on_spot += 1
            in_break = False
            continue
        if in_break and rows_on_spot >= 1:
            if _post_gap_prefers_next_plan_spot(
                cols,
                dead,
                plan_xy,
                plan_i,
                ri,
                window=win,
                margin_mm2=margin,
            ):
                plan_i += 1
                rows_on_spot = 0
            in_break = False
        out[ri] = plan_i
        rows_on_spot += 1
    return out


def sequential_spans_from_plan_indices(plan_idx: PlanIndexArray) -> list[EpisodeSpan]:
    """Contiguous on-spot runs sharing one plan index (deadtime rows omitted)."""
    n = int(plan_idx.size)
    spans: list[EpisodeSpan] = []
    i = 0
    while i < n:
        while i < n and int(plan_idx[i]) < 0:
            i += 1
        if i >= n:
            break
        pi = int(plan_idx[i])
        s = i
        i += 1
        while i < n and int(plan_idx[i]) == pi:
            i += 1
        spans.append((s, i))
    return spans


def plan_spot_index_per_span(
    plan_idx: PlanIndexArray, spans: list[EpisodeSpan]
) -> np.ndarray:
    """Plan delivery index for each span (``len(spans)``)."""
    if not spans:
        return np.zeros(0, dtype=np.int64)
    return np.fromiter((int(plan_idx[s]) for s, _ in spans), dtype=np.int64, count=len(spans))


class PlanSequentialAssigner:
    """Auto sub-assigner: plan-order delivery with deadtime gaps."""

    method = "plan_sequential"

    def assign(
        self,
        cols: AutoFitColumns,
        *,
        n_plan_spots: int,
        plan_xy: np.ndarray,
        spots_per_layer: Sequence[int],
        params: PlanSequentialAssignParams | None = None,
    ) -> AutoAssignResult:
        from spot_check.analysis.layers import delivery_layer_indices

        p = params or PlanSequentialAssignParams(
            cluster_window=int(AUTO_PLAN_SEQ_CLUSTER_WINDOW),
            advance_margin_mm2=float(AUTO_PLAN_SEQ_ADVANCE_MARGIN_MM2),
        )
        plan_idx = assign_plan_indices(cols, plan_xy, params=p)
        spans = sequential_spans_from_plan_indices(plan_idx)
        spot_pi = plan_spot_index_per_span(plan_idx, spans)
        plan_layers = delivery_layer_indices(n_plan_spots, spots_per_layer)
        layers = plan_layers[spot_pi]
        return AutoAssignResult(
            spans=spans,
            layer_index_per_span=layers,
            plan_index_per_row=plan_idx,
        )


assigner = PlanSequentialAssigner()

# Back-compat names used by tests and diagnostics.
assign_plan_indices_sequential = assign_plan_indices
