"""Sequential plan-order spot assignment for ``layer_mode='auto'``."""

from __future__ import annotations

import numpy as np

from spot_check.analysis.auto_columns import AutoFitColumns, position_fit_deadtime_mask
from spot_check.analysis.episodes import (
    EpisodeSpan,
    _xy_sqdist,
    align_episode_spans_to_plan_count_cols,
    refine_spans_with_plan_xy,
)
from spot_check.constants import (
    AUTO_PLAN_SEQ_ADVANCE_MARGIN_MM2,
    AUTO_PLAN_SEQ_CLUSTER_RADIUS_MM_DEFAULT,
    AUTO_PLAN_SEQ_CLUSTER_RADIUS_SCALE,
    AUTO_PLAN_SEQ_CLUSTER_WINDOW,
)

PlanIndexArray = np.ndarray


def infer_plan_seq_cluster_radius_mm2(plan_xy: np.ndarray) -> float:
    """On-spot cluster radius from median consecutive plan XY spacing."""
    pts = np.asarray(plan_xy, dtype=np.float64)
    if pts.shape[0] < 2:
        r_mm = float(AUTO_PLAN_SEQ_CLUSTER_RADIUS_MM_DEFAULT)
        return r_mm * r_mm
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    d = d[(d > 1e-6) & np.isfinite(d)]
    step = float(np.median(d)) if d.size else float(AUTO_PLAN_SEQ_CLUSTER_RADIUS_MM_DEFAULT)
    r_mm = max(3.0, step * float(AUTO_PLAN_SEQ_CLUSTER_RADIUS_SCALE))
    return r_mm * r_mm


def _spans_from_position_deadtime(dead: np.ndarray, n: int) -> list[EpisodeSpan]:
    """Contiguous on-spot runs where ``dead`` is false."""
    spans: list[EpisodeSpan] = []
    i = 0
    while i < n:
        while i < n and dead[i]:
            i += 1
        if i >= n:
            break
        s = i
        while i < n and not dead[i]:
            i += 1
        spans.append((s, i))
    return spans


def _post_gap_centroid(
    cols: AutoFitColumns,
    dead: np.ndarray,
    end_i: int,
    *,
    window: int,
) -> tuple[float, float] | None:
    """Weighted centroid from on-spot rows at/just before ``end_i`` (post-gap window only)."""
    idxs: list[int] = []
    j = int(end_i)
    win = max(1, int(window))
    while j >= 0 and len(idxs) < win:
        if not dead[j]:
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


def _assign_plan_indices_streaming(
    cols: AutoFitColumns,
    plan_xy: np.ndarray,
    dead: np.ndarray,
    *,
    min_rows_on_spot: int,
    cluster_window: int,
    margin_mm2: float,
) -> PlanIndexArray:
    """Row walk: advance by +1 after a gap only when post-gap XY favors the next plan spot."""
    n = len(cols)
    m = int(plan_xy.shape[0])
    out = np.full(n, -1, dtype=np.int32)
    plan_i = 0
    in_break = False
    rows_on_spot = 0
    min_on = max(1, int(min_rows_on_spot))
    win = max(1, int(cluster_window))

    for ri in range(n):
        if dead[ri]:
            in_break = True
            continue
        if plan_i >= m - 1:
            out[ri] = plan_i
            rows_on_spot += 1
            in_break = False
            continue
        if in_break and rows_on_spot >= min_on:
            if _post_gap_prefers_next_plan_spot(
                cols,
                dead,
                plan_xy,
                plan_i,
                ri,
                window=win,
                margin_mm2=margin_mm2,
            ):
                plan_i += 1
                rows_on_spot = 0
            in_break = False
        out[ri] = plan_i
        rows_on_spot += 1
    return out


def assign_plan_indices_sequential(
    cols: AutoFitColumns,
    plan_xy: np.ndarray,
    *,
    min_rows_on_spot: int = 1,
    cluster_radius_mm2: float | None = None,
    cluster_window: int = AUTO_PLAN_SEQ_CLUSTER_WINDOW,
    advance_margin_mm2: float = AUTO_PLAN_SEQ_ADVANCE_MARGIN_MM2,
) -> PlanIndexArray:
    """Per-row plan spot index in delivery order; ``-1`` for deadtime rows.

    Deadtime = neither Fit Mean Position A nor B (see :func:`position_fit_deadtime_mask`).
    Walk acquisition time from the first plan spot (index 0 = first spot of the
    highest nominal-energy layer). Position-fit spans are merged to the plan spot count
    and boundaries are nudged with plan XY; each contiguous on-spot burst (as few as one row)
    maps to one delivery index
    (``+1`` only between consecutive spans). Rows in the same burst always share the same
    plan index. If alignment fails, falls back to advancing by one plan slot after a gap
    only when post-gap fit XY favors the next spot.
    """
    n = len(cols)
    m = int(plan_xy.shape[0])
    out = np.full(n, -1, dtype=np.int32)
    if n == 0 or m == 0:
        return out

    dead = position_fit_deadtime_mask(cols)
    spans = _spans_from_position_deadtime(dead, n)
    if not spans:
        return out

    aligned, diag = align_episode_spans_to_plan_count_cols(
        cols, spans, m, plan_xy=plan_xy
    )
    if diag.count_align_ok and len(aligned) == m:
        aligned = refine_spans_with_plan_xy(cols, aligned, plan_xy)
        for pi, (s, e) in enumerate(aligned):
            if pi >= m or e <= s:
                continue
            out[s:e] = pi
        return out

    if cluster_radius_mm2 is None:
        cluster_radius_mm2 = infer_plan_seq_cluster_radius_mm2(plan_xy)
    del cluster_radius_mm2  # used only by callers / future tuning
    return _assign_plan_indices_streaming(
        cols,
        plan_xy,
        dead,
        min_rows_on_spot=min_rows_on_spot,
        cluster_window=cluster_window,
        margin_mm2=float(advance_margin_mm2),
    )


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
