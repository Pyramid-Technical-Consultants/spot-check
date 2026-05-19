"""Signal-based episode segmentation for ``layer_mode='auto'`` (no Gate Counter)."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from spot_check.analysis.auto_columns import AutoFitColumns
from spot_check.constants import (
    AUTO_EDGE_DEAD_RATIO_DEFAULT,
    AUTO_EDGE_ON_WEIGHT_FRAC,
    AUTO_EDGE_ROLLING_WINDOW,
    AUTO_EDGE_TINY_MERGE_ROWS,
    AUTO_EPISODE_MERGE_DT_MM2_PER_S,
    AUTO_PLAN_BOUNDARY_REFINE_PASSES,
    AUTO_PLAN_BOUNDARY_REFINE_ROWS,
    AUTO_PLAN_MERGE_BLEND,
)

# Half-open row index range into a contiguous table.
EpisodeSpan = tuple[int, int]

FinBuf = tuple[float, float, float, float, int, float, float, float]


@dataclass(frozen=True)
class SpanAggregate:
    """One plan spot after collapsing an episode span."""

    a: float
    b: float
    mx: float
    my: float
    weight: float
    ch_n: float
    pcd: int
    sa: float | None
    sb: float | None
    mx_pp: float | None
    my_pp: float | None


@dataclass(frozen=True)
class AutoFitRow:
    """One CSV row after fit filtering (plan-imputed XY + detector A/B)."""

    t: float
    mx: float
    my: float
    a: float
    b: float
    mx_p: float | None
    my_p: float | None
    weight: float
    ch_n: float
    pcd: int
    sa: float | None
    sb: float | None


@dataclass(frozen=True)
class AutoEpisodeDiagnostics:
    n_raw_episodes: int
    n_after_align: int
    n_plan: int
    count_align_ok: bool


_last_diag: AutoEpisodeDiagnostics | None = None


def last_auto_episode_diagnostics() -> AutoEpisodeDiagnostics | None:
    return _last_diag


def _set_last_diag(d: AutoEpisodeDiagnostics | None) -> None:
    global _last_diag
    _last_diag = d


def _ratio_or_nan(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full(num.shape, np.nan, dtype=np.float64)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def _weighted_mean_nan(vals: np.ndarray, weights: np.ndarray) -> float | None:
    ok = np.isfinite(vals)
    if not ok.any():
        return None
    w = weights[ok]
    v = vals[ok]
    den = float(w.sum())
    if den <= 0.0:
        return None
    return float(np.dot(v, w) / den)


def _episode_bincount_stats(
    cols: AutoFitColumns, spans: list[EpisodeSpan]
) -> tuple[np.ndarray, ...]:
    """Weighted per-episode sums for all rows (``O(n_rows)``)."""
    k = len(spans)
    n = len(cols)
    ep = np.full(n, -1, dtype=np.int32)
    for i, (s, e) in enumerate(spans):
        ep[s:e] = i

    ok = ep >= 0
    if not ok.any():
        z = np.zeros(k, dtype=np.float64)
        return (z, z, z, z, z, np.zeros(k, dtype=np.int32), z, z, z, z, z, z, z, z, z, z)

    ep_ok = ep[ok]
    w_all = np.maximum(cols.weight, 1e-18)
    w = w_all[ok]
    sum_w = np.bincount(ep_ok, weights=w, minlength=k)
    sum_aw = np.bincount(ep_ok, weights=w * cols.a[ok], minlength=k)
    sum_bw = np.bincount(ep_ok, weights=w * cols.b[ok], minlength=k)
    sum_mx = np.bincount(ep_ok, weights=w * cols.mx[ok], minlength=k)
    sum_my = np.bincount(ep_ok, weights=w * cols.my[ok], minlength=k)

    pcd_max = np.zeros(k, dtype=np.int32)
    np.maximum.at(pcd_max, ep_ok, cols.pcd[ok])

    sa_m = ok & np.isfinite(cols.sa)
    sb_m = ok & np.isfinite(cols.sb)
    sum_sa = np.zeros(k, dtype=np.float64)
    sum_sa_w = np.zeros(k, dtype=np.float64)
    sum_sb = np.zeros(k, dtype=np.float64)
    sum_sb_w = np.zeros(k, dtype=np.float64)
    if sa_m.any():
        e_sa = ep[sa_m]
        w_sa = w_all[sa_m]
        np.add.at(sum_sa, e_sa, cols.sa[sa_m] * w_sa)
        np.add.at(sum_sa_w, e_sa, w_sa)
    if sb_m.any():
        e_sb = ep[sb_m]
        w_sb = w_all[sb_m]
        np.add.at(sum_sb, e_sb, cols.sb[sb_m] * w_sb)
        np.add.at(sum_sb_w, e_sb, w_sb)

    ch_m = ok & np.isfinite(cols.ch_n) & (cols.ch_n > 0)
    sum_ch = np.zeros(k, dtype=np.float64)
    sum_ch_w = np.zeros(k, dtype=np.float64)
    if ch_m.any():
        e_ch = ep[ch_m]
        w_ch = w_all[ch_m]
        np.add.at(sum_ch, e_ch, cols.ch_n[ch_m] * w_ch)
        np.add.at(sum_ch_w, e_ch, w_ch)

    pp_m = ok & (np.isfinite(cols.mx_p) | np.isfinite(cols.my_p))
    sum_pp_mx = np.zeros(k, dtype=np.float64)
    sum_pp_mx_w = np.zeros(k, dtype=np.float64)
    sum_pp_my = np.zeros(k, dtype=np.float64)
    sum_pp_my_w = np.zeros(k, dtype=np.float64)
    if pp_m.any():
        epp = ep[pp_m]
        wpp = w_all[pp_m]
        mxv = cols.mx_p[pp_m]
        myv = cols.my_p[pp_m]
        mx_ok = np.isfinite(mxv)
        my_ok = np.isfinite(myv)
        if mx_ok.any():
            np.add.at(sum_pp_mx, epp[mx_ok], wpp[mx_ok] * mxv[mx_ok])
            np.add.at(sum_pp_mx_w, epp[mx_ok], wpp[mx_ok])
        if my_ok.any():
            np.add.at(sum_pp_my, epp[my_ok], wpp[my_ok] * myv[my_ok])
            np.add.at(sum_pp_my_w, epp[my_ok], wpp[my_ok])

    return (
        sum_w,
        sum_aw,
        sum_bw,
        sum_mx,
        sum_my,
        pcd_max,
        sum_sa,
        sum_sa_w,
        sum_sb,
        sum_sb_w,
        sum_ch,
        sum_ch_w,
        sum_pp_mx,
        sum_pp_mx_w,
        sum_pp_my,
        sum_pp_my_w,
    )


def fin_bufs_from_spans_batch(cols: AutoFitColumns, spans: list[EpisodeSpan]) -> list[FinBuf]:
    """Episode fin buffers for merge/split alignment (one ``O(n_rows)`` pass)."""
    k = len(spans)
    if k == 0:
        return []
    st = _episode_bincount_stats(cols, spans)
    (
        sum_w,
        sum_aw,
        sum_bw,
        _sum_mx,
        _sum_my,
        pcd_max,
        sum_sa,
        sum_sa_w,
        sum_sb,
        sum_sb_w,
        sum_ch,
        sum_ch_w,
        _pp_mx,
        _pp_mx_w,
        _pp_my,
        _pp_my_w,
    ) = st
    sw = sum_w.astype(np.float64, copy=False)
    sa_v = _ratio_or_nan(sum_sa, sum_sa_w)
    sb_v = _ratio_or_nan(sum_sb, sum_sb_w)
    ch_n = _ratio_or_nan(sum_ch, sum_ch_w)
    ch_n = np.where(sum_ch_w > 0, ch_n, sw)
    a_mean = sum_aw / sw
    b_mean = sum_bw / sw
    pcd_out = np.where(pcd_max > 0, pcd_max, 0)
    return [
        (
            float(a_mean[i]),
            float(b_mean[i]),
            0.0,
            float(sw[i]),
            int(pcd_out[i]),
            float(sa_v[i]),
            float(sb_v[i]),
            float(ch_n[i]),
        )
        for i in range(k)
    ]


def aggregate_spans_batch(
    cols: AutoFitColumns, spans: list[EpisodeSpan]
) -> list[SpanAggregate]:
    """Collapse all episode spans in one pass over rows (``O(n_rows)``)."""
    k = len(spans)
    if k == 0:
        return []
    (
        sum_w,
        sum_aw,
        sum_bw,
        sum_mx,
        sum_my,
        pcd_max,
        sum_sa,
        sum_sa_w,
        sum_sb,
        sum_sb_w,
        sum_ch,
        sum_ch_w,
        sum_pp_mx,
        sum_pp_mx_w,
        sum_pp_my,
        sum_pp_my_w,
    ) = _episode_bincount_stats(cols, spans)

    out: list[SpanAggregate] = []
    for i in range(k):
        sw = float(sum_w[i])
        sa_v = sum_sa[i] / sum_sa_w[i] if sum_sa_w[i] > 0 else float("nan")
        sb_v = sum_sb[i] / sum_sb_w[i] if sum_sb_w[i] > 0 else float("nan")
        sa_o = None if sa_v != sa_v else float(sa_v)
        sb_o = None if sb_v != sb_v else float(sb_v)
        if sum_ch_w[i] > 0:
            ch_n = float(sum_ch[i] / sum_ch_w[i])
        else:
            ch_n = sw
        pcd_i = int(pcd_max[i])
        mx_pp = float(sum_pp_mx[i] / sum_pp_mx_w[i]) if sum_pp_mx_w[i] > 0 else None
        my_pp = float(sum_pp_my[i] / sum_pp_my_w[i]) if sum_pp_my_w[i] > 0 else None
        if mx_pp is not None and not (mx_pp == mx_pp):
            mx_pp = None
        if my_pp is not None and not (my_pp == my_pp):
            my_pp = None
        out.append(
            SpanAggregate(
                a=float(sum_aw[i] / sw),
                b=float(sum_bw[i] / sw),
                mx=float(sum_mx[i] / sw),
                my=float(sum_my[i] / sw),
                weight=sw,
                ch_n=ch_n,
                pcd=pcd_i if pcd_i > 0 else 0,
                sa=sa_o,
                sb=sb_o,
                mx_pp=mx_pp,
                my_pp=my_pp,
            )
        )
    return out


def _weighted_merge_pair(xl: float, xr: float, sw_l: float, sw_r: float) -> float:
    """Merge two episode scalars; skip NaN sides (matches batch bincount aggregates)."""
    xl_ok = xl == xl
    xr_ok = xr == xr
    if xl_ok and xr_ok:
        return (xl * sw_l + xr * sw_r) / (sw_l + sw_r)
    if xl_ok:
        return xl
    if xr_ok:
        return xr
    return float("nan")


def _merge_fin_bufs(lt: FinBuf, rt: FinBuf) -> FinBuf:
    sw_l = max(float(lt[3]), 1e-18)
    sw_r = max(float(rt[3]), 1e-18)
    sw = sw_l + sw_r
    pcd_l, pcd_r = int(lt[4]), int(rt[4])
    pcd_out = max(pcd_l, pcd_r) if (pcd_l > 0 or pcd_r > 0) else 0
    merge = _weighted_merge_pair
    return (
        merge(float(lt[0]), float(rt[0]), sw_l, sw_r),
        merge(float(lt[1]), float(rt[1]), sw_l, sw_r),
        merge(float(lt[2]), float(rt[2]), sw_l, sw_r),
        sw,
        pcd_out,
        merge(float(lt[5]), float(rt[5]), sw_l, sw_r),
        merge(float(lt[6]), float(rt[6]), sw_l, sw_r),
        merge(float(lt[7]), float(rt[7]), sw_l, sw_r),
    )


def _merge_pair_cost_cols(
    fins: Sequence[FinBuf],
    cols: AutoFitColumns,
    spans: Sequence[EpisodeSpan],
    i: int,
    *,
    dt_coeff: float,
    plan_xy: np.ndarray | None = None,
    row_w: np.ndarray | None = None,
) -> float:
    if plan_xy is not None and row_w is not None and i + 1 < len(plan_xy):
        sig = _merge_pair_cost_signal(fins, cols, spans, i, dt_coeff=dt_coeff)
        plan_c = _merge_pair_cost_plan(cols, spans, i, plan_xy, row_w)
        blend = float(AUTO_PLAN_MERGE_BLEND)
        return blend * plan_c + (1.0 - blend) * sig
    return _merge_pair_cost_signal(fins, cols, spans, i, dt_coeff=dt_coeff)


def _merge_pair_cost_signal(
    fins: Sequence[FinBuf],
    cols: AutoFitColumns,
    spans: Sequence[EpisodeSpan],
    i: int,
    *,
    dt_coeff: float,
) -> float:
    lt, rt = fins[i], fins[i + 1]
    da = float(lt[0]) - float(rt[0])
    db = float(lt[1]) - float(rt[1])
    dist_sq = da * da + db * db
    left_end = spans[i][1] - 1
    right_start = spans[i + 1][0]
    t_gap = float(cols.t[right_start]) - float(cols.t[left_end])
    return dist_sq + float(dt_coeff) * (max(0.0, t_gap) ** 2)


def _xy_sqdist(ax: float, ay: float, bx: float, by: float) -> float:
    dx = ax - bx
    dy = ay - by
    return dx * dx + dy * dy


def _banded_monotone_plan_path(
    cents: np.ndarray,
    plan: np.ndarray,
    *,
    window: int = 3,
    dup_penalty_mm2: float = 0.0,
) -> np.ndarray:
    """Monotone plan-index path for episode centroids within a diagonal band.

    Episode *i* maps to plan index ``path[i]`` (non-decreasing) minimizing squared
    XY distance to ``plan[path[i]]``, with ``|path[i] - i| <= window`` and an optional
    penalty when ``path[i] == path[i - 1]``.
    """
    n = int(cents.shape[0])
    m = int(plan.shape[0])
    if n == 0:
        return np.zeros(0, dtype=np.int32)
    if m == 0:
        raise ValueError("plan must have at least one spot")
    w = max(0, int(window))
    inf = np.inf
    emit = np.empty((n, m), dtype=np.float64)
    for i in range(n):
        cx, cy = float(cents[i, 0]), float(cents[i, 1])
        for j in range(m):
            emit[i, j] = _xy_sqdist(cx, cy, float(plan[j, 0]), float(plan[j, 1]))
    C = np.full((n, m), inf, dtype=np.float64)
    back = np.zeros((n, m), dtype=np.int32)
    j0_lo = max(0, -w)
    j0_hi = min(m - 1, w)
    for j in range(j0_lo, j0_hi + 1):
        C[0, j] = emit[0, j]
        back[0, j] = j
    for i in range(1, n):
        j_lo = max(0, i - w)
        j_hi = min(m - 1, i + w)
        for j in range(j_lo, j_hi + 1):
            k_lo = max(0, j - w, j_lo)
            best = inf
            best_k = j_lo
            for k in range(k_lo, j + 1):
                if k > j_hi:
                    break
                if C[i - 1, k] >= inf:
                    continue
                trans = float(C[i - 1, k])
                if j == k and dup_penalty_mm2 > 0.0:
                    trans += float(dup_penalty_mm2)
                if trans < best:
                    best = trans
                    best_k = k
            if best < inf:
                C[i, j] = emit[i, j] + best
                back[i, j] = best_k
    j_end_lo = max(0, n - 1 - w)
    j_end_hi = min(m - 1, n - 1 + w)
    j_end = int(j_end_lo + np.argmin(C[n - 1, j_end_lo : j_end_hi + 1]))
    path = np.zeros(n, dtype=np.int32)
    j = j_end
    for i in range(n - 1, -1, -1):
        path[i] = j
        if i > 0:
            j = int(back[i, j])
    return path


def _span_xy_centroid(
    cols: AutoFitColumns,
    span: EpisodeSpan,
    row_w: np.ndarray,
) -> tuple[float, float]:
    s, e = span
    if e <= s:
        return float(cols.mx[s]), float(cols.my[s])
    ww = row_w[s:e]
    den = float(ww.sum())
    if den <= 0.0:
        return float(cols.mx[s]), float(cols.my[s])
    mx = float(np.dot(cols.mx[s:e], ww) / den)
    my = float(np.dot(cols.my[s:e], ww) / den)
    return mx, my


def _plan_pair_cost(
    cols: AutoFitColumns,
    left: EpisodeSpan,
    right: EpisodeSpan,
    plan_a: np.ndarray,
    plan_b: np.ndarray,
    row_w: np.ndarray,
) -> float:
    c0 = _span_xy_centroid(cols, left, row_w)
    c1 = _span_xy_centroid(cols, right, row_w)
    return _xy_sqdist(c0[0], c0[1], float(plan_a[0]), float(plan_a[1])) + _xy_sqdist(
        c1[0], c1[1], float(plan_b[0]), float(plan_b[1])
    )


def _merge_pair_cost_plan(
    cols: AutoFitColumns,
    spans: Sequence[EpisodeSpan],
    i: int,
    plan_xy: np.ndarray,
    row_w: np.ndarray,
) -> float:
    """Low cost when merging episodes *i* and *i+1* best matches plan spot *i*."""
    s0, e0 = spans[i]
    s1, e1 = spans[i + 1]
    ww0 = float(row_w[s0:e0].sum())
    ww1 = float(row_w[s1:e1].sum())
    sw = ww0 + ww1
    if sw <= 0.0:
        cm = _span_xy_centroid(cols, spans[i], row_w)
    else:
        c0 = _span_xy_centroid(cols, spans[i], row_w)
        c1 = _span_xy_centroid(cols, spans[i + 1], row_w)
        cm = (
            (c0[0] * ww0 + c1[0] * ww1) / sw,
            (c0[1] * ww0 + c1[1] * ww1) / sw,
        )
    p0 = plan_xy[i]
    return min(
        _xy_sqdist(cm[0], cm[1], float(p0[0]), float(p0[1])),
        _xy_sqdist(cm[0], cm[1], float(plan_xy[i + 1][0]), float(plan_xy[i + 1][1])),
    )


def refine_spans_with_plan_xy(
    cols: AutoFitColumns,
    spans: list[EpisodeSpan],
    plan_xy: np.ndarray,
    *,
    window_rows: int = AUTO_PLAN_BOUNDARY_REFINE_ROWS,
    passes: int = AUTO_PLAN_BOUNDARY_REFINE_PASSES,
) -> list[EpisodeSpan]:
    """Nudge inter-spot row boundaries using plan XY + measured centroids."""
    k = len(spans)
    if k < 2 or plan_xy.shape[0] < k:
        return spans
    row_w = delivery_row_weights(cols)
    win = max(2, int(window_rows))
    out = list(spans)
    for _ in range(max(1, int(passes))):
        changed = False
        for i in range(k - 1):
            if i + 1 >= plan_xy.shape[0]:
                break
            s0, e0 = out[i]
            s1, e1 = out[i + 1]
            if e1 <= s0 + 1:
                continue
            lo = max(s0 + 1, e0 - win, s1 - win)
            hi = min(e1 - 1, e0 + win, s1 + win)
            if hi <= lo:
                continue
            pa = plan_xy[i]
            pb = plan_xy[i + 1]
            best_split = e0
            best_cost = _plan_pair_cost(cols, (s0, e0), (s1, e1), pa, pb, row_w)
            for split in range(lo, hi + 1):
                if split <= s0 or split >= e1:
                    continue
                cost = _plan_pair_cost(cols, (s0, split), (split, e1), pa, pb, row_w)
                if cost < best_cost - 1e-9:
                    best_cost = cost
                    best_split = split
            if best_split != e0:
                out[i] = (s0, best_split)
                out[i + 1] = (best_split, e1)
                changed = True
        if not changed:
            break
    return out


def _best_plan_split_in_span(
    cols: AutoFitColumns,
    span: EpisodeSpan,
    plan_a: np.ndarray,
    plan_b: np.ndarray,
    row_w: np.ndarray,
) -> int:
    """Row index *k* in ``[s+1, e-1]`` minimizing plan-pair cost for a two-spot split."""
    s, e = span
    if e - s < 2:
        return -1
    best_k = -1
    best_cost = float("inf")
    for k in range(s + 1, e):
        cost = _plan_pair_cost(cols, (s, k), (k, e), plan_a, plan_b, row_w)
        if cost < best_cost:
            best_cost = cost
            best_k = k
    return best_k


def _span_max_internal_gap_cols(cols: AutoFitColumns, span: EpisodeSpan) -> tuple[int, float]:
    s, e = span
    n = e - s
    if n < 2:
        return -1, -1.0
    dt = np.diff(cols.t[s:e])
    k = int(np.argmax(dt)) + 1
    return k, float(dt[k - 1])


def _merge_short_spans(spans: list[EpisodeSpan], min_rows: int) -> None:
    if min_rows <= 1:
        return
    changed = True
    safety = 0
    max_passes = max(len(spans) + 4, 4)
    while changed and safety < max_passes:
        safety += 1
        changed = False
        for i, (s, e) in enumerate(spans):
            if e - s >= min_rows:
                continue
            if i > 0:
                ps, _ = spans[i - 1]
                spans[i - 1] = (ps, e)
                spans.pop(i)
                changed = True
                break
            if i + 1 < len(spans):
                _, ne = spans[i + 1]
                spans[i] = (s, ne)
                spans.pop(i + 1)
                changed = True
                break


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling mean; output length always matches *x* (unlike ``convolve(..., same)``)."""
    n = int(x.size)
    if n == 0:
        return x
    w = max(1, min(int(window), n))
    if w == 1:
        return np.asarray(x, dtype=np.float64)
    k = np.ones(w, dtype=np.float64) / w
    full = np.convolve(x, k, mode="full")
    start = (w - 1) // 2
    return full[start : start + n]


def _deadtime_mask(
    ch: np.ndarray,
    fa: np.ndarray,
    *,
    dead_ratio: float,
    window: int = AUTO_EDGE_ROLLING_WINDOW,
) -> np.ndarray:
    """Rows with both IX512 sum and fit-A below a rolling baseline × *dead_ratio*."""
    rm_ch = _rolling_mean(ch, window)
    rm_fa = _rolling_mean(fa, window)
    ratio = float(dead_ratio)
    return (ch <= rm_ch * ratio) & (fa <= rm_fa * ratio)


def _on_spans_from_deadtime(dead: np.ndarray) -> list[EpisodeSpan]:
    """Contiguous non-deadtime runs (each delivered spot)."""
    n = int(dead.size)
    if n <= 0:
        return []
    spans: list[EpisodeSpan] = []
    s = 0
    while s < n:
        while s < n and dead[s]:
            s += 1
        if s >= n:
            break
        e = s + 1
        while e < n and not dead[e]:
            e += 1
        if e > s:
            spans.append((s, e))
        s = e
    return spans


def _merge_tiny_forward_spans(
    spans: list[EpisodeSpan],
    *,
    max_rows: int,
) -> list[EpisodeSpan]:
    """Merge short false on-spot glitches into the following episode."""
    cap = max(0, int(max_rows))
    if cap <= 0 or len(spans) < 2:
        return spans
    out: list[EpisodeSpan] = []
    i = 0
    while i < len(spans):
        s, e = spans[i]
        while (e - s) <= cap and i + 1 < len(spans):
            i += 1
            e = spans[i][1]
        out.append((s, e))
        i += 1
    return out


def delivery_row_weights(
    cols: AutoFitColumns,
    *,
    window: int = AUTO_EDGE_ROLLING_WINDOW,
    on_frac: float = AUTO_EDGE_ON_WEIGHT_FRAC,
) -> np.ndarray:
    """Down-weight deadtime / off-spot rows for episode aggregation (numpy-only rolling mean)."""
    ch, fa = cols.ch_n, cols.fit_a
    rm_ch = _rolling_mean(ch, window)
    rm_fa = _rolling_mean(fa, window)
    on = (ch >= rm_ch * on_frac) & (fa >= rm_fa * on_frac)
    return np.maximum(cols.weight, 1e-18) * on.astype(np.float64)


def cols_with_delivery_weights(cols: AutoFitColumns) -> AutoFitColumns:
    """Copy with aggregation weights scaled by on-spot signal (channel sum + fit A)."""
    from dataclasses import replace

    return replace(cols, weight=delivery_row_weights(cols))


def segment_into_episodes_cols(
    cols: AutoFitColumns,
    *,
    episode_gap_s: float,
    min_on_spot_weight_na: float,
    spot_xy_jump_mm: float,
    min_episode_rows: int,
    dead_ratio: float = AUTO_EDGE_DEAD_RATIO_DEFAULT,
    tiny_merge_rows: int = AUTO_EDGE_TINY_MERGE_ROWS,
) -> list[EpisodeSpan]:
    """Segment on deadtime between spots (low IX512 sum + fit-A vs rolling baseline)."""
    del episode_gap_s, min_on_spot_weight_na, spot_xy_jump_mm  # legacy kwargs
    n = len(cols)
    if n == 0:
        return []
    dead = _deadtime_mask(cols.ch_n, cols.fit_a, dead_ratio=dead_ratio)
    spans = _on_spans_from_deadtime(dead)
    spans = _merge_tiny_forward_spans(spans, max_rows=tiny_merge_rows)
    _merge_short_spans(spans, min_episode_rows)
    return [sp for sp in spans if sp[1] > sp[0]]


def align_episode_spans_to_plan_count_cols(
    cols: AutoFitColumns,
    spans: list[EpisodeSpan],
    n_plan: int,
    *,
    merge_dt_coeff: float = AUTO_EPISODE_MERGE_DT_MM2_PER_S,
    plan_xy: np.ndarray | None = None,
) -> tuple[list[EpisodeSpan], AutoEpisodeDiagnostics]:
    """Merge/split spans until ``len(spans) == n_plan``; optional plan XY refines boundaries."""
    raw_n = len(spans)
    if n_plan <= 0:
        diag_bad = AutoEpisodeDiagnostics(
            n_raw_episodes=raw_n,
            n_after_align=len(spans),
            n_plan=n_plan,
            count_align_ok=False,
        )
        _set_last_diag(diag_bad)
        return spans, diag_bad

    row_w = delivery_row_weights(cols) if plan_xy is not None else None
    work = list(spans)

    if raw_n == n_plan:
        if plan_xy is not None and row_w is not None:
            work = refine_spans_with_plan_xy(cols, work, plan_xy)
        diag_ok = AutoEpisodeDiagnostics(
            n_raw_episodes=raw_n,
            n_after_align=len(work),
            n_plan=n_plan,
            count_align_ok=True,
        )
        _set_last_diag(diag_ok)
        return work, diag_ok

    fins = fin_bufs_from_spans_batch(cols, work)
    count_align_ok = True

    def merge_cost(i: int) -> float:
        return _merge_pair_cost_cols(
            fins,
            cols,
            work,
            i,
            dt_coeff=merge_dt_coeff,
            plan_xy=plan_xy,
            row_w=row_w,
        )

    heap: list[tuple[float, int]] = []
    for i in range(len(work) - 1):
        heapq.heappush(heap, (merge_cost(i), i))

    while len(work) > n_plan and heap:
        c_heap, i = heapq.heappop(heap)
        if i >= len(work) - 1:
            continue
        c_now = merge_cost(i)
        if c_now > c_heap + 1e-12:
            heapq.heappush(heap, (c_now, i))
            continue
        s0, _ = work[i]
        _, e1 = work[i + 1]
        work[i] = (s0, e1)
        work.pop(i + 1)
        fins[i] = _merge_fin_bufs(fins[i], fins.pop(i + 1))
        for j in (i - 1, i):
            if 0 <= j < len(work) - 1:
                heapq.heappush(heap, (merge_cost(j), j))

    while len(work) < n_plan:
        best_ei = -1
        best_k = -1
        best_key = -1.0
        for ei, (s, e) in enumerate(work):
            if e - s < 2:
                continue
            if plan_xy is not None and row_w is not None and ei + 1 < plan_xy.shape[0]:
                k = _best_plan_split_in_span(
                    cols, (s, e), plan_xy[ei], plan_xy[ei + 1], row_w
                )
                if k < 0:
                    continue
                left, right = (s, k), (k, e)
                key = _plan_pair_cost(cols, left, right, plan_xy[ei], plan_xy[ei + 1], row_w)
            else:
                k, dt = _span_max_internal_gap_cols(cols, (s, e))
                if k < 0:
                    continue
                key = dt
            if best_ei < 0 or key < best_key:
                best_ei, best_k, best_key = ei, k, key
        if best_ei < 0:
            count_align_ok = False
            break
        s, e = work[best_ei]
        left: EpisodeSpan = (s, best_k)
        right: EpisodeSpan = (best_k, e)
        work[best_ei : best_ei + 1] = [left, right]

    if len(work) != n_plan:
        count_align_ok = False

    if count_align_ok and plan_xy is not None and len(work) == n_plan:
        work = refine_spans_with_plan_xy(cols, work, plan_xy)

    diag = AutoEpisodeDiagnostics(
        n_raw_episodes=raw_n,
        n_after_align=len(work),
        n_plan=n_plan,
        count_align_ok=count_align_ok and len(work) == n_plan,
    )
    _set_last_diag(diag)
    return work, diag


def segment_align_auto_columns(
    cols: AutoFitColumns,
    *,
    n_plan_spots: int,
    episode_gap_s: float,
    min_on_spot_weight_na: float,
    spot_xy_jump_mm: float,
    min_episode_rows: int,
    dead_ratio: float = AUTO_EDGE_DEAD_RATIO_DEFAULT,
    tiny_merge_rows: int = AUTO_EDGE_TINY_MERGE_ROWS,
    plan_xy: np.ndarray | None = None,
) -> tuple[list[EpisodeSpan], AutoEpisodeDiagnostics]:
    spans = segment_into_episodes_cols(
        cols,
        episode_gap_s=episode_gap_s,
        min_on_spot_weight_na=min_on_spot_weight_na,
        spot_xy_jump_mm=spot_xy_jump_mm,
        min_episode_rows=min_episode_rows,
        dead_ratio=dead_ratio,
        tiny_merge_rows=tiny_merge_rows,
    )
    if plan_xy is not None and len(spans) >= 2 and n_plan_spots >= 2:
        n_ref = min(len(spans), n_plan_spots)
        head = refine_spans_with_plan_xy(
            cols, spans[:n_ref], plan_xy[:n_ref]
        )
        spans = head + spans[n_ref:]
    return align_episode_spans_to_plan_count_cols(
        cols, spans, n_plan_spots, plan_xy=plan_xy
    )


def _optional_float(v: float | None) -> float:
    if v is None:
        return float("nan")
    x = float(v)
    return x if x == x else float("nan")


def _rows_to_columns(rows: Sequence[AutoFitRow]) -> AutoFitColumns:
    n = len(rows)
    if n == 0:
        z = np.zeros(0, dtype=np.float64)
        return AutoFitColumns(
            t=z,
            mx=z,
            my=z,
            a=z,
            b=z,
            mx_p=z,
            my_p=z,
            weight=z,
            ch_n=z,
            fit_a=z,
            pcd=np.zeros(0, dtype=np.int32),
            sa=z,
            sb=z,
        )
    return AutoFitColumns(
        t=np.fromiter((r.t for r in rows), dtype=np.float64, count=n),
        mx=np.fromiter((r.mx for r in rows), dtype=np.float64, count=n),
        my=np.fromiter((r.my for r in rows), dtype=np.float64, count=n),
        a=np.fromiter((r.a for r in rows), dtype=np.float64, count=n),
        b=np.fromiter((r.b for r in rows), dtype=np.float64, count=n),
        mx_p=np.fromiter((_optional_float(r.mx_p) for r in rows), dtype=np.float64, count=n),
        my_p=np.fromiter((_optional_float(r.my_p) for r in rows), dtype=np.float64, count=n),
        weight=np.fromiter((r.weight for r in rows), dtype=np.float64, count=n),
        ch_n=np.fromiter((r.ch_n for r in rows), dtype=np.float64, count=n),
        fit_a=np.fromiter((r.weight for r in rows), dtype=np.float64, count=n),
        pcd=np.fromiter((r.pcd for r in rows), dtype=np.int32, count=n),
        sa=np.fromiter((_optional_float(r.sa) for r in rows), dtype=np.float64, count=n),
        sb=np.fromiter((_optional_float(r.sb) for r in rows), dtype=np.float64, count=n),
    )


def segment_into_episodes(
    rows: Sequence[AutoFitRow],
    *,
    episode_gap_s: float,
    min_on_spot_weight_na: float,
    spot_xy_jump_mm: float,
    min_episode_rows: int,
) -> list[EpisodeSpan]:
    return segment_into_episodes_cols(
        _rows_to_columns(rows),
        episode_gap_s=episode_gap_s,
        min_on_spot_weight_na=min_on_spot_weight_na,
        spot_xy_jump_mm=spot_xy_jump_mm,
        min_episode_rows=min_episode_rows,
    )


def segment_align_auto_episodes(
    rows: Sequence[AutoFitRow],
    *,
    n_plan_spots: int,
    episode_gap_s: float,
    min_on_spot_weight_na: float,
    spot_xy_jump_mm: float,
    min_episode_rows: int,
    dead_ratio: float = AUTO_EDGE_DEAD_RATIO_DEFAULT,
    tiny_merge_rows: int = AUTO_EDGE_TINY_MERGE_ROWS,
) -> tuple[list[EpisodeSpan], AutoEpisodeDiagnostics]:
    return segment_align_auto_columns(
        _rows_to_columns(rows),
        n_plan_spots=n_plan_spots,
        episode_gap_s=episode_gap_s,
        min_on_spot_weight_na=min_on_spot_weight_na,
        spot_xy_jump_mm=spot_xy_jump_mm,
        min_episode_rows=min_episode_rows,
        dead_ratio=dead_ratio,
        tiny_merge_rows=tiny_merge_rows,
    )


