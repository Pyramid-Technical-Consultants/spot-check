"""Signal-based episode segmentation for ``layer_mode='auto'`` (no Gate Counter)."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from spot_check.analysis.auto_columns import AutoFitColumns
from spot_check.constants import AUTO_EPISODE_MERGE_DT_MM2_PER_S

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


def _finalize_span_cols(cols: AutoFitColumns, span: EpisodeSpan) -> FinBuf:
    s, e = span
    sl = slice(s, e)
    w = np.maximum(cols.weight[sl], 1e-18)
    sw = float(w.sum())
    a_mean = float(np.dot(cols.a[sl], w) / sw)
    b_mean = float(np.dot(cols.b[sl], w) / sw)
    pcd_seg = cols.pcd[sl]
    pcd_out = int(pcd_seg.max()) if (pcd_seg > 0).any() else 0
    sig_a = _weighted_mean_nan(cols.sa[sl], w)
    sig_b = _weighted_mean_nan(cols.sb[sl], w)
    sa = float("nan") if sig_a is None else float(sig_a)
    sb = float("nan") if sig_b is None else float(sig_b)
    ch_seg = cols.ch_n[sl]
    ch_ok = np.isfinite(ch_seg) & (ch_seg > 0)
    if ch_ok.any():
        ch_mean = float(np.dot(ch_seg[ch_ok], w[ch_ok]) / w[ch_ok].sum())
    else:
        ch_mean = sw
    return (a_mean, b_mean, 0.0, sw, pcd_out, sa, sb, ch_mean)


def _episode_bincount_stats(
    cols: AutoFitColumns, spans: list[EpisodeSpan]
) -> tuple[np.ndarray, ...]:
    """Weighted per-episode sums for all rows (``O(n_rows)``)."""
    k = len(spans)
    n = len(cols)
    ep = np.empty(n, dtype=np.int32)
    for i, (s, e) in enumerate(spans):
        ep[s:e] = i

    w = np.maximum(cols.weight, 1e-18)
    sum_w = np.bincount(ep, weights=w, minlength=k)
    sum_aw = np.bincount(ep, weights=w * cols.a, minlength=k)
    sum_bw = np.bincount(ep, weights=w * cols.b, minlength=k)
    sum_mx = np.bincount(ep, weights=w * cols.mx, minlength=k)
    sum_my = np.bincount(ep, weights=w * cols.my, minlength=k)

    pcd_max = np.zeros(k, dtype=np.int32)
    np.maximum.at(pcd_max, ep, cols.pcd)

    sa_ok = np.isfinite(cols.sa)
    sb_ok = np.isfinite(cols.sb)
    sum_sa = np.zeros(k, dtype=np.float64)
    sum_sa_w = np.zeros(k, dtype=np.float64)
    sum_sb = np.zeros(k, dtype=np.float64)
    sum_sb_w = np.zeros(k, dtype=np.float64)
    if sa_ok.any():
        np.add.at(sum_sa, ep[sa_ok], (cols.sa[sa_ok] * w[sa_ok]))
        np.add.at(sum_sa_w, ep[sa_ok], w[sa_ok])
    if sb_ok.any():
        np.add.at(sum_sb, ep[sb_ok], (cols.sb[sb_ok] * w[sb_ok]))
        np.add.at(sum_sb_w, ep[sb_ok], w[sb_ok])

    ch_ok = np.isfinite(cols.ch_n) & (cols.ch_n > 0)
    sum_ch = np.zeros(k, dtype=np.float64)
    sum_ch_w = np.zeros(k, dtype=np.float64)
    if ch_ok.any():
        np.add.at(sum_ch, ep[ch_ok], (cols.ch_n[ch_ok] * w[ch_ok]))
        np.add.at(sum_ch_w, ep[ch_ok], w[ch_ok])

    has_pp = np.isfinite(cols.mx_p) | np.isfinite(cols.my_p)
    sum_pp_mx = np.zeros(k, dtype=np.float64)
    sum_pp_mx_w = np.zeros(k, dtype=np.float64)
    sum_pp_my = np.zeros(k, dtype=np.float64)
    sum_pp_my_w = np.zeros(k, dtype=np.float64)
    if has_pp.any():
        epp = ep[has_pp]
        wpp = w[has_pp]
        mxv = cols.mx_p[has_pp]
        myv = cols.my_p[has_pp]
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


def aggregate_span_cols(cols: AutoFitColumns, span: EpisodeSpan) -> SpanAggregate:
    """Vectorized weighted collapse of one episode span (``O(span length)`` NumPy)."""
    s, e = span
    sl = slice(s, e)
    w = np.maximum(cols.weight[sl], 1e-18)
    sw = float(w.sum())
    fin = _finalize_span_cols(cols, span)
    mx_m = float(np.dot(cols.mx[sl], w) / sw)
    my_m = float(np.dot(cols.my[sl], w) / sw)
    mx_p = cols.mx_p[sl]
    my_p = cols.my_p[sl]
    has_p = np.isfinite(mx_p) | np.isfinite(my_p)
    if has_p.any():
        wp = w[has_p]
        mx_pp = _weighted_mean_nan(mx_p[has_p], wp)
        my_pp = _weighted_mean_nan(my_p[has_p], wp)
    else:
        mx_pp, my_pp = None, None
    sig_a = float(fin[5])
    sig_b = float(fin[6])
    sa_o = None if sig_a != sig_a else sig_a
    sb_o = None if sig_b != sig_b else sig_b
    return SpanAggregate(
        a=float(fin[0]),
        b=float(fin[1]),
        mx=mx_m,
        my=my_m,
        weight=float(fin[3]),
        ch_n=float(fin[7]),
        pcd=int(fin[4]),
        sa=sa_o,
        sb=sb_o,
        mx_pp=mx_pp,
        my_pp=my_pp,
    )


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
) -> float:
    lt, rt = fins[i], fins[i + 1]
    da = float(lt[0]) - float(rt[0])
    db = float(lt[1]) - float(rt[1])
    dist_sq = da * da + db * db
    left_end = spans[i][1] - 1
    right_start = spans[i + 1][0]
    t_gap = float(cols.t[right_start]) - float(cols.t[left_end])
    return dist_sq + float(dt_coeff) * (max(0.0, t_gap) ** 2)


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


def segment_into_episodes_cols(
    cols: AutoFitColumns,
    *,
    episode_gap_s: float,
    min_on_spot_weight_na: float,
    spot_xy_jump_mm: float,
    min_episode_rows: int,
) -> list[EpisodeSpan]:
    """Vectorized segmentation on column arrays."""
    n = len(cols)
    if n == 0:
        return []
    dt = np.diff(cols.t)
    dxy_sq = np.diff(cols.a) ** 2 + np.diff(cols.b) ** 2
    w = cols.weight
    min_w = float(min_on_spot_weight_na)
    jump_sq = float(spot_xy_jump_mm) ** 2
    br = (dt >= float(episode_gap_s)) | (w[1:] < min_w)
    br |= (w[:-1] >= min_w) & (w[1:] >= min_w) & (dxy_sq > jump_sq)
    if not br.any():
        spans = [(0, n)]
    else:
        starts = np.concatenate(([0], np.nonzero(br)[0] + 1))
        ends = np.concatenate((np.nonzero(br)[0] + 1, [n]))
        spans = [(int(s), int(e)) for s, e in zip(starts, ends)]
    _merge_short_spans(spans, min_episode_rows)
    return [sp for sp in spans if sp[1] > sp[0]]


def align_episode_spans_to_plan_count_cols(
    cols: AutoFitColumns,
    spans: list[EpisodeSpan],
    n_plan: int,
    *,
    merge_dt_coeff: float = AUTO_EPISODE_MERGE_DT_MM2_PER_S,
) -> tuple[list[EpisodeSpan], AutoEpisodeDiagnostics]:
    """Merge/split spans until ``len(spans) == n_plan`` when possible."""
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

    work = list(spans)
    fins = fin_bufs_from_spans_batch(cols, work)
    count_align_ok = True

    heap: list[tuple[float, int]] = []
    for i in range(len(work) - 1):
        heapq.heappush(
            heap, (_merge_pair_cost_cols(fins, cols, work, i, dt_coeff=merge_dt_coeff), i)
        )

    while len(work) > n_plan and heap:
        c_heap, i = heapq.heappop(heap)
        if i >= len(work) - 1:
            continue
        c_now = _merge_pair_cost_cols(fins, cols, work, i, dt_coeff=merge_dt_coeff)
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
                heapq.heappush(
                    heap,
                    (_merge_pair_cost_cols(fins, cols, work, j, dt_coeff=merge_dt_coeff), j),
                )

    gap_meta = [_span_max_internal_gap_cols(cols, sp) for sp in work]
    while len(work) < n_plan:
        best_ei = -1
        best_k = -1
        best_dt = -1.0
        for ei, (k, dt) in enumerate(gap_meta):
            if k >= 0 and dt > best_dt:
                best_dt = dt
                best_k = k
                best_ei = ei
        if best_ei < 0:
            count_align_ok = False
            break
        s, e = work[best_ei]
        left: EpisodeSpan = (s, s + best_k)
        right: EpisodeSpan = (s + best_k, e)
        work[best_ei : best_ei + 1] = [left, right]
        gap_meta[best_ei : best_ei + 1] = [
            _span_max_internal_gap_cols(cols, left),
            _span_max_internal_gap_cols(cols, right),
        ]

    if len(work) != n_plan:
        count_align_ok = False

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
) -> tuple[list[EpisodeSpan], AutoEpisodeDiagnostics]:
    spans = segment_into_episodes_cols(
        cols,
        episode_gap_s=episode_gap_s,
        min_on_spot_weight_na=min_on_spot_weight_na,
        spot_xy_jump_mm=spot_xy_jump_mm,
        min_episode_rows=min_episode_rows,
    )
    return align_episode_spans_to_plan_count_cols(cols, spans, n_plan_spots)


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
) -> tuple[list[EpisodeSpan], AutoEpisodeDiagnostics]:
    return segment_align_auto_columns(
        _rows_to_columns(rows),
        n_plan_spots=n_plan_spots,
        episode_gap_s=episode_gap_s,
        min_on_spot_weight_na=min_on_spot_weight_na,
        spot_xy_jump_mm=spot_xy_jump_mm,
        min_episode_rows=min_episode_rows,
    )


