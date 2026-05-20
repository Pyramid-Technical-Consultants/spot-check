"""Layer-based assignment for ``auto_assign_method='layer_em'``.

Plan layer 0 is the **highest nominal energy** and is delivered **first in time**; each
successive layer is a later contiguous block on the acquisition timeline (no time travel).

Pipeline:

1. Build a delivery-weighted on-spot timeline (same weights as auto episodes).
2. Place monotone layer boundaries (MU or spot-count fractions), then refine them.
3. Within each layer, partition the time block onto plan spots in plan order (contiguous).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from spot_check.analysis.auto_columns import AutoFitColumns
from spot_check.analysis.episodes import (
    EpisodeSpan,
    _clamp_monotone_plan_edges,
    _spans_from_row_weight_quantiles,
    delivery_row_weights,
)
from spot_check.constants import (
    AUTO_LAYER_EM_BOUNDARY_SHIFT_ROWS,
    AUTO_LAYER_EM_DP_MAX_CELLS,
    AUTO_LAYER_EM_DP_MAX_SPOTS,
    AUTO_LAYER_EM_FAST_MIN_ROWS,
    AUTO_LAYER_EM_LAYER_BOUNDARY_BAND_ROWS,
    AUTO_LAYER_EM_REFINE_PASSES_DEFAULT,
    AUTO_LAYER_EM_SPOT_REFINE_PASSES,
)

_last_layer_em_diag: LayerEmDiagnostics | None = None


@dataclass(frozen=True)
class LayerEmDiagnostics:
    n_plan: int
    n_on_rows: int
    n_layers: int
    total_cost: float
    refine_passes: int
    layer_on_bounds: tuple[int, ...]


def last_layer_em_diagnostics() -> LayerEmDiagnostics | None:
    return _last_layer_em_diag


def _set_diag(d: LayerEmDiagnostics | None) -> None:
    global _last_layer_em_diag
    _last_layer_em_diag = d


def _xy_sq(ax: float, ay: float, bx: float, by: float) -> float:
    dx = ax - bx
    dy = ay - by
    return dx * dx + dy * dy


def _timeline_row_indices(cols: AutoFitColumns) -> np.ndarray:
    """All acquisition rows in time order (layer 0 = highest energy = earliest segment)."""
    n = len(cols)
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    return np.arange(n, dtype=np.int64)


def _layer_fractions(
    spots_per_layer: Sequence[int],
    plan_mu: np.ndarray | None,
) -> np.ndarray:
    weights: list[float] = []
    p0 = 0
    for n_sp in spots_per_layer:
        p1 = p0 + int(n_sp)
        if plan_mu is not None and plan_mu.size >= p1:
            weights.append(float(plan_mu[p0:p1].sum()))
        else:
            weights.append(float(n_sp))
        p0 = p1
    arr = np.asarray(weights, dtype=np.float64)
    tot = float(arr.sum())
    if tot <= 0:
        n = max(1, len(spots_per_layer))
        return np.ones(n, dtype=np.float64) / n
    return arr / tot


def _spot_fractions(
    spot0: int,
    n_sp: int,
    plan_mu: np.ndarray | None,
) -> np.ndarray:
    if n_sp <= 0:
        return np.zeros(0, dtype=np.float64)
    if plan_mu is not None and plan_mu.size >= spot0 + n_sp:
        w = np.asarray(plan_mu[spot0 : spot0 + n_sp], dtype=np.float64)
        tot = float(w.sum())
        if tot > 0:
            return w / tot
    return np.full(n_sp, 1.0 / float(n_sp), dtype=np.float64)


def _clamp_layer_bounds(edges: np.ndarray, n_on: int, n_layers: int) -> np.ndarray:
    out = edges.astype(np.int64, copy=True)
    out[0] = 0
    out[-1] = n_on
    for i in range(1, n_layers + 1):
        lo = out[i - 1] + 1
        hi = n_on - (n_layers - i)
        out[i] = int(np.clip(out[i], lo, hi))
    out[-1] = n_on
    return out


def _edges_from_fractions(n: int, frac: np.ndarray) -> np.ndarray:
    """Monotone partition of ``n`` on-timeline slots into ``len(frac)`` blocks (count prior)."""
    k = int(frac.size)
    edges = np.zeros(k + 1, dtype=np.int64)
    if n <= 0 or k == 0:
        return edges
    if k == 1:
        edges[-1] = n
        return edges
    if n <= k:
        for i in range(k + 1):
            edges[i] = min(i, n)
        return _clamp_layer_bounds(edges, n, k)
    layer_cum = np.cumsum(frac)
    edges[1:k] = np.round(float(n) * layer_cum[:-1]).astype(np.int64)
    return _clamp_layer_bounds(edges, n, k)


def _edges_from_on_weights(
    on_idx: np.ndarray,
    row_w: np.ndarray,
    frac: np.ndarray,
) -> np.ndarray:
    n_on = int(on_idx.size)
    k = int(frac.size)
    edges = np.zeros(k + 1, dtype=np.int64)
    if n_on <= 0 or k == 0:
        return edges
    if k == 1:
        edges[-1] = n_on
        return edges
    w_on = row_w[on_idx]
    cum = np.cumsum(w_on)
    tot = float(cum[-1])
    if tot <= 0:
        return _edges_from_fractions(n_on, frac)
    layer_cum = np.cumsum(frac)
    targets = tot * layer_cum[:-1]
    edges[1:k] = np.searchsorted(cum, targets, side="right")
    return _clamp_layer_bounds(edges, n_on, k)


def _segment_rows_cost(
    cols: AutoFitColumns,
    row_idxs: np.ndarray,
    i0: int,
    i1: int,
    plan_pt: np.ndarray,
    row_w: np.ndarray,
) -> float:
    if i1 <= i0:
        return 0.0
    idx = row_idxs[i0:i1]
    ww = row_w[idx]
    den = float(ww.sum())
    px, py = float(plan_pt[0]), float(plan_pt[1])
    if den <= 0.0 or idx.size == 0:
        return 0.0
    mx = float(np.dot(cols.mx[idx], ww) / den)
    my = float(np.dot(cols.my[idx], ww) / den)
    return _xy_sq(mx, my, px, py)


def _partition_layer_dp(
    cols: AutoFitColumns,
    row_idxs: np.ndarray,
    plan_xy: np.ndarray,
    row_w: np.ndarray,
) -> tuple[list[tuple[int, int]], float]:
    n = int(row_idxs.size)
    k = int(plan_xy.shape[0])
    if k <= 0:
        return [], 0.0
    if k == 1:
        c = _segment_rows_cost(cols, row_idxs, 0, n, plan_xy[0], row_w)
        return [(0, n)], c
    if k > int(AUTO_LAYER_EM_DP_MAX_SPOTS) or n * k > int(AUTO_LAYER_EM_DP_MAX_CELLS):
        return [], float("inf")
    inf = 1e300
    dp = np.full((n + 1, k + 1), inf, dtype=np.float64)
    back = np.full((n + 1, k + 1), -1, dtype=np.int32)
    dp[0, 0] = 0.0
    for j in range(1, k + 1):
        pj = plan_xy[j - 1]
        for i in range(j, n + 1):
            for t in range(j - 1, i):
                if dp[t, j - 1] >= inf:
                    continue
                c = _segment_rows_cost(cols, row_idxs, t, i, pj, row_w)
                v = float(dp[t, j - 1]) + c
                if v < dp[i, j]:
                    dp[i, j] = v
                    back[i, j] = t
    if dp[n, k] >= inf:
        return [], float("inf")
    parts: list[tuple[int, int]] = []
    i, j = n, k
    while j > 0:
        t = int(back[i, j])
        if t < 0:
            return [], float("inf")
        parts.append((t, i))
        i, j = t, j - 1
    parts.reverse()
    return parts, float(dp[n, k])


def _assign_spots_in_layer(
    cols: AutoFitColumns,
    on_idx: np.ndarray,
    a: int,
    b: int,
    plan_xy: np.ndarray,
    row_w: np.ndarray,
    spot0: int,
    n_sp: int,
    plan_mu: np.ndarray | None,
    *,
    spot_shift: int,
    spot_refine_passes: int,
) -> tuple[list[tuple[int, int]], float]:
    """Contiguous on-timeline parts for ``n_sp`` plan spots (plan delivery order)."""
    if n_sp <= 0:
        return [], 0.0
    slice_idx = on_idx[a:b]
    n = int(slice_idx.size)
    if n <= 0:
        return [], 0.0
    plan_layer = plan_xy[spot0 : spot0 + n_sp]

    parts, c = _partition_layer_dp(cols, slice_idx, plan_layer, row_w)
    if c < float("inf") and len(parts) == n_sp:
        return parts, c

    frac = _spot_fractions(spot0, n_sp, plan_mu)
    spot_edges = _edges_from_on_weights(slice_idx, row_w, frac)
    if n_sp == 1:
        return [(0, n)], _segment_rows_cost(cols, slice_idx, 0, n, plan_layer[0], row_w)

    spot_edges = _clamp_monotone_plan_edges(spot_edges, n)
    if n < n_sp:
        parts = [(int(spot_edges[i]), int(spot_edges[i + 1])) for i in range(n_sp)]
        return parts, 0.0
    win = max(1, int(spot_shift))
    for _ in range(max(0, int(spot_refine_passes))):
        for si in range(1, n_sp):
            best_b = int(spot_edges[si])
            best_c = float("inf")
            lo = max(int(spot_edges[si - 1]) + 1, best_b - win)
            hi = min(int(spot_edges[si + 1]) - 1, best_b + win)
            for trial in range(lo, hi + 1):
                c_try = _segment_rows_cost(
                    cols,
                    slice_idx,
                    int(spot_edges[si - 1]),
                    trial,
                    plan_layer[si - 1],
                    row_w,
                )
                c_try += _segment_rows_cost(
                    cols,
                    slice_idx,
                    trial,
                    int(spot_edges[si + 1]),
                    plan_layer[si],
                    row_w,
                )
                if c_try < best_c:
                    best_c = c_try
                    best_b = trial
            spot_edges[si] = best_b
        spot_edges = _clamp_monotone_plan_edges(spot_edges, n)
    parts = [(int(spot_edges[i]), int(spot_edges[i + 1])) for i in range(n_sp)]
    cost = sum(
        _segment_rows_cost(
            cols,
            slice_idx,
            int(spot_edges[i]),
            int(spot_edges[i + 1]),
            plan_layer[i],
            row_w,
        )
        for i in range(n_sp)
    )
    return parts, cost


def _layer_block_cost(
    cols: AutoFitColumns,
    on_idx: np.ndarray,
    row_w: np.ndarray,
    plan_xy: np.ndarray,
    spot0: int,
    n_sp: int,
    a: int,
    b: int,
    plan_mu: np.ndarray | None,
    *,
    spot_shift: int,
    spot_refine_passes: int,
) -> float:
    _, c = _assign_spots_in_layer(
        cols,
        on_idx,
        a,
        b,
        plan_xy,
        row_w,
        spot0,
        n_sp,
        plan_mu,
        spot_shift=spot_shift,
        spot_refine_passes=spot_refine_passes,
    )
    return c


def _optimize_layer_bounds_band(
    cols: AutoFitColumns,
    on_idx: np.ndarray,
    row_w: np.ndarray,
    plan_xy: np.ndarray,
    spots_per_layer: Sequence[int],
    plan_mu: np.ndarray | None,
    init: np.ndarray,
    *,
    band: int,
    layer_passes: int,
    spot_shift: int,
    spot_refine_passes: int,
) -> np.ndarray:
    l_n = len(spots_per_layer)
    n_on = int(on_idx.size)
    if l_n < 2:
        return _clamp_layer_bounds(init, n_on, l_n)
    out = _clamp_layer_bounds(init.astype(np.int64, copy=True), n_on, l_n)
    win = max(1, int(band))
    spot_starts = np.cumsum([0, *spots_per_layer[:-1]], dtype=np.int64)
    for _ in range(max(0, int(layer_passes))):
        for li in range(1, l_n):
            best_b = int(out[li])
            best_c = float("inf")
            lo = max(int(out[li - 1]) + 1, best_b - win)
            hi = min(int(out[li + 1]) - 1, best_b + win)
            for trial in range(lo, hi + 1):
                c_try = _layer_block_cost(
                    cols,
                    on_idx,
                    row_w,
                    plan_xy,
                    int(spot_starts[li - 1]),
                    spots_per_layer[li - 1],
                    int(out[li - 1]),
                    trial,
                    plan_mu,
                    spot_shift=spot_shift,
                    spot_refine_passes=spot_refine_passes,
                )
                c_try += _layer_block_cost(
                    cols,
                    on_idx,
                    row_w,
                    plan_xy,
                    int(spot_starts[li]),
                    spots_per_layer[li],
                    trial,
                    int(out[li + 1]),
                    plan_mu,
                    spot_shift=spot_shift,
                    spot_refine_passes=spot_refine_passes,
                )
                if c_try < best_c:
                    best_c = c_try
                    best_b = trial
            out[li] = best_b
    return _clamp_layer_bounds(out, n_on, l_n)


def _fallback_plan_spans(
    cols: AutoFitColumns,
    n_plan: int,
    row_w: np.ndarray,
) -> list[EpisodeSpan]:
    n_rows = len(cols)
    if n_plan <= 0 or n_rows <= 0:
        return []
    cum = np.cumsum(row_w)
    return _spans_from_row_weight_quantiles(n_rows, n_plan, cum)


def _on_parts_to_spans(
    on_idx: np.ndarray,
    base: int,
    parts: list[tuple[int, int]],
) -> list[EpisodeSpan]:
    """Map slice-local ``[i0, i1)`` parts to acquisition row spans."""
    spans: list[EpisodeSpan] = []
    n_on = int(on_idx.size)
    for i0, i1 in parts:
        g0 = int(base) + int(i0)
        if i1 <= i0:
            ri = int(on_idx[g0]) if 0 <= g0 < n_on else 0
            spans.append((ri, ri + 1))
        else:
            g1 = int(base) + int(i1) - 1
            spans.append((int(on_idx[g0]), int(on_idx[g1]) + 1))
    return spans


def layer_em_plan_spans(
    cols: AutoFitColumns,
    plan_xy: np.ndarray,
    spots_per_layer: Sequence[int],
    *,
    plan_mu: np.ndarray | None = None,
    refine_passes: int = AUTO_LAYER_EM_REFINE_PASSES_DEFAULT,
    boundary_shift_rows: int = AUTO_LAYER_EM_BOUNDARY_SHIFT_ROWS,
) -> tuple[list[EpisodeSpan], LayerEmDiagnostics]:
    """One contiguous acquisition span per plan spot (plan / energy order)."""
    p_n = int(plan_xy.shape[0])
    row_w = delivery_row_weights(cols)
    on_idx = _timeline_row_indices(cols)
    n_on = int(on_idx.size)
    l_n = len(spots_per_layer)
    if p_n == 0 or l_n == 0:
        diag = LayerEmDiagnostics(0, n_on, l_n, 0.0, 0, (0,))
        _set_diag(diag)
        return [], diag

    layer_frac = _layer_fractions(spots_per_layer, plan_mu)
    # Layer cuts follow acquisition time (highest-energy layer first), not weight mass alone.
    init = _edges_from_fractions(n_on, layer_frac)
    layer_passes = max(0, int(refine_passes))
    spot_shift = max(1, int(boundary_shift_rows))
    spot_refine = int(AUTO_LAYER_EM_SPOT_REFINE_PASSES)
    band = int(AUTO_LAYER_EM_LAYER_BOUNDARY_BAND_ROWS)
    max_sp = max(int(s) for s in spots_per_layer) if spots_per_layer else 0
    if max_sp > int(AUTO_LAYER_EM_DP_MAX_SPOTS):
        spot_refine = 0
    fast = n_on > int(AUTO_LAYER_EM_FAST_MIN_ROWS)
    if fast:
        spot_refine = 0
        bounds = init
    elif layer_passes > 0:
        bounds = _optimize_layer_bounds_band(
            cols,
            on_idx,
            row_w,
            plan_xy,
            spots_per_layer,
            plan_mu,
            init,
            band=band,
            layer_passes=layer_passes,
            spot_shift=spot_shift,
            spot_refine_passes=spot_refine,
        )
    else:
        bounds = init

    spans: list[EpisodeSpan] = []
    cost = 0.0
    spot0 = 0
    for li, n_sp in enumerate(spots_per_layer):
        a = int(bounds[li])
        b = int(bounds[li + 1])
        if n_sp <= 0:
            continue
        parts, c = _assign_spots_in_layer(
            cols,
            on_idx,
            a,
            b,
            plan_xy,
            row_w,
            spot0,
            n_sp,
            plan_mu,
            spot_shift=spot_shift,
            spot_refine_passes=spot_refine,
        )
        cost += c
        spans.extend(_on_parts_to_spans(on_idx, a, parts))
        spot0 += n_sp

    if len(spans) != p_n:
        spans = _fallback_plan_spans(cols, p_n, row_w)
        cost = float("inf")

    diag = LayerEmDiagnostics(
        n_plan=p_n,
        n_on_rows=n_on,
        n_layers=l_n,
        total_cost=cost,
        refine_passes=layer_passes,
        layer_on_bounds=tuple(int(x) for x in bounds),
    )
    _set_diag(diag)
    return spans, diag
