"""Layers."""

from __future__ import annotations

from spot_check.analysis._imports import *  # noqa: F403
from spot_check.analysis.spatial import _min_xy_dist_to_nominal_energy


def _opt_float_cell(row: dict[str, str], key: str) -> float | None:
    raw = (row.get(key) or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None

class _PlanImputeLookup:
    """Nearest plan spot along X or Y only; sorted-axis queries O(log N)."""

    pts: np.ndarray  # (N, 2) float64
    ord_x: np.ndarray  # indices, argsort pts[:, 0]
    ord_y: np.ndarray  # indices, argsort pts[:, 1]

    @classmethod
    def from_xy(cls, xy: np.ndarray) -> _PlanImputeLookup | None:
        arr = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        if arr.shape[0] == 0:
            return None
        return cls(
            pts=arr,
            ord_x=np.argsort(arr[:, 0]),
            ord_y=np.argsort(arr[:, 1]),
        )

def _impute_plan_axis_fast(
    lk: _PlanImputeLookup,
    mx: float | None,
    my: float | None,
) -> tuple[float, float]:
    """Copy measured axis; fill the other from the closest plan spot on that axis (1D).

    Matches the original linear scan: all spots achieving min |Δaxis| are considered;
    tie-break is the smallest row index (plan / bucket order).
    """
    px, py = lk.pts[:, 0], lk.pts[:, 1]
    ox, oy = lk.ord_x, lk.ord_y
    if mx is not None and my is not None:
        return float(mx), float(my)
    if mx is None and my is None:
        i0 = int(ox[0])
        return float(px[i0]), float(py[i0])
    tol_coord = 1e-9

    def _pick_after_expanding(
        spv: np.ndarray, ordv: np.ndarray, mcoord: float
    ) -> tuple[float, float]:
        """spv = sorted coordinate; ordv maps sorted pos -> row index."""
        n = int(spv.shape[0])
        if n == 0:
            return float("nan"), float("nan")
        idx = int(np.searchsorted(spv, mcoord, side="left"))
        pos_cand: list[int] = []
        if idx > 0:
            pos_cand.append(idx - 1)
        if idx < n:
            pos_cand.append(idx)
        best_d = min(abs(float(spv[i]) - mcoord) for i in pos_cand)
        tol = max(tol_coord, 1e-12 * max(1.0, abs(mcoord)))
        lo = min(pos_cand)
        hi = max(pos_cand)
        while lo > 0 and abs(float(spv[lo - 1]) - mcoord) <= best_d + tol:
            lo -= 1
        while hi + 1 < n and abs(float(spv[hi + 1]) - mcoord) <= best_d + tol:
            hi += 1
        cand_rows = ordv[lo : hi + 1].astype(np.int64, copy=False)
        best_j = int(np.min(cand_rows))
        return float(px[best_j]), float(py[best_j])

    if mx is not None:
        spx = px[ox]
        bx, by = _pick_after_expanding(spx, ox, float(mx))
        if math.isnan(bx):
            return float(mx), 0.0
        return bx, by
    spv_y = py[oy]
    bx, by = _pick_after_expanding(spv_y, oy, float(my))  # type: ignore[arg-type]
    if math.isnan(bx):
        return 0.0, float(my)  # type: ignore[arg-type]
    return bx, by

def _plan_impute_lookups_per_layer(layer_xy: list[np.ndarray]) -> list[_PlanImputeLookup | None]:
    out: list[_PlanImputeLookup | None] = []
    for arr in layer_xy:
        out.append(_PlanImputeLookup.from_xy(np.asarray(arr, dtype=np.float64)))
    return out

def _layer_advance_plausible_vs_refill(
    planned_xyz: list[tuple[float, float, float]],
    layer_energies: list[float],
    layer: int,
    mx: float,
    my: float,
    *,
    trust_time_gap_stay_dist_mm: float = REFILL_TRUST_TIME_GAP_STAY_DIST_MM,
    layer_trees: list[Any] | None = None,
) -> bool:
    """After a gap that is not a same-spot XY return: True → advance nominal energy layer."""
    if layer >= len(layer_energies) - 1:
        return False
    if layer_trees is not None and len(layer_trees) == len(layer_energies):

        def _min_dist_mm(li: int) -> float:
            if li < 0 or li >= len(layer_trees):
                return float("inf")
            tr = layer_trees[li]
            if tr is None:
                return float("inf")
            q = np.array([[mx, my]], dtype=np.float64)
            d, _ = _kdtree_query_k1(tr, q)
            return float(np.asarray(d, dtype=np.float64).reshape(-1)[0])

        d_stay = _min_dist_mm(layer)
        d_next = _min_dist_mm(layer + 1)
        if not math.isfinite(d_stay):
            d_stay = float("inf")
        if not math.isfinite(d_next):
            d_next = float("inf")
        if d_stay > trust_time_gap_stay_dist_mm:
            return True
        ratio_base = max(d_stay, 1.0)
        worse = (d_next > d_stay + REFILL_REJECT_EXTRA_MM) or (
            d_next > REFILL_REJECT_RATIO * ratio_base
        )
        return not worse
    e0 = layer_energies[layer]
    e1 = layer_energies[layer + 1]
    d_stay = _min_xy_dist_to_nominal_energy(planned_xyz, e0, mx, my)
    d_next = _min_xy_dist_to_nominal_energy(planned_xyz, e1, mx, my)
    if not math.isfinite(d_stay):
        d_stay = float("inf")
    if not math.isfinite(d_next):
        d_next = float("inf")
    if d_stay > trust_time_gap_stay_dist_mm:
        return True
    ratio_base = max(d_stay, 1.0)
    worse = (d_next > d_stay + REFILL_REJECT_EXTRA_MM) or (
        d_next > REFILL_REJECT_RATIO * ratio_base
    )
    return not worse

def energies_for_measured_time_layers(
    layer_energies: list[float],
    measured_abc: list[tuple[float, ...]],
) -> list[float]:
    if not layer_energies or not measured_abc:
        return []
    hi = len(layer_energies) - 1
    out: list[float] = []
    for tup in measured_abc:
        idx = int(round(float(tup[2])))
        if idx < 0:
            idx = 0
        elif idx > hi:
            idx = hi
        out.append(layer_energies[idx])
    return out

def viterbi_monotone_layer_assign(
    emit_sq_mm2: np.ndarray,
    advance_penalty_mm2: float | np.ndarray,
) -> np.ndarray:
    """
    Minimize sum_i emit[i, ell_i] + sum of advance penalties at steps where ell increases,
    subject to ell non-decreasing and ell[i] - ell[i-1] in {0, 1}.

    ``advance_penalty_mm2`` may be a scalar (same cost every step) or length-``n`` vector:
    penalty applied when transitioning into row ``i`` (index ``i``), for ``i >= 1``.
    """
    n, L = emit_sq_mm2.shape
    if L == 0:
        raise ValueError("no nominal layers")
    ap = np.asarray(advance_penalty_mm2, dtype=np.float64)
    if ap.size == 1:
        pen_row = np.full(n, float(ap.flat[0]), dtype=np.float64)
    else:
        ap = ap.reshape(-1)
        if ap.shape[0] != n:
            raise ValueError(
                f"advance_penalty_mm2 must be scalar or length n={n}, got length {ap.shape[0]}"
            )
        pen_row = ap
    if (pen_row < 0).any():
        raise ValueError("advance_penalty_mm2 must be >= 0")
    inf = np.inf
    C = np.full((n, L), inf, dtype=np.float64)
    back = np.zeros((n, L), dtype=np.int32)
    C[0, 0] = float(emit_sq_mm2[0, 0])
    C[0, 1:] = inf
    idx_hi = np.arange(1, L, dtype=np.int32)
    idx_lo = np.arange(0, L - 1, dtype=np.int32)
    for i in range(1, n):
        p_add = float(pen_row[i])
        C[i, 0] = float(emit_sq_mm2[i, 0]) + C[i - 1, 0]
        back[i, 0] = 0
        stay = C[i - 1, 1:L]
        adv = C[i - 1, : L - 1] + p_add
        stay_better = stay <= adv
        C[i, 1:L] = emit_sq_mm2[i, 1:L] + np.where(stay_better, stay, adv)
        back[i, 1:L] = np.where(stay_better, idx_hi, idx_lo)
    k_end = int(np.argmin(C[n - 1]))
    layers = np.zeros(n, dtype=np.int32)
    k = k_end
    layers[n - 1] = k
    for i in range(n - 1, 0, -1):
        k = int(back[i, k])
        layers[i - 1] = k
    return layers

def build_unified_advance_penalty_mm2(
    times_s: np.ndarray,
    meas_xy: np.ndarray,
    *,
    base_penalty_mm2: float,
    layer_gap_s: float,
    refill_same_spot_xy_tol_mm: float,
    short_dt_extra_mm2: float = UNIFIED_SHORT_DT_EXTRA_MM2,
    same_spot_refill_block_mm2: float = UNIFIED_SAME_SPOT_REFILL_BLOCK_MM2,
) -> np.ndarray:
    """
    Per-row Viterbi advance penalty: ``base`` + time/refill modifiers on steps ``i>=1``.
    Row index ``i`` uses ``Δt`` and ``ΔXY`` from valid row ``i-1`` to ``i``.
    """
    n = meas_xy.shape[0]
    pen = np.full(n, float(base_penalty_mm2), dtype=np.float64)
    if n <= 1:
        return pen
    dt = np.diff(times_s.astype(np.float64))
    dxy = np.linalg.norm(np.diff(meas_xy.astype(np.float64), axis=0), axis=1)
    long_gap = dt >= float(layer_gap_s)
    same_small = dxy <= float(refill_same_spot_xy_tol_mm)
    pen[1:] += np.where(~long_gap, float(short_dt_extra_mm2), 0.0)
    pen[1:] += np.where(long_gap & same_small, float(same_spot_refill_block_mm2), 0.0)
    return pen
