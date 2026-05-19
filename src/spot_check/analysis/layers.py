"""Layers."""

from __future__ import annotations

import bisect
from typing import Sequence

from spot_check.analysis._imports import *  # noqa: F403
from spot_check.analysis.spatial import _kdtree_query_k1, _min_xy_dist_to_nominal_energy


def _opt_float_cell(row: dict[str, str], key: str) -> float | None:
    raw = (row.get(key) or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None

@dataclass
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

    def impute_xy_arrays(
        self, mx_p: np.ndarray, my_p: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fill missing plan axes for many rows (partial rows only call 1D nearest-plan)."""
        mx = np.asarray(mx_p, dtype=np.float64, order="C").copy()
        my = np.asarray(my_p, dtype=np.float64, order="C").copy()
        both = np.isfinite(mx) & np.isfinite(my)
        mx[both] = mx_p[both]
        my[both] = my_p[both]
        px, py = self.pts[:, 0], self.pts[:, 1]
        spx = px[self.ord_x]
        spy = py[self.ord_y]
        need_x = np.isfinite(mx) & ~np.isfinite(my)
        for i in np.flatnonzero(need_x):
            bx, by = _pick_axis_band(spx, self.ord_x, px, py, float(mx[i]))
            mx[i] = bx
            my[i] = by
        need_y = ~np.isfinite(mx) & np.isfinite(my)
        for i in np.flatnonzero(need_y):
            bx, by = _pick_axis_band(spy, self.ord_y, px, py, float(my[i]))
            mx[i] = bx
            my[i] = by
        neither = ~np.isfinite(mx) & ~np.isfinite(my)
        if neither.any():
            i0 = int(self.ord_x[0])
            mx[neither] = px[i0]
            my[neither] = py[i0]
        return mx, my


def _pick_axis_band(
    spv: np.ndarray,
    ordv: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    mcoord: float,
) -> tuple[float, float]:
    """1D nearest plan spot on ``spv`` with tie-band + smallest plan index."""
    n = int(spv.shape[0])
    if n == 0:
        return float("nan"), float("nan")
    idx = int(np.searchsorted(spv, mcoord, side="left"))
    if idx <= 0:
        lo_i = hi_i = 0
    elif idx >= n:
        lo_i = hi_i = n - 1
    else:
        lo_i, hi_i = idx - 1, idx
    best_d = min(abs(float(spv[lo_i]) - mcoord), abs(float(spv[hi_i]) - mcoord))
    lo, hi = min(lo_i, hi_i), max(lo_i, hi_i)
    tol = max(1e-9, 1e-12 * max(1.0, abs(mcoord)))
    while lo > 0 and abs(float(spv[lo - 1]) - mcoord) <= best_d + tol:
        lo -= 1
    while hi + 1 < n and abs(float(spv[hi + 1]) - mcoord) <= best_d + tol:
        hi += 1
    best_j = int(np.min(ordv[lo : hi + 1]))
    return float(px[best_j]), float(py[best_j])


def _impute_plan_axis_fast(
    lk: _PlanImputeLookup,
    mx: float | None,
    my: float | None,
) -> tuple[float, float]:
    """Copy measured axis; fill the other from the closest plan spot on that axis (1D)."""
    if mx is not None and my is not None:
        return float(mx), float(my)
    px, py = lk.pts[:, 0], lk.pts[:, 1]
    if mx is None and my is None:
        i0 = int(lk.ord_x[0])
        return float(px[i0]), float(py[i0])
    if mx is not None:
        spx = px[lk.ord_x]
        bx, by = _pick_axis_band(spx, lk.ord_x, px, py, float(mx))
        if math.isnan(bx):
            return float(mx), 0.0
        return bx, by
    spy = py[lk.ord_y]
    bx, by = _pick_axis_band(spy, lk.ord_y, px, py, float(my))  # type: ignore[arg-type]
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

def delivery_layer_indices(n_spots: int, spots_per_layer: Sequence[int]) -> np.ndarray:
    """Nominal layer index per delivered spot in DICOM/plan order (matches gate_counter)."""
    if n_spots <= 0:
        return np.zeros(0, dtype=np.int64)
    cumul: list[int] = [0]
    for c in spots_per_layer:
        cumul.append(cumul[-1] + int(c))
    hi = max(0, len(spots_per_layer) - 1)
    out = np.empty(n_spots, dtype=np.int64)
    for i in range(n_spots):
        out[i] = max(0, min(bisect.bisect_right(cumul, i) - 1, hi))
    return out


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
