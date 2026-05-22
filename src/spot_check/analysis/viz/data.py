"""Data."""

from __future__ import annotations

import math

from spot_check.analysis._imports import *  # noqa: F403
from spot_check.analysis.layers import energies_for_measured_time_layers
from spot_check.analysis.measured import measured_row_time_s
from spot_check.analysis.spatial import nominal_layer_energies_mev
from spot_check.analysis.viz.glyphs import _plan_energy_bounds_mev


def prepare_comparison_3d_data(
    planned_xyz: list[tuple[float, float, float]],
    measured_abc: list[tuple[float, ...]],
    *,
    a_is_x: bool,
    max_measured_draw: int | None = None,
    plan_fwhm_xy_mm: np.ndarray | None = None,
) -> Comparison3DData:
    if not planned_xyz and not measured_abc:
        raise PlanDataError("No plan spots and no measured points to display")
    if a_is_x:
        xlab, ylab = "Fit A (mm)", "Fit B (mm)"
    else:
        xlab, ylab = "Fit B (mm)", "Fit A (mm)"

    n_plan = len(planned_xyz)
    fwhm_arr: np.ndarray | None = None
    if n_plan:
        if plan_fwhm_xy_mm is not None:
            fa = np.asarray(plan_fwhm_xy_mm, dtype=np.float64).reshape(-1)
            if fa.size != 2 * n_plan:
                raise ValueError(
                    "plan_fwhm_xy_mm must have length 2 * n_plan or shape (n_plan, 2)"
                )
            fwhm_arr = fa.reshape(n_plan, 2)
        e_hi, e_lo = _plan_energy_bounds_mev(planned_xyz)
        plan_xyz = np.asarray(planned_xyz, dtype=np.float64).reshape(-1, 3)
    else:
        e_hi, e_lo = 0.0, 0.0
        plan_xyz = np.zeros((0, 3), dtype=np.float64)

    rows = list(measured_abc)
    if max_measured_draw is not None and len(rows) > max_measured_draw:
        rows = rows[:max_measured_draw]

    if rows:
        if planned_xyz:
            layer_e = nominal_layer_energies_mev(planned_xyz)
            z_mapped = energies_for_measured_time_layers(layer_e, rows)
        else:
            z_mapped = [max(0.0, float(round(float(t[2])))) for t in rows]
            e_vals = [float(z) for z in z_mapped]
            e_hi, e_lo = (max(e_vals), min(e_vals)) if e_vals else (0.0, 0.0)
        if a_is_x:
            mx = [t[0] for t in rows]
            my = [t[1] for t in rows]
        else:
            mx = [t[1] for t in rows]
            my = [t[0] for t in rows]
        wts: list[float] = []
        parts: list[int] = []
        for t in rows:
            wts.append(float(t[3]) if len(t) >= 4 else 1.0)
            parts.append(int(t[4]) if len(t) >= 5 else 0)
        meas_weight = np.asarray(wts, dtype=np.float64)
        meas_partial_raw = np.asarray(parts, dtype=np.int8)
        meas_xyz = np.column_stack([mx, my, z_mapped]).astype(np.float64)
        sig_plot_x: list[float] = []
        sig_plot_y: list[float] = []
        for t in rows:
            if len(t) >= 7:
                try:
                    sa = float(t[5])
                    if not math.isfinite(sa):
                        sa = float("nan")
                except (TypeError, ValueError):
                    sa = float("nan")
                try:
                    sb = float(t[6])
                    if not math.isfinite(sb):
                        sb = float("nan")
                except (TypeError, ValueError):
                    sb = float("nan")
            else:
                sa, sb = float("nan"), float("nan")
            if a_is_x:
                sig_plot_x.append(sa)
                sig_plot_y.append(sb)
            else:
                sig_plot_x.append(sb)
                sig_plot_y.append(sa)
        meas_sigma_xy = np.column_stack(
            [np.asarray(sig_plot_x, dtype=np.float64), np.asarray(sig_plot_y, dtype=np.float64)]
        )
        meas_time_s = np.asarray([measured_row_time_s(t) for t in rows], dtype=np.float64)
    else:
        meas_weight = np.zeros(0, dtype=np.float64)
        meas_partial_raw = np.zeros(0, dtype=np.int8)
        meas_xyz = np.zeros((0, 3), dtype=np.float64)
        meas_sigma_xy = np.zeros((0, 2), dtype=np.float64)
        meas_time_s = np.zeros(0, dtype=np.float64)

    return Comparison3DData(
        plan_xyz=plan_xyz,
        meas_xyz=meas_xyz,
        xlab=xlab,
        ylab=ylab,
        e_hi=e_hi,
        e_lo=e_lo,
        meas_weight=meas_weight,
        meas_partial_raw=meas_partial_raw,
        plan_fwhm_xy_mm=fwhm_arr,
        meas_sigma_xy_mm=meas_sigma_xy,
        meas_time_s=meas_time_s,
    )

def _energy_slice_mask(energy_mev: np.ndarray, lo_mev: float, hi_mev: float) -> np.ndarray:
    """Inclusive nominal-energy band; ``lo_mev``, ``hi_mev`` may be in either order."""
    a, b = sorted((float(lo_mev), float(hi_mev)))
    e = np.asarray(energy_mev, dtype=np.float64).reshape(-1)
    return (e >= a) & (e <= b)


def _time_slice_mask(
    time_s: np.ndarray,
    start_s: float,
    *,
    window_s: float,
) -> np.ndarray:
    """Inclusive acquisition-time window ``[start_s, start_s + window_s]``."""
    t = np.asarray(time_s, dtype=np.float64).reshape(-1)
    lo = float(start_s)
    hi = lo + float(window_s)
    return np.isfinite(t) & (t >= lo) & (t <= hi)


def _time_slice_range_ms(
    time_s: np.ndarray,
    *,
    window_s: float,
) -> tuple[int, int, float, float] | None:
    """Slider range in ms and timeline bounds in seconds; ``None`` when no finite times."""
    t = np.asarray(time_s, dtype=np.float64).reshape(-1)
    ok = np.isfinite(t)
    if not bool(np.any(ok)):
        return None
    t_min = float(np.min(t[ok]))
    t_max = float(np.max(t[ok]))
    win = max(float(window_s), 1e-9)
    start_min_ms = int(math.floor(t_min * 1000.0))
    start_max_ms = int(math.floor(max(t_min, t_max - win) * 1000.0))
    if start_max_ms < start_min_ms:
        start_max_ms = start_min_ms
    return start_min_ms, start_max_ms, t_min, t_max


def _timeline_range_ms(
    meas_time_s: np.ndarray,
    plan_time_s: np.ndarray | None = None,
    *,
    window_s: float,
) -> tuple[int, int, float, float] | None:
    """Slider range from measured and/or plan delivery times."""
    parts: list[np.ndarray] = []
    mt = np.asarray(meas_time_s, dtype=np.float64).reshape(-1)
    if mt.size:
        parts.append(mt[np.isfinite(mt)])
    if plan_time_s is not None:
        pt = np.asarray(plan_time_s, dtype=np.float64).reshape(-1)
        if pt.size:
            parts.append(pt[np.isfinite(pt)])
    if not parts:
        return None
    combined = np.concatenate(parts) if len(parts) > 1 else parts[0]
    if combined.size == 0:
        return None
    return _time_slice_range_ms(combined, window_s=window_s)


def build_plan_spot_delivery_times_s(
    n_plan: int,
    rows: Sequence[tuple[float, ...]],
    plan_index_per_row: Sequence[int],
) -> np.ndarray:
    """Per-plan-slot delivery time (weighted mean of row times); ``NaN`` when unassigned."""
    out = np.full(int(n_plan), np.nan, dtype=np.float64)
    if n_plan <= 0 or not rows:
        return out
    if len(plan_index_per_row) != len(rows):
        raise ValueError(
            f"plan_index_per_row length {len(plan_index_per_row)} != rows {len(rows)}"
        )
    w_sum = np.zeros(n_plan, dtype=np.float64)
    t_sum = np.zeros(n_plan, dtype=np.float64)
    for row, pi_raw in zip(rows, plan_index_per_row, strict=True):
        pi = int(pi_raw)
        if pi < 0 or pi >= n_plan:
            continue
        t = measured_row_time_s(row)
        if not math.isfinite(t):
            continue
        w = float(row[3]) if len(row) >= 4 else 1.0
        if not math.isfinite(w) or w <= 0.0:
            w = 1.0
        w = max(w, 1e-18)
        w_sum[pi] += w
        t_sum[pi] += w * t
    ok = w_sum > 0.0
    out[ok] = t_sum[ok] / w_sum[ok]
    return out


def _nominal_layer_index_band_mev(
    layer_energies_mev: Sequence[float],
    center_index: int,
    *,
    half_width: int = 2,
) -> tuple[float, float]:
    """Inclusive MeV range for up to ``2 * half_width + 1`` consecutive plan layers around
    ``center_index``."""
    n = len(layer_energies_mev)
    if n == 0:
        return 0.0, 0.0
    c = int(np.clip(int(center_index), 0, n - 1))
    hw = int(max(0, half_width))
    i0 = max(0, c - hw)
    i1 = min(n - 1, c + hw)
    band = [float(layer_energies_mev[j]) for j in range(i0, i1 + 1)]
    return float(min(band)), float(max(band))
