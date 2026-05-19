"""Data."""

from __future__ import annotations

from spot_check.analysis._imports import *  # noqa: F403
from spot_check.analysis.layers import energies_for_measured_time_layers
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
    if not planned_xyz:
        raise PlanDataError("No planned spots extracted from DICOM")
    if not measured_abc:
        raise AcquisitionDataError("No measured points found in CSV")
    n_plan = len(planned_xyz)
    fwhm_arr: np.ndarray | None = None
    if plan_fwhm_xy_mm is not None:
        fa = np.asarray(plan_fwhm_xy_mm, dtype=np.float64).reshape(-1)
        if fa.size != 2 * n_plan:
            raise ValueError("plan_fwhm_xy_mm must have length 2 * n_plan or shape (n_plan, 2)")
        fwhm_arr = fa.reshape(n_plan, 2)
    e_hi, e_lo = _plan_energy_bounds_mev(planned_xyz)
    plan_xyz = np.asarray(planned_xyz, dtype=np.float64).reshape(-1, 3)
    layer_e = nominal_layer_energies_mev(planned_xyz)

    rows = list(measured_abc)
    if max_measured_draw is not None and len(rows) > max_measured_draw:
        rows = rows[:max_measured_draw]

    z_mapped = energies_for_measured_time_layers(layer_e, rows)

    if a_is_x:
        xlab, ylab = "Fit A (mm)", "Fit B (mm)"
        mx = [t[0] for t in rows]
        my = [t[1] for t in rows]
    else:
        xlab, ylab = "Fit B (mm)", "Fit A (mm)"
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
    )

def _energy_slice_mask(energy_mev: np.ndarray, lo_mev: float, hi_mev: float) -> np.ndarray:
    """Inclusive nominal-energy band; ``lo_mev``, ``hi_mev`` may be in either order."""
    a, b = sorted((float(lo_mev), float(hi_mev)))
    e = np.asarray(energy_mev, dtype=np.float64).reshape(-1)
    return (e >= a) & (e <= b)

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
