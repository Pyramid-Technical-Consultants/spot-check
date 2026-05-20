"""Plan Qa."""

from __future__ import annotations

from spot_check.analysis._imports import *  # noqa: F403
from spot_check.analysis.colors import _hex_to_rgb_u8
from spot_check.analysis.measured import (
    measured_charge_na_from_tuple,
    measured_spot_weight_caption,
)
from spot_check.analysis.pyvista_backend import pv
from spot_check.analysis.spatial import (
    _layer_plan_mu_by_energy_layer,
    distances_measured_xy_to_layer_nn_plan_mm,
    layer_nn_local_spot_index_on_layer,
    layer_nn_plan_xy_distances_and_expected_xyz,
    nominal_layer_energies_mev,
)


def measured_rgba_by_plan_qa(
    dist_mm: np.ndarray,
    *,
    pass_mm: float,
    warn_mm: float,
    alpha_u8: np.ndarray | None = None,
) -> np.ndarray:
    """RGBA per point: pass (≤pass_mm), warn (pass..warn], fail (>warn_mm). Alpha from ``alpha_u8``
    or opaque."""
    if warn_mm <= pass_mm or pass_mm < 0:
        raise ValueError("plan QA: require 0 ≤ pass_mm < warn_mm")
    d = np.asarray(dist_mm, dtype=np.float64).reshape(-1)
    n = int(d.shape[0])
    if alpha_u8 is not None:
        au = np.asarray(alpha_u8, dtype=np.uint8).reshape(-1)
        if au.shape[0] != n:
            raise ValueError("alpha_u8 length must match dist_mm")
    rp, gp, bp = _hex_to_rgb_u8(_PLAN_QA_PASS_HEX)
    rw, gw, bw = _hex_to_rgb_u8(_PLAN_QA_WARN_HEX)
    rf, gf, bf = _hex_to_rgb_u8(_PLAN_QA_FAIL_HEX)
    rgba = np.zeros((n, 4), dtype=np.uint8)
    pass_m = d <= pass_mm
    fail_m = d > warn_mm
    warn_m = ~pass_m & ~fail_m
    rgba[pass_m, 0] = np.uint8(rp)
    rgba[pass_m, 1] = np.uint8(gp)
    rgba[pass_m, 2] = np.uint8(bp)
    rgba[warn_m, 0] = np.uint8(rw)
    rgba[warn_m, 1] = np.uint8(gw)
    rgba[warn_m, 2] = np.uint8(bw)
    rgba[fail_m, 0] = np.uint8(rf)
    rgba[fail_m, 1] = np.uint8(gf)
    rgba[fail_m, 2] = np.uint8(bf)
    if alpha_u8 is None:
        rgba[:, 3] = np.uint8(255)
    else:
        rgba[:, 3] = au
    return rgba

def measured_rgba_by_plan_dose_qa(
    signed_delta_pp: np.ndarray,
    *,
    pass_pp: float,
    warn_pp: float,
    alpha_u8: np.ndarray | None = None,
) -> np.ndarray:
    """RGBA per point from signed layer dose error (pp): + = over-dose, − = under-dose.

    Over: yellow warn / red fail (same as position QA). Under: sky warn / violet fail.
    """
    if warn_pp <= pass_pp or pass_pp < 0:
        raise ValueError("dose QA: require 0 ≤ pass_pp < warn_pp")
    s = np.asarray(signed_delta_pp, dtype=np.float64).reshape(-1)
    n = int(s.shape[0])
    if alpha_u8 is not None:
        au = np.asarray(alpha_u8, dtype=np.uint8).reshape(-1)
        if au.shape[0] != n:
            raise ValueError("alpha_u8 length must match signed_delta_pp")
    rp, gp, bp = _hex_to_rgb_u8(_PLAN_QA_PASS_HEX)
    rw, gw, bw = _hex_to_rgb_u8(_PLAN_QA_WARN_HEX)
    rf, gf, bf = _hex_to_rgb_u8(_PLAN_QA_FAIL_HEX)
    ruw, guw, buw = _hex_to_rgb_u8(_PLAN_QA_DOSE_UNDER_WARN_HEX)
    ruf, guf, buf = _hex_to_rgb_u8(_PLAN_QA_DOSE_UNDER_FAIL_HEX)
    rgba = np.zeros((n, 4), dtype=np.uint8)
    finite = np.isfinite(s)
    a = np.abs(s)
    pass_m = finite & (a <= pass_pp)
    over_warn_m = finite & (s > pass_pp) & (s <= warn_pp)
    over_fail_m = finite & (s > warn_pp)
    under_warn_m = finite & (s < -pass_pp) & (s >= -warn_pp)
    under_fail_m = finite & (s < -warn_pp)
    rgba[pass_m, 0] = np.uint8(rp)
    rgba[pass_m, 1] = np.uint8(gp)
    rgba[pass_m, 2] = np.uint8(bp)
    rgba[over_warn_m, 0] = np.uint8(rw)
    rgba[over_warn_m, 1] = np.uint8(gw)
    rgba[over_warn_m, 2] = np.uint8(bw)
    rgba[over_fail_m, 0] = np.uint8(rf)
    rgba[over_fail_m, 1] = np.uint8(gf)
    rgba[over_fail_m, 2] = np.uint8(bf)
    rgba[under_warn_m, 0] = np.uint8(ruw)
    rgba[under_warn_m, 1] = np.uint8(guw)
    rgba[under_warn_m, 2] = np.uint8(buw)
    rgba[under_fail_m, 0] = np.uint8(ruf)
    rgba[under_fail_m, 1] = np.uint8(guf)
    rgba[under_fail_m, 2] = np.uint8(buf)
    if alpha_u8 is None:
        rgba[:, 3] = np.uint8(255)
    else:
        rgba[:, 3] = au
    return rgba

def plan_dose_qa_tier_counts(
    signed_delta_pp: np.ndarray,
    *,
    pass_pp: float,
    warn_pp: float,
) -> tuple[int, int, int, int, int]:
    """Counts: pass, over_warn, over_fail, under_warn, under_fail (non-finite excluded)."""
    s = np.asarray(signed_delta_pp, dtype=np.float64).reshape(-1)
    finite = np.isfinite(s)
    a = np.abs(s)
    n_pass = int(np.count_nonzero(finite & (a <= pass_pp)))
    n_over_warn = int(np.count_nonzero(finite & (s > pass_pp) & (s <= warn_pp)))
    n_over_fail = int(np.count_nonzero(finite & (s > warn_pp)))
    n_under_warn = int(np.count_nonzero(finite & (s < -pass_pp) & (s >= -warn_pp)))
    n_under_fail = int(np.count_nonzero(finite & (s < -warn_pp)))
    return n_pass, n_over_warn, n_over_fail, n_under_warn, n_under_fail

def _plan_qa_error_line_polylines(
    meas_pts_view: np.ndarray,
    expected_plan_xyz: np.ndarray,
    dist_mm: np.ndarray,
    *,
    pass_mm: float,
    warn_mm: float,
    use_proton_water_depth_mm: bool = False,
    upstream_wet_mm: float = 0.0,
    z_depth_metric: str = "csda",
    plan_e_lo_mev: float | None = None,
    plan_e_hi_mev: float | None = None,
) -> tuple[Any, Any]:
    """Separate line sets for warn-tier and fail-tier points (measured → NN plan spot), view Z."""
    if pv is None:
        return None, None
    d = np.asarray(dist_mm, dtype=np.float64).reshape(-1)
    n = int(d.shape[0])
    if meas_pts_view.shape[0] != n or expected_plan_xyz.shape != (n, 3):
        raise ValueError("shape mismatch building plan QA error lines")
    pass_m = d <= pass_mm
    fail_m = d > warn_mm
    warn_m = ~pass_m & ~fail_m

    if plan_e_lo_mev is None or plan_e_hi_mev is None:
        raise ValueError("plan_e_lo_mev and plan_e_hi_mev required for plan QA error lines")

    exp = np.asarray(expected_plan_xyz, dtype=np.float64).reshape(-1, 3)
    meas = np.asarray(meas_pts_view, dtype=np.float64).reshape(-1, 3)
    exp_view = exp.copy()
    _z_cfg = ZAxisDisplayConfig(
        use_water_depth_mm=bool(use_proton_water_depth_mm),
        upstream_wet_mm=float(upstream_wet_mm),
        z_depth_metric=str(z_depth_metric),
    )
    if use_proton_water_depth_mm:
        d_lo, d_hi = plan_depth_bounds_mm(
            float(plan_e_lo_mev),
            float(plan_e_hi_mev),
            upstream_wet_mm=upstream_wet_mm,
            z_depth_metric=z_depth_metric,
        )
        exp_view[:, 2] = nominal_energy_to_scene_z(
            exp[:, 2],
            plan_e_lo=float(plan_e_lo_mev),
            plan_e_hi=float(plan_e_hi_mev),
            config=_z_cfg,
            depth_lo_mm=d_lo,
            depth_hi_mm=d_hi,
        )
    else:
        exp_view[:, 2] = nominal_energy_to_scene_z(
            exp[:, 2],
            plan_e_lo=float(plan_e_lo_mev),
            plan_e_hi=float(plan_e_hi_mev),
            config=_z_cfg,
        )

    def _build(idxs: np.ndarray) -> Any:
        pts_list: list[np.ndarray] = []
        lines_list: list[int] = []
        v = 0
        for i in idxs:
            p1 = exp_view[i]
            if not np.all(np.isfinite(p1)):
                continue
            p0 = meas[i]
            pts_list.append(p0)
            pts_list.append(p1)
            lines_list.extend((2, v, v + 1))
            v += 2
        if not pts_list:
            return None
        points = np.vstack(pts_list)
        lines = np.asarray(lines_list, dtype=np.int64)
        return pv.PolyData(points, lines=lines)

    warn_idx = np.flatnonzero(warn_m)
    fail_idx = np.flatnonzero(fail_m)
    return _build(warn_idx), _build(fail_idx)

def plan_qa_pass_warn_fail_counts(
    dist_mm: np.ndarray,
    *,
    pass_mm: float,
    warn_mm: float,
) -> tuple[int, int, int]:
    d = np.asarray(dist_mm, dtype=np.float64).reshape(-1)
    n_pass = int(np.count_nonzero(d <= pass_mm))
    n_fail = int(np.count_nonzero(d > warn_mm))
    n_warn = int(d.size) - n_pass - n_fail
    return n_pass, n_warn, n_fail

def plan_qa_measured_spot_pass_warn_fail(
    planned_xyz: list[tuple[float, float, float]],
    measured_rows: list[tuple[float, ...]],
    *,
    qa_mode: str,
    pass_thr: float,
    warn_thr: float,
    plan_mu: np.ndarray | None = None,
    a_is_x: bool = False,
) -> tuple[int, int, int]:
    """Pass / warn / fail counts for measured spots (same tiers as 3D plan QA coloring).

    Position: XY distance (mm) to nearest plan spot on the row's nominal layer.
    Dose: signed layer MU-fraction error (pp); warn and fail combine over- and under-dose.
    """
    mode = str(qa_mode).strip().lower().replace("-", "_")
    if mode == "dose":
        _dev_pp, plan_frac, meas_frac, _dist = plan_dose_fraction_deviation_pp(
            planned_xyz, plan_mu, measured_rows, a_is_x=a_is_x
        )
        signed_pp = np.where(
            np.isfinite(plan_frac) & np.isfinite(meas_frac),
            (meas_frac - plan_frac) * 100.0,
            np.nan,
        )
        npass, now, nof, nuw, nuf = plan_dose_qa_tier_counts(
            signed_pp, pass_pp=float(pass_thr), warn_pp=float(warn_thr)
        )
        return npass, int(now + nuw), int(nof + nuf)
    if mode != "position":
        raise ValueError("qa_mode must be 'position' or 'dose'")
    dist_mm = distances_measured_xy_to_layer_nn_plan_mm(
        planned_xyz, measured_rows, a_is_x=a_is_x
    )
    return plan_qa_pass_warn_fail_counts(
        dist_mm, pass_mm=float(pass_thr), warn_mm=float(warn_thr)
    )

def format_plan_qa_caption(
    *,
    pass_mm: float,
    warn_mm: float,
    n_pass: int,
    n_warn: int,
    n_fail: int,
) -> str:
    return (
        f"Plan QA: pass d≤{pass_mm:g} mm; warn {pass_mm:g}<d≤{warn_mm:g} mm; fail d>{warn_mm:g} mm "
        f"({n_pass} pass / {n_warn} warn / {n_fail} fail)."
    )

def plan_dose_fraction_deviation_pp(
    planned_xyz: list[tuple[float, float, float]],
    plan_mu: np.ndarray | None,
    measured_rows: list[tuple[float, ...]],
    *,
    a_is_x: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Layer-relative dose QA: |meas_frac − plan_frac| in percentage points per row."""
    n = len(measured_rows)
    dev_pp = np.full(n, np.nan, dtype=np.float64)
    plan_frac_out = np.full(n, np.nan, dtype=np.float64)
    meas_frac_out = np.full(n, np.nan, dtype=np.float64)
    dist_mm, _exp_xyz = layer_nn_plan_xy_distances_and_expected_xyz(
        planned_xyz, measured_rows, a_is_x=a_is_x
    )
    if plan_mu is None or len(plan_mu) != len(planned_xyz) or n == 0 or not planned_xyz:
        return dev_pp, plan_frac_out, meas_frac_out, dist_mm

    layer_e = nominal_layer_energies_mev(planned_xyz)
    if not layer_e:
        return dev_pp, plan_frac_out, meas_frac_out, dist_mm

    plan_mu_arr = np.asarray(plan_mu, dtype=np.float64)
    layer_mu = _layer_plan_mu_by_energy_layer(planned_xyz, plan_mu_arr, layer_e)
    plan_frac_by_layer: list[np.ndarray] = []
    for mu_arr in layer_mu:
        if mu_arr.size == 0:
            plan_frac_by_layer.append(np.zeros(0, dtype=np.float64))
            continue
        total = float(np.nansum(mu_arr))
        if not math.isfinite(total) or total <= 0.0:
            plan_frac_by_layer.append(np.full(mu_arr.shape[0], np.nan, dtype=np.float64))
        else:
            plan_frac_by_layer.append(mu_arr / total)

    li_raw = np.rint(np.asarray([float(t[2]) for t in measured_rows], dtype=np.float64)).astype(
        np.intp, copy=False
    )
    hi = len(layer_e) - 1
    np.clip(li_raw, 0, hi, out=li_raw)
    local_idx = layer_nn_local_spot_index_on_layer(
        planned_xyz, measured_rows, a_is_x=a_is_x
    )
    charges_by_layer = [
        np.zeros(layer_mu[ell].shape[0], dtype=np.float64) for ell in range(len(layer_e))
    ]
    for i, tup in enumerate(measured_rows):
        ell = int(li_raw[i])
        j = int(local_idx[i])
        if j < 0 or ell >= len(charges_by_layer) or j >= charges_by_layer[ell].shape[0]:
            continue
        ch = measured_charge_na_from_tuple(tup)
        if math.isfinite(ch) and ch > 0.0:
            charges_by_layer[ell][j] += ch

    for i, _tup in enumerate(measured_rows):
        ell = int(li_raw[i])
        j = int(local_idx[i])
        if ell >= len(plan_frac_by_layer) or j < 0 or j >= plan_frac_by_layer[ell].shape[0]:
            continue
        pf = float(plan_frac_by_layer[ell][j])
        layer_total = float(np.sum(charges_by_layer[ell]))
        if not math.isfinite(pf) or not math.isfinite(layer_total) or layer_total <= 0.0:
            continue
        mf = float(charges_by_layer[ell][j]) / layer_total
        plan_frac_out[i] = pf
        meas_frac_out[i] = mf
        dev_pp[i] = abs(mf - pf) * 100.0

    return dev_pp, plan_frac_out, meas_frac_out, dist_mm

def format_plan_dose_qa_caption(
    *,
    pass_pp: float,
    warn_pp: float,
    n_pass: int,
    n_over_warn: int,
    n_over_fail: int,
    n_under_warn: int,
    n_under_fail: int,
    spot_weight_mode: str = SPOT_WEIGHT_MODE_DEFAULT,
) -> str:
    w_lbl = measured_spot_weight_caption(spot_weight_mode)
    return (
        f"Dose QA (layer %): pass |Δ|≤{pass_pp:g} pp ({n_pass}). "
        f"Over: yellow {pass_pp:g}<Δ≤{warn_pp:g} pp ({n_over_warn}), "
        f"red Δ>{warn_pp:g} pp ({n_over_fail}). "
        f"Under: cyan −{warn_pp:g}≤Δ<−{pass_pp:g} pp ({n_under_warn}), "
        f"violet Δ<−{warn_pp:g} pp ({n_under_fail}). "
        f"Plan MU vs measured {w_lbl}."
    )
