"""Format plan/measured spot metadata for the inspect popup."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from spot_check.analysis.measured import measured_row_time_s
from spot_check.analysis.plan_qa import plan_dose_fraction_deviation_pp
from spot_check.analysis.spatial import layer_nn_plan_match_for_measured, nominal_layer_energies_mev

SpotKind = Literal["plan", "measured"]


@dataclass(frozen=True)
class SpotInfoRow:
    label: str
    value: str


def _fmt_num(v: float, *, digits: int = 3) -> str:
    if not math.isfinite(v):
        return "—"
    return f"{v:.{digits}g}"


def _fmt_time_s(t: float) -> str:
    if not math.isfinite(t):
        return "—"
    return f"{t:.3f} s"


def format_plan_spot_info(
    spot_index: int,
    planned_xyz: list[tuple[float, float, float]],
    *,
    xlab: str,
    ylab: str,
    plan_mu: np.ndarray | None = None,
    plan_fwhm_xy_mm: np.ndarray | None = None,
    plan_time_s: np.ndarray | None = None,
    plan_spots_no_data: np.ndarray | None = None,
) -> list[SpotInfoRow]:
    """Build label/value rows for a plan spot (``spot_index`` is 0-based)."""
    n = len(planned_xyz)
    if spot_index < 0 or spot_index >= n:
        return [SpotInfoRow("Error", f"Invalid plan spot index {spot_index + 1}")]

    px, py, pe = planned_xyz[spot_index]
    layer_e = nominal_layer_energies_mev(planned_xyz)
    li = next((i for i, e in enumerate(layer_e) if abs(float(e) - float(pe)) < 1e-6), 0)

    rows: list[SpotInfoRow] = [
        SpotInfoRow("Type", "Plan spot"),
        SpotInfoRow("Index", str(spot_index + 1)),
        SpotInfoRow(xlab, f"{px:.3f} mm"),
        SpotInfoRow(ylab, f"{py:.3f} mm"),
        SpotInfoRow("Nominal energy", f"{pe:.3f} MeV"),
        SpotInfoRow("Layer index", str(li)),
    ]
    if plan_mu is not None and spot_index < int(plan_mu.shape[0]):
        rows.append(SpotInfoRow("Plan MU", _fmt_num(float(plan_mu[spot_index]), digits=4)))
    if plan_fwhm_xy_mm is not None and spot_index < int(plan_fwhm_xy_mm.shape[0]):
        fx, fy = float(plan_fwhm_xy_mm[spot_index, 0]), float(plan_fwhm_xy_mm[spot_index, 1])
        rows.append(SpotInfoRow("FWHM X", f"{_fmt_num(fx)} mm"))
        rows.append(SpotInfoRow("FWHM Y", f"{_fmt_num(fy)} mm"))
    if plan_time_s is not None and spot_index < int(plan_time_s.shape[0]):
        rows.append(SpotInfoRow("Delivery time", _fmt_time_s(float(plan_time_s[spot_index]))))
    if plan_spots_no_data is not None and spot_index < int(plan_spots_no_data.shape[0]):
        if bool(plan_spots_no_data[spot_index]):
            rows.append(SpotInfoRow("Measured data", "None assigned"))
    return rows


def format_measured_spot_info(
    spot_index: int,
    measured_rows: list[tuple[float, ...]],
    planned_xyz: list[tuple[float, float, float]],
    *,
    a_is_x: bool,
    plan_mu: np.ndarray | None = None,
    qa_mode: str = "position",
    display_state: dict[str, Any] | None = None,
) -> list[SpotInfoRow]:
    """Build label/value rows for a measured spot (``spot_index`` is 0-based source row)."""
    n = len(measured_rows)
    if spot_index < 0 or spot_index >= n:
        return [SpotInfoRow("Error", f"Invalid measured spot index {spot_index + 1}")]

    tup = measured_rows[spot_index]
    a_mm = float(tup[0])
    b_mm = float(tup[1])
    layer = int(round(float(tup[2])))
    weight = float(tup[3]) if len(tup) > 3 else float("nan")
    partial = int(tup[4]) if len(tup) > 4 else 0
    sig_a = float(tup[5]) if len(tup) > 5 else float("nan")
    sig_b = float(tup[6]) if len(tup) > 6 else float("nan")

    rows: list[SpotInfoRow] = [
        SpotInfoRow("Type", "Measured spot"),
        SpotInfoRow("Index", str(spot_index + 1)),
        SpotInfoRow("Fit A", f"{a_mm:.3f} mm"),
        SpotInfoRow("Fit B", f"{b_mm:.3f} mm"),
        SpotInfoRow("Layer index", str(layer)),
        SpotInfoRow("Spot weight", _fmt_num(weight, digits=4)),
        SpotInfoRow("Partial code", str(partial)),
        SpotInfoRow("σ A", f"{_fmt_num(sig_a)} mm"),
        SpotInfoRow("σ B", f"{_fmt_num(sig_b)} mm"),
    ]
    t_s = measured_row_time_s(tup)
    if math.isfinite(t_s):
        rows.append(SpotInfoRow("Delivery time", _fmt_time_s(t_s)))
    elif display_state is not None:
        meas_time = display_state.get("meas_time_final")
        meas_src_idx = display_state.get("meas_src_idx")
        if meas_time is not None and meas_src_idx is not None:
            src = np.asarray(meas_src_idx, dtype=np.int64).reshape(-1)
            mt = np.asarray(meas_time, dtype=np.float64).reshape(-1)
            hit = np.flatnonzero(src == spot_index)
            if hit.size and hit[0] < mt.size:
                rows.append(SpotInfoRow("Delivery time", _fmt_time_s(float(mt[hit[0]]))))

    if planned_xyz:
        dist, exp_xyz, exp_mu = layer_nn_plan_match_for_measured(
            planned_xyz, plan_mu, measured_rows, a_is_x=a_is_x
        )
        if spot_index < dist.shape[0]:
            rows.append(SpotInfoRow("Plan XY distance", f"{_fmt_num(float(dist[spot_index]))} mm"))
        if spot_index < exp_xyz.shape[0]:
            ex = float(exp_xyz[spot_index, 0])
            ey = float(exp_xyz[spot_index, 1])
            ee = float(exp_xyz[spot_index, 2])
            rows.append(SpotInfoRow("Expected plan X", f"{ex:.3f} mm"))
            rows.append(SpotInfoRow("Expected plan Y", f"{ey:.3f} mm"))
            rows.append(SpotInfoRow("Expected plan E", f"{ee:.3f} MeV"))
        if exp_mu is not None and spot_index < exp_mu.shape[0]:
            mu_v = float(exp_mu[spot_index])
            rows.append(SpotInfoRow("Expected plan MU", _fmt_num(mu_v, digits=4)))

        if qa_mode == "dose" and plan_mu is not None:
            dev_pp, _, _, _ = plan_dose_fraction_deviation_pp(
                planned_xyz, plan_mu, measured_rows, a_is_x=a_is_x
            )
            if spot_index < dev_pp.shape[0]:
                dv = float(dev_pp[spot_index])
                rows.append(SpotInfoRow("Dose deviation", f"{_fmt_num(dv)} pp"))

    return rows


def format_spot_info(
    kind: SpotKind,
    spot_index: int,
    *,
    planned_xyz: list[tuple[float, float, float]],
    measured_rows: list[tuple[float, ...]],
    xlab: str,
    ylab: str,
    a_is_x: bool,
    plan_mu: np.ndarray | None = None,
    plan_fwhm_xy_mm: np.ndarray | None = None,
    plan_time_s: np.ndarray | None = None,
    plan_spots_no_data: np.ndarray | None = None,
    qa_mode: str = "position",
    display_state: dict[str, Any] | None = None,
) -> list[SpotInfoRow]:
    if kind == "plan":
        return format_plan_spot_info(
            spot_index,
            planned_xyz,
            xlab=xlab,
            ylab=ylab,
            plan_mu=plan_mu,
            plan_fwhm_xy_mm=plan_fwhm_xy_mm,
            plan_time_s=plan_time_s,
            plan_spots_no_data=plan_spots_no_data,
        )
    return format_measured_spot_info(
        spot_index,
        measured_rows,
        planned_xyz,
        a_is_x=a_is_x,
        plan_mu=plan_mu,
        qa_mode=qa_mode,
        display_state=display_state,
    )
