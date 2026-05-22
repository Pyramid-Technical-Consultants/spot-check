"""Time-gap layer assignment from acquisition CSV."""

from __future__ import annotations

import csv
from typing import Any

import numpy as np

from spot_check.analysis.assign._post import apply_coarse_flat_to_rows
from spot_check.analysis.assign.base import LayerAssignerBase
from spot_check.analysis.assign.types import AssignCsvParams, MeasuredAssignResult
from spot_check.analysis.csv_io import open_acquisition_csv
from spot_check.analysis.layers import (
    _impute_plan_axis_fast,
    _layer_advance_plausible_vs_refill,
    _opt_float_cell,
    _plan_impute_lookups_per_layer,
    _PlanImputeLookup,
)
from spot_check.analysis.measured import (
    _channel_sum_na_from_row,
    _gate_int_from_row,
    _measured_row_with_sigma,
    measured_spot_weight_from_row,
    normalize_measured_spot_weight_mode,
)
from spot_check.analysis.spatial import (
    _ab_from_plan_xy,
    _build_layer_kdtrees,
    _plan_xy_by_energy_layer,
    _plan_xy_from_optional_ab,
    fit_position_row_ok,
    nominal_layer_energies_mev,
)
from spot_check.constants import GATE_COUNTER_KEY, SIGMA_A_KEY, SIGMA_B_KEY


class TimeGapAssigner(LayerAssignerBase):
    layer_mode = "time_gap"

    def validate(self, params: AssignCsvParams) -> None:
        if params.layer_gap_s <= 0:
            raise ValueError("layer_gap_s must be > 0")
        if params.refill_same_spot_xy_tol_mm <= 0:
            raise ValueError("refill_same_spot_xy_tol_mm must be > 0")
        if params.refill_trust_time_gap_stay_dist_mm <= 0:
            raise ValueError("refill_trust_time_gap_stay_dist_mm must be > 0")

    def assign(self, params: AssignCsvParams) -> MeasuredAssignResult:
        swm = normalize_measured_spot_weight_mode(params.spot_weight_mode)
        planned_xyz = params.planned_xyz
        layer_energies: list[float] | None = None
        max_layer: int | None = None
        if planned_xyz:
            layer_energies = nominal_layer_energies_mev(planned_xyz)
            if layer_energies:
                max_layer = len(layer_energies) - 1

        out: list[tuple[float, ...]] = []
        spot_ids_tg: list[int] = []
        gates_tg: list[int] = []
        layer = 0
        timing_spot_id = 0
        prev_t: float | None = None
        prev_mx: float | None = None
        prev_my: float | None = None
        plan_xy2_tg: np.ndarray | None = None
        global_lk_tg: _PlanImputeLookup | None = None
        layer_lks_tg: list[_PlanImputeLookup | None] | None = None
        layer_trees_tg: list[Any] | None = None
        if planned_xyz:
            plan_xy2_tg = np.asarray(
                [(float(px), float(py)) for px, py, _ in planned_xyz],
                dtype=np.float64,
            )
            global_lk_tg = _PlanImputeLookup.from_xy(plan_xy2_tg)
            if layer_energies:
                layer_xy_tg = _plan_xy_by_energy_layer(planned_xyz, layer_energies)
                layer_lks_tg = _plan_impute_lookups_per_layer(layer_xy_tg)
                layer_trees_tg = _build_layer_kdtrees(layer_xy_tg)

        fa_key = "Fit Amplitude A (nA)"
        a_key = "Fit Mean Position A (mm)"
        b_key = "Fit Mean Position B (mm)"

        with open_acquisition_csv(params.csv_path) as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return MeasuredAssignResult([], [], layer_mode=self.layer_mode)
            time_key = reader.fieldnames[0]

            for row in reader:
                if not (row.get(fa_key) or "").strip():
                    continue
                try:
                    t = float(row[time_key])
                except ValueError:
                    continue
                a_opt = _opt_float_cell(row, a_key)
                b_opt = _opt_float_cell(row, b_key)
                mx_p, my_p, pcd = _plan_xy_from_optional_ab(
                    a_opt, b_opt, a_is_x=params.a_is_x
                )
                if not fit_position_row_ok(
                    pcd, heal_partial_fit_axes=params.heal_partial_fit_axes
                ):
                    continue

                li_use = layer if max_layer is None else min(layer, max_layer)
                if global_lk_tg is None:
                    mx = float(mx_p or 0.0)
                    my = float(my_p or 0.0)
                else:
                    lk_row: _PlanImputeLookup | None = None
                    if layer_lks_tg and 0 <= li_use < len(layer_lks_tg):
                        lk_row = layer_lks_tg[li_use]
                    lk_row = lk_row or global_lk_tg
                    mx, my = _impute_plan_axis_fast(lk_row, mx_p, my_p)
                a_fin, b_fin = _ab_from_plan_xy(mx, my, a_is_x=params.a_is_x)

                if prev_t is not None and (t - prev_t) >= params.layer_gap_s:
                    same_spot_refill = (
                        prev_mx is not None
                        and prev_my is not None
                        and float(np.hypot(mx - prev_mx, my - prev_my))
                        <= params.refill_same_spot_xy_tol_mm
                    )
                    if not same_spot_refill:
                        timing_spot_id += 1
                        if max_layer is None:
                            layer += 1
                        elif (
                            layer < max_layer
                            and layer_energies
                            and planned_xyz is not None
                            and _layer_advance_plausible_vs_refill(
                                planned_xyz,
                                layer_energies,
                                layer,
                                mx,
                                my,
                                trust_time_gap_stay_dist_mm=(
                                    params.refill_trust_time_gap_stay_dist_mm
                                ),
                                layer_trees=layer_trees_tg,
                            )
                        ):
                            layer += 1

                eff_layer = layer if max_layer is None else min(layer, max_layer)
                sa = _opt_float_cell(row, SIGMA_A_KEY)
                sb = _opt_float_cell(row, SIGMA_B_KEY)
                out.append(
                    _measured_row_with_sigma(
                        a_fin,
                        b_fin,
                        float(eff_layer),
                        measured_spot_weight_from_row(row, swm),
                        int(pcd),
                        sa,
                        sb,
                        channel_sum_na=_channel_sum_na_from_row(row),
                        time_s=t,
                    )
                )
                g_cell = _gate_int_from_row(row, GATE_COUNTER_KEY)
                gates_tg.append(int(g_cell) if g_cell is not None else -1)
                spot_ids_tg.append(timing_spot_id)
                prev_t = t
                prev_mx, prev_my = mx, my
                if params.max_points is not None and len(out) >= params.max_points:
                    break

        out = apply_coarse_flat_to_rows(
            out, transform=params.coarse_flat_transform, a_is_x=params.a_is_x
        )
        return MeasuredAssignResult(
            out,
            spot_ids_tg,
            layer_mode=self.layer_mode,
            gates=gates_tg,
        )


assigner = TimeGapAssigner()
