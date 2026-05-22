"""Gate-counter layer/spot assignment from acquisition CSV."""

from __future__ import annotations

import bisect
import csv

import numpy as np

from spot_check.analysis.assign._post import apply_coarse_flat_to_rows
from spot_check.analysis.assign.base import LayerAssignerBase
from spot_check.analysis.assign.types import AssignCsvParams, MeasuredAssignResult
from spot_check.analysis.csv_io import open_acquisition_csv
from spot_check.analysis.layers import (
    _impute_plan_axis_fast,
    _opt_float_cell,
    _plan_impute_lookups_per_layer,
    _PlanImputeLookup,
)
from spot_check.analysis.measured import (
    _channel_sum_na_from_row,
    _measured_row_with_sigma,
    measured_spot_weight_from_row,
    normalize_measured_spot_weight_mode,
)
from spot_check.analysis.spatial import (
    _ab_from_plan_xy,
    _plan_xy_by_energy_layer,
    _plan_xy_from_optional_ab,
    nominal_layer_energies_mev,
)
from spot_check.constants import GATE_COUNTER_KEY, SIGMA_A_KEY, SIGMA_B_KEY


class GateCounterAssigner(LayerAssignerBase):
    layer_mode = "gate_counter"

    def validate(self, params: AssignCsvParams) -> None:
        if not params.planned_xyz:
            raise ValueError("gate_counter requires planned_xyz from the RT plan")

    def assign(self, params: AssignCsvParams) -> MeasuredAssignResult:
        planned_xyz = params.planned_xyz
        assert planned_xyz is not None
        swm = normalize_measured_spot_weight_mode(params.spot_weight_mode)
        layer_energies = nominal_layer_energies_mev(planned_xyz)
        if not layer_energies:
            raise ValueError("gate_counter requires a plan with at least one nominal energy layer")
        max_layer = len(layer_energies) - 1
        layer_xy_gc = _plan_xy_by_energy_layer(planned_xyz, layer_energies)
        spots_per = [
            int(np.asarray(arr, dtype=np.float64).reshape(-1, 2).shape[0]) for arr in layer_xy_gc
        ]
        if sum(spots_per) == 0:
            raise ValueError("gate_counter: plan has no spots")
        cumul: list[int] = [0]
        for c in spots_per:
            cumul.append(cumul[-1] + c)
        plan_xy2_gc = np.asarray(
            [(float(px), float(py)) for px, py, _ in planned_xyz],
            dtype=np.float64,
        )
        global_lk_gc = _PlanImputeLookup.from_xy(plan_xy2_gc)
        if global_lk_gc is None:
            raise ValueError("gate_counter: plan has no XY spots")
        layer_lks_gc = _plan_impute_lookups_per_layer(layer_xy_gc)
        hi_gc = max_layer
        out_gc: list[tuple[float, ...]] = []
        spot_ids_gc: list[int] = []
        prev_gate: int | None = None
        spot_id_gc = -1
        i_spot = 0
        eff_li = 0
        fa_key = "Fit Amplitude A (nA)"
        a_key = "Fit Mean Position A (mm)"
        b_key = "Fit Mean Position B (mm)"
        with open_acquisition_csv(params.csv_path) as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return MeasuredAssignResult([], [], layer_mode=self.layer_mode)
            gc_key = GATE_COUNTER_KEY
            if gc_key not in reader.fieldnames:
                raise ValueError(
                    f"CSV has no “{GATE_COUNTER_KEY}” column (columns: {list(reader.fieldnames)!r})"
                )
            time_key = "time (s)" if "time (s)" in reader.fieldnames else reader.fieldnames[0]
            for row in reader:
                g_raw = (row.get(gc_key) or "").strip()
                if not g_raw:
                    continue
                try:
                    g = int(float(g_raw))
                except ValueError:
                    continue
                if g != prev_gate:
                    if g % 2 == 1:
                        spot_id_gc += 1
                        eff_li = max(0, min(bisect.bisect_right(cumul, i_spot) - 1, hi_gc))
                        i_spot += 1
                    prev_gate = g
                if g % 2 == 0:
                    continue
                if not (row.get(fa_key) or "").strip():
                    continue
                try:
                    t_row = float(row[time_key])
                except (KeyError, ValueError, TypeError):
                    t_row = float("nan")
                a_opt = _opt_float_cell(row, a_key)
                b_opt = _opt_float_cell(row, b_key)
                mx_p, my_p, pcd = _plan_xy_from_optional_ab(
                    a_opt, b_opt, a_is_x=params.a_is_x
                )
                if pcd < 0:
                    continue
                lk_row = layer_lks_gc[eff_li] if eff_li < len(layer_lks_gc) else None
                lk_use = lk_row or global_lk_gc
                mx, my = _impute_plan_axis_fast(lk_use, mx_p, my_p)
                a_fin, b_fin = _ab_from_plan_xy(mx, my, a_is_x=params.a_is_x)
                w_ch = measured_spot_weight_from_row(row, swm)
                ch_n = _channel_sum_na_from_row(row)
                sa = _opt_float_cell(row, SIGMA_A_KEY)
                sb = _opt_float_cell(row, SIGMA_B_KEY)
                out_gc.append(
                    _measured_row_with_sigma(
                        a_fin,
                        b_fin,
                        float(eff_li),
                        w_ch,
                        int(pcd),
                        sa,
                        sb,
                        channel_sum_na=ch_n,
                        time_s=t_row,
                    )
                )
                spot_ids_gc.append(spot_id_gc)
                if params.max_points is not None and len(out_gc) >= params.max_points:
                    break
        out_gc = apply_coarse_flat_to_rows(
            out_gc, transform=params.coarse_flat_transform, a_is_x=params.a_is_x
        )
        return MeasuredAssignResult(
            out_gc,
            spot_ids_gc,
            layer_mode=self.layer_mode,
        )


assigner = GateCounterAssigner()
