"""Plan-guided Viterbi layer assignment from acquisition CSV."""

from __future__ import annotations

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
    viterbi_monotone_layer_assign,
)
from spot_check.analysis.measured import (
    _channel_sum_na_from_row,
    _gate_int_from_row,
    _gate_phase_spot_ids_for_rows,
    _measured_row_with_sigma,
    measured_spot_weight_from_row,
    normalize_measured_spot_weight_mode,
)
from spot_check.analysis.spatial import (
    _ab_from_plan_xy,
    _emit_sqdist_to_layers_mm2,
    _plan_xy_by_energy_layer,
    _plan_xy_from_optional_ab,
    fit_position_row_ok,
    nominal_layer_energies_mev,
)
from spot_check.constants import GATE_COUNTER_KEY, SIGMA_A_KEY, SIGMA_B_KEY


class PlanViterbiAssigner(LayerAssignerBase):
    layer_mode = "plan_viterbi"

    def validate(self, params: AssignCsvParams) -> None:
        if params.viterbi_advance_penalty_mm2 < 0:
            raise ValueError("viterbi_advance_penalty_mm2 must be >= 0")
        if not params.planned_xyz:
            raise ValueError("plan_viterbi requires planned_xyz from the RT plan")
        layer_energies = nominal_layer_energies_mev(params.planned_xyz)
        if not layer_energies:
            raise ValueError(
                "plan-based layer modes require a plan with at least one nominal energy layer"
            )

    def spot_ids_are_plan_slots(self, result: MeasuredAssignResult) -> bool:
        return bool(result.gates) and any(int(g) >= 0 for g in result.gates)

    def assign(self, params: AssignCsvParams) -> MeasuredAssignResult:
        planned_xyz = params.planned_xyz
        assert planned_xyz is not None
        swm = normalize_measured_spot_weight_mode(params.spot_weight_mode)
        layer_energies = nominal_layer_energies_mev(planned_xyz)
        max_layer = len(layer_energies) - 1
        layer_xy = _plan_xy_by_energy_layer(planned_xyz, layer_energies)
        plan_xy2 = np.asarray(
            [(float(px), float(py)) for px, py, _ in planned_xyz],
            dtype=np.float64,
        )
        global_lk = _PlanImputeLookup.from_xy(plan_xy2)
        if global_lk is None:
            raise ValueError("plan has no scan spots for imputation / Viterbi")
        layer_lks = _plan_impute_lookups_per_layer(layer_xy)
        ab_buf: list[tuple[float, float]] = []
        xy_buf: list[list[float]] = []
        w_buf: list[float] = []
        ch_buf: list[float] = []
        partial_plan_xy: list[tuple[float | None, float | None]] = []
        partial_codes: list[int] = []
        gates_acc: list[int] = []
        sig_acc: list[tuple[float | None, float | None]] = []
        time_acc: list[float] = []
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
                    t_row = float(row[time_key])
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
                mx_i, my_i = _impute_plan_axis_fast(global_lk, mx_p, my_p)
                a_fin, b_fin = _ab_from_plan_xy(mx_i, my_i, a_is_x=params.a_is_x)
                ab_buf.append((a_fin, b_fin))
                xy_buf.append([mx_i, my_i])
                w_buf.append(measured_spot_weight_from_row(row, swm))
                ch_buf.append(_channel_sum_na_from_row(row))
                partial_plan_xy.append((mx_p, my_p))
                partial_codes.append(pcd)
                sig_acc.append(
                    (_opt_float_cell(row, SIGMA_A_KEY), _opt_float_cell(row, SIGMA_B_KEY))
                )
                time_acc.append(float(t_row))
                g_cell = _gate_int_from_row(row, GATE_COUNTER_KEY)
                gates_acc.append(int(g_cell) if g_cell is not None else -1)
                if params.max_points is not None and len(ab_buf) >= params.max_points:
                    break
        if not ab_buf:
            return MeasuredAssignResult([], [], layer_mode=self.layer_mode)
        meas_xy = np.asarray(xy_buf, dtype=np.float64)
        emit = _emit_sqdist_to_layers_mm2(meas_xy, layer_xy)
        layers_idx = viterbi_monotone_layer_assign(emit, params.viterbi_advance_penalty_mm2)
        hi = max_layer
        for i, (mx_p, my_p) in enumerate(partial_plan_xy):
            if partial_codes[i] == 0:
                continue
            efi = int(layers_idx[i])
            if efi < 0:
                efi = 0
            elif efi > hi:
                efi = hi
            lk_ref = layer_lks[efi]
            if lk_ref is None:
                lk_ref = global_lk
            mx_f, my_f = _impute_plan_axis_fast(lk_ref, mx_p, my_p)
            a_fin, b_fin = _ab_from_plan_xy(mx_f, my_f, a_is_x=params.a_is_x)
            ab_buf[i] = (a_fin, b_fin)

        out: list[tuple[float, ...]] = []
        for i, ((a, b), ell, wch, ch_n, pcd) in enumerate(
            zip(ab_buf, layers_idx, w_buf, ch_buf, partial_codes)
        ):
            efi = int(ell)
            if efi < 0:
                efi = 0
            elif efi > hi:
                efi = hi
            sa, sb = sig_acc[i]
            out.append(
                _measured_row_with_sigma(
                    a,
                    b,
                    float(efi),
                    wch,
                    pcd,
                    sa,
                    sb,
                    channel_sum_na=ch_n,
                    time_s=time_acc[i],
                )
            )
        agg_ids = (
            _gate_phase_spot_ids_for_rows(gates_acc)
            if any(g >= 0 for g in gates_acc)
            else [int(layers_idx[i]) for i in range(len(out))]
        )
        out = apply_coarse_flat_to_rows(
            out, transform=params.coarse_flat_transform, a_is_x=params.a_is_x
        )
        return MeasuredAssignResult(
            out,
            list(agg_ids),
            layer_mode=self.layer_mode,
            gates=gates_acc,
        )


assigner = PlanViterbiAssigner()
