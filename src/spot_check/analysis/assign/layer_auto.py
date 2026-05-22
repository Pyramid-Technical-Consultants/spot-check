"""Auto layer assignment (episodes or plan_sequential sub-assigners)."""

from __future__ import annotations

import numpy as np

from spot_check.analysis.assign.base import LayerAssignerBase
from spot_check.analysis.assign.types import (
    AssignCsvParams,
    EpisodeAssignParams,
    MeasuredAssignResult,
    PlanSequentialAssignParams,
)
from spot_check.analysis.layers import (
    _impute_plan_axis_fast,
    _plan_impute_lookups_per_layer,
    _PlanImputeLookup,
)
from spot_check.analysis.measured import (
    _measured_row_with_sigma,
    normalize_measured_spot_weight_mode,
)
from spot_check.analysis.spatial import (
    _ab_from_plan_xy,
    _plan_xy_by_energy_layer,
    nominal_layer_energies_mev,
)
from spot_check.constants import (
    AUTO_ASSIGN_METHODS,
    AUTO_EDGE_DEAD_RATIO_DEFAULT,
    AUTO_EDGE_TINY_MERGE_ROWS,
)


def _normalize_auto_assign_method(method: str) -> str:
    assign_m = str(method).strip().lower().replace("-", "_")
    if assign_m == "sequential":
        assign_m = "plan_sequential"
    return assign_m


class AutoAssigner(LayerAssignerBase):
    layer_mode = "auto"

    def validate(self, params: AssignCsvParams) -> None:
        if params.auto_episode_gap_s <= 0:
            raise ValueError("auto_episode_gap_s must be > 0")
        if params.auto_spot_xy_jump_mm <= 0:
            raise ValueError("auto_spot_xy_jump_mm must be > 0")
        if params.auto_min_on_spot_weight_na < 0:
            raise ValueError("auto_min_on_spot_weight_na must be >= 0")
        if int(params.auto_min_episode_rows) < 1:
            raise ValueError("auto_min_episode_rows must be >= 1")
        if params.viterbi_advance_penalty_mm2 < 0:
            raise ValueError("viterbi_advance_penalty_mm2 must be >= 0")
        if not params.planned_xyz:
            raise ValueError("auto requires planned_xyz from the RT plan")
        assign_m = _normalize_auto_assign_method(params.auto_assign_method)
        if assign_m not in AUTO_ASSIGN_METHODS:
            raise ValueError(
                f"auto_assign_method must be one of {sorted(AUTO_ASSIGN_METHODS)}, "
                f"got {assign_m!r}"
            )

    def assign(self, params: AssignCsvParams) -> MeasuredAssignResult:
        from spot_check.analysis.assign import run_auto_assignment
        from spot_check.analysis.auto_columns import load_auto_fit_columns_from_csv
        from spot_check.analysis.episodes import cols_with_delivery_weights

        planned_xyz = params.planned_xyz
        assert planned_xyz is not None
        assign_m_run = _normalize_auto_assign_method(params.auto_assign_method)
        layer_energies = nominal_layer_energies_mev(planned_xyz)
        if not layer_energies:
            raise ValueError(
                "plan-based layer modes require a plan with at least one nominal energy layer"
            )
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

        cols = params.preloaded_auto_columns
        swm = normalize_measured_spot_weight_mode(params.spot_weight_mode)
        if cols is None:
            cols = load_auto_fit_columns_from_csv(
                params.csv_path,
                global_lk=global_lk,
                a_is_x=params.a_is_x,
                spot_weight_mode=swm,
                max_points=params.max_points,
                include_deadtime_rows=assign_m_run == "plan_sequential",
                heal_partial_fit_axes=params.heal_partial_fit_axes,
            )
        if len(cols) == 0:
            return MeasuredAssignResult(
                [],
                [],
                layer_mode=self.layer_mode,
                assign_method=assign_m_run,
            )

        if params.coarse_flat_transform is not None:
            from spot_check.analysis.alignment import (
                apply_coarse_flat_transform_to_auto_fit_columns,
            )

            cols = apply_coarse_flat_transform_to_auto_fit_columns(
                cols,
                params.coarse_flat_transform,
                a_is_x=params.a_is_x,
            )

        n_plan_spots = len(planned_xyz)
        spots_per_layer = [
            int(np.asarray(arr, dtype=np.float64).reshape(-1, 2).shape[0])
            for arr in layer_xy
        ]

        gap_s = float(params.auto_episode_gap_s)
        xy_jump = float(params.auto_spot_xy_jump_mm)
        min_w = float(params.auto_min_on_spot_weight_na)
        min_rows = int(params.auto_min_episode_rows)
        dead_ratio = float(AUTO_EDGE_DEAD_RATIO_DEFAULT)
        tiny_merge = int(AUTO_EDGE_TINY_MERGE_ROWS)
        if params.auto_infer_params:
            from spot_check.analysis.auto_params import infer_auto_layer_params

            auto_p = infer_auto_layer_params(cols, planned_xyz)
            gap_s = auto_p.episode_gap_s
            xy_jump = auto_p.spot_xy_jump_mm
            min_w = auto_p.min_on_spot_weight_na
            min_rows = auto_p.min_episode_rows
            dead_ratio = auto_p.dead_ratio
            tiny_merge = auto_p.tiny_merge_rows

        ep_params = EpisodeAssignParams(
            episode_gap_s=gap_s,
            min_on_spot_weight_na=min_w,
            spot_xy_jump_mm=xy_jump,
            min_episode_rows=min_rows,
            dead_ratio=dead_ratio,
            tiny_merge_rows=tiny_merge,
        )
        ps_params = PlanSequentialAssignParams(min_rows_on_spot=1)
        assign_out = run_auto_assignment(
            assign_m_run,
            cols,
            n_plan_spots=n_plan_spots,
            plan_xy=plan_xy2,
            spots_per_layer=spots_per_layer,
            episode_params=ep_params,
            plan_sequential_params=ps_params,
        )
        aligned_groups = assign_out.spans
        layers_idx_auto = assign_out.layer_index_per_span
        plan_idx = assign_out.plan_index_per_row

        cols_w = cols_with_delivery_weights(cols)
        hi_auto = max_layer

        def _clamp_layer(efi: int) -> int:
            if efi < 0:
                return 0
            if efi > hi_auto:
                return hi_auto
            return efi

        out_rows: list[tuple[float, ...]] = []
        spot_ids: list[int] = []
        for ei, (s, e) in enumerate(aligned_groups):
            efi = _clamp_layer(int(layers_idx_auto[ei]))
            lk_ref = layer_lks[efi] if efi < len(layer_lks) else None
            if lk_ref is None:
                lk_ref = global_lk
            for ri in range(s, e):
                pcd = int(cols.pcd[ri])
                mx_pp = float(cols.mx_p[ri])
                my_pp = float(cols.my_p[ri])
                if pcd == 0:
                    a_fin, b_fin = float(cols.a[ri]), float(cols.b[ri])
                else:
                    mx_f, my_f = _impute_plan_axis_fast(
                        lk_ref,
                        mx_pp if mx_pp == mx_pp else None,
                        my_pp if my_pp == my_pp else None,
                    )
                    a_fin, b_fin = _ab_from_plan_xy(mx_f, my_f, a_is_x=params.a_is_x)
                sa_v = float(cols.sa[ri])
                sb_v = float(cols.sb[ri])
                sa = sa_v if sa_v == sa_v else None
                sb = sb_v if sb_v == sb_v else None
                wch = float(cols_w.weight[ri])
                ch_n = float(cols_w.ch_n[ri])
                out_rows.append(
                    _measured_row_with_sigma(
                        a_fin,
                        b_fin,
                        float(efi),
                        wch,
                        pcd,
                        sa,
                        sb,
                        channel_sum_na=ch_n,
                        time_s=float(cols.t[ri]),
                    )
                )
                if assign_m_run == "plan_sequential" and plan_idx is not None:
                    spot_ids.append(int(plan_idx[ri]))
                else:
                    spot_ids.append(ei)
                if params.max_points is not None and len(out_rows) >= params.max_points:
                    return MeasuredAssignResult(
                        out_rows,
                        spot_ids,
                        layer_mode=self.layer_mode,
                        assign_method=assign_m_run,
                        n_plan_spots=n_plan_spots,
                        planned_xyz=list(planned_xyz),
                        spots_per_layer=list(spots_per_layer),
                        a_is_x=params.a_is_x,
                    )

        return MeasuredAssignResult(
            out_rows,
            spot_ids,
            layer_mode=self.layer_mode,
            assign_method=assign_m_run,
            n_plan_spots=n_plan_spots,
            planned_xyz=list(planned_xyz),
            spots_per_layer=list(spots_per_layer),
            a_is_x=params.a_is_x,
        )


assigner = AutoAssigner()
