"""Shared acquisition CSV preload for coarse alignment and auto assignment."""

from __future__ import annotations

import numpy as np

from spot_check.analysis.auto_columns import load_auto_fit_columns_from_csv
from spot_check.analysis.layers import _PlanImputeLookup
from spot_check.pipeline.types import PipelineConfig, PipelineState


def _normalize_auto_assign_method(auto_assign_method: str) -> str:
    assign_m = str(auto_assign_method).strip().lower().replace("-", "_")
    if assign_m == "sequential":
        return "plan_sequential"
    return assign_m


def preload_auto_fit_columns_if_needed(
    state: PipelineState,
    config: PipelineConfig,
    *,
    layer_mode_run: str,
    auto_assign_method: str,
) -> None:
    """Parse acquisition CSV once when coarse flat and/or auto assign need column arrays."""
    if state.auto_fit_columns is not None:
        return
    csv_path = config.csv_path
    planned = state.planned
    if csv_path is None or not planned:
        return

    mode = layer_mode_run.strip().lower().replace("-", "_")
    need_for_auto = mode == "auto"
    need_for_coarse = bool(config.coarse_flat_align)
    if not (need_for_auto or need_for_coarse):
        return

    plan_xy2 = np.asarray(
        [(float(px), float(py)) for px, py, _ in planned],
        dtype=np.float64,
    )
    global_lk = _PlanImputeLookup.from_xy(plan_xy2)
    if global_lk is None:
        return

    assign_m = _normalize_auto_assign_method(auto_assign_method)
    include_deadtime = need_for_auto and assign_m == "plan_sequential"
    state.auto_fit_columns = load_auto_fit_columns_from_csv(
        csv_path,
        global_lk=global_lk,
        a_is_x=False,
        spot_weight_mode=config.spot_weight_mode,
        max_points=None,
        include_deadtime_rows=include_deadtime,
        heal_partial_fit_axes=config.heal_partial_fit_axes,
    )
