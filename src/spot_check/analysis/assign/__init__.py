"""Spot/layer assignment algorithms (mutually isolated; use only this package entry point)."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from spot_check.analysis.assign import episodes as _episodes
from spot_check.analysis.assign import plan_sequential as _plan_sequential
from spot_check.analysis.assign.base import (
    finalize_measured_assign_coverage,
    normalize_layer_mode,
    plan_spots_without_assignment_data,
)
from spot_check.analysis.assign.types import (
    LAYER_ASSIGN_MODES,
    AssignCsvParams,
    AutoAssignResult,
    EpisodeAssignParams,
    EpisodeSpan,
    LayerAssigner,
    MeasuredAssignResult,
    PlanIndexArray,
    PlanSequentialAssignParams,
    SpotAssigner,
)
from spot_check.analysis.auto_columns import AutoFitColumns
from spot_check.constants import AUTO_ASSIGN_METHODS

__all__ = [
    "AUTO_ASSIGN_METHODS",
    "AssignCsvParams",
    "AutoAssignResult",
    "EpisodeAssignParams",
    "EpisodeSpan",
    "LAYER_ASSIGN_MODES",
    "LayerAssigner",
    "MeasuredAssignResult",
    "PlanIndexArray",
    "PlanSequentialAssignParams",
    "SpotAssigner",
    "assign_plan_indices_sequential",
    "finalize_measured_assign_coverage",
    "get_layer_assigner",
    "plan_spot_index_per_span",
    "plan_spots_without_assignment_data",
    "run_auto_assignment",
    "run_layer_assignment",
    "sequential_spans_from_plan_indices",
]

# Re-export plan-sequential helpers for tests (implementation lives in assign.plan_sequential).
assign_plan_indices_sequential = _plan_sequential.assign_plan_indices_sequential
sequential_spans_from_plan_indices = _plan_sequential.sequential_spans_from_plan_indices
plan_spot_index_per_span = _plan_sequential.plan_spot_index_per_span

_ASSIGNERS: dict[str, SpotAssigner] = {
    _episodes.assigner.method: _episodes.assigner,
    _plan_sequential.assigner.method: _plan_sequential.assigner,
}

_LAYER_ASSIGNERS: dict[str, LayerAssigner] | None = None


def _layer_assigners() -> dict[str, LayerAssigner]:
    global _LAYER_ASSIGNERS
    if _LAYER_ASSIGNERS is None:
        from spot_check.analysis.assign import layer_auto as _layer_auto
        from spot_check.analysis.assign import layer_gate_counter as _layer_gate_counter
        from spot_check.analysis.assign import layer_plan_viterbi as _layer_plan_viterbi
        from spot_check.analysis.assign import layer_time_gap as _layer_time_gap

        _LAYER_ASSIGNERS = {
            _layer_time_gap.assigner.layer_mode: _layer_time_gap.assigner,
            _layer_gate_counter.assigner.layer_mode: _layer_gate_counter.assigner,
            _layer_plan_viterbi.assigner.layer_mode: _layer_plan_viterbi.assigner,
            _layer_auto.assigner.layer_mode: _layer_auto.assigner,
        }
    return _LAYER_ASSIGNERS


def get_layer_assigner(mode: str) -> LayerAssigner:
    """Return the layer assigner registered for ``mode``."""
    m = normalize_layer_mode(mode)
    return _layer_assigners()[m]


def run_layer_assignment(
    layer_mode: str,
    params: AssignCsvParams,
) -> MeasuredAssignResult:
    """Validate, probe CSV columns if needed, and run one layer-mode assigner."""
    from spot_check.analysis.measured import (
        _probe_csv_columns_for_measured_weights,
        normalize_measured_spot_weight_mode,
    )

    impl = get_layer_assigner(layer_mode)
    impl.validate(params)
    swm = normalize_measured_spot_weight_mode(params.spot_weight_mode)
    if not params.skip_column_probe:
        _probe_csv_columns_for_measured_weights(params.csv_path, spot_weight_mode=swm)
    return impl.assign(params)


def run_auto_assignment(
    method: str,
    cols: AutoFitColumns,
    *,
    n_plan_spots: int,
    plan_xy: np.ndarray,
    spots_per_layer: Sequence[int],
    episode_params: EpisodeAssignParams | None = None,
    plan_sequential_params: PlanSequentialAssignParams | None = None,
) -> AutoAssignResult:
    """Dispatch to one auto assigner; algorithms do not call each other."""
    m = str(method).strip().lower().replace("-", "_")
    if m == "sequential":
        m = "plan_sequential"
    if m not in AUTO_ASSIGN_METHODS:
        raise ValueError(f"unknown auto assign method {method!r}")
    impl = _ASSIGNERS[m]
    if m == "episodes":
        if episode_params is None:
            raise ValueError("episode_params required for episodes assignment")
        return impl.assign(
            cols,
            n_plan_spots=n_plan_spots,
            plan_xy=plan_xy,
            spots_per_layer=spots_per_layer,
            params=episode_params,
        )
    return impl.assign(
        cols,
        n_plan_spots=n_plan_spots,
        plan_xy=plan_xy,
        spots_per_layer=spots_per_layer,
        params=plan_sequential_params,
    )
