"""Spot/layer assignment algorithms (mutually isolated; use only this package entry point)."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from spot_check.analysis.assign import episodes as _episodes
from spot_check.analysis.assign import plan_sequential as _plan_sequential
from spot_check.analysis.assign.types import (
    AutoAssignResult,
    AutoSpotAssigner,
    EpisodeAssignParams,
    EpisodeSpan,
    PlanIndexArray,
    PlanSequentialAssignParams,
)
from spot_check.analysis.auto_columns import AutoFitColumns
from spot_check.constants import AUTO_ASSIGN_METHODS

__all__ = [
    "AUTO_ASSIGN_METHODS",
    "AutoAssignResult",
    "AutoSpotAssigner",
    "EpisodeAssignParams",
    "EpisodeSpan",
    "PlanIndexArray",
    "PlanSequentialAssignParams",
    "assign_plan_indices_sequential",
    "plan_spot_index_per_span",
    "run_auto_assignment",
    "sequential_spans_from_plan_indices",
]

# Re-export plan-sequential helpers for tests (implementation lives in assign.plan_sequential).
assign_plan_indices_sequential = _plan_sequential.assign_plan_indices_sequential
sequential_spans_from_plan_indices = _plan_sequential.sequential_spans_from_plan_indices
plan_spot_index_per_span = _plan_sequential.plan_spot_index_per_span

_ASSIGNERS: dict[str, AutoSpotAssigner] = {
    _episodes.method: _episodes,
    _plan_sequential.method: _plan_sequential,
}


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
