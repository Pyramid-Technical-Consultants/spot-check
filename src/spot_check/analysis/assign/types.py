"""Shared types for spot/layer assignment algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from spot_check.analysis.auto_columns import AutoFitColumns
from spot_check.constants import (
    AUTO_MIN_EPISODE_ROWS_DEFAULT,
    AUTO_MIN_ON_SPOT_WEIGHT_NA_DEFAULT,
    AUTO_SPOT_XY_JUMP_MM_DEFAULT,
    REFILL_SAME_SPOT_XY_TOLERANCE_MM,
    REFILL_TRUST_TIME_GAP_STAY_DIST_MM,
    SPOT_WEIGHT_MODE_DEFAULT,
    TIME_LAYER_GAP_S_DEFAULT,
    VITERBI_LAYER_ADVANCE_PENALTY_MM2_DEFAULT,
)
from spot_check.models import DetectorRigidAlign2D

EpisodeSpan = tuple[int, int]
PlanIndexArray = np.ndarray

LAYER_ASSIGN_MODES: frozenset[str] = frozenset(
    {"time_gap", "plan_viterbi", "auto", "gate_counter"}
)


@dataclass
class MeasuredAssignResult:
    """Assigned measured rows before optional spot aggregation."""

    rows: list[tuple[float, ...]]
    spot_ids: list[int]
    layer_mode: str = "time_gap"
    assign_method: str = ""
    n_plan_spots: int = 0
    planned_xyz: list[tuple[float, float, float]] | None = None
    spots_per_layer: list[int] | None = None
    a_is_x: bool = False
    gates: list[int] = field(default_factory=list)
    plan_index_per_row: list[int] | None = None
    plan_spots_no_data: np.ndarray | None = None


@dataclass
class AssignCsvParams:
    """Inputs shared by all layer-mode CSV assigners."""

    csv_path: Path
    max_points: int | None = None
    planned_xyz: list[tuple[float, float, float]] | None = None
    a_is_x: bool = False
    spot_weight_mode: str = SPOT_WEIGHT_MODE_DEFAULT
    skip_column_probe: bool = False
    heal_partial_fit_axes: bool = False
    coarse_flat_transform: DetectorRigidAlign2D | None = None
    preloaded_auto_columns: Any = None
    layer_gap_s: float = TIME_LAYER_GAP_S_DEFAULT
    refill_same_spot_xy_tol_mm: float = REFILL_SAME_SPOT_XY_TOLERANCE_MM
    refill_trust_time_gap_stay_dist_mm: float = REFILL_TRUST_TIME_GAP_STAY_DIST_MM
    viterbi_advance_penalty_mm2: float = VITERBI_LAYER_ADVANCE_PENALTY_MM2_DEFAULT
    auto_episode_gap_s: float = TIME_LAYER_GAP_S_DEFAULT
    auto_min_on_spot_weight_na: float = AUTO_MIN_ON_SPOT_WEIGHT_NA_DEFAULT
    auto_spot_xy_jump_mm: float = AUTO_SPOT_XY_JUMP_MM_DEFAULT
    auto_min_episode_rows: int = AUTO_MIN_EPISODE_ROWS_DEFAULT
    auto_infer_params: bool = True
    auto_assign_method: str = "episodes"


@dataclass(frozen=True)
class AutoAssignResult:
    """Output of an auto-mode spot assignment (``layer_mode='auto'`` only)."""

    spans: list[EpisodeSpan]
    layer_index_per_span: np.ndarray
    plan_index_per_row: PlanIndexArray | None = None


@dataclass(frozen=True)
class EpisodeAssignParams:
    episode_gap_s: float
    min_on_spot_weight_na: float
    spot_xy_jump_mm: float
    min_episode_rows: int
    dead_ratio: float
    tiny_merge_rows: int


@dataclass(frozen=True)
class PlanSequentialAssignParams:
    min_rows_on_spot: int = 1
    cluster_window: int = 5
    advance_margin_mm2: float = 4.0


class SpotAssigner(Protocol):
    """Contract for auto sub-assigners (episodes, plan_sequential)."""

    method: str

    def assign(
        self,
        cols: AutoFitColumns,
        *,
        n_plan_spots: int,
        plan_xy: np.ndarray,
        spots_per_layer: Sequence[int],
        params: EpisodeAssignParams | PlanSequentialAssignParams | None = None,
    ) -> AutoAssignResult: ...


class LayerAssigner(Protocol):
    """Contract for CSV layer assignment modes (time_gap, gate_counter, auto, plan_viterbi)."""

    layer_mode: str

    def validate(self, params: AssignCsvParams) -> None: ...

    def assign(self, params: AssignCsvParams) -> MeasuredAssignResult: ...

    def spot_ids_are_plan_slots(self, result: MeasuredAssignResult) -> bool: ...

    def finalize(
        self,
        result: MeasuredAssignResult,
        *,
        planned_xyz: list[tuple[float, float, float]] | None,
    ) -> MeasuredAssignResult: ...
