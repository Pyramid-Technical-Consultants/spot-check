"""Shared types for spot/layer assignment algorithms (no algorithm logic here)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from spot_check.analysis.auto_columns import AutoFitColumns

EpisodeSpan = tuple[int, int]
PlanIndexArray = np.ndarray


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


class AutoSpotAssigner(Protocol):
    """Contract for auto assign methods; implementations must not import each other."""

    method: str

    def assign(
        self,
        cols: AutoFitColumns,
        *,
        n_plan_spots: int,
        plan_xy: np.ndarray,
        spots_per_layer: Sequence[int],
    ) -> AutoAssignResult: ...
