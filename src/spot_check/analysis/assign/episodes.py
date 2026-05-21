"""Signal-episode assignment (deadtime segmentation + plan spot-count alignment).

Does not import any other assignment implementation.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from spot_check.analysis.assign.types import (
    AutoAssignResult,
    EpisodeAssignParams,
)
from spot_check.analysis.auto_columns import AutoFitColumns

method = "episodes"


def assign(
    cols: AutoFitColumns,
    *,
    n_plan_spots: int,
    plan_xy: np.ndarray,
    spots_per_layer: Sequence[int],
    params: EpisodeAssignParams,
) -> AutoAssignResult:
    from spot_check.analysis.episodes import segment_align_auto_columns
    from spot_check.analysis.layers import layer_indices_by_acquisition_time

    aligned_groups, _diag = segment_align_auto_columns(
        cols,
        n_plan_spots=n_plan_spots,
        episode_gap_s=params.episode_gap_s,
        min_on_spot_weight_na=params.min_on_spot_weight_na,
        spot_xy_jump_mm=params.spot_xy_jump_mm,
        min_episode_rows=params.min_episode_rows,
        dead_ratio=params.dead_ratio,
        tiny_merge_rows=params.tiny_merge_rows,
        plan_xy=plan_xy,
    )
    layers = layer_indices_by_acquisition_time(aligned_groups, spots_per_layer)
    return AutoAssignResult(
        spans=aligned_groups,
        layer_index_per_span=np.asarray(layers, dtype=np.int64),
        plan_index_per_row=None,
    )
