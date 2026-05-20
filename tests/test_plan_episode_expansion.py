"""Plan-order expansion: raw episodes map to every plan spot (cube under-segmentation)."""

from __future__ import annotations

import numpy as np

from spot_check.analysis.auto_columns import AutoFitColumns
from spot_check.analysis.episodes import (
    _partition_plans_across_episodes,
    _spans_from_plan_episode_expansion,
    align_episode_spans_to_plan_count_cols,
    segment_into_episodes_cols,
)


def _flat_cols(n: int, *, xy_step: float = 0.0) -> AutoFitColumns:
    t = np.arange(n, dtype=np.float64) * 1e-3
    mx = np.arange(n, dtype=np.float64) * xy_step
    sig = np.full(n, 100.0, dtype=np.float64)
    return AutoFitColumns(
        t=t,
        mx=mx,
        my=np.zeros(n, dtype=np.float64),
        a=mx,
        b=np.zeros(n, dtype=np.float64),
        mx_p=mx,
        my_p=np.zeros(n, dtype=np.float64),
        weight=sig,
        ch_n=sig,
        fit_a=sig * 0.1,
        pcd=np.zeros(n, dtype=np.int32),
        sa=np.full(n, np.nan, dtype=np.float64),
        sb=np.full(n, np.nan, dtype=np.float64),
    )


def test_partition_covers_all_plan_spots() -> None:
    cents = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]], dtype=np.float64)
    plan = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [5.0, 0.0], [9.0, 0.0]], dtype=np.float64)
    emit = np.sum((cents[:, np.newaxis, :] - plan[np.newaxis, :, :]) ** 2, axis=2)
    groups = _partition_plans_across_episodes(emit, max_seg=3)
    assert len(groups) == 3
    assert groups[0][0] == 0 and groups[-1][1] == 5
    assert sum(b - a for a, b in groups) == 5


def test_align_expansion_reaches_plan_count() -> None:
    n_rows = 4000
    cols = _flat_cols(n_rows, xy_step=0.01)
    spans = segment_into_episodes_cols(
        cols,
        episode_gap_s=0.05,
        min_on_spot_weight_na=1e-12,
        spot_xy_jump_mm=500.0,
        min_episode_rows=1,
        dead_ratio=0.55,
    )
    n_plan = 3000
    plan_xy = np.column_stack(
        (
            np.linspace(0.0, 30.0, n_plan),
            np.zeros(n_plan, dtype=np.float64),
        )
    )
    work, diag = align_episode_spans_to_plan_count_cols(
        cols, spans, n_plan, plan_xy=plan_xy
    )
    assert diag.count_align_ok
    assert len(work) == n_plan
    expanded = _spans_from_plan_episode_expansion(cols, spans, plan_xy)
    assert len(expanded) == n_plan
