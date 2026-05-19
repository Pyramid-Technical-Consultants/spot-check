"""Unit tests for episode fin-buffer merge and column conversion."""

from __future__ import annotations

import math

import numpy as np

from spot_check.analysis.auto_columns import AutoFitColumns
from spot_check.analysis.episodes import (
    _merge_fin_bufs,
    fin_bufs_from_spans_batch,
    segment_align_auto_episodes,
)


def _fin(
    a: float,
    b: float,
    weight: float,
    *,
    sa: float = float("nan"),
    sb: float = float("nan"),
    ch_n: float | None = None,
) -> tuple[float, float, float, float, int, float, float, float]:
    ch = weight if ch_n is None else ch_n
    return (a, b, 0.0, weight, 0, sa, sb, ch)


def test_merge_fin_bufs_preserves_finite_sigma_when_other_nan() -> None:
    left = _fin(1.0, 2.0, 2.0, sa=1.0, sb=float("nan"))
    right = _fin(3.0, 4.0, 1.0, sa=float("nan"), sb=3.0)
    merged = _merge_fin_bufs(left, right)
    assert merged[5] == 1.0
    assert merged[6] == 3.0


def test_merge_fin_bufs_matches_batch_aggregate_for_spans() -> None:
    cols = AutoFitColumns(
        t=np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
        mx=np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
        my=np.zeros(4, dtype=np.float64),
        a=np.array([1.0, 2.0, 10.0, 20.0], dtype=np.float64),
        b=np.zeros(4, dtype=np.float64),
        mx_p=np.array([1.0, 2.0, 10.0, 20.0], dtype=np.float64),
        my_p=np.zeros(4, dtype=np.float64),
        weight=np.ones(4, dtype=np.float64),
        ch_n=np.ones(4, dtype=np.float64),
        fit_a=np.ones(4, dtype=np.float64),
        pcd=np.zeros(4, dtype=np.int32),
        sa=np.array([1.0, float("nan"), 3.0, float("nan")], dtype=np.float64),
        sb=np.array([float("nan"), 2.0, float("nan"), 4.0], dtype=np.float64),
    )
    spans = [(0, 2), (2, 4)]
    batch = fin_bufs_from_spans_batch(cols, spans)
    whole = fin_bufs_from_spans_batch(cols, [(0, 4)])[0]
    chained = _merge_fin_bufs(batch[0], batch[1])
    assert math.isclose(chained[0], whole[0], rel_tol=0, abs_tol=1e-9)
    assert math.isclose(chained[5], whole[5], rel_tol=0, abs_tol=1e-9)
    assert math.isclose(chained[6], whole[6], rel_tol=0, abs_tol=1e-9)


def test_rows_to_columns_empty() -> None:
    groups, diag = segment_align_auto_episodes(
        [],
        n_plan_spots=0,
        episode_gap_s=0.5,
        min_on_spot_weight_na=1e-12,
        spot_xy_jump_mm=50.0,
        min_episode_rows=1,
    )
    assert groups == []
    assert not diag.count_align_ok
