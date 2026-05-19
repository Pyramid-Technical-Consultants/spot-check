"""Tests for plan/CSV-adaptive auto layer parameters."""

from __future__ import annotations

import numpy as np

from spot_check.analysis.auto_columns import AutoFitColumns
from spot_check.analysis.auto_params import infer_auto_layer_params
from tests.conftest import MINIMAL_PLANNED_XYZ


def _cols_from_dt(
    dt: float,
    n: int,
    *,
    weight: float = 1.0,
    step_xy: float = 1.0,
) -> AutoFitColumns:
    t = np.arange(n, dtype=np.float64) * dt
    a = np.arange(n, dtype=np.float64) * step_xy
    z = np.zeros(n, dtype=np.float64)
    w = np.full(n, weight, dtype=np.float64)
    nan = np.full(n, np.nan, dtype=np.float64)
    pcd = np.zeros(n, dtype=np.int64)
    return AutoFitColumns(
        t=t,
        mx=a.copy(),
        my=z.copy(),
        a=a,
        b=z.copy(),
        mx_p=a.copy(),
        my_p=z.copy(),
        weight=w,
        ch_n=w,
        fit_a=w,
        pcd=pcd,
        sa=nan,
        sb=nan,
    )


def _cols_with_spots(n_spots: int, *, spot_len: int = 12, dead_len: int = 1) -> AutoFitColumns:
    high, low = 100.0, 5.0
    chunks: list[tuple[float, float]] = []
    for _ in range(n_spots):
        chunks.extend([(high, high)] * spot_len + [(low, low)] * dead_len)
    ch = np.array([c[0] for c in chunks], dtype=np.float64)
    fa = np.array([c[1] for c in chunks], dtype=np.float64)
    n = ch.size
    t = np.arange(n, dtype=np.float64) * 0.01
    z = np.zeros(n, dtype=np.float64)
    nan = np.full(n, np.nan, dtype=np.float64)
    return AutoFitColumns(
        t=t,
        mx=t.copy(),
        my=z,
        a=t,
        b=z,
        mx_p=t,
        my_p=z,
        weight=ch,
        ch_n=ch,
        fit_a=fa,
        pcd=np.zeros(n, dtype=np.int64),
        sa=nan,
        sb=nan,
    )


def test_infer_episode_gap_from_timing() -> None:
    cols = _cols_from_dt(2.0, 8)
    p = infer_auto_layer_params(cols, MINIMAL_PLANNED_XYZ)
    assert 0.005 <= p.episode_gap_s <= 5.0


def test_infer_dead_ratio_calibrated() -> None:
    cols = _cols_with_spots(40)
    p = infer_auto_layer_params(cols, MINIMAL_PLANNED_XYZ)
    assert 0.52 <= p.dead_ratio <= 0.64
    assert p.tiny_merge_rows >= 1


def test_infer_min_weight_from_distribution() -> None:
    w = np.array([0.01, 0.02, 1.0, 2.0, 3.0], dtype=np.float64)
    cols = AutoFitColumns(
        t=np.arange(5, dtype=np.float64),
        mx=w,
        my=np.zeros(5),
        a=w,
        b=np.zeros(5),
        mx_p=w,
        my_p=np.zeros(5),
        weight=w,
        ch_n=w,
        fit_a=w,
        pcd=np.zeros(5, dtype=np.int64),
        sa=np.full(5, np.nan),
        sb=np.full(5, np.nan),
    )
    p = infer_auto_layer_params(cols, MINIMAL_PLANNED_XYZ)
    assert p.min_on_spot_weight_na > 0
    assert p.min_on_spot_weight_na < 0.5
