"""Regression: zero-weight episode spans must not divide by zero."""

from __future__ import annotations

import warnings

import numpy as np

from spot_check.analysis.auto_columns import AutoFitColumns
from spot_check.analysis.episodes import aggregate_spans_batch


def test_aggregate_zero_weight_span_no_runtime_warning() -> None:
    cols = AutoFitColumns(
        t=np.array([0.0, 1.0], dtype=np.float64),
        mx=np.array([1.0, 2.0], dtype=np.float64),
        my=np.zeros(2, dtype=np.float64),
        a=np.array([1.0, 2.0], dtype=np.float64),
        b=np.zeros(2, dtype=np.float64),
        mx_p=np.array([1.0, 2.0], dtype=np.float64),
        my_p=np.zeros(2, dtype=np.float64),
        weight=np.zeros(2, dtype=np.float64),
        ch_n=np.ones(2, dtype=np.float64),
        fit_a=np.ones(2, dtype=np.float64),
        pcd=np.zeros(2, dtype=np.int32),
        sa=np.full(2, np.nan, dtype=np.float64),
        sb=np.full(2, np.nan, dtype=np.float64),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        aggs = aggregate_spans_batch(cols, [(0, 2)])
    assert len(aggs) == 1
    assert aggs[0].a == 1.5
    assert aggs[0].weight < 1e-12
