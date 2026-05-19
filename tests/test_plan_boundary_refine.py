"""Plan XY boundary refinement for auto episode spans."""

from __future__ import annotations

import numpy as np

from spot_check.analysis.auto_columns import AutoFitColumns
from spot_check.analysis.episodes import refine_spans_with_plan_xy


def test_refine_shifts_boundary_toward_plan() -> None:
    """Split point moves to the row window that best matches sequential plan XY."""
    n = 20
    t = np.arange(n, dtype=np.float64)
    mx = np.array([0.0] * 8 + [10.0] * 12, dtype=np.float64)
    my = np.zeros(n, dtype=np.float64)
    w = np.ones(n, dtype=np.float64)
    nan = np.full(n, np.nan, dtype=np.float64)
    cols = AutoFitColumns(
        t=t,
        mx=mx,
        my=my,
        a=mx,
        b=my,
        mx_p=mx,
        my_p=my,
        weight=w,
        ch_n=w,
        fit_a=w,
        pcd=np.zeros(n, dtype=np.int32),
        sa=nan,
        sb=nan,
    )
    spans = [(0, 10), (10, 20)]
    plan_xy = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float64)
    out = refine_spans_with_plan_xy(cols, spans, plan_xy, window_rows=6, passes=1)
    assert out[0][1] == 8
    assert out[1][0] == 8
