"""Layer-EM auto assignment (time-monotone layers, highest energy first)."""

from __future__ import annotations

import numpy as np

from spot_check.analysis.auto_columns import AutoFitColumns
from spot_check.analysis.auto_layer_em import layer_em_plan_spans


def _synthetic_cols(n_rows: int, n_layers: int) -> AutoFitColumns:
    t = np.arange(n_rows, dtype=np.float64) * 0.01
    layer_row = np.minimum((t * 10).astype(np.int64), n_layers - 1)
    mx = layer_row.astype(np.float64) * 6.0
    my = np.zeros(n_rows, dtype=np.float64)
    w = np.full(n_rows, 50.0, dtype=np.float64)
    return AutoFitColumns(
        t=t,
        mx=mx,
        my=my,
        a=mx,
        b=my,
        mx_p=mx,
        my_p=my,
        weight=w,
        ch_n=w,
        fit_a=w * 0.1,
        pcd=np.zeros(n_rows, dtype=np.int32),
        sa=np.full(n_rows, np.nan, dtype=np.float64),
        sb=np.full(n_rows, np.nan, dtype=np.float64),
    )


def test_layer_em_one_span_per_plan_spot() -> None:
    n_layers = 3
    spots_per = [4, 4, 4]
    n_plan = sum(spots_per)
    plan_xy = []
    for li in range(n_layers):
        for si in range(spots_per[li]):
            plan_xy.append((float(li * 6 + si), 0.0))
    plan_xy_arr = np.asarray(plan_xy, dtype=np.float64)
    cols = _synthetic_cols(120, n_layers)
    spans, diag = layer_em_plan_spans(cols, plan_xy_arr, spots_per, refine_passes=0)
    assert len(spans) == n_plan
    assert diag.n_plan == n_plan
    assert diag.total_cost < float("inf")


def test_layer_em_spans_are_time_monotone_by_layer() -> None:
    """Layer 0 rows precede layer 1 on the acquisition timeline."""
    n_layers = 3
    spots_per = [3, 3, 3]
    plan_xy = np.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [10.0, 0.0], [11.0, 0.0], [12.0, 0.0],
         [20.0, 0.0], [21.0, 0.0], [22.0, 0.0]],
        dtype=np.float64,
    )
    cols = _synthetic_cols(90, n_layers)
    spans, _ = layer_em_plan_spans(cols, plan_xy, spots_per, refine_passes=1)
    assert len(spans) == 9
    layer_ends = [spans[2][1], spans[5][1], spans[8][1]]
    layer_starts = [spans[0][0], spans[3][0], spans[6][0]]
    assert layer_starts[0] < layer_starts[1] < layer_starts[2]
    assert layer_ends[0] <= layer_starts[1]
    assert layer_ends[1] <= layer_starts[2]
