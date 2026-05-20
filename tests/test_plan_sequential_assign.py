"""Sequential plan-order auto assignment (break + XY cluster on next spot)."""

from __future__ import annotations

import numpy as np

from spot_check.analysis.auto_columns import AutoFitColumns, position_fit_deadtime_mask
from spot_check.analysis.episodes import AutoFitRow, _rows_to_columns
from spot_check.analysis.plan_sequential import (
    assign_plan_indices_sequential,
    plan_spot_index_per_span,
    sequential_spans_from_plan_indices,
)


def _cols_along_x(n_spots: int, *, spot_len: int = 8, dead_len: int = 2) -> AutoFitColumns:
    """Plan spots on x = 0, 10, 20, …; measured mx follows plan index × 10."""
    rows: list[AutoFitRow] = []
    idx = 0
    for si in range(n_spots):
        x = float(si * 10)
        for _ in range(spot_len):
            rows.append(
                AutoFitRow(
                    t=float(idx),
                    mx=x,
                    my=0.0,
                    a=x,
                    b=0.0,
                    mx_p=x,
                    my_p=0.0,
                    weight=100.0,
                    ch_n=100.0,
                    pcd=0,
                    sa=None,
                    sb=None,
                )
            )
            idx += 1
        for _ in range(dead_len):
            rows.append(
                AutoFitRow(
                    t=float(idx),
                    mx=float("nan"),
                    my=float("nan"),
                    a=float("nan"),
                    b=float("nan"),
                    mx_p=float("nan"),
                    my_p=float("nan"),
                    weight=5.0,
                    ch_n=5.0,
                    pcd=-1,
                    sa=None,
                    sb=None,
                )
            )
            idx += 1
    return _rows_to_columns(rows)


def test_plan_indices_never_skip_a_slot() -> None:
    cols = _cols_along_x(4)
    plan_xy = np.column_stack([np.arange(4, dtype=float) * 10.0, np.zeros(4)])
    plan_idx = assign_plan_indices_sequential(cols, plan_xy, min_rows_on_spot=1)
    on = plan_idx[plan_idx >= 0]
    assert on.size > 0
    assert int(on.max()) <= 3
    steps = np.diff(on.astype(np.int64))
    assert np.all(steps >= 0)
    assert np.all(steps <= 1)


def test_every_plan_slot_gets_at_least_one_row() -> None:
    n_plan = 4
    cols = _cols_along_x(n_plan)
    plan_xy = np.column_stack([np.arange(n_plan, dtype=float) * 10.0, np.zeros(n_plan)])
    plan_idx = assign_plan_indices_sequential(cols, plan_xy, min_rows_on_spot=1)
    seen = np.zeros(n_plan, dtype=bool)
    on = plan_idx[plan_idx >= 0]
    seen[on] = True
    assert seen.all()


def test_assign_advances_through_plan_order() -> None:
    n_plan = 4
    cols = _cols_along_x(n_plan)
    plan_xy = np.column_stack([np.arange(n_plan, dtype=float) * 10.0, np.zeros(n_plan)])
    plan_idx = assign_plan_indices_sequential(cols, plan_xy, min_rows_on_spot=1)
    spans = sequential_spans_from_plan_indices(plan_idx)
    spot_pi = plan_spot_index_per_span(plan_idx, spans)
    assert len(spans) == n_plan
    assert np.array_equal(spot_pi, np.arange(n_plan, dtype=np.int64))


def test_deadtime_rows_unassigned() -> None:
    cols = _cols_along_x(2)
    plan_xy = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float64)
    plan_idx = assign_plan_indices_sequential(cols, plan_xy)
    assert int(np.sum(plan_idx < 0)) > 0


def test_position_fit_deadtime_mask() -> None:
    rows: list[AutoFitRow] = [
        AutoFitRow(
            t=0.0,
            mx=0.0,
            my=0.0,
            a=0.0,
            b=0.0,
            mx_p=float("nan"),
            my_p=float("nan"),
            weight=1.0,
            ch_n=1.0,
            pcd=-1,
            sa=None,
            sb=None,
        ),
        AutoFitRow(
            t=1.0,
            mx=0.0,
            my=0.0,
            a=0.0,
            b=0.0,
            mx_p=float("nan"),
            my_p=1.0,
            weight=1.0,
            ch_n=1.0,
            pcd=1,
            sa=None,
            sb=None,
        ),
    ]
    cols = _rows_to_columns(rows)
    dead = position_fit_deadtime_mask(cols)
    assert bool(dead[0])
    assert not bool(dead[1])


def test_plan_sequential_measured_csv(tmp_path) -> None:
    """Four spaced spots in CSV align to four plan spots via plan_sequential."""
    from spot_check import analysis
    from tests.conftest import write_measured_csv

    rows_csv = []
    idx = 0
    for spot in range(4):
        for _ in range(8):
            rows_csv.append(
                {
                    "time (s)": str(idx * 0.01),
                    "IX512 Channel Sum (nA)": "100.0",
                    "Fit Amplitude A (nA)": "10.0",
                    "Fit Mean Position A (mm)": str(float(spot * 10)),
                    "Fit Mean Position B (mm)": "0.0",
                }
            )
            idx += 1
        rows_csv.append(
            {
                "time (s)": str(idx * 0.01),
                "IX512 Channel Sum (nA)": "",
                "Fit Amplitude A (nA)": "",
                "Fit Mean Position A (mm)": "",
                "Fit Mean Position B (mm)": "",
            }
        )
        idx += 1
    plan = [(float(i * 10), 0.0, 100.0) for i in range(4)]
    csv_path = write_measured_csv(tmp_path / "seq.csv", rows_csv)
    out = analysis.measured_spot_abc_from_csv(
        csv_path,
        planned_xyz=plan,
        layer_mode="auto",
        auto_infer_params=False,
        auto_assign_method="plan_sequential",
        auto_min_episode_rows=1,
        aggregate_spots=True,
    )
    assert len(out) == 4
