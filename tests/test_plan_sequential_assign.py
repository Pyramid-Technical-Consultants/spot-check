"""Sequential plan-order auto assignment (position deadtime + plan XY spans)."""

from __future__ import annotations

import math

import numpy as np

from spot_check.analysis.auto_columns import AutoFitColumns, position_fit_deadtime_mask
from spot_check.analysis.episodes import AutoFitRow, _rows_to_columns
from spot_check.analysis.plan_sequential import (
    assign_plan_indices_sequential,
    plan_spot_index_per_span,
    sequential_spans_from_plan_indices,
)


def _cols_from_on_off_pattern(
    pattern: list[tuple[str, int, float]],
) -> AutoFitColumns:
    """Build timeline from ``("on"|"off", row_count, plan_x_mm)`` segments in order."""
    rows: list[AutoFitRow] = []
    idx = 0
    for kind, count, x in pattern:
        for _ in range(count):
            if kind == "on":
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
            else:
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


def _assert_on_bursts_share_one_plan_index(
    plan_idx: np.ndarray, dead: np.ndarray
) -> None:
    """Each contiguous position-fit on-burst has a single plan index."""
    n = int(plan_idx.size)
    i = 0
    while i < n:
        while i < n and (bool(dead[i]) or int(plan_idx[i]) < 0):
            i += 1
        if i >= n:
            break
        pi = int(plan_idx[i])
        s = i
        i += 1
        while i < n and not dead[i] and int(plan_idx[i]) == pi:
            i += 1
        burst = plan_idx[s:i]
        assert burst.size >= 1
        assert np.all(burst == pi)


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


def test_single_row_on_burst_is_assigned() -> None:
    """One on-row between deadtime gaps still counts as a plan spot dwell."""
    cols = _cols_from_on_off_pattern(
        [("on", 1, 0.0), ("off", 1, 0.0), ("on", 1, 10.0), ("off", 1, 0.0), ("on", 1, 20.0)]
    )
    plan_xy = np.column_stack([np.arange(3, dtype=float) * 10.0, np.zeros(3)])
    plan_idx = assign_plan_indices_sequential(cols, plan_xy, min_rows_on_spot=1)
    dead = position_fit_deadtime_mask(cols)
    assert int(plan_idx[0]) == 0
    assert bool(dead[1])
    assert int(plan_idx[1]) < 0
    assert int(plan_idx[2]) == 1
    assert int(plan_idx[4]) == 2
    _assert_on_bursts_share_one_plan_index(plan_idx, dead)


def test_single_row_off_is_deadtime() -> None:
    cols = _cols_from_on_off_pattern([("on", 2, 0.0), ("off", 1, 0.0), ("on", 2, 10.0)])
    dead = position_fit_deadtime_mask(cols)
    assert bool(dead[2])
    assert int(np.sum(dead)) == 1


def test_every_on_burst_has_uniform_plan_index() -> None:
    """All rows in each on-spot run map to the same delivery index."""
    pattern: list[tuple[str, int, float]] = []
    for si in range(5):
        pattern.append(("on", 1, float(si * 10)))
        pattern.append(("off", 1, float(si * 10)))
    cols = _cols_from_on_off_pattern(pattern)
    n_plan = 5
    plan_xy = np.column_stack([np.arange(n_plan, dtype=float) * 10.0, np.zeros(n_plan)])
    plan_idx = assign_plan_indices_sequential(cols, plan_xy, min_rows_on_spot=1)
    dead = position_fit_deadtime_mask(cols)
    _assert_on_bursts_share_one_plan_index(plan_idx, dead)
    on = plan_idx[plan_idx >= 0]
    assert np.array_equal(on, np.arange(n_plan, dtype=np.int64))


def test_high_min_rows_on_spot_does_not_fold_single_row_burst() -> None:
    """``min_rows_on_spot`` > 1 must not steal a one-row burst from its plan slot."""
    cols = _cols_from_on_off_pattern([("on", 1, 0.0), ("off", 1, 0.0), ("on", 1, 10.0)])
    plan_xy = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float64)
    plan_idx = assign_plan_indices_sequential(cols, plan_xy, min_rows_on_spot=5)
    assert int(plan_idx[0]) == 0
    assert int(plan_idx[2]) == 1


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


def test_plan_sequential_measured_csv_single_row_bursts(tmp_path) -> None:
    """End-to-end: one CSV row per on-burst still yields one aggregated spot per plan slot."""
    from spot_check import analysis
    from tests.conftest import write_measured_csv

    rows_csv = []
    idx = 0
    for spot in range(3):
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
    plan = [(float(i * 10), 0.0, 100.0) for i in range(3)]
    csv_path = write_measured_csv(tmp_path / "seq1.csv", rows_csv)
    out = analysis.measured_spot_abc_from_csv(
        csv_path,
        planned_xyz=plan,
        layer_mode="auto",
        auto_infer_params=False,
        auto_assign_method="plan_sequential",
        auto_min_episode_rows=8,
        aggregate_spots=True,
    )
    assert len(out) == 3


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
    for row in out:
        assert math.isfinite(float(row[2]))
