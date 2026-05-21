"""Pre-assignment flat-plan detector align (before plan-sequential / episodes)."""

from __future__ import annotations

import math

import numpy as np

from spot_check.analysis.alignment import (
    align_auto_fit_columns_to_plan_xy,
    last_detector_align_info,
)
from spot_check.analysis.assign import assign_plan_indices_sequential
from spot_check.analysis.auto_columns import (
    AutoFitColumns,
    position_fit_deadtime_mask,
)
from spot_check.analysis.episodes import AutoFitRow, _rows_to_columns


def _cols_along_x_rotated(
    n_spots: int,
    *,
    spot_len: int = 5,
    dead_len: int = 1,
    theta_deg: float,
    tx: float,
    ty: float,
) -> AutoFitColumns:
    """Delivery-order spots along x=0,10,… with rigid detector mis-map."""
    th = math.radians(theta_deg)
    c, s = math.cos(th), math.sin(th)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    rows: list[AutoFitRow] = []
    idx = 0
    for si in range(n_spots):
        plan_x, plan_y = float(si * 10), 0.0
        det = rot @ np.array([plan_x, plan_y]) + np.array([tx, ty])
        mx, my = float(det[0]), float(det[1])
        for _ in range(spot_len):
            rows.append(
                AutoFitRow(
                    t=float(idx),
                    mx=mx,
                    my=my,
                    a=mx,
                    b=my,
                    mx_p=mx,
                    my_p=my,
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


def _mean_assign_xy_error(
    cols: AutoFitColumns,
    plan_idx: np.ndarray,
    plan_xy: np.ndarray,
) -> float:
    dead = position_fit_deadtime_mask(cols)
    errs: list[float] = []
    for i in range(len(cols)):
        if dead[i] or int(plan_idx[i]) < 0:
            continue
        pi = int(plan_idx[i])
        errs.append(
            float(
                np.hypot(
                    float(cols.mx[i]) - float(plan_xy[pi, 0]),
                    float(cols.my[i]) - float(plan_xy[pi, 1]),
                )
            )
        )
    return float(np.mean(errs)) if errs else float("inf")


def test_pre_align_reduces_plan_sequential_xy_error() -> None:
    n = 12
    plan_xy = np.asarray([(float(i * 10), 0.0) for i in range(n)], dtype=np.float64)
    planned = [(float(x), float(y), 177.5) for x, y in plan_xy]
    cols = _cols_along_x_rotated(n, theta_deg=22.0, tx=4.0, ty=-3.0)

    raw_idx = assign_plan_indices_sequential(cols, plan_xy, min_rows_on_spot=1)
    err_before = _mean_assign_xy_error(cols, raw_idx, plan_xy)
    assert err_before > 5.0

    aligned, info = align_auto_fit_columns_to_plan_xy(cols, planned, a_is_x=False)
    assert info.pre_assignment
    assert last_detector_align_info() is info

    fixed_idx = assign_plan_indices_sequential(aligned, plan_xy, min_rows_on_spot=1)
    err_after = _mean_assign_xy_error(aligned, fixed_idx, plan_xy)
    assert err_after < 1.0
    assert err_after < err_before * 0.2
