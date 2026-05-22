"""Coarse flat alignment orientation search (A/B swap and plan-axis mirrors)."""

from __future__ import annotations

import numpy as np

from spot_check.analysis.alignment import (
    apply_detector_rigid2d_xy_to_measured_rows,
    fit_coarse_flat_align_from_auto_columns,
)
from spot_check.analysis.auto_columns import AutoFitColumns
from spot_check.analysis.spatial import layer_nn_plan_xy_distances_and_expected_xyz


def _synthetic_auto_cols(
    planned: list[tuple[float, float, float]],
    *,
    mirror_x: bool = False,
    mirror_y: bool = False,
) -> AutoFitColumns:
    n = len(planned)
    mx = np.array(
        [(-float(px) if mirror_x else float(px)) for px, _py, _ in planned],
        dtype=np.float64,
    )
    my = np.array(
        [(-float(py) if mirror_y else float(py)) for _px, py, _ in planned],
        dtype=np.float64,
    )
    z = np.zeros(n, dtype=np.float64)
    pcd = np.zeros(n, dtype=np.int32)
    return AutoFitColumns(
        t=z,
        mx=mx,
        my=my,
        a=my.copy(),
        b=mx.copy(),
        mx_p=z.copy(),
        my_p=z.copy(),
        weight=np.ones(n, dtype=np.float64),
        ch_n=np.ones(n, dtype=np.float64),
        fit_a=np.ones(n, dtype=np.float64),
        pcd=pcd,
        sa=z.copy(),
        sb=z.copy(),
    )


def _row_from_plan_xy(mx: float, my: float) -> tuple[float, ...]:
    """a_is_x=False storage: Fit A=my, Fit B=mx."""
    return (float(my), float(mx), 0.0, 1.0, 0, float("nan"), float("nan"), 1.0)


def test_coarse_flat_recovers_plan_x_mirror() -> None:
    planned = [
        (10.0, 0.0, 140.0),
        (14.5, -2.25, 140.0),
        (11.0, 6.75, 140.0),
        (17.75, 4.125, 140.0),
    ]
    cols = _synthetic_auto_cols(planned, mirror_x=True)
    info = fit_coarse_flat_align_from_auto_columns(cols, planned)
    assert info.rms_residual_mm < 0.2
    assert info.flip_plan_x or (info.flip_plan_y and abs(abs(info.theta_deg) - 180.0) < 3.0)


def test_coarse_flat_recovers_plan_y_mirror() -> None:
    planned = [
        (10.0, 0.0, 140.0),
        (14.5, -2.25, 140.0),
        (11.0, 6.75, 140.0),
        (17.75, 4.125, 140.0),
    ]
    cols = _synthetic_auto_cols(planned, mirror_y=True)
    info = fit_coarse_flat_align_from_auto_columns(cols, planned)
    assert info.rms_residual_mm < 0.2
    assert info.flip_plan_y or (info.flip_plan_x and abs(abs(info.theta_deg) - 180.0) < 3.0)


def test_coarse_flat_apply_corrects_mirrored_rows() -> None:
    planned = [
        (10.0, 0.0, 140.0),
        (14.5, -2.25, 140.0),
        (11.0, 6.75, 140.0),
        (17.75, 4.125, 140.0),
    ]
    cols = _synthetic_auto_cols(planned, mirror_x=True)
    info = fit_coarse_flat_align_from_auto_columns(cols, planned)
    rows = [_row_from_plan_xy(float(cols.mx[i]), float(cols.my[i])) for i in range(len(planned))]
    aligned = apply_detector_rigid2d_xy_to_measured_rows(rows, info, a_is_x=False)
    dist, _ = layer_nn_plan_xy_distances_and_expected_xyz(planned, aligned, a_is_x=False)
    dist = dist[np.isfinite(dist)]
    assert dist.size == len(planned)
    assert float(np.max(dist)) < 0.2
