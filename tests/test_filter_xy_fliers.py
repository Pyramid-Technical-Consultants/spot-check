"""Tests for σ-normalized XY flier filtering vs layer-NN plan."""

from __future__ import annotations

from spot_check.analysis.measured import MeasuredAssignResult, filter_assigned_xy_fliers
from spot_check.analysis.spatial import xy_sigma_flier_keep_mask
from spot_check.gui.parsers import (
    filter_xy_flier_sigma_input_in_progress,
    parse_filter_xy_flier_sigma,
)


def _row(
    a_mm: float,
    b_mm: float,
    *,
    layer: float = 0.0,
    sigma_a: float = 1.0,
    sigma_b: float = 1.0,
) -> tuple[float, ...]:
    return (a_mm, b_mm, layer, 1.0, 0, sigma_a, sigma_b, 1.0)


def test_xy_sigma_flier_keep_mask_at_limit() -> None:
    plan = [(0.0, 0.0, 100.0)]
    rows = [_row(0.0, 3.0, sigma_a=1.0, sigma_b=1.0)]
    keep = xy_sigma_flier_keep_mask(rows, plan, n_sigma=3.0, a_is_x=False)
    assert keep.tolist() == [True]
    keep_tight = xy_sigma_flier_keep_mask(rows, plan, n_sigma=2.9, a_is_x=False)
    assert keep_tight.tolist() == [False]


def test_xy_sigma_flier_keep_mask_on_target() -> None:
    plan = [(0.0, 0.0, 100.0)]
    rows = [_row(0.0, 0.0)]
    keep = xy_sigma_flier_keep_mask(rows, plan, n_sigma=3.0, a_is_x=False)
    assert keep.tolist() == [True]


def test_filter_assigned_xy_fliers_preserves_metadata() -> None:
    plan = [(0.0, 0.0, 100.0), (10.0, 0.0, 100.0)]
    result = MeasuredAssignResult(
        rows=[
            _row(0.0, 0.0),
            _row(0.0, 3.0, sigma_a=0.5, sigma_b=0.5),
        ],
        spot_ids=[0, 1],
        layer_mode="time_gap",
        assign_method="",
        n_plan_spots=2,
        planned_xyz=plan,
        a_is_x=False,
        gates=[1, 3],
    )
    filtered = filter_assigned_xy_fliers(result, plan, n_sigma=3.0)
    assert len(filtered.rows) == 1
    assert filtered.spot_ids == [0]
    assert filtered.gates == [1]
    assert filtered.layer_mode == "time_gap"


def test_filter_assigned_xy_fliers_noop_without_plan() -> None:
    result = MeasuredAssignResult(rows=[_row(0.0, 0.0)], spot_ids=[0])
    out = filter_assigned_xy_fliers(result, [], n_sigma=3.0)
    assert out is result


def test_parse_filter_xy_flier_sigma_helpers() -> None:
    assert parse_filter_xy_flier_sigma("3") == 3.0
    assert parse_filter_xy_flier_sigma("0.4") is None
    assert parse_filter_xy_flier_sigma("21") is None
    assert filter_xy_flier_sigma_input_in_progress("3.") is True
