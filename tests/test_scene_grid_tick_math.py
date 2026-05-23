"""Tests for tick value planning."""

from __future__ import annotations

from spot_check.analysis.viz.scene_grid.tick_math import (
    bounds_expanded_to_tick_step,
    tick_values_centered_on_zero,
)


def test_bounds_expanded_to_tick_step() -> None:
    assert bounds_expanded_to_tick_step(-10.0, 20.0, 10.0) == (-10.0, 20.0)
    assert bounds_expanded_to_tick_step(-7.0, 23.0, 10.0) == (-10.0, 30.0)
    assert bounds_expanded_to_tick_step(-5.0, 15.0, 10.0) == (-10.0, 20.0)
    assert bounds_expanded_to_tick_step(-25.0, 25.0, 10.0) == (-30.0, 30.0)


def test_tick_values_centered_on_zero_symmetric() -> None:
    assert tick_values_centered_on_zero(-25.0, 25.0, 10.0) == (-20.0, -10.0, 10.0, 20.0)


def test_tick_values_centered_on_zero_asymmetric() -> None:
    assert tick_values_centered_on_zero(-10.0, 20.0, 10.0) == (-10.0, 10.0, 20.0)


def test_tick_values_centered_on_zero_none_when_narrow() -> None:
    assert tick_values_centered_on_zero(-1.0, 1.0, 10.0) == ()


def test_tick_values_centered_on_zero_excludes_zero() -> None:
    assert 0.0 not in tick_values_centered_on_zero(-15.0, 15.0, 10.0)


def test_tick_values_invalid_step() -> None:
    assert tick_values_centered_on_zero(-10.0, 10.0, 0.0) == ()
    assert tick_values_centered_on_zero(-10.0, 10.0, float("nan")) == ()
