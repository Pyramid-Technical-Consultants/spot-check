"""Regression: clinic IC256 cube CSV must load without hanging."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from spot_check import analysis
from spot_check.gui.pipeline import resolve_csv_load_layer_mode
from tests.conftest import MINIMAL_PLANNED_XYZ

_RUN12 = (
    Path(__file__).resolve().parents[1]
    / "test_data"
    / "run12-cube-ic256-42-11377-data acquisition-2026-05-19-23-07-25.csv"
)

pytestmark = pytest.mark.local_data


@pytest.mark.skipif(not _RUN12.is_file(), reason="run12 cube fixture not present")
def test_run12_gate_counter_fallback_time_gap_finishes_quickly() -> None:
    mode, agg = resolve_csv_load_layer_mode(
        layer_mode="gate_counter",
        plan_path=Path("plan.dcm"),
        csv_path=_RUN12,
        aggregate_spots=True,
    )
    assert mode == "time_gap"
    assert agg is False

    planned = list(MINIMAL_PLANNED_XYZ)
    t0 = time.perf_counter()
    rows = analysis.measured_spot_abc_from_csv(
        _RUN12,
        planned_xyz=planned,
        layer_mode=mode,
        aggregate_spots=agg,
        auto_infer_params=False,
    )
    elapsed = time.perf_counter() - t0
    assert rows
    assert elapsed < 15.0


@pytest.mark.skipif(not _RUN12.is_file(), reason="run12 cube fixture not present")
def test_run12_auto_mode_large_plan_finishes_quickly() -> None:
    planned = list(MINIMAL_PLANNED_XYZ) * 5000
    t0 = time.perf_counter()
    rows = analysis.measured_spot_abc_from_csv(
        _RUN12,
        planned_xyz=planned,
        layer_mode="auto",
        a_is_x=False,
    )
    elapsed = time.perf_counter() - t0
    assert rows
    assert elapsed < 12.0
