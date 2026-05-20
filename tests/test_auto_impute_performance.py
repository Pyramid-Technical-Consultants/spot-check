"""Auto-mode performance regressions (large plan imputation, calibration)."""

from __future__ import annotations

import time

import numpy as np
import pytest

from spot_check.analysis.auto_columns import load_auto_fit_columns_from_csv
from spot_check.analysis.auto_params import infer_auto_layer_params
from spot_check.analysis.layers import _PlanImputeLookup
from spot_check.analysis.measured import measured_spot_abc_from_csv

_RUN12 = (
    __import__("pathlib").Path(__file__).resolve().parents[1]
    / "test_data"
    / "run12-cube-ic256-42-11377-data acquisition-2026-05-19-23-07-25.csv"
)

pytestmark = pytest.mark.local_data


@pytest.mark.skipif(not _RUN12.is_file(), reason="run12 cube CSV not under test_data/")
def test_large_plan_impute_finishes_quickly() -> None:
    n_plan = 20_000
    plan_xy = np.column_stack(
        (
            np.linspace(0.0, 200.0, n_plan),
            np.linspace(0.0, 150.0, n_plan),
        )
    )
    lk = _PlanImputeLookup.from_xy(plan_xy)
    assert lk is not None
    t0 = time.perf_counter()
    cols = load_auto_fit_columns_from_csv(
        _RUN12,
        global_lk=lk,
        a_is_x=False,
        spot_weight_mode="channel_sum",
    )
    assert time.perf_counter() - t0 < 4.0
    assert len(cols) > 0


@pytest.mark.skipif(not _RUN12.is_file(), reason="run12 cube CSV not under test_data/")
def test_auto_infer_and_measured_large_plan_under_budget() -> None:
    n_plan = 20_000
    planned = [
        (float(i % 200) * 0.1, float(i % 151) * 0.1, 100.0 + (i % 50))
        for i in range(n_plan)
    ]
    plan_xy = np.asarray([(p[0], p[1]) for p in planned], dtype=np.float64)
    lk = _PlanImputeLookup.from_xy(plan_xy)
    cols = load_auto_fit_columns_from_csv(
        _RUN12, global_lk=lk, a_is_x=False, spot_weight_mode="channel_sum"
    )
    t0 = time.perf_counter()
    infer_auto_layer_params(cols, planned)
    assert time.perf_counter() - t0 < 2.0

    t1 = time.perf_counter()
    rows = measured_spot_abc_from_csv(
        _RUN12,
        planned_xyz=planned,
        layer_mode="auto",
        a_is_x=False,
    )
    assert rows
    assert time.perf_counter() - t1 < 12.0
