"""Run12 cube: plan-sequential auto assign must cover all nominal energy layers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spot_check import analysis
from spot_check.analysis.auto_columns import load_auto_fit_columns_from_csv
from spot_check.analysis.auto_params import last_auto_layer_params
from spot_check.analysis.layers import _PlanImputeLookup
from spot_check.analysis.plan_sequential import assign_plan_indices_sequential
from spot_check.analysis.spatial import nominal_layer_energies_mev
from spot_check.plan import planned_spot_xyz_and_counts_from_pyramid_csv

_RUN12 = (
    Path(__file__).resolve().parents[1]
    / "test_data"
    / "run12-cube-ic256-42-11377-data acquisition-2026-05-19-23-07-25.csv"
)
_CUBE_PLAN = Path(__file__).resolve().parents[1] / "test_data" / "R20M10_cube_original.csv"

pytestmark = pytest.mark.local_data


@pytest.mark.skipif(
    not _RUN12.is_file() or not _CUBE_PLAN.is_file(),
    reason="run12 cube fixture or R20M10 plan not present",
)
def test_run12_plan_sequential_spans_all_nominal_layers() -> None:
    planned_xyz, _, _, _, _ = planned_spot_xyz_and_counts_from_pyramid_csv(_CUBE_PLAN)
    n_layers = len(nominal_layer_energies_mev(planned_xyz))
    assert n_layers == 25

    rows = analysis.measured_spot_abc_from_csv(
        _RUN12,
        planned_xyz=planned_xyz,
        layer_mode="auto",
        auto_assign_method="plan_sequential",
        a_is_x=False,
        aggregate_spots=True,
    )
    n_plan = len(planned_xyz)
    assert len(rows) > 1000
    layers = np.asarray([int(r[2]) for r in rows], dtype=np.int64)
    assert int(layers.min()) == 0
    assert int(layers.max()) == n_layers - 1
    assert len(np.unique(layers)) == n_layers
    assert len(rows) >= n_plan - 1

    plan_xy = np.asarray([(p[0], p[1]) for p in planned_xyz], dtype=np.float64)
    lk = _PlanImputeLookup.from_xy(plan_xy)
    cols = load_auto_fit_columns_from_csv(
        _RUN12,
        global_lk=lk,
        a_is_x=False,
        spot_weight_mode="channel_sum",
        include_deadtime_rows=True,
    )
    plan_idx = assign_plan_indices_sequential(cols, plan_xy, min_rows_on_spot=1)
    seen = np.zeros(n_plan, dtype=bool)
    on = plan_idx[plan_idx >= 0]
    seen[on] = True
    assert int(on.max()) == n_plan - 1
    assert bool(seen.all())

    assert last_auto_layer_params() is not None
