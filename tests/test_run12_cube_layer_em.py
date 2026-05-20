"""IC256 cube: layer EM must emit one measured spot per plan spot."""

from __future__ import annotations

from pathlib import Path

import pytest

from spot_check import analysis
from spot_check.analysis.auto_layer_em import last_layer_em_diagnostics
from spot_check.plan import planned_spot_xyz_and_counts_from_pyramid_csv

_RUN12 = (
    Path(__file__).resolve().parents[1]
    / "test_data"
    / "run12-cube-ic256-42-11377-data acquisition-2026-05-19-23-07-25.csv"
)
_CUBE_PLAN = Path(__file__).resolve().parents[1] / "test_data" / "R20M10_cube_original.csv"


@pytest.mark.skipif(
    not _RUN12.is_file() or not _CUBE_PLAN.is_file(),
    reason="run12 cube fixture or R20M10 plan not present",
)
def test_run12_layer_em_aligns_spot_count_to_cube_plan() -> None:
    planned_xyz, _, plan_mu, _, _ = planned_spot_xyz_and_counts_from_pyramid_csv(_CUBE_PLAN)
    n_plan = len(planned_xyz)
    assert n_plan > 1000

    rows = analysis.measured_spot_abc_from_csv(
        _RUN12,
        planned_xyz=planned_xyz,
        planned_mu=plan_mu,
        layer_mode="auto",
        auto_assign_method="layer_em",
        a_is_x=False,
        aggregate_spots=True,
    )
    diag = last_layer_em_diagnostics()
    assert diag is not None
    assert len(rows) == n_plan
    assert diag.n_plan == n_plan
