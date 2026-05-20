"""Run12 cube: earliest measured spots must be low layer index (high energy)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spot_check import analysis
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
def test_run12_auto_earliest_spots_are_high_energy_layer() -> None:
    planned_xyz, _, _, _, _ = planned_spot_xyz_and_counts_from_pyramid_csv(_CUBE_PLAN)
    rows = analysis.measured_spot_abc_from_csv(
        _RUN12,
        planned_xyz=planned_xyz,
        layer_mode="auto",
        auto_assign_method="episodes",
        a_is_x=False,
        aggregate_spots=True,
    )
    assert len(rows) == len(planned_xyz)
    layers = np.asarray([int(r[2]) for r in rows], dtype=np.int64)
    n = layers.size
    head = layers[: max(50, n // 20)]
    tail = layers[-max(50, n // 20) :]
    assert float(np.median(head)) < float(np.median(tail))
    assert int(np.min(head)) == 0
