"""IC256 cube acquisition: auto mode must align episode count to the plan."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spot_check import analysis
from spot_check.analysis.episodes import last_auto_episode_diagnostics
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
def test_run12_auto_aligns_episode_count_to_cube_plan() -> None:
    planned_xyz, _, _, _, _ = planned_spot_xyz_and_counts_from_pyramid_csv(_CUBE_PLAN)
    n_plan = len(planned_xyz)
    assert n_plan > 1000

    rows = analysis.measured_spot_abc_from_csv(
        _RUN12,
        planned_xyz=planned_xyz,
        layer_mode="auto",
        a_is_x=False,
        aggregate_spots=True,
    )
    diag = last_auto_episode_diagnostics()
    assert diag is not None
    assert diag.count_align_ok
    assert diag.n_plan == n_plan
    assert len(rows) == n_plan

    layers = np.asarray([int(r[2]) for r in rows], dtype=np.int64)
    weights = np.asarray([float(r[3]) for r in rows], dtype=np.float64)
    assert int(np.min(layers)) == 0
    assert int(np.max(layers)) >= 1
    assert len(np.unique(layers)) >= 2

    from spot_check.analysis.measured import _plan_xy_by_energy_layer, nominal_layer_energies_mev

    layer_energies = nominal_layer_energies_mev(planned_xyz)
    layer_xy = _plan_xy_by_energy_layer(planned_xyz, layer_energies)
    spots_per_layer = [
        int(np.asarray(arr, dtype=np.float64).reshape(-1, 2).shape[0]) for arr in layer_xy
    ]
    off = 0
    for li, n_sp in enumerate(spots_per_layer):
        w_layer = weights[off : off + n_sp]
        off += n_sp
        assert int(np.sum(w_layer > 1e-6)) >= int(0.9 * n_sp), f"layer {li} missing spots"
