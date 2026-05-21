"""Run12 cube: plan-sequential auto assign must cover all nominal energy layers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spot_check import analysis
from spot_check.analysis.assign import assign_plan_indices_sequential
from spot_check.analysis.auto_columns import load_auto_fit_columns_from_csv
from spot_check.analysis.auto_params import last_auto_layer_params
from spot_check.analysis.layers import _PlanImputeLookup
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
    """Plan defines 25 layers; Run12 CSV only acquired a subset (no plan-only measured rows)."""
    from spot_check.analysis.layers import delivery_layer_indices
    from spot_check.analysis.spatial import _plan_xy_by_energy_layer

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
        align_detector_xy_before_assign=True,
    )
    n_plan = len(planned_xyz)
    layers = np.asarray([int(r[2]) for r in rows], dtype=np.int64)
    assert int(layers.min()) == 0
    assert len(rows) <= n_plan
    assert len(rows) > 300
    assert int(np.sum([float(r[3]) for r in rows])) > 0

    plan_xy = np.asarray([(p[0], p[1]) for p in planned_xyz], dtype=np.float64)
    lk = _PlanImputeLookup.from_xy(plan_xy)
    cols = load_auto_fit_columns_from_csv(
        _RUN12,
        global_lk=lk,
        a_is_x=False,
        spot_weight_mode="channel_sum",
        include_deadtime_rows=True,
    )
    from spot_check.analysis.alignment import align_auto_fit_columns_to_plan_xy

    cols, _ = align_auto_fit_columns_to_plan_xy(cols, planned_xyz, a_is_x=False)
    plan_idx = assign_plan_indices_sequential(cols, plan_xy, min_rows_on_spot=1)
    on = plan_idx[plan_idx >= 0]
    assert int(on.size) > 1000
    assert int(on[0]) == 0
    steps = np.diff(on.astype(np.int64))
    assert int(np.sum(steps < 0)) == 0
    assert int(np.max(steps)) <= 1
    assert len(np.unique(on)) > 300
    layer_e = nominal_layer_energies_mev(planned_xyz)
    layer_xy = _plan_xy_by_energy_layer(planned_xyz, layer_e)
    spots_per = [int(a.reshape(-1, 2).shape[0]) for a in layer_xy]
    plan_layers = delivery_layer_indices(n_plan, spots_per)
    assigned_layers = plan_layers[on]
    assert len(np.unique(assigned_layers)) >= 2

    assert last_auto_layer_params() is not None


@pytest.mark.skipif(
    not _RUN12.is_file() or not _CUBE_PLAN.is_file(),
    reason="run12 cube fixture or R20M10 plan not present",
)
def test_run12_plan_sequential_detector_align() -> None:
    """Measured rows are real assignments only (no plan-only padding)."""
    planned_xyz, _, _, _, _ = planned_spot_xyz_and_counts_from_pyramid_csv(_CUBE_PLAN)
    rows = analysis.measured_spot_abc_from_csv(
        _RUN12,
        planned_xyz=planned_xyz,
        layer_mode="auto",
        auto_assign_method="plan_sequential",
        aggregate_spots=True,
        align_detector_xy_before_assign=True,
    )
    assert 0 < len(rows) < len(planned_xyz)
    info = analysis.last_detector_align_info()
    assert info is not None
    assert info.pre_assignment
    assert info.n_pairs >= 2


@pytest.mark.skipif(
    not _RUN12.is_file() or not _CUBE_PLAN.is_file(),
    reason="run12 cube fixture or R20M10 plan not present",
)
def test_run12_layer0_aggregated_xy_near_plan() -> None:
    """Layer 0 grid: aggregated measured XY should sit on plan spots after pre-align + NN."""
    import numpy as np

    from spot_check.analysis.layers import delivery_layer_indices
    from spot_check.analysis.spatial import _plan_xy_by_energy_layer, nominal_layer_energies_mev

    planned_xyz, _, _, _, _ = planned_spot_xyz_and_counts_from_pyramid_csv(_CUBE_PLAN)
    rows = analysis.measured_spot_abc_from_csv(
        _RUN12,
        planned_xyz=planned_xyz,
        layer_mode="auto",
        auto_assign_method="plan_sequential",
        aggregate_spots=True,
        align_detector_xy_before_assign=True,
    )
    layer_e = nominal_layer_energies_mev(planned_xyz)
    layer_xy = _plan_xy_by_energy_layer(planned_xyz, layer_e)
    spots_per = [int(a.reshape(-1, 2).shape[0]) for a in layer_xy]
    plan_layers = delivery_layer_indices(len(planned_xyz), spots_per)
    l0_xy = np.asarray(
        [(p[0], p[1]) for p, ly in zip(planned_xyz, plan_layers, strict=True) if int(ly) == 0],
        dtype=np.float64,
    )
    l0_rows = [r for r in rows if int(r[2]) == 0]
    assert len(l0_rows) >= 300
    # GUI plot uses a_is_x=False: plan X = Fit B (tuple[1]), plan Y = Fit A (tuple[0]).
    meas = np.asarray([(r[1], r[0]) for r in l0_rows], dtype=np.float64)
    from scipy.spatial import cKDTree

    dist, _ = cKDTree(l0_xy).query(meas)
    assert int(np.count_nonzero(dist <= 3.0)) >= 300
    assert float(np.median(dist)) < 2.0
