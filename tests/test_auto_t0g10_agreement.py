"""Regression: signal-only auto vs gate_counter on local T0G10 test_data (optional).

Auto must never read Gate Counter; gate_counter mode is the reference baseline.
Layer indices follow plan delivery order; gate_counter may differ (~13%) on layer index
because it weights per-row layer tags inside each spot.
"""

from __future__ import annotations

import numpy as np
import pytest

from spot_check import analysis
from spot_check.analysis.auto_params import nominal_layer_energies_mev
from spot_check.analysis.layers import delivery_layer_indices
from spot_check.analysis.spatial import _plan_xy_by_energy_layer
from spot_check.constants import project_root

_T0G10_DCM = project_root() / "test_data" / "RN.15186535.T0G10.dcm"
_T0G10_CSV = (
    project_root()
    / "test_data"
    / "15186535_T0G10_ic256-45-9018-data acquisition-2026-05-06-16-27-25.csv"
)

pytestmark = pytest.mark.local_data


@pytest.mark.skipif(
    not _T0G10_DCM.is_file() or not _T0G10_CSV.is_file(),
    reason="T0G10 plan/CSV not present under test_data/",
)
def test_auto_matches_gate_counter_t0g10() -> None:
    planned, _, _, _, _ = analysis.planned_spot_xyz_and_counts_from_dicom(_T0G10_DCM)
    energies = nominal_layer_energies_mev(planned)
    layer_xy = _plan_xy_by_energy_layer(planned, energies)
    spots_per_layer = [
        int(np.asarray(arr, dtype=np.float64).reshape(-1, 2).shape[0]) for arr in layer_xy
    ]
    delivery_layers = delivery_layer_indices(len(planned), spots_per_layer)

    gate = analysis.measured_spot_abc_from_csv(
        _T0G10_CSV,
        planned_xyz=list(planned),
        layer_mode="gate_counter",
        a_is_x=False,
        aggregate_spots=True,
    )
    auto = analysis.measured_spot_abc_from_csv(
        _T0G10_CSV,
        planned_xyz=list(planned),
        layer_mode="auto",
        a_is_x=False,
        aggregate_spots=True,
    )
    assert len(auto) == len(gate) == len(planned)

    diag = analysis.last_auto_episode_diagnostics()
    assert diag is not None
    assert diag.count_align_ok

    la = np.array([int(r[2]) for r in auto], dtype=np.int64)
    assert float(np.mean(la == delivery_layers)) >= 0.99

    ga = np.array([[float(r[0]), float(r[1])] for r in gate], dtype=np.float64)
    aa = np.array([[float(r[0]), float(r[1])] for r in auto], dtype=np.float64)
    d = np.sqrt(np.sum((ga - aa) ** 2, axis=1))
    assert float(d[0]) < 0.5
    assert float(d[1]) < 1.5
    assert float((d < 1.0).mean()) > 0.02

    params = analysis.last_auto_layer_params()
    assert params is not None
    assert 0.48 <= params.dead_ratio <= 0.85
