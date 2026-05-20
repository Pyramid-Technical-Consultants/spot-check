"""Fast gate_counter layer index path vs full aggregation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spot_check import analysis
from spot_check.analysis.measured import gate_counter_aggregated_layer_indices_from_csv
from spot_check.analysis.spatial import _plan_xy_by_energy_layer, nominal_layer_energies_mev
from spot_check.constants import project_root
from tests.conftest import MINIMAL_PLANNED_XYZ, minimal_measured_rows, write_measured_csv

_T0G10_CSV = (
    project_root()
    / "test_data"
    / "15186535_T0G10_ic256-45-9018-data acquisition-2026-05-06-16-27-25.csv"
)


def test_gate_layer_fast_path_matches_gate_counter(tmp_path: Path) -> None:
    csv_path = write_measured_csv(tmp_path / "gates.csv", minimal_measured_rows())
    planned = list(MINIMAL_PLANNED_XYZ)
    le = nominal_layer_energies_mev(planned)
    layer_xy = _plan_xy_by_energy_layer(planned, le)
    spots_per_layer = [int(a.reshape(-1, 2).shape[0]) for a in layer_xy]
    gate = analysis.measured_spot_abc_from_csv(
        csv_path,
        planned_xyz=planned,
        layer_mode="gate_counter",
        a_is_x=False,
        aggregate_spots=True,
    )
    fast = gate_counter_aggregated_layer_indices_from_csv(
        csv_path,
        spots_per_layer,
        max_layer=len(le) - 1,
        a_is_x=False,
        spot_weight_mode="channel_sum",
    )
    lg = np.array([int(r[2]) for r in gate], dtype=np.int64)
    assert len(fast) == len(gate)
    assert np.array_equal(fast, lg)


@pytest.mark.local_data
@pytest.mark.skipif(not _T0G10_CSV.is_file(), reason="T0G10 CSV not under test_data/")
def test_gate_layer_fast_path_matches_t0g10_gate_counter() -> None:
    from spot_check.constants import project_root as root

    dcm = root() / "test_data" / "RN.15186535.T0G10.dcm"
    if not dcm.is_file():
        pytest.skip("T0G10 DICOM not present")
    planned, _, _, _, _ = analysis.planned_spot_xyz_and_counts_from_dicom(dcm)
    le = nominal_layer_energies_mev(planned)
    layer_xy = _plan_xy_by_energy_layer(planned, le)
    spots_per_layer = [int(a.reshape(-1, 2).shape[0]) for a in layer_xy]
    gate = analysis.measured_spot_abc_from_csv(
        _T0G10_CSV,
        planned_xyz=list(planned),
        layer_mode="gate_counter",
        a_is_x=False,
        aggregate_spots=True,
    )
    fast = gate_counter_aggregated_layer_indices_from_csv(
        _T0G10_CSV,
        spots_per_layer,
        max_layer=len(le) - 1,
        a_is_x=False,
        spot_weight_mode="channel_sum",
    )
    lg = np.array([int(r[2]) for r in gate], dtype=np.int64)
    assert len(fast) == len(gate)
    assert float(np.mean(fast == lg)) >= 0.999
