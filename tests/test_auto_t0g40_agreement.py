"""Optional regression: auto vs gate_counter on T0G40 test_data."""

from __future__ import annotations

import numpy as np
import pytest

from spot_check import analysis
from spot_check.constants import project_root

_T0G40_DCM = project_root() / "test_data" / "RN.15186535.T0G40.dcm"
_T0G40_CSV = (
    project_root()
    / "test_data"
    / "15186535_T0G40_ic256-45-9018-data acquisition-2026-05-06-16-19-12.csv"
)

pytestmark = pytest.mark.local_data


@pytest.mark.skipif(
    not _T0G40_DCM.is_file() or not _T0G40_CSV.is_file(),
    reason="T0G40 plan/CSV not present under test_data/",
)
def test_auto_reasonable_on_t0g40() -> None:
    planned, _, _, _, _ = analysis.planned_spot_xyz_and_counts_from_dicom(_T0G40_DCM)
    gate = analysis.measured_spot_abc_from_csv(
        _T0G40_CSV,
        planned_xyz=list(planned),
        layer_mode="gate_counter",
        a_is_x=False,
        aggregate_spots=True,
    )
    auto = analysis.measured_spot_abc_from_csv(
        _T0G40_CSV,
        planned_xyz=list(planned),
        layer_mode="auto",
        a_is_x=False,
        aggregate_spots=True,
    )
    assert len(auto) == len(gate) == len(planned)
    assert analysis.last_auto_episode_diagnostics() is not None

    ga = np.array([[float(r[0]), float(r[1])] for r in gate], dtype=np.float64)
    aa = np.array([[float(r[0]), float(r[1])] for r in auto], dtype=np.float64)
    d = np.sqrt(np.sum((ga - aa) ** 2, axis=1))
    assert float(d[0]) < 1.0
    assert float((d < 1.0).mean()) > 0.03
