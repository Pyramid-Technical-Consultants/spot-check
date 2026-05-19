"""Regression: signal-only auto vs gate_counter on local T0G10 test_data (optional).

Auto must never read Gate Counter; gate_counter mode is the reference baseline.
"""

from __future__ import annotations

import numpy as np
import pytest

from spot_check import analysis
from spot_check.constants import project_root

_T0G10_DCM = project_root() / "test_data" / "RN.15186535.T0G10.dcm"
_T0G10_CSV = (
    project_root()
    / "test_data"
    / "15186535_T0G10_ic256-45-9018-data acquisition-2026-05-06-16-27-25.csv"
)


@pytest.mark.skipif(
    not _T0G10_DCM.is_file() or not _T0G10_CSV.is_file(),
    reason="T0G10 plan/CSV not present under test_data/",
)
def test_auto_matches_gate_counter_t0g10() -> None:
    planned, _, _, _, _ = analysis.planned_spot_xyz_and_counts_from_dicom(_T0G10_DCM)
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
    )
    assert len(auto) == len(gate) == len(planned)

    diag = analysis.last_auto_episode_diagnostics()
    assert diag is not None
    assert diag.count_align_ok

    lg = np.array([int(r[2]) for r in gate], dtype=np.int64)
    la = np.array([int(r[2]) for r in auto], dtype=np.int64)
    layer_match = float(np.mean(lg == la))
    assert layer_match >= 0.85

    params = analysis.last_auto_layer_params()
    assert params is not None
