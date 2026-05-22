"""Fine align must improve layer-NN QA on clinic-scale fixtures (optional local_data)."""

from __future__ import annotations

import numpy as np
import pytest

from spot_check.analysis.spatial import layer_nn_plan_xy_distances_and_expected_xyz
from spot_check.constants import project_root
from spot_check.pipeline import run_data_phases
from spot_check.pipeline.types import PipelineConfig

_T0G10_DCM = project_root() / "test_data" / "RN.15186535.T0G10.dcm"
_T0G10_CSV = (
    project_root()
    / "test_data"
    / "15186535_T0G10_ic256-45-9018-data acquisition-2026-05-06-16-27-25.csv"
)


def _qa_mean_mm(planned, rows) -> float:
    dist, _ = layer_nn_plan_xy_distances_and_expected_xyz(planned, rows, a_is_x=False)
    dist = dist[np.isfinite(dist)]
    assert dist.size > 0
    return float(np.mean(dist))


@pytest.mark.local_data
@pytest.mark.skipif(
    not _T0G10_DCM.is_file() or not _T0G10_CSV.is_file(),
    reason="T0G10 plan/CSV not present under test_data/",
)
def test_t0g10_fine_align_improves_layer_nn_qa_when_aggregated() -> None:
    """Regression: fine fit must use layer-NN pairing (not delivery index when counts match)."""
    ok = run_data_phases(
        PipelineConfig(
            plan_path=_T0G10_DCM,
            csv_path=_T0G10_CSV,
            layer_assign_mode="gate_counter",
            aggregate_spots=True,
            spot_weight_mode="channel_sum",
            coarse_flat_align=True,
            fine_align_xy=True,
            fine_align_rotation=True,
            fine_align_scale=True,
        )
    )
    planned = ok.planned
    base = ok.measured_unaligned
    fine = ok.measured_fine_aligned
    assert fine is not None and ok.fine_align_info is not None
    qa_before = _qa_mean_mm(planned, base)
    qa_after = _qa_mean_mm(planned, fine)
    assert qa_after <= qa_before + 0.02
    assert ok.fine_align_info.rms_after_mm <= ok.fine_align_info.rms_before_mm + 0.05


@pytest.mark.local_data
@pytest.mark.skipif(
    not _T0G10_DCM.is_file() or not _T0G10_CSV.is_file(),
    reason="T0G10 plan/CSV not present under test_data/",
)
def test_t0g10_fine_align_improves_layer_nn_qa_when_not_aggregated() -> None:
    ok = run_data_phases(
        PipelineConfig(
            plan_path=_T0G10_DCM,
            csv_path=_T0G10_CSV,
            layer_assign_mode="gate_counter",
            aggregate_spots=False,
            spot_weight_mode="channel_sum",
            coarse_flat_align=True,
            fine_align_xy=True,
            fine_align_rotation=True,
            fine_align_scale=True,
        )
    )
    planned = ok.planned
    base = ok.measured_unaligned
    fine = ok.measured_fine_aligned
    assert fine is not None and ok.fine_align_info is not None
    qa_before = _qa_mean_mm(planned, base)
    qa_after = _qa_mean_mm(planned, fine)
    assert qa_after <= qa_before + 0.02
    assert ok.fine_align_info.rms_after_mm <= ok.fine_align_info.rms_before_mm + 0.05
