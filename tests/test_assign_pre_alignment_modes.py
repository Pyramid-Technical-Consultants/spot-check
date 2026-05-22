"""Coarse flat alignment is a pipeline phase; assign only applies the stored transform."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from spot_check.analysis.alignment import fit_coarse_flat_align_from_auto_columns
from spot_check.analysis.auto_columns import load_auto_fit_columns_from_csv
from spot_check.analysis.layers import _PlanImputeLookup
from spot_check.analysis.measured import assign_measured_from_csv
from spot_check.pipeline import NullProgressSink, run_data_phases
from spot_check.pipeline.types import PipelineConfig
from tests.conftest import MINIMAL_PLANNED_XYZ, minimal_measured_rows, write_measured_csv


def _write_acq_csv(tmp_path: Path) -> Path:
    return write_measured_csv(tmp_path / "coarse_modes.csv", minimal_measured_rows())


def _fit_coarse_for_minimal_plan(csv_path: Path) -> object:
    planned = list(MINIMAL_PLANNED_XYZ)
    plan_xy2 = np.asarray([(float(px), float(py)) for px, py, _ in planned], dtype=np.float64)
    global_lk = _PlanImputeLookup.from_xy(plan_xy2)
    assert global_lk is not None
    cols = load_auto_fit_columns_from_csv(
        csv_path,
        global_lk=global_lk,
        a_is_x=False,
        spot_weight_mode="channel_sum",
        include_deadtime_rows=False,
    )
    return fit_coarse_flat_align_from_auto_columns(cols, planned)


def test_assign_applies_coarse_flat_transform_to_rows(tmp_path: Path) -> None:
    csv_path = _write_acq_csv(tmp_path)
    transform = _fit_coarse_for_minimal_plan(csv_path)
    raw = assign_measured_from_csv(
        csv_path,
        layer_mode="time_gap",
        planned_xyz=list(MINIMAL_PLANNED_XYZ),
        coarse_flat_transform=None,
    )
    aligned = assign_measured_from_csv(
        csv_path,
        layer_mode="time_gap",
        planned_xyz=list(MINIMAL_PLANNED_XYZ),
        coarse_flat_transform=transform,
    )
    assert raw.rows and aligned.rows
    assert len(raw.rows) == len(aligned.rows)
    assert transform.from_coarse_phase


def test_orchestrator_coarse_flat_skipped_without_plan(tmp_path: Path) -> None:
    csv_path = _write_acq_csv(tmp_path)
    ok = run_data_phases(
        PipelineConfig(
            plan_path=None,
            csv_path=csv_path,
            layer_assign_mode="time_gap",
            aggregate_spots=False,
            spot_weight_mode="channel_sum",
            coarse_flat_align=True,
        ),
        progress=NullProgressSink(),
    )
    assert ok.coarse_flat_align_info is None


def test_orchestrator_without_coarse_leaves_info_none(tmp_path: Path) -> None:
    csv_path = _write_acq_csv(tmp_path)
    ok = run_data_phases(
        PipelineConfig(
            plan_path=None,
            csv_path=csv_path,
            layer_assign_mode="time_gap",
            aggregate_spots=False,
            spot_weight_mode="channel_sum",
            coarse_flat_align=False,
        ),
        progress=NullProgressSink(),
    )
    assert ok.coarse_flat_align_info is None
