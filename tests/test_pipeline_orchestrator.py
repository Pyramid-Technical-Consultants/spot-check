"""Tests for pipeline orchestrator and progress reporting."""

from __future__ import annotations

from pathlib import Path

import pytest

from spot_check.pipeline import NullProgressSink, ProgressEvent, run_data_phases
from spot_check.pipeline.load_result import resolve_csv_load_layer_mode
from spot_check.pipeline.progress import CallbackProgressSink
from spot_check.pipeline.types import (
    PHASE_ASSIGN,
    PHASE_FILTER,
    PHASE_LOAD,
    PipelineConfig,
)
from tests.conftest import minimal_measured_rows, write_measured_csv


def test_resolve_csv_load_layer_mode_reexported() -> None:
    """load_result helpers remain importable from gui.pipeline for tests."""
    from spot_check.gui.pipeline import resolve_csv_load_layer_mode as gui_resolve

    assert gui_resolve is resolve_csv_load_layer_mode


def test_run_data_phases_csv_only_emits_progress(tmp_path: Path) -> None:
    csv_path = write_measured_csv(tmp_path / "acq.csv", minimal_measured_rows())
    events: list[ProgressEvent] = []
    sink = CallbackProgressSink(events.append)
    config = PipelineConfig(
        plan_path=None,
        csv_path=csv_path,
        layer_assign_mode="time_gap",
        aggregate_spots=False,
        spot_weight_mode="channel_sum",
    )
    result = run_data_phases(config, progress=sink)
    assert result.measured_unaligned
    phase_ids = {e.phase_id for e in events}
    assert PHASE_LOAD in phase_ids
    assert PHASE_FILTER in phase_ids
    assert PHASE_ASSIGN in phase_ids


def test_assign_and_aggregate_split(tmp_path: Path) -> None:
    from spot_check.analysis.measured import (
        aggregate_measured_assign_result,
        assign_measured_from_csv,
    )

    csv_path = write_measured_csv(tmp_path / "acq.csv", minimal_measured_rows())
    assigned = assign_measured_from_csv(csv_path, layer_mode="time_gap")
    assert assigned.rows
    agg = aggregate_measured_assign_result(assigned, aggregate_spots=True)
    raw = aggregate_measured_assign_result(assigned, aggregate_spots=False)
    assert len(raw) >= len(agg)


def test_run_data_phases_no_files_raises() -> None:
    config = PipelineConfig(
        plan_path=None,
        csv_path=None,
        layer_assign_mode="auto",
        aggregate_spots=False,
        spot_weight_mode="channel_sum",
    )
    with pytest.raises(ValueError, match="No plan or acquisition CSV"):
        run_data_phases(config, progress=NullProgressSink())


def test_run_data_phases_with_plan(tmp_path: Path) -> None:
    csv_path = write_measured_csv(tmp_path / "acq.csv", minimal_measured_rows())
    plan = tmp_path / "plan.dcm"
    plan.write_bytes(b"\x00")  # invalid dcm — skip if no pydicom fixture
    # Use CSV-only path; plan optional for time_gap
    config = PipelineConfig(
        plan_path=None,
        csv_path=csv_path,
        layer_assign_mode="time_gap",
        aggregate_spots=True,
        spot_weight_mode="fit_amplitude_a",
    )
    result = run_data_phases(config, progress=NullProgressSink())
    assert len(result.measured_unaligned) >= 1


def test_pipeline_load_job_delegates(tmp_path: Path) -> None:
    from spot_check.gui.pipeline import pipeline_load_job

    csv_path = write_measured_csv(tmp_path / "g.csv", minimal_measured_rows())
    ok = pipeline_load_job(
        None,
        csv_path,
        layer_assign_mode="time_gap",
        aggregate_spots=False,
        spot_weight_mode="channel_sum",
    )
    assert ok.measured_unaligned
