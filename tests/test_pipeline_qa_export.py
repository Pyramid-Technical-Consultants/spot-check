"""Tests for QA phase and export pipeline."""

from __future__ import annotations

from pathlib import Path

from spot_check.pipeline.diagnostics import QAResult
from spot_check.pipeline.export_job import pipeline_export_load
from spot_check.pipeline.phases.qa import run_qa_phase
from spot_check.pipeline.progress import NullProgressSink
from spot_check.pipeline.types import PipelineConfig, PipelineState
from tests.conftest import MINIMAL_PLANNED_XYZ, minimal_measured_rows, write_measured_csv


def test_run_qa_phase_position_counts() -> None:
    planned = list(MINIMAL_PLANNED_XYZ)
    rows = [
        (1.0, 2.0, 0.0, 1.0, 0, float("nan"), float("nan"), 1.0),
    ]
    state = PipelineState()
    qa = run_qa_phase(
        state,
        NullProgressSink(),
        planned=planned,
        measured=rows,
        qa_mode="position",
        pass_thr=1.0,
        warn_thr=5.0,
        plan_mu=None,
        enabled=True,
    )
    assert qa is not None
    assert isinstance(qa, QAResult)
    assert qa.n_pass + qa.n_warn + qa.n_fail == 1


def test_run_qa_phase_disabled_returns_none() -> None:
    state = PipelineState()
    qa = run_qa_phase(
        state,
        NullProgressSink(),
        planned=list(MINIMAL_PLANNED_XYZ),
        measured=minimal_measured_rows(),
        qa_mode="position",
        pass_thr=1.0,
        warn_thr=5.0,
        plan_mu=None,
        enabled=False,
    )
    assert qa is None


def test_pipeline_export_load(tmp_path: Path) -> None:
    csv_path = write_measured_csv(tmp_path / "acq.csv", minimal_measured_rows())
    config = PipelineConfig(
        plan_path=None,
        csv_path=csv_path,
        layer_assign_mode="time_gap",
        aggregate_spots=False,
        spot_weight_mode="channel_sum",
    )
    ok, measured = pipeline_export_load(config)
    assert measured
    assert ok.csv_display_name == csv_path.name
