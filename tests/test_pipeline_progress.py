"""Tests for pipeline progress widget."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from spot_check.gui.pipeline_progress import PipelineProgressWidget
from spot_check.pipeline.progress import ProgressEvent
from spot_check.pipeline.types import PHASE_ASSIGN, PHASE_LOAD


@pytest.fixture(scope="module")
def _qt_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_pipeline_progress_reset_and_percent(_qt_app: QApplication) -> None:
    w = PipelineProgressWidget()
    w.reset(message="Loading plan.dcm…")
    assert w._msg.text() == "Loading plan.dcm…"
    assert w._pct.text() == "0%"


def test_pipeline_progress_apply_event_updates_bar(_qt_app: QApplication) -> None:
    w = PipelineProgressWidget()
    w.reset(message="Starting…")
    w.apply_event(
        ProgressEvent(
            phase_id=PHASE_LOAD,
            step="load_start",
            message="Reading plan…",
            current=50,
            total=100,
        )
    )
    assert "Reading plan" in w._msg.text()
    assert w._bar.value() > 0
    assert w._pct.text().endswith("%")


def test_pipeline_progress_status_message_reserves_two_lines(_qt_app: QApplication) -> None:
    w = PipelineProgressWidget()
    one_line_h = w._msg.sizeHint().height()
    w._msg.setText("First line\nSecond line")
    two_line_h = w._msg.sizeHint().height()
    assert w._msg.minimumHeight() >= two_line_h
    assert w._msg.minimumHeight() >= one_line_h


def test_pipeline_progress_marks_prior_phases_done(_qt_app: QApplication) -> None:
    w = PipelineProgressWidget()
    w.reset(message="Starting…")
    w.apply_event(
        ProgressEvent(
            phase_id=PHASE_LOAD,
            step="load_done",
            message="Plan loaded",
            current=None,
            total=None,
        )
    )
    w.apply_event(
        ProgressEvent(
            phase_id=PHASE_ASSIGN,
            step="assign_start",
            message="Assigning spots…",
            current=None,
            total=None,
        )
    )
    assert w._phase_status[PHASE_LOAD] == "✓"
