"""Modular data processing pipeline."""

from spot_check.pipeline.diagnostics import AssignDiagnostics, PipelineDiagnostics, QAResult
from spot_check.pipeline.export_job import pipeline_export_load
from spot_check.pipeline.orchestrator import run_data_phases
from spot_check.pipeline.progress import (
    CallbackProgressSink,
    NullProgressSink,
    ProgressEvent,
    ProgressSink,
)
from spot_check.pipeline.types import (
    ALL_PHASE_IDS,
    DATA_PHASE_IDS,
    PHASE_LABELS,
    PHASE_WEIGHTS,
    PipelineConfig,
    PipelinePhase,
    PipelineState,
)

__all__ = [
    "ALL_PHASE_IDS",
    "AssignDiagnostics",
    "DATA_PHASE_IDS",
    "PHASE_LABELS",
    "PHASE_WEIGHTS",
    "CallbackProgressSink",
    "NullProgressSink",
    "PipelineConfig",
    "PipelineDiagnostics",
    "PipelinePhase",
    "PipelineState",
    "ProgressEvent",
    "ProgressSink",
    "QAResult",
    "pipeline_export_load",
    "run_data_phases",
]
