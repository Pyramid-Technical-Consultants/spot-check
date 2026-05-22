"""Export pipeline — load data via orchestrator for combined CSV export."""

from __future__ import annotations

from spot_check.pipeline.load_result import PipelineLoadOK
from spot_check.pipeline.orchestrator import run_data_phases
from spot_check.pipeline.progress import NullProgressSink, ProgressSink
from spot_check.pipeline.types import PipelineConfig


def pipeline_export_load(
    config: PipelineConfig,
    progress: ProgressSink | None = None,
) -> tuple[PipelineLoadOK, list[tuple[float, ...]]]:
    """Run the data pipeline and return load result plus rows for export."""
    ok = run_data_phases(config, progress=progress or NullProgressSink())
    measured = list(
        ok.measured_fine_aligned or ok.measured_unaligned
    )
    if not measured:
        raise ValueError("No measured rows to export.")
    return ok, measured
