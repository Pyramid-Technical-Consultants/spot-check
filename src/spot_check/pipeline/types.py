"""Pipeline configuration, state, and phase contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from spot_check.models import DetectorRigidAlign2D
from spot_check.pipeline.diagnostics import AssignDiagnostics, PipelineDiagnostics, QAResult
from spot_check.pipeline.progress import ProgressSink

# Ordered phase ids (data phases run on worker thread; visualize on main thread).
PHASE_LOAD = "load"
PHASE_FILTER = "filter"
PHASE_COARSE_FLAT = "coarse_flat"
PHASE_ASSIGN = "assign"
PHASE_AGGREGATE = "aggregate"
PHASE_FINE_ALIGN = "fine_align"
PHASE_QA = "qa"
PHASE_VISUALIZE = "visualize"

DATA_PHASE_IDS: tuple[str, ...] = (
    PHASE_LOAD,
    PHASE_FILTER,
    PHASE_COARSE_FLAT,
    PHASE_ASSIGN,
    PHASE_AGGREGATE,
    PHASE_FINE_ALIGN,
    PHASE_QA,
)

ALL_PHASE_IDS: tuple[str, ...] = (*DATA_PHASE_IDS, PHASE_VISUALIZE)

PHASE_LABELS: dict[str, str] = {
    PHASE_LOAD: "Loading",
    PHASE_FILTER: "Filtering",
    PHASE_COARSE_FLAT: "Coarse flat alignment",
    PHASE_ASSIGN: "Spot assignment",
    PHASE_AGGREGATE: "Aggregation",
    PHASE_FINE_ALIGN: "Fine detector alignment",
    PHASE_QA: "QA calculation",
    PHASE_VISUALIZE: "Visualization",
}

# Relative weights for overall progress bar (sum = 1.0).
PHASE_WEIGHTS: dict[str, float] = {
    PHASE_LOAD: 0.15,
    PHASE_FILTER: 0.18,
    PHASE_COARSE_FLAT: 0.07,
    PHASE_ASSIGN: 0.25,
    PHASE_AGGREGATE: 0.075,
    PHASE_FINE_ALIGN: 0.075,
    PHASE_QA: 0.10,
    PHASE_VISUALIZE: 0.05,
}


@dataclass(frozen=True)
class PipelineConfig:
    """Inputs that drive the data-processing pipeline."""

    plan_path: Path | None
    csv_path: Path | None
    layer_assign_mode: str
    aggregate_spots: bool
    spot_weight_mode: str
    coarse_flat_align: bool = False
    heal_partial_fit_axes: bool = False
    fine_align_xy: bool = True
    fine_align_rotation: bool = True
    fine_align_scale: bool = True
    filter_xy_fliers: bool = False
    filter_xy_flier_sigma: float = 3.0


@dataclass
class PipelineState:
    """Artifacts produced by pipeline phases (grows as phases run)."""

    pipeline_key: tuple[Any, ...] = field(default_factory=tuple)
    label: str = ""
    planned: list[tuple[float, float, float]] = field(default_factory=list)
    plan_fwhm_xy: Any = None
    plan_mu: Any = None
    n_plan_kept: int = 0
    n_plan_raw: int = 0
    measured_unaligned: list[tuple[float, ...]] = field(default_factory=list)
    csv_display_name: str = ""
    csv_columns_validated: bool = False
    coarse_flat_align: DetectorRigidAlign2D | None = None
    measured_fine_aligned: list[tuple[float, ...]] | None = None
    coarse_flat_align_info: DetectorRigidAlign2D | None = None
    fine_align_info: Any | None = None
    layer_mode_run: str = "time_gap"
    auto_assign_method: str = "episodes"
    aggregate_run: bool = False
    assign_result: Any = None
    assign_diagnostics: AssignDiagnostics | None = None
    auto_fit_columns: Any = None
    qa_result: QAResult | None = None
    diagnostics: PipelineDiagnostics | None = None


class PipelinePhase(Protocol):
    """One step in the data pipeline."""

    id: str
    label: str

    def run(
        self,
        state: PipelineState,
        config: PipelineConfig,
        progress: ProgressSink,
    ) -> PipelineState: ...
