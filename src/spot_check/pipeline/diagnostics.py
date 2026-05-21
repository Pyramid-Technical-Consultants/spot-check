"""Structured diagnostics produced by pipeline phases."""

from __future__ import annotations

from dataclasses import dataclass

from spot_check.analysis.auto_params import AutoLayerParams
from spot_check.analysis.episodes import AutoEpisodeDiagnostics
from spot_check.models import DetectorRigidAlign2D


@dataclass(frozen=True)
class AssignDiagnostics:
    """Auto-mode tuning and episode alignment outcomes from the assign phase."""

    auto_layer_params: AutoLayerParams | None = None
    episode_diagnostics: AutoEpisodeDiagnostics | None = None


@dataclass(frozen=True)
class QAResult:
    """Plan QA tier counts for measured spots."""

    n_pass: int
    n_warn: int
    n_fail: int
    qa_mode: str
    pass_thr: float
    warn_thr: float


@dataclass(frozen=True)
class PipelineDiagnostics:
    """All structured diagnostics from a completed data pipeline run."""

    assign: AssignDiagnostics | None = None
    align_info: DetectorRigidAlign2D | None = None
    detector_pre_aligned: bool = False
