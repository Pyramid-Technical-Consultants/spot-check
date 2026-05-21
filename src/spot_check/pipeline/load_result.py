"""Pipeline load result types and layer-mode resolution (GUI-agnostic)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spot_check.analysis.csv_io import acquisition_csv_has_gate_counter
from spot_check.pipeline.diagnostics import AssignDiagnostics


@dataclass(frozen=True)
class PipelineLoadOK:
    pipeline_key: tuple[Any, ...]
    label: str
    planned: list[tuple[float, float, float]]
    plan_fwhm_xy: Any
    plan_mu: Any
    n_plan_kept: int
    n_plan_raw: int
    measured_unaligned: list[tuple[float, ...]]
    csv_display_name: str
    measured_aligned: list[tuple[float, ...]] | None = None
    align_info: Any | None = None
    detector_pre_aligned: bool = False
    layer_mode_run: str = "time_gap"
    auto_assign_method: str = "episodes"
    aggregate_run: bool = False
    assign_diagnostics: AssignDiagnostics | None = None


def aggregation_applies(*, layer_mode: str, aggregate_spots: bool) -> bool:
    """True when aggregation toggle is on (post-assignment weighted mean)."""
    del layer_mode  # kept for call-site stability
    return bool(aggregate_spots)


def resolve_csv_load_layer_mode(
    *,
    layer_mode: str,
    plan_path: Path | None,
    csv_path: Path,
    aggregate_spots: bool,
    auto_assign_method: str = "episodes",
) -> tuple[str, bool]:
    """Layer mode and spot aggregation for :func:`measured_spot_abc_from_csv`."""
    mode = layer_mode.strip().lower().replace("-", "_")
    if mode not in ("auto", "gate_counter", "plan_viterbi", "time_gap"):
        mode = "gate_counter"

    if mode == "gate_counter" and not acquisition_csv_has_gate_counter(csv_path):
        mode = "time_gap"

    if plan_path is None and mode in ("auto", "gate_counter", "plan_viterbi"):
        mode = "time_gap"

    assign_m = str(auto_assign_method).strip().lower().replace("-", "_")
    if assign_m == "sequential":
        assign_m = "plan_sequential"

    agg = aggregation_applies(layer_mode=mode, aggregate_spots=aggregate_spots)
    return mode, agg


def file_mtime(path: Path) -> float:
    try:
        return float(path.stat().st_mtime)
    except OSError:
        return -1.0


def is_acquisition_csv_file(path: Path) -> bool:
    if not path.is_file():
        return False
    lower = path.name.lower()
    return lower.endswith(".csv") or lower.endswith(".csv.gz")
