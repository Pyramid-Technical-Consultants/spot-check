"""Plan + CSV load pipeline (runs off the GUI thread)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spot_check import analysis
from spot_check.analysis.csv_io import acquisition_csv_has_gate_counter
from spot_check.gui.layer_assign import resolve_layer_assign_mode
from spot_check.plan import plan_label_from_path, planned_spot_xyz_and_counts_from_plan


@dataclass(frozen=True)
class GuiRefreshContext:
    plan_path: Path | None
    csv_path: Path | None
    xy_tick_use: float
    qa_mode: str
    qa_pass_f: float
    qa_warn_f: float
    layer_assign_mode: str
    aggregate_spots: bool
    agg_even_n: int
    spot_weight_mode_run: str
    pipeline_key: tuple[Any, ...]


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
    layer_mode_run: str = "time_gap"
    auto_assign_method: str = "episodes"


def resolve_csv_load_layer_mode(
    *,
    layer_mode: str,
    plan_path: Path | None,
    csv_path: Path,
    aggregate_spots: bool,
) -> tuple[str, bool]:
    """Layer mode and gate aggregation for :func:`measured_spot_abc_from_csv`.

    Aggregation is honored in **auto** (weighted mean per signal episode) and **gate_counter**
    (odd ``Gate Counter`` phases; requires that column). Gate-counter mode without it falls back
    to **time_gap** when a plan is loaded.
    """
    mode = layer_mode.strip().lower().replace("-", "_")
    if mode not in ("auto", "gate_counter", "plan_viterbi", "time_gap"):
        mode = "gate_counter"

    agg = bool(aggregate_spots) and mode in ("auto", "gate_counter")
    if mode == "gate_counter" and not acquisition_csv_has_gate_counter(csv_path):
        if plan_path is not None:
            mode = "time_gap"
        agg = False
    elif mode not in ("auto", "gate_counter"):
        agg = False

    if plan_path is None and mode in ("auto", "gate_counter", "plan_viterbi"):
        mode = "time_gap"
        agg = False

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


def pipeline_load_job(
    plan_path: Path | None,
    csv_path: Path | None,
    *,
    layer_assign_mode: str,
    aggregate_spots: bool,
    aggregate_even_rows_after_odd: int,
    spot_weight_mode: str,
    auto_align: bool = False,
) -> PipelineLoadOK:
    if plan_path is None and csv_path is None:
        raise ValueError("No plan or acquisition CSV to load.")

    label = ""
    planned: list[tuple[float, float, float]] = []
    plan_fwhm_xy = None
    plan_mu: Any = None
    n_plan_kept = 0
    n_plan_raw = 0

    if plan_path is not None:
        label = plan_label_from_path(plan_path)
        planned, plan_fwhm_xy, plan_mu, n_plan_kept, n_plan_raw = (
            planned_spot_xyz_and_counts_from_plan(plan_path)
        )

    measured_unaligned: list[tuple[float, ...]] = []
    csv_display_name = ""
    layer_mode_req, auto_assign_method, auto_infer = resolve_layer_assign_mode(
        layer_assign_mode
    )
    layer_mode_run = layer_mode_req
    if csv_path is not None:
        csv_display_name = csv_path.name
        layer_mode_run, aggregate_run = resolve_csv_load_layer_mode(
            layer_mode=layer_mode_req,
            plan_path=plan_path,
            csv_path=csv_path,
            aggregate_spots=aggregate_spots,
        )
        measured_unaligned = analysis.measured_spot_abc_from_csv(
            csv_path,
            max_points=None,
            planned_xyz=planned if planned else None,
            a_is_x=False,
            layer_mode=layer_mode_run,
            aggregate_spots=aggregate_run,
            aggregate_even_rows_after_odd=int(aggregate_even_rows_after_odd),
            spot_weight_mode=spot_weight_mode,
            auto_infer_params=auto_infer and layer_mode_run == "auto",
            auto_assign_method=auto_assign_method,
            planned_mu=plan_mu,
        )
        if not measured_unaligned:
            raise ValueError("No measured rows to plot.")

    if not planned and not measured_unaligned:
        raise ValueError("No plan spots and no measured rows to plot.")

    measured_aligned: list[tuple[float, ...]] | None = None
    align_info: Any | None = None
    if auto_align and planned and measured_unaligned:
        measured_aligned, align_info = analysis.align_measured_to_plan_detector_xy(
            planned,
            measured_unaligned,
            a_is_x=False,
        )

    pipeline_key = (
        str(plan_path.resolve()) if plan_path is not None else "",
        file_mtime(plan_path) if plan_path is not None else -1.0,
        str(csv_path.resolve()) if csv_path is not None else "",
        file_mtime(csv_path) if csv_path is not None else -1.0,
        layer_assign_mode,
        bool(aggregate_spots),
        int(aggregate_even_rows_after_odd),
        spot_weight_mode,
    )
    return PipelineLoadOK(
        pipeline_key=pipeline_key,
        label=label,
        planned=planned,
        plan_fwhm_xy=plan_fwhm_xy,
        plan_mu=plan_mu,
        n_plan_kept=n_plan_kept,
        n_plan_raw=n_plan_raw,
        measured_unaligned=list(measured_unaligned),
        csv_display_name=csv_display_name,
        measured_aligned=list(measured_aligned) if measured_aligned is not None else None,
        align_info=align_info,
        layer_mode_run=layer_mode_run,
        auto_assign_method=auto_assign_method,
    )
