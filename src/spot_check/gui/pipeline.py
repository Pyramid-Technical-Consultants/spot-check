"""DICOM + CSV load pipeline (runs off the GUI thread)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pydicom

from spot_check import analysis


@dataclass(frozen=True)
class GuiRefreshContext:
    dcm: Path
    csv_path: Path
    xy_tick_use: float
    qa_pass_f: float
    qa_warn_f: float
    layer_mode: str
    gap: float
    xy_tol: float
    trust_stay: float
    vp_f: float
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
    n_plan_kept: int
    n_plan_raw: int
    measured_unaligned: list[tuple[float, ...]]
    csv_display_name: str
    measured_aligned: list[tuple[float, ...]] | None = None
    align_info: Any | None = None


def file_mtime(path: Path) -> float:
    try:
        return float(path.stat().st_mtime)
    except OSError:
        return -1.0


def pipeline_load_job(
    dcm: Path,
    csv_path: Path,
    *,
    layer_mode: str,
    gap: float,
    xy_tol: float,
    trust_stay: float,
    vp_f: float,
    aggregate_spots: bool,
    aggregate_even_rows_after_odd: int,
    spot_weight_mode: str,
    auto_align: bool = False,
) -> PipelineLoadOK:
    label = str(pydicom.dcmread(dcm, stop_before_pixels=True, force=True).get("RTPlanLabel", ""))
    planned, plan_fwhm_xy, n_plan_kept, n_plan_raw = (
        analysis.planned_spot_xyz_and_counts_from_dicom(dcm)
    )
    measured_unaligned = analysis.measured_spot_abc_from_csv(
        csv_path,
        max_points=None,
        planned_xyz=planned,
        a_is_x=False,
        layer_mode=layer_mode,
        layer_gap_s=gap,
        refill_same_spot_xy_tol_mm=xy_tol,
        refill_trust_time_gap_stay_dist_mm=trust_stay,
        viterbi_advance_penalty_mm2=vp_f,
        aggregate_spots=aggregate_spots,
        aggregate_even_rows_after_odd=int(aggregate_even_rows_after_odd),
        spot_weight_mode=spot_weight_mode,
    )
    if not measured_unaligned:
        raise ValueError("No measured rows to plot.")
    measured_aligned: list[tuple[float, ...]] | None = None
    align_info: Any | None = None
    if auto_align:
        measured_aligned, align_info = analysis.align_measured_to_plan_detector_xy(
            planned,
            measured_unaligned,
            a_is_x=False,
        )
    pipeline_key = (
        str(dcm.resolve()),
        file_mtime(dcm),
        str(csv_path.resolve()),
        file_mtime(csv_path),
        layer_mode,
        float(gap),
        float(xy_tol),
        float(vp_f),
        bool(aggregate_spots),
        int(aggregate_even_rows_after_odd),
        spot_weight_mode,
    )
    return PipelineLoadOK(
        pipeline_key=pipeline_key,
        label=label,
        planned=planned,
        plan_fwhm_xy=plan_fwhm_xy,
        n_plan_kept=n_plan_kept,
        n_plan_raw=n_plan_raw,
        measured_unaligned=list(measured_unaligned),
        csv_display_name=csv_path.name,
        measured_aligned=list(measured_aligned) if measured_aligned is not None else None,
        align_info=align_info,
    )
