"""Coarse flat 2D alignment phase — rigid fit after filter, before assignment."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from spot_check.analysis.alignment import fit_coarse_flat_align_from_auto_columns
from spot_check.analysis.auto_columns import load_auto_fit_columns_from_csv
from spot_check.analysis.layers import _PlanImputeLookup
from spot_check.analysis.spatial import nominal_layer_energies_mev
from spot_check.models import DetectorRigidAlign2D
from spot_check.pipeline.progress import ProgressSink, report_phase_progress
from spot_check.pipeline.types import PHASE_COARSE_FLAT, PipelineConfig, PipelineState


def run_coarse_flat_align_phase(
    state: PipelineState,
    config: PipelineConfig,
    progress: ProgressSink,
    *,
    csv_path: Path,
) -> PipelineState:
    """Optionally fit coarse flat rigid 2D map; store transform on state for assign only."""
    state.coarse_flat_align = None
    state.coarse_flat_align_info = None

    planned = state.planned
    if not config.coarse_flat_align or not planned:
        return state

    layer_energies = nominal_layer_energies_mev(planned)
    if not layer_energies:
        return state

    plan_xy2 = np.asarray(
        [(float(px), float(py)) for px, py, _ in planned],
        dtype=np.float64,
    )
    global_lk = _PlanImputeLookup.from_xy(plan_xy2)
    if global_lk is None:
        return state

    report_phase_progress(
        progress, PHASE_COARSE_FLAT, step="coarse_start", message="Coarse flat 2D alignment…"
    )
    try:
        cols = state.auto_fit_columns
        if cols is None:
            cols = load_auto_fit_columns_from_csv(
                csv_path,
                global_lk=global_lk,
                a_is_x=False,
                spot_weight_mode=config.spot_weight_mode,
                max_points=None,
                include_deadtime_rows=False,
                heal_partial_fit_axes=config.heal_partial_fit_axes,
            )
            state.auto_fit_columns = cols
        if len(cols) == 0:
            report_phase_progress(
                progress,
                PHASE_COARSE_FLAT,
                step="coarse_skip",
                message="Coarse alignment skipped (no fit rows).",
            )
            return state
        info: DetectorRigidAlign2D = fit_coarse_flat_align_from_auto_columns(cols, planned)
    except ValueError as ex:
        report_phase_progress(
            progress,
            PHASE_COARSE_FLAT,
            step="coarse_skip",
            message=f"Coarse alignment skipped: {ex}",
        )
        return state

    state.coarse_flat_align = info
    state.coarse_flat_align_info = info
    report_phase_progress(
        progress,
        PHASE_COARSE_FLAT,
        step="coarse_done",
        message=(
            f"Coarse flat alignment — RMS={info.rms_residual_mm:.4g} mm "
            f"(n={info.n_pairs})."
        ),
    )
    return state
