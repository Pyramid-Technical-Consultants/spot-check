"""Pipeline orchestration — sequences phases and reports progress."""

from __future__ import annotations

from spot_check.analysis.viz.data import build_plan_spot_delivery_times_s
from spot_check.pipeline.diagnostics import PipelineDiagnostics
from spot_check.pipeline.layer_mode import resolve_layer_mode_for_csv
from spot_check.pipeline.load_result import PipelineLoadOK, file_mtime
from spot_check.pipeline.phases import (
    run_aggregate_phase,
    run_assign_phase,
    run_coarse_flat_align_phase,
    run_filter_phase,
    run_fine_align_phase,
    run_load_phase,
)
from spot_check.pipeline.phases.csv_preload import preload_auto_fit_columns_if_needed
from spot_check.pipeline.phases.filter import apply_xy_flier_filter_if_enabled
from spot_check.pipeline.progress import NullProgressSink, ProgressEvent, ProgressSink
from spot_check.pipeline.types import PHASE_FILTER, PipelineConfig, PipelineState


def run_data_phases(
    config: PipelineConfig,
    progress: ProgressSink | None = None,
) -> PipelineLoadOK:
    """Run load → filter → coarse → assign → aggregate → fine on the worker thread."""
    sink: ProgressSink = progress if progress is not None else NullProgressSink()
    plan_path = config.plan_path
    csv_path = config.csv_path

    if plan_path is None and csv_path is None:
        raise ValueError("No plan or acquisition CSV to load.")

    state = PipelineState()
    state = run_load_phase(state, config, sink)

    measured_unaligned: list[tuple[float, ...]] = []
    plan_spots_no_data = None
    plan_spot_time_s = None
    layer_mode_run = "time_gap"
    auto_assign_method = "episodes"
    aggregate_run = False

    if csv_path is not None:
        layer_mode_run, aggregate_run, auto_assign_method, auto_infer = (
            resolve_layer_mode_for_csv(config)
        )
        state = run_filter_phase(
            state, config, sink, layer_mode_run=layer_mode_run
        )
        preload_auto_fit_columns_if_needed(
            state,
            config,
            layer_mode_run=layer_mode_run,
            auto_assign_method=auto_assign_method,
        )
        state = run_coarse_flat_align_phase(state, config, sink, csv_path=csv_path)
        assigned = run_assign_phase(
            state,
            config,
            sink,
            csv_path=csv_path,
            layer_mode_run=layer_mode_run,
            auto_assign_method=auto_assign_method,
            auto_infer=auto_infer,
        )
        assigned = apply_xy_flier_filter_if_enabled(assigned, state, config, sink)
        if state.planned and assigned.plan_index_per_row:
            plan_spot_time_s = build_plan_spot_delivery_times_s(
                len(state.planned),
                assigned.rows,
                assigned.plan_index_per_row,
            )
        n_filtered = len(assigned.rows)
        sink.report(
            ProgressEvent(
                phase_id=PHASE_FILTER,
                step="filter_done",
                message=(
                    f"Filtered — {n_filtered} measured row"
                    f"{'s' if n_filtered != 1 else ''}."
                ),
                current=n_filtered,
                total=n_filtered,
            )
        )
        measured_unaligned = run_aggregate_phase(
            state,
            sink,
            assigned=assigned,
            aggregate_spots=aggregate_run,
        )
        plan_spots_no_data = assigned.plan_spots_no_data
        layer_mode_run = state.layer_mode_run
        auto_assign_method = state.auto_assign_method
        aggregate_run = state.aggregate_run

    if not state.planned and not measured_unaligned:
        raise ValueError("No plan spots and no measured rows to plot.")

    measured_fine_aligned, fine_align_info = run_fine_align_phase(state, config, sink)

    state.diagnostics = PipelineDiagnostics(
        assign=state.assign_diagnostics,
        coarse_flat_align_info=state.coarse_flat_align_info,
        fine_align_info=fine_align_info,
    )

    pipeline_key = (
        str(plan_path.resolve()) if plan_path is not None else "",
        file_mtime(plan_path) if plan_path is not None else -1.0,
        str(csv_path.resolve()) if csv_path is not None else "",
        file_mtime(csv_path) if csv_path is not None else -1.0,
        config.layer_assign_mode,
        bool(config.aggregate_spots),
        config.spot_weight_mode,
        bool(config.heal_partial_fit_axes),
        bool(config.coarse_flat_align),
        bool(config.fine_align_xy),
        bool(config.fine_align_rotation),
        bool(config.fine_align_scale),
        bool(config.filter_xy_fliers),
        float(config.filter_xy_flier_sigma),
    )

    return PipelineLoadOK(
        pipeline_key=pipeline_key,
        label=state.label,
        planned=state.planned,
        plan_fwhm_xy=state.plan_fwhm_xy,
        plan_mu=state.plan_mu,
        n_plan_kept=state.n_plan_kept,
        n_plan_raw=state.n_plan_raw,
        measured_unaligned=list(measured_unaligned),
        csv_display_name=state.csv_display_name,
        measured_fine_aligned=(
            list(measured_fine_aligned) if measured_fine_aligned is not None else None
        ),
        coarse_flat_align_info=state.coarse_flat_align_info,
        fine_align_info=fine_align_info,
        layer_mode_run=layer_mode_run,
        auto_assign_method=auto_assign_method,
        aggregate_run=bool(aggregate_run),
        assign_diagnostics=state.assign_diagnostics,
        plan_spots_no_data=plan_spots_no_data,
        plan_spot_time_s=plan_spot_time_s,
    )
