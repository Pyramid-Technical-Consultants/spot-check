"""Layer-mode resolution for acquisition CSV processing."""

from __future__ import annotations

from spot_check.gui.layer_assign import resolve_layer_assign_mode
from spot_check.pipeline.load_result import resolve_csv_load_layer_mode
from spot_check.pipeline.types import PipelineConfig


def resolve_layer_mode_for_csv(
    config: PipelineConfig,
) -> tuple[str, bool, str, bool]:
    """Resolve effective layer mode, aggregation, assign method, and auto-infer flag."""
    if config.csv_path is None:
        raise ValueError("csv_path required")
    layer_mode_req, auto_assign_method, auto_infer = resolve_layer_assign_mode(
        config.layer_assign_mode
    )
    layer_mode_run, aggregate_run = resolve_csv_load_layer_mode(
        layer_mode=layer_mode_req,
        plan_path=config.plan_path,
        csv_path=config.csv_path,
        aggregate_spots=config.aggregate_spots,
        auto_assign_method=auto_assign_method,
    )
    return layer_mode_run, aggregate_run, auto_assign_method, auto_infer
