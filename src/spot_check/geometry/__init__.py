"""Scene geometry helpers (Z axis, scene bounds)."""

from __future__ import annotations

from spot_check.geometry.proton_csda_water import (
    normalize_z_depth_metric,
    proton_csda_water_range_mm,
    proton_water_depth_mm,
)
from spot_check.geometry.scene_bounds import (
    cube_axes_ranges,
    plan_cube_scene_bounds_and_axes_ranges,
)
from spot_check.geometry.z_axis import (
    apply_z_display_to_comparison_clouds,
    cube_z_axis_label_endpoints,
    cube_z_axis_spec,
    cube_z_axis_spec_for_display,
    label_at_scene_z,
    n_cube_axis_labels_for_mm_step,
    nominal_depth_to_scene_z_cube,
    nominal_energy_to_scene_z,
    nominal_mev_column_to_scene_z,
    nominal_mev_to_scene_z_mev_cube,
    plan_depth_bounds_mm,
    plan_depth_bounds_mm_config,
    z_display_config_for_plotter,
)

__all__ = [
    "apply_z_display_to_comparison_clouds",
    "cube_axes_ranges",
    "cube_z_axis_label_endpoints",
    "cube_z_axis_spec",
    "cube_z_axis_spec_for_display",
    "label_at_scene_z",
    "n_cube_axis_labels_for_mm_step",
    "nominal_energy_to_scene_z",
    "nominal_depth_to_scene_z_cube",
    "nominal_mev_column_to_scene_z",
    "nominal_mev_to_scene_z_mev_cube",
    "normalize_z_depth_metric",
    "plan_cube_scene_bounds_and_axes_ranges",
    "plan_depth_bounds_mm",
    "plan_depth_bounds_mm_config",
    "proton_csda_water_range_mm",
    "proton_water_depth_mm",
    "z_display_config_for_plotter",
]
