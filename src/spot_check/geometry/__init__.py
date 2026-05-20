"""Scene geometry helpers (Z axis, cube-axes ticks)."""

from __future__ import annotations

from .cube_axes_style import (
    PYVISTA_CUBE_AXES_GRID,
    PYVISTA_CUBE_AXES_LOCATION,
    PYVISTA_CUBE_AXES_PADDING,
    PYVISTA_CUBE_AXES_TICKS,
    apply_pyvista_cube_axes_style,
    cube_axes_ranges,
    disable_pyvista_cube_axes_label_lod,
    pin_pyvista_cube_bounds,
    pyvista_show_bounds_kwargs,
    refresh_pyvista_cube_axes,
)
from .proton_csda_water import (
    normalize_z_depth_metric,
    proton_csda_water_range_mm,
    proton_water_depth_mm,
)
from .z_axis import (
    apply_z_display_to_comparison_clouds,
    cube_z_axis_label_endpoints,
    cube_z_axis_spec,
    cube_z_axis_spec_for_display,
    label_at_scene_z,
    n_cube_axis_labels_for_mm_step,
    nominal_depth_to_scene_z_cube,
    nominal_energy_to_scene_z,
    nominal_mev_to_scene_z_mev_cube,
    plan_depth_bounds_mm,
    plan_depth_bounds_mm_config,
)

__all__ = [
    "PYVISTA_CUBE_AXES_GRID",
    "PYVISTA_CUBE_AXES_LOCATION",
    "PYVISTA_CUBE_AXES_PADDING",
    "PYVISTA_CUBE_AXES_TICKS",
    "apply_pyvista_cube_axes_style",
    "apply_z_display_to_comparison_clouds",
    "cube_axes_ranges",
    "disable_pyvista_cube_axes_label_lod",
    "cube_z_axis_label_endpoints",
    "cube_z_axis_spec",
    "cube_z_axis_spec_for_display",
    "label_at_scene_z",
    "n_cube_axis_labels_for_mm_step",
    "nominal_energy_to_scene_z",
    "nominal_depth_to_scene_z_cube",
    "nominal_mev_to_scene_z_mev_cube",
    "plan_depth_bounds_mm",
    "plan_depth_bounds_mm_config",
    "normalize_z_depth_metric",
    "proton_csda_water_range_mm",
    "proton_water_depth_mm",
    "pyvista_show_bounds_kwargs",
    "pin_pyvista_cube_bounds",
    "refresh_pyvista_cube_axes",
]
