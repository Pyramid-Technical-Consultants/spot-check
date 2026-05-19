"""Scene geometry helpers (Z axis, cube-axes ticks)."""

from __future__ import annotations

from .cube_axes_style import (
    PYVISTA_CUBE_AXES_GRID,
    PYVISTA_CUBE_AXES_LOCATION,
    PYVISTA_CUBE_AXES_PADDING,
    PYVISTA_CUBE_AXES_TICKS,
    apply_pyvista_cube_axes_style,
    pyvista_show_bounds_kwargs,
)
from .proton_csda_water import (
    normalize_z_depth_metric,
    proton_cda_water_range_mm,
    proton_csda_water_range_mm,
    proton_water_depth_mm,
)
from .z_axis import (
    apply_pyvista_cube_z_axis,
    cube_z_axis_spec,
    n_cube_axis_labels_for_mm_step,
    nominal_mev_to_plot_z,
)

__all__ = [
    "PYVISTA_CUBE_AXES_GRID",
    "PYVISTA_CUBE_AXES_LOCATION",
    "PYVISTA_CUBE_AXES_PADDING",
    "PYVISTA_CUBE_AXES_TICKS",
    "apply_pyvista_cube_axes_style",
    "apply_pyvista_cube_z_axis",
    "cube_z_axis_spec",
    "n_cube_axis_labels_for_mm_step",
    "nominal_mev_to_plot_z",
    "normalize_z_depth_metric",
    "proton_cda_water_range_mm",
    "proton_csda_water_range_mm",
    "proton_water_depth_mm",
    "pyvista_show_bounds_kwargs",
]
