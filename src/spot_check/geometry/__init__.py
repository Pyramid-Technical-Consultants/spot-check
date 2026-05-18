"""Scene geometry helpers (Z axis, cube-axes ticks)."""

from __future__ import annotations

from .z_axis import (
    PYVISTA_CUBE_AXES_GRID,
    PYVISTA_CUBE_AXES_LOCATION,
    PYVISTA_CUBE_AXES_TICKS,
    apply_pyvista_cube_z_axis,
    cube_z_axis_spec,
    n_cube_axis_labels_for_mm_step,
    nominal_mev_to_plot_z,
    proton_cda_water_range_mm,
)

__all__ = [
    "PYVISTA_CUBE_AXES_GRID",
    "PYVISTA_CUBE_AXES_LOCATION",
    "PYVISTA_CUBE_AXES_TICKS",
    "apply_pyvista_cube_z_axis",
    "cube_z_axis_spec",
    "n_cube_axis_labels_for_mm_step",
    "nominal_mev_to_plot_z",
    "proton_cda_water_range_mm",
]
