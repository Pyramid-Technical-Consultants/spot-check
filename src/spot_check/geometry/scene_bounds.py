"""Scene bounds for the comparison 3D view."""

from __future__ import annotations

import numpy as np

from spot_check.geometry.z_axis import cube_z_axis_spec_for_display
from spot_check.models import CubeZAxisSpec, ZAxisDisplayConfig

Bounds6 = tuple[float, float, float, float, float, float]

__all__ = [
    "Bounds6",
    "cube_axes_ranges",
    "plan_cube_scene_bounds_and_axes_ranges",
    "scene_bounds_for_plan",
]


def cube_axes_ranges(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_spec: CubeZAxisSpec,
) -> Bounds6:
    """Scene 6-tuple from XY extent and Z spec."""
    return (
        float(x_min),
        float(x_max),
        float(y_min),
        float(y_max),
        float(z_spec.zmin_scene),
        float(z_spec.zmax_scene),
    )


def plan_cube_scene_bounds_and_axes_ranges(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_spec: CubeZAxisSpec,
) -> tuple[Bounds6, Bounds6]:
    """Return identical ``(bounds, axes_ranges)`` in scene coordinates."""
    scene_bounds = cube_axes_ranges(x_min, x_max, y_min, y_max, z_spec)
    return scene_bounds, scene_bounds


def scene_bounds_for_plan(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    plan_scene_z: np.ndarray,
    plan_energies_mev: np.ndarray,
    z_display_cfg: ZAxisDisplayConfig,
    *,
    sanity: bool = False,
) -> tuple[Bounds6, CubeZAxisSpec | None]:
    """Compute scene bounds and optional Z spec for grid/camera setup."""
    if sanity:
        return (0.0, 10.0, 0.0, 10.0, 0.0, 10.0), None
    z_spec = cube_z_axis_spec_for_display(
        np.asarray(plan_scene_z, dtype=np.float64).reshape(-1),
        np.asarray(plan_energies_mev, dtype=np.float64).reshape(-1),
        z_display_cfg,
    )
    bounds, _axes = plan_cube_scene_bounds_and_axes_ranges(
        float(x_min),
        float(x_max),
        float(y_min),
        float(y_max),
        z_spec,
    )
    return bounds, z_spec
