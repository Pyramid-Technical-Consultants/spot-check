"""Scene bounds helpers (re-export from geometry)."""

from spot_check.geometry.scene_bounds import (
    Bounds6,
    cube_axes_ranges,
    plan_cube_scene_bounds_and_axes_ranges,
    scene_bounds_for_plan,
)

__all__ = [
    "Bounds6",
    "cube_axes_ranges",
    "plan_cube_scene_bounds_and_axes_ranges",
    "scene_bounds_for_plan",
]
