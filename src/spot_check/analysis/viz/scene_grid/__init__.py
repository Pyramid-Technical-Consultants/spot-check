"""Custom scene grid for the comparison 3D view."""

from spot_check.analysis.viz.scene_grid.bounds import (
    cube_axes_ranges,
    plan_cube_scene_bounds_and_axes_ranges,
    scene_bounds_for_plan,
)
from spot_check.analysis.viz.scene_grid.controller import PlanSceneGridController
from spot_check.analysis.viz.scene_grid.planner import (
    plan_xy_grid,
    xy_boundary_perimeter,
    xy_zero_axes,
)
from spot_check.analysis.viz.scene_grid.types import (
    AxisLineSpec,
    Bounds6,
    GridPlan,
    GridStyle,
    LabelAnchor,
    SceneFrame,
)

__all__ = [
    "AxisLineSpec",
    "Bounds6",
    "GridPlan",
    "GridStyle",
    "LabelAnchor",
    "PlanSceneGridController",
    "SceneFrame",
    "cube_axes_ranges",
    "plan_cube_scene_bounds_and_axes_ranges",
    "plan_xy_grid",
    "scene_bounds_for_plan",
    "xy_boundary_perimeter",
    "xy_zero_axes",
]
