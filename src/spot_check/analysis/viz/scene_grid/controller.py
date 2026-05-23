"""Plotter-facing controller for custom scene grid."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from spot_check.models import CubeZAxisSpec, ZAxisDisplayConfig

from .bounds import scene_bounds_for_plan
from .planner import plan_xy_grid
from .render import GridRenderState, clear_grid_actors, render_grid_plan
from .types import Bounds6, GridStyle, SceneFrame


@dataclass
class PlanSceneGridController:
    """Owns comparison-view scene grid state and VTK actor lifecycle."""

    @staticmethod
    def clear_on_plotter(plotter: Any) -> None:
        """Remove grid actors from a reused plotter before ``clear()``."""
        prev = getattr(plotter, "_spot_check_scene_grid", None)
        if prev is not None:
            prev.clear(plotter)

    xlab: str
    ylab: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_display_cfg: ZAxisDisplayConfig
    style: GridStyle = field(default_factory=GridStyle)
    sanity: bool = False
    ready: bool = False
    z_spec: CubeZAxisSpec | None = field(default=None, repr=False)
    scene_bounds: Bounds6 | None = field(default=None, repr=False)
    _render: GridRenderState = field(default_factory=GridRenderState, repr=False)

    def clear(self, plotter: Any) -> None:
        """Remove grid actors from the plotter."""
        clear_grid_actors(plotter, self._render)

    def _update_bounds(
        self,
        plan_scene_z: np.ndarray,
        plan_energies_mev: np.ndarray,
    ) -> Bounds6:
        bounds, z_spec = scene_bounds_for_plan(
            float(self.x_min),
            float(self.x_max),
            float(self.y_min),
            float(self.y_max),
            plan_scene_z,
            plan_energies_mev,
            self.z_display_cfg,
            sanity=self.sanity,
        )
        self.z_spec = z_spec
        self.scene_bounds = bounds
        return bounds

    def refresh(self, plotter: Any) -> None:
        """Re-draw grid from current scene bounds."""
        if not self.ready or self.scene_bounds is None:
            return
        frame = SceneFrame.from_bounds6(self.scene_bounds)
        plan = plan_xy_grid(
            frame,
            label_pad_mm=float(self.style.label_endpoint_pad_mm),
            label_format=str(self.style.label_format),
            minor_tick_mm=float(self.style.minor_tick_mm),
        )
        render_grid_plan(plotter, plan, style=self.style, state=self._render)

    def show(
        self,
        plotter: Any,
        plan_scene_z: np.ndarray,
        plan_energies_mev: np.ndarray,
        *,
        force: bool = False,
    ) -> None:
        """Create or refresh the scene grid for the full plan extent."""
        self._update_bounds(plan_scene_z, plan_energies_mev)
        if force or self._render.major_line_actor is None:
            self.clear(plotter)
        self.refresh(plotter)
        try:
            plotter.render()
        except Exception:
            pass

    def camera_bounds(self) -> Bounds6 | None:
        return self.scene_bounds
