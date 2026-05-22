"""PyVista cube axes for the comparison 3D view — single owner for bounds, ticks, and VTK guard.

Scene Z mapping (MeV / water depth → plot coordinates) lives in :mod:`spot_check.geometry.z_axis`.
This module owns everything that touches ``CubeAxesActor``: ``show_bounds``, Z label range, grid
style, and the ``update_bounds_axes`` guard.

Production setup matches ``scripts/cube_axes_10_cube_test.py``:
- ``bounds`` and ``axes_ranges`` are the **same** scene 6-tuple passed to ``show_bounds``.
- ``mesh=None``, ``location='outer'``, ``padding=0``.
- Inverted depth/MeV tick text is applied only via ``z_axis_range`` (no ``SetAxisLabels``).
- ``CubeAxesActor.bounds = …`` resets ``z_axis_range`` to ascending scene Z; guard + ``refresh``
  restore the inverted label range.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from spot_check.geometry.z_axis import cube_z_axis_spec_for_display
from spot_check.models import CubeZAxisSpec, ZAxisDisplayConfig

# vtkAxisActor rejects 0 / negative / huge label counts (often logged as -2147483647).
_CUBE_AXIS_LABEL_COUNT_MIN = 2
_CUBE_AXIS_LABEL_COUNT_MAX = 64

# Outer edges (confirmed on bare 10³ cube test). ``padding`` must stay 0: PyVista
# expands ``bounds`` but not ``axes_ranges``, which skews ticks on the box.
PYVISTA_CUBE_AXES_LOCATION: str = "outer"
PYVISTA_CUBE_AXES_GRID: str = "back"
PYVISTA_CUBE_AXES_TICKS: str = "inside"
PYVISTA_CUBE_AXES_PADDING: float = 0.0
PYVISTA_CUBE_Z_TICK_LABEL_ORIENTATION: float = 90.0
PYVISTA_CUBE_AXES_LABEL_OFFSET: float = 36.0
CUBE_AXES_LABEL_FORMAT: str = "%.0f"
# Alias kept for callers/tests written during the label-format study.
PYVISTA_CUBE_Z_LABEL_FORMAT: str = CUBE_AXES_LABEL_FORMAT

Bounds6 = tuple[float, float, float, float, float, float]


def normalize_cube_axes_label_counts(actor: Any) -> None:
    """Clamp ``n_{x,y,z}labels`` before PyVista regenerates ticks."""
    for name, dflt in (("n_xlabels", 6), ("n_ylabels", 6), ("n_zlabels", 6)):
        if not hasattr(actor, name):
            continue
        try:
            cur = int(getattr(actor, name))
        except (TypeError, ValueError):
            cur = dflt
        if cur < _CUBE_AXIS_LABEL_COUNT_MIN or cur > _CUBE_AXIS_LABEL_COUNT_MAX:
            cur = dflt
        try:
            setattr(actor, name, cur)
        except Exception:
            pass


def _pinned_axis_tick_values(vmin: float, vmax: float, n: int) -> list[float]:
    """Evenly spaced ticks with exact ``vmin`` / ``vmax`` at the ends (PyVista linspace drift)."""
    vmin = float(vmin)
    vmax = float(vmax)
    n = int(n)
    if n < _CUBE_AXIS_LABEL_COUNT_MIN:
        n = _CUBE_AXIS_LABEL_COUNT_MIN
    if n > _CUBE_AXIS_LABEL_COUNT_MAX:
        n = _CUBE_AXIS_LABEL_COUNT_MAX
    if not math.isfinite(vmin) or not math.isfinite(vmax):
        return []
    if vmin == vmax:
        return [float(vmin)] * n
    vals = np.linspace(vmin, vmax, n, dtype=np.float64)
    vals[0] = vmin
    vals[-1] = vmax
    return [float(x) for x in vals]


def _format_axis_tick_string(value: float, fmt_s: str) -> str:
    if fmt_s.startswith("%"):
        return fmt_s % value
    if fmt_s:
        return fmt_s.format(value)
    return str(value)


def disable_pyvista_cube_axes_label_lod(actor: Any) -> None:
    """VTK hides axis labels at some view angles unless distance/view-angle LOD is off."""
    try:
        actor.SetEnableViewAngleLOD(False)
        actor.SetEnableDistanceLOD(False)
    except Exception:
        pass


def _set_cube_axes_fly_mode(actor: Any, location: str) -> None:
    loc = str(location).lower()
    if loc in ("all",):
        actor.SetFlyModeToStaticEdges()
    elif loc in ("origin",):
        actor.SetFlyModeToStaticTriad()
    elif loc in ("outer",):
        actor.SetFlyModeToOuterEdges()
    elif loc in ("default", "closest", "front"):
        actor.SetFlyModeToClosestTriad()
    elif loc in ("furthest", "back"):
        actor.SetFlyModeToFurthestTriad()
    else:
        actor.SetFlyModeToOuterEdges()


def apply_pyvista_cube_axes_style(actor: Any) -> None:
    """Fly mode (outer), grid, and no label LOD — matches bare ``show_bounds`` setup."""
    _set_cube_axes_fly_mode(actor, PYVISTA_CUBE_AXES_LOCATION)
    grid_loc = getattr(actor, "VTK_GRID_LINES_FURTHEST", None)
    if grid_loc is not None and hasattr(actor, "SetGridLineLocation"):
        actor.SetGridLineLocation(grid_loc)
    actor.SetDrawXGridlines(True)
    actor.SetDrawYGridlines(True)
    actor.SetDrawZGridlines(True)
    normalize_cube_axes_label_counts(actor)
    disable_pyvista_cube_axes_label_lod(actor)


def _apply_z_label_range(actor: Any, z_spec: CubeZAxisSpec) -> None:
    """Invert Z tick numbers via ``z_axis_range`` (deep / high MeV at scene zmin)."""
    normalize_cube_axes_label_counts(actor)
    disable_pyvista_cube_axes_label_lod(actor)
    try:
        actor.z_label_format = PYVISTA_CUBE_Z_LABEL_FORMAT
        actor.z_label_visibility = True
        actor.x_label_visibility = True
        actor.y_label_visibility = True
    except Exception:
        pass
    try:
        actor.z_axis_range = (
            float(z_spec.z_label_at_min),
            float(z_spec.z_label_at_max),
        )
    except Exception:
        pass


def _apply_inverted_scene_z_label_range(
    actor: Any,
    *,
    z_scene_min: float,
    z_scene_max: float,
) -> None:
    """Fallback: invert scene-coordinate Z labels without ``SetAxisLabels``."""
    lo = float(min(z_scene_min, z_scene_max))
    hi = float(max(z_scene_min, z_scene_max))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return
    normalize_cube_axes_label_counts(actor)
    disable_pyvista_cube_axes_label_lod(actor)
    try:
        actor.z_label_format = PYVISTA_CUBE_Z_LABEL_FORMAT
        actor.z_label_visibility = True
    except Exception:
        pass
    try:
        actor.z_axis_range = (hi, lo)
    except Exception:
        pass


def _restore_plan_cube_axes(
    actor: Any,
    bounds: Bounds6,
    *,
    z_spec: CubeZAxisSpec | None = None,
) -> None:
    """Restore full-plan scene bounds and inverted Z labels after VTK mutates the actor."""
    normalize_cube_axes_label_counts(actor)
    disable_pyvista_cube_axes_label_lod(actor)
    try:
        actor.z_label_format = PYVISTA_CUBE_Z_LABEL_FORMAT
        actor.x_label_format = CUBE_AXES_LABEL_FORMAT
        actor.y_label_format = CUBE_AXES_LABEL_FORMAT
        actor.z_label_visibility = True
        actor.x_label_visibility = True
        actor.y_label_visibility = True
    except Exception:
        pass
    try:
        if hasattr(actor, "bounds"):
            actor.bounds = bounds
        else:
            actor.SetBounds(bounds)
    except Exception:
        try:
            actor.SetBounds(bounds)
        except Exception:
            return
    if z_spec is not None:
        _apply_z_label_range(actor, z_spec)
    else:
        _apply_inverted_scene_z_label_range(
            actor,
            z_scene_min=float(bounds[4]),
            z_scene_max=float(bounds[5]),
        )


def _pin_xy_tick_endpoints(actor: Any) -> None:
    """Legacy XY pin via ``SetAxisLabels`` — tests/scripts only; not used in production."""
    normalize_cube_axes_label_counts(actor)
    try:
        from pyvista import _vtk
    except Exception:
        return

    def _one(axis: int, n_name: str, fmt_name: str, range_name: str) -> None:
        try:
            vr = getattr(actor, range_name)
            vmin, vmax = float(vr[0]), float(vr[1])
        except Exception:
            return
        if not math.isfinite(vmin) or not math.isfinite(vmax) or vmin == vmax:
            return
        try:
            n_lbl = int(getattr(actor, n_name, 6))
        except Exception:
            n_lbl = 6
        if n_lbl < _CUBE_AXIS_LABEL_COUNT_MIN or n_lbl > _CUBE_AXIS_LABEL_COUNT_MAX:
            n_lbl = 6
        vals = _pinned_axis_tick_values(vmin, vmax, n_lbl)
        if len(vals) < 2:
            return
        fmt_s = str(getattr(actor, fmt_name, None) or CUBE_AXES_LABEL_FORMAT)
        labels = _vtk.vtkStringArray()
        for v in vals:
            labels.InsertNextValue(_format_axis_tick_string(v, fmt_s))
        try:
            actor.SetAxisLabels(axis, labels)
        except Exception:
            return

    _one(0, "n_xlabels", "x_label_format", "x_axis_range")
    _one(1, "n_ylabels", "y_label_format", "y_axis_range")


def _invert_z_tick_labels(actor: Any, *, z_scene_min: float, z_scene_max: float) -> None:
    """Legacy Z invert via ``SetAxisLabels`` — tests/scripts only."""
    z0 = float(z_scene_min)
    z1 = float(z_scene_max)
    lo = float(min(z0, z1))
    hi = float(max(z0, z1))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return
    try:
        from pyvista import _vtk
    except Exception:
        return
    normalize_cube_axes_label_counts(actor)
    disable_pyvista_cube_axes_label_lod(actor)
    try:
        actor.z_label_visibility = True
    except Exception:
        pass
    try:
        n_lbl = int(getattr(actor, "n_zlabels", 6))
        if n_lbl < _CUBE_AXIS_LABEL_COUNT_MIN or n_lbl > _CUBE_AXIS_LABEL_COUNT_MAX:
            n_lbl = 6
        vals = _pinned_axis_tick_values(z0, z1, n_lbl)
        if len(vals) < _CUBE_AXIS_LABEL_COUNT_MIN:
            return
        labels = _vtk.vtkStringArray()
        for v in vals:
            inv = float(lo + hi - float(v))
            labels.InsertNextValue(_format_axis_tick_string(inv, PYVISTA_CUBE_Z_LABEL_FORMAT))
        if labels.GetNumberOfValues() < _CUBE_AXIS_LABEL_COUNT_MIN:
            return
        actor.SetAxisLabels(2, labels)
    except Exception:
        return


def _finalize_cube_axes_after_show_bounds(
    actor: Any,
    *,
    z_spec: CubeZAxisSpec | None,
    apply_style: bool = True,
) -> None:
    """Style + inverted Z labels after ``show_bounds`` (``bounds == axes_ranges`` already set)."""
    if apply_style:
        apply_pyvista_cube_axes_style(actor)
    if z_spec is not None:
        _apply_z_label_range(actor, z_spec)


def heal_plan_cube_axes(
    actor: Any,
    bounds: Bounds6,
    *,
    z_spec: CubeZAxisSpec | None = None,
    axes_ranges: Bounds6 | None = None,
    apply_style: bool = False,
) -> None:
    """Restore full-plan bounds and Z label range after VTK mutates the cube axes actor."""
    if apply_style:
        apply_pyvista_cube_axes_style(actor)
    _restore_plan_cube_axes(actor, bounds, z_spec=z_spec)


# --- Back-compat names (tests + scripts) -----------------------------------


def pin_xy_cube_axis_tick_endpoints(actor: Any) -> None:
    _pin_xy_tick_endpoints(actor)


def invert_z_cube_axis_tick_labels(
    actor: Any,
    *,
    z_scene_min: float,
    z_scene_max: float,
) -> None:
    _invert_z_tick_labels(actor, z_scene_min=z_scene_min, z_scene_max=z_scene_max)


def pin_pyvista_cube_bounds(actor: Any, bounds: Bounds6) -> None:
    heal_plan_cube_axes(actor, bounds, apply_style=False)


def apply_plan_cube_axis_labels(
    actor: Any,
    bounds: Bounds6,
    *,
    z_spec: CubeZAxisSpec | None = None,
    apply_style: bool = False,
) -> None:
    heal_plan_cube_axes(actor, bounds, z_spec=z_spec, apply_style=apply_style)


def refresh_pyvista_cube_axes(
    actor: Any,
    bounds: Bounds6,
    axes_ranges: Bounds6,
    *,
    z_spec: CubeZAxisSpec | None = None,
) -> None:
    heal_plan_cube_axes(actor, bounds, z_spec=z_spec, apply_style=False)


def cube_axes_ranges(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_spec: CubeZAxisSpec,
) -> Bounds6:
    """Scene ``axes_ranges`` for ``show_bounds`` (identical to ``bounds``)."""
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
    """Return identical ``(bounds, axes_ranges)`` in scene coordinates for ``show_bounds``."""
    scene_bounds = cube_axes_ranges(x_min, x_max, y_min, y_max, z_spec)
    return scene_bounds, scene_bounds


def pyvista_show_bounds_kwargs() -> dict[str, Any]:
    return {
        "location": PYVISTA_CUBE_AXES_LOCATION,
        "padding": float(PYVISTA_CUBE_AXES_PADDING),
    }


def z_labels_inverted(actor: Any) -> bool:
    """True when Z tick strings run high→low (deep / high MeV at scene zmin)."""
    try:
        zl = [float(actor.z_labels[i]) for i in range(len(actor.z_labels))]
    except Exception:
        return False
    return len(zl) >= 2 and zl[0] > zl[-1]


# --- Plotter-facing controller ------------------------------------------------


@dataclass
class PlanCubeAxesController:
    """Owns comparison-view cube axes state, ``show_bounds``, guard, and Z label restore."""

    xlab: str
    ylab: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_display_cfg: ZAxisDisplayConfig
    sanity: bool = False
    ready: bool = False
    z_spec: CubeZAxisSpec | None = field(default=None, repr=False)
    scene_bounds: Bounds6 | None = field(default=None, repr=False)
    axes_ranges: Bounds6 | None = field(default=None, repr=False)
    actor: Any = field(default=None, repr=False)
    _guard_installed: bool = field(default=False, repr=False)

    @staticmethod
    def detach_guard_on_plotter(plotter: Any) -> None:
        """Restore VTK ``update_bounds_axes`` before plotter reuse (no controller instance yet)."""
        orig_pl = getattr(plotter, "_spot_check_orig_update_bounds_axes", None)
        orig_ren = getattr(plotter, "_spot_check_orig_renderer_update_bounds_axes", None)
        if orig_pl is not None:
            plotter.update_bounds_axes = orig_pl
        if orig_ren is not None:
            plotter.renderer.update_bounds_axes = orig_ren
        plotter._spot_check_cube_axes_guard = False

    def detach_guard(self, plotter: Any) -> None:
        """Restore VTK ``update_bounds_axes`` before plotter reuse."""
        PlanCubeAxesController.detach_guard_on_plotter(plotter)
        self._guard_installed = False

    def install_guard(self, plotter: Any) -> None:
        """Heal cube axes after VTK ``update_bounds_axes`` (visible-mesh shrink)."""
        if self._guard_installed or getattr(plotter, "_spot_check_cube_axes_guard", False):
            self._guard_installed = True
            return
        orig_pl = plotter.update_bounds_axes
        orig_ren = plotter.renderer.update_bounds_axes
        ctrl = self

        def _after_update() -> None:
            ctrl.refresh(plotter, apply_style=False)

        def _guarded_pl() -> None:
            orig_pl()
            _after_update()

        def _guarded_ren() -> None:
            orig_ren()
            _after_update()

        plotter._spot_check_orig_update_bounds_axes = orig_pl
        plotter._spot_check_orig_renderer_update_bounds_axes = orig_ren
        plotter.update_bounds_axes = _guarded_pl
        plotter.renderer.update_bounds_axes = _guarded_ren
        plotter._spot_check_cube_axes_guard = True
        self._guard_installed = True

    def _bounds_for_plan(
        self,
        plan_scene_z: np.ndarray,
        plan_energies_mev: np.ndarray,
    ) -> tuple[Bounds6, str, int]:
        if self.sanity:
            self.z_spec = None
            box: Bounds6 = (0.0, 10.0, 0.0, 10.0, 0.0, 10.0)
            return box, "Z", 6
        z_spec = cube_z_axis_spec_for_display(
            np.asarray(plan_scene_z, dtype=np.float64).reshape(-1),
            np.asarray(plan_energies_mev, dtype=np.float64).reshape(-1),
            self.z_display_cfg,
        )
        self.z_spec = z_spec
        scene_bounds, _axes = plan_cube_scene_bounds_and_axes_ranges(
            float(self.x_min),
            float(self.x_max),
            float(self.y_min),
            float(self.y_max),
            z_spec,
        )
        return scene_bounds, str(z_spec.ztitle), int(z_spec.n_zlabels)

    def refresh(self, plotter: Any, *, apply_style: bool = False) -> None:
        """Re-apply scene bounds and inverted Z labels without rebuilding ``show_bounds``."""
        if not self.ready or self.sanity:
            return
        actor = plotter.renderer.cube_axes_actor
        if actor is None or self.scene_bounds is None:
            return
        heal_plan_cube_axes(
            actor,
            self.scene_bounds,
            z_spec=self.z_spec,
            apply_style=apply_style,
        )
        self.actor = actor

    def show(
        self,
        plotter: Any,
        plan_scene_z: np.ndarray,
        plan_energies_mev: np.ndarray,
        *,
        force: bool = False,
    ) -> None:
        """Create or refresh cube axes for the full plan Z extent."""
        scene_bounds, ztitle, n_z = self._bounds_for_plan(plan_scene_z, plan_energies_mev)
        prev_bounds = self.scene_bounds
        self.scene_bounds = scene_bounds
        self.axes_ranges = scene_bounds
        actor = plotter.renderer.cube_axes_actor
        if (
            not force
            and actor is not None
            and prev_bounds is not None
            and tuple(scene_bounds) == tuple(prev_bounds)
        ):
            self.refresh(plotter, apply_style=False)
            return
        plotter.show_bounds(
            mesh=None,
            bounds=scene_bounds,
            axes_ranges=scene_bounds,
            location=PYVISTA_CUBE_AXES_LOCATION,
            padding=PYVISTA_CUBE_AXES_PADDING,
            grid=PYVISTA_CUBE_AXES_GRID,
            ticks=PYVISTA_CUBE_AXES_TICKS,
            xtitle=self.xlab,
            ytitle=self.ylab,
            ztitle=ztitle,
            n_xlabels=6,
            n_ylabels=6,
            n_zlabels=n_z,
            fmt=CUBE_AXES_LABEL_FORMAT,
            color="white",
        )
        actor = plotter.renderer.cube_axes_actor
        self.actor = actor
        if actor is not None:
            if self.sanity:
                apply_pyvista_cube_axes_style(actor)
            else:
                _finalize_cube_axes_after_show_bounds(
                    actor,
                    z_spec=self.z_spec,
                    apply_style=True,
                )
        try:
            plotter.render()
        except Exception:
            pass

    def camera_bounds(self) -> Bounds6 | None:
        return self.scene_bounds
