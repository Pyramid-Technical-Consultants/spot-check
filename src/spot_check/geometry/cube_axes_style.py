"""PyVista cube-axes fly mode, grid, ticks, and padding for 3D bounds."""

from __future__ import annotations

import math
from typing import Any

from spot_check.models import CubeZAxisSpec

# vtkAxisActor rejects 0 / negative / huge label counts (often logged as -2147483647).
_CUBE_AXIS_LABEL_COUNT_MIN = 2
_CUBE_AXIS_LABEL_COUNT_MAX = 64


def normalize_cube_axes_label_counts(actor: Any) -> None:
    """Clamp ``n_{x,y,z}labels`` before PyVista regenerates ticks (0 → empty ``SetAxisLabels``)."""
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
    try:
        import numpy as np
    except Exception:
        if n < 2:
            return [vmin, vmax]
        return [
            float(vmin + (vmax - vmin) * i / (n - 1)) for i in range(n)
        ]
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


def pin_xy_cube_axis_tick_endpoints(actor: Any) -> None:
    """Pin first/last X and Y tick labels to ``x_axis_range`` / ``y_axis_range`` endpoints."""
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
        fmt_s = str(getattr(actor, fmt_name, None) or "%.0f")
        labels = _vtk.vtkStringArray()
        for v in vals:
            labels.InsertNextValue(_format_axis_tick_string(v, fmt_s))
        try:
            actor.SetAxisLabels(axis, labels)
        except Exception:
            return

    _one(0, "n_xlabels", "x_label_format", "x_axis_range")
    _one(1, "n_ylabels", "y_label_format", "y_axis_range")


# Outer edges (confirmed on bare 10³ cube test). ``padding`` must stay 0: PyVista
# expands ``bounds`` but not ``axes_ranges``, which skews ticks on the box.
PYVISTA_CUBE_AXES_LOCATION: str = "outer"
PYVISTA_CUBE_AXES_GRID: str = "back"
PYVISTA_CUBE_AXES_TICKS: str = "inside"
PYVISTA_CUBE_AXES_PADDING: float = 0.0
PYVISTA_CUBE_Z_TICK_LABEL_ORIENTATION: float = 90.0
PYVISTA_CUBE_AXES_LABEL_OFFSET: float = 36.0


def cube_axes_ranges(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_spec: CubeZAxisSpec,
) -> tuple[float, float, float, float, float, float]:
    """Z tick corners in depth/mm (or MeV) for a *split* ``axes_ranges`` tuple.

    The 3D plotter uses ``bounds == axes_ranges`` (scene Z on all corners) like
    ``scripts/cube_axes_10_cube_test.py`` so VTK ticks stay aligned; this helper remains for
    callers that need explicit label endpoints vs scene bounds.
    """
    return (
        float(x_min),
        float(x_max),
        float(y_min),
        float(y_max),
        float(z_spec.z_label_at_min),
        float(z_spec.z_label_at_max),
    )


def pin_pyvista_cube_bounds(
    actor: Any,
    bounds: tuple[float, float, float, float, float, float],
) -> None:
    """Re-pin full-plan cube bounds after VTK ``update_bounds_axes``."""
    if hasattr(actor, "bounds"):
        actor.bounds = bounds
    else:
        actor.SetBounds(bounds)
    normalize_cube_axes_label_counts(actor)
    try:
        actor.z_label_visibility = True
        actor.x_label_visibility = True
        actor.y_label_visibility = True
    except Exception:
        pass
    disable_pyvista_cube_axes_label_lod(actor)
    pin_xy_cube_axis_tick_endpoints(actor)


def invert_z_cube_axis_tick_labels(
    actor: Any,
    *,
    z_scene_min: float,
    z_scene_max: float,
) -> None:
    """Put larger tick numbers toward the global origin end of the Z axis (labels only).

    Scene ``bounds`` and ``z_axis_range`` stay ordered as VTK expects. Tick **positions** follow
    ``(z_scene_min, z_scene_max)`` with exact endpoints; label text uses
    ``min+max-v`` per tick (inverted display).
    """
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
    # LOD must be off before custom labels: ``SetEnable*LOD`` rebuild can wipe ``SetAxisLabels``.
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
        fmt_s = str(getattr(actor, "z_label_format", None) or "%.0f")
        labels = _vtk.vtkStringArray()
        for v in vals:
            inv = float(lo + hi - float(v))
            labels.InsertNextValue(_format_axis_tick_string(inv, fmt_s))
        if labels.GetNumberOfValues() < _CUBE_AXIS_LABEL_COUNT_MIN:
            return
        actor.SetAxisLabels(2, labels)
    except Exception:
        return


def refresh_pyvista_cube_axes(
    actor: Any,
    bounds: tuple[float, float, float, float, float, float],
    axes_ranges: tuple[float, float, float, float, float, float],
) -> None:
    """Re-pin bounds and axis ranges (unit tests); plotter uses :func:`pin_pyvista_cube_bounds`."""
    if hasattr(actor, "bounds"):
        actor.bounds = bounds
    else:
        actor.SetBounds(bounds)
    actor.x_axis_range = float(axes_ranges[0]), float(axes_ranges[1])
    actor.y_axis_range = float(axes_ranges[2]), float(axes_ranges[3])
    normalize_cube_axes_label_counts(actor)
    try:
        actor.z_label_visibility = True
    except Exception:
        pass
    disable_pyvista_cube_axes_label_lod(actor)
    pin_xy_cube_axis_tick_endpoints(actor)
    invert_z_cube_axis_tick_labels(
        actor, z_scene_min=float(bounds[4]), z_scene_max=float(bounds[5])
    )


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


def disable_pyvista_cube_axes_label_lod(actor: Any) -> None:
    """VTK hides axis labels at some view angles unless distance/view-angle LOD is off."""
    try:
        actor.SetEnableViewAngleLOD(False)
        actor.SetEnableDistanceLOD(False)
    except Exception:
        pass


def pyvista_show_bounds_kwargs() -> dict[str, Any]:
    """Bare ``show_bounds`` options (matches ``scripts/cube_axes_10_cube_test.py``)."""
    return {
        "location": PYVISTA_CUBE_AXES_LOCATION,
        "padding": float(PYVISTA_CUBE_AXES_PADDING),
    }


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
