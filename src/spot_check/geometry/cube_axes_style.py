"""PyVista cube-axes fly mode, grid, ticks, and padding for 3D bounds."""

from __future__ import annotations

from typing import Any

from spot_check.models import CubeZAxisSpec

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
    """``axes_ranges`` for :meth:`pyvista.Plotter.show_bounds` (scene XY, Z tick corners)."""
    return (
        float(x_min),
        float(x_max),
        float(y_min),
        float(y_max),
        float(z_spec.z_label_at_min),
        float(z_spec.z_label_at_max),
    )


def refresh_pyvista_cube_axes(
    actor: Any,
    bounds: tuple[float, float, float, float, float, float],
    axes_ranges: tuple[float, float, float, float, float, float],
) -> None:
    """Re-pin full-plan bounds and Z corner labels after VTK ``update_bounds_axes``.

    Use the PyVista ``bounds`` property (not raw ``SetBounds``) so tick strings regenerate.
    ``axes_ranges`` Z may differ from scene ``bounds`` (positive MeV/mm at corners).
    """
    if hasattr(actor, "bounds"):
        actor.bounds = bounds
    else:
        actor.SetBounds(bounds)
    actor.x_axis_range = float(axes_ranges[0]), float(axes_ranges[1])
    actor.y_axis_range = float(axes_ranges[2]), float(axes_ranges[3])
    actor.z_axis_range = float(axes_ranges[4]), float(axes_ranges[5])
    try:
        actor.z_label_visibility = True
    except Exception:
        pass
    disable_pyvista_cube_axes_label_lod(actor)


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
    disable_pyvista_cube_axes_label_lod(actor)
