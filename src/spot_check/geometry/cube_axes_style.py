"""PyVista cube-axes fly mode, grid, ticks, and padding for 3D bounds."""

from __future__ import annotations

from typing import Any

PYVISTA_CUBE_AXES_LOCATION: str = "outer"
PYVISTA_CUBE_AXES_GRID: str = "back"
PYVISTA_CUBE_AXES_TICKS: str = "inside"
PYVISTA_CUBE_AXES_PADDING: float = 0.06
# Degrees for vtkTextProperty on Z tick labels (vertical along the Z edge).
PYVISTA_CUBE_Z_TICK_LABEL_ORIENTATION: float = 90.0
# vtkCubeAxesActor default label offset is 20 px; 90° Z ticks need more standoff.
PYVISTA_CUBE_AXES_LABEL_OFFSET: float = 36.0


def pyvista_show_bounds_kwargs() -> dict[str, Any]:
    """Keyword args for :meth:`pyvista.Plotter.show_bounds` (except bounds/ranges)."""
    return {
        "location": PYVISTA_CUBE_AXES_LOCATION,
        "grid": PYVISTA_CUBE_AXES_GRID,
        "ticks": PYVISTA_CUBE_AXES_TICKS,
        "padding": float(PYVISTA_CUBE_AXES_PADDING),
    }


def apply_pyvista_cube_axes_style(actor: Any) -> None:
    """Apply fly mode and grid lines (used on create and on Z-axis refresh)."""
    actor.SetFlyModeToOuterEdges()
    grid_loc = getattr(actor, "VTK_GRID_LINES_FURTHEST", None)
    if grid_loc is not None and hasattr(actor, "SetGridLineLocation"):
        actor.SetGridLineLocation(grid_loc)
    actor.SetDrawXGridlines(True)
    actor.SetDrawYGridlines(True)
    actor.SetDrawZGridlines(True)
