"""Add/remove custom grid actors on a PyVista plotter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .geometry import label_anchor_points, line_segments_tube_mesh
from .types import GridPlan, GridStyle


@dataclass
class GridRenderState:
    """Tracks actors created for one grid draw."""

    major_line_actor: Any | None = field(default=None, repr=False)
    minor_line_actor: Any | None = field(default=None, repr=False)
    label_actors: tuple[Any, ...] = field(default_factory=tuple, repr=False)

    @property
    def label_actor(self) -> Any | None:
        """First label actor, for tests that only check presence."""
        return self.label_actors[0] if self.label_actors else None


def clear_grid_actors(plotter: Any, state: GridRenderState) -> None:
    """Remove tracked grid actors from the plotter."""
    actors = (state.major_line_actor, state.minor_line_actor, *state.label_actors)
    for actor in actors:
        if actor is None:
            continue
        try:
            plotter.remove_actor(actor)
        except Exception:
            pass
    state.major_line_actor = None
    state.minor_line_actor = None
    state.label_actors = ()


def _configure_surface_line_actor(actor: Any) -> None:
    """Tubed grid lines must not show points or wireframe extras."""
    if actor is None:
        return
    try:
        prop = actor.GetProperty()
        prop.SetPointSize(0)
        prop.SetRenderPointsAsSpheres(False)
        prop.SetRepresentationToSurface()
        if hasattr(prop, "SetVertexVisibility"):
            prop.SetVertexVisibility(False)
        if hasattr(prop, "SetEdgeVisibility"):
            prop.SetEdgeVisibility(False)
    except Exception:
        pass


def _add_line_mesh(
    plotter: Any,
    segments: tuple,
    *,
    style: GridStyle,
    opacity: float,
) -> Any | None:
    mesh = line_segments_tube_mesh(segments, radius_mm=float(style.line_tube_radius_mm))
    if int(mesh.n_points) <= 0:
        return None
    actor = plotter.add_mesh(
        mesh,
        color=style.color,
        opacity=float(opacity),
        smooth_shading=False,
        lighting=False,
        pickable=False,
    )
    _configure_surface_line_actor(actor)
    return actor


def _configure_billboard_text_property(tp: Any, *, style: GridStyle) -> None:
    r, g, b = style.color
    opacity = float(style.label_opacity)
    tp.SetFontSize(int(style.label_font_size))
    tp.SetColor(float(r), float(g), float(b))
    tp.SetOpacity(opacity)
    tp.SetBackgroundOpacity(0.0)
    if hasattr(tp, "SetJustificationToCentered"):
        tp.SetJustificationToCentered()
    elif hasattr(tp, "SetJustification"):
        tp.SetJustification(1)
    if hasattr(tp, "SetVerticalJustificationToCentered"):
        tp.SetVerticalJustificationToCentered()
    elif hasattr(tp, "SetVerticalJustification"):
        tp.SetVerticalJustification(1)
    if hasattr(tp, "SetFrame"):
        tp.SetFrame(0)
    if hasattr(tp, "SetFrameWidth"):
        tp.SetFrameWidth(0)


def _add_billboard_labels(
    plotter: Any,
    points: np.ndarray,
    texts: list[str],
    *,
    style: GridStyle,
) -> tuple[Any, ...]:
    """Billboard text only — no point glyphs at label anchors."""
    if points.shape[0] == 0:
        return ()
    try:
        from vtkmodules.vtkRenderingCore import vtkBillboardTextActor3D
    except ImportError:  # pragma: no cover
        from pyvista import _vtk

        vtkBillboardTextActor3D = _vtk.vtkBillboardTextActor3D

    actors: list[Any] = []
    for i, text in enumerate(texts):
        actor = vtkBillboardTextActor3D()
        actor.SetInput(str(text))
        actor.SetPosition(float(points[i, 0]), float(points[i, 1]), float(points[i, 2]))
        _configure_billboard_text_property(actor.GetTextProperty(), style=style)
        plotter.add_actor(actor)
        actors.append(actor)
    return tuple(actors)


def render_grid_plan(
    plotter: Any,
    plan: GridPlan,
    *,
    style: GridStyle | None = None,
    state: GridRenderState | None = None,
) -> GridRenderState:
    """Draw grid lines and billboard labels; returns updated render state."""
    st = state if state is not None else GridRenderState()
    clear_grid_actors(plotter, st)
    style = style or GridStyle()

    st.major_line_actor = _add_line_mesh(
        plotter,
        plan.major_lines,
        style=style,
        opacity=float(style.opacity),
    )
    st.minor_line_actor = _add_line_mesh(
        plotter,
        plan.minor_lines,
        style=style,
        opacity=float(style.minor_opacity),
    )

    pts, texts = label_anchor_points(
        plan.labels,
        below_plane_mm=float(style.label_below_plane_mm),
    )
    st.label_actors = _add_billboard_labels(plotter, pts, texts, style=style)
    return st
