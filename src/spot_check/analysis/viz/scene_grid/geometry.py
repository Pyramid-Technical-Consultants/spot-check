"""Build PyVista meshes from pure grid plans."""

from __future__ import annotations

from typing import Any

import numpy as np

from spot_check.analysis.pyvista_backend import pv

from .types import AxisLineSpec, GridPlan, LabelAnchor


def line_segments_polydata(segments: tuple[AxisLineSpec, ...]) -> Any:
    """Single polydata with one line cell per segment (no verts / faces)."""
    if not segments:
        return pv.PolyData(np.zeros((0, 3), dtype=np.float64))
    points: list[list[float]] = []
    lines: list[int] = []
    for spec in segments:
        i0 = len(points)
        points.append(list(spec.start))
        points.append(list(spec.end))
        lines.extend([2, i0, i0 + 1])
    mesh = pv.PolyData(np.asarray(points, dtype=np.float64))
    mesh.lines = np.asarray(lines, dtype=np.int64)
    return mesh


def line_segments_tube_mesh(segments: tuple[AxisLineSpec, ...], *, radius_mm: float) -> Any:
    """Thin surface tubes — avoids OpenGL wide-line square caps at segment ends."""
    wire = line_segments_polydata(segments)
    if int(wire.n_lines) <= 0:
        return wire
    radius = float(radius_mm)
    if radius <= 0.0:
        return wire
    return wire.tube(radius=radius, n_sides=6, capping=False)


def axis_lines_polydata(plan: GridPlan) -> Any:
    """Polydata for all major and minor line segments."""
    return line_segments_polydata(plan.major_lines + plan.minor_lines)


def label_anchor_points(
    labels: tuple[LabelAnchor, ...],
    *,
    below_plane_mm: float = 0.0,
) -> tuple[np.ndarray, list[str]]:
    """World positions and strings for billboard labels."""
    if not labels:
        return np.zeros((0, 3), dtype=np.float64), []
    pts = np.asarray([a.position for a in labels], dtype=np.float64)
    drop = float(below_plane_mm)
    if drop != 0.0:
        pts = pts.copy()
        pts[:, 2] -= drop
    texts = [a.text for a in labels]
    return pts, texts
