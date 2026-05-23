"""Pure grid planning — tick/line layout without VTK."""

from __future__ import annotations

from spot_check.constants import (
    SCENE_GRID_LABEL_ENDPOINT_PAD_MM,
    SCENE_GRID_LABEL_FORMAT,
    SCENE_GRID_MINOR_TICK_MM,
)

from .label_format import format_grid_label
from .label_layout import anchor_beyond_line_end, anchor_padded_xy, segment_endpoint_labels
from .tick_math import bounds_expanded_to_tick_step, tick_values_centered_on_zero
from .types import AxisLineSpec, GridPlan, LabelAnchor, SceneFrame


def xy_grid_extent(
    frame: SceneFrame,
    *,
    tick_mm: float = SCENE_GRID_MINOR_TICK_MM,
) -> tuple[float, float, float, float]:
    """XY bounds expanded outward to tick-step multiples for a symmetric grid."""
    x0, x1 = bounds_expanded_to_tick_step(frame.x_min, frame.x_max, tick_mm)
    y0, y1 = bounds_expanded_to_tick_step(frame.y_min, frame.y_max, tick_mm)
    return x0, x1, y0, y1


def xy_boundary_perimeter(
    frame: SceneFrame,
    *,
    tick_mm: float = SCENE_GRID_MINOR_TICK_MM,
) -> tuple[AxisLineSpec, ...]:
    """Minor lines: XY rectangle on the scene z-min plane."""
    z = float(frame.z_min)
    x0, x1, y0, y1 = xy_grid_extent(frame, tick_mm=tick_mm)
    bl = (x0, y0, z)
    br = (x1, y0, z)
    tr = (x1, y1, z)
    tl = (x0, y1, z)
    return (
        AxisLineSpec(start=bl, end=br),
        AxisLineSpec(start=br, end=tr),
        AxisLineSpec(start=tr, end=tl),
        AxisLineSpec(start=tl, end=bl),
    )


def xy_boundary_labels(
    frame: SceneFrame,
    edges: tuple[AxisLineSpec, ...],
    *,
    tick_mm: float = SCENE_GRID_MINOR_TICK_MM,
    label_pad_mm: float = SCENE_GRID_LABEL_ENDPOINT_PAD_MM,
    label_format: str = SCENE_GRID_LABEL_FORMAT,
) -> tuple[LabelAnchor, ...]:
    """Perimeter labels padded outward normal to each edge (same as tick grid ends)."""
    if len(edges) != 4:
        raise ValueError("xy_boundary_labels expects four perimeter edges")
    z = float(frame.z_min)
    x0, x1, y0, y1 = xy_grid_extent(frame, tick_mm=tick_mm)
    pad = float(label_pad_mm)
    fmt = label_format
    bl = (x0, y0, z)
    br = (x1, y0, z)
    tr = (x1, y1, z)
    tl = (x0, y1, z)
    return (
        LabelAnchor(anchor_padded_xy(bl, pad_y_mm=-pad), format_grid_label(x0, fmt)),
        LabelAnchor(anchor_padded_xy(br, pad_y_mm=-pad), format_grid_label(x1, fmt)),
        LabelAnchor(anchor_padded_xy(br, pad_x_mm=pad), format_grid_label(y0, fmt)),
        LabelAnchor(anchor_padded_xy(tr, pad_x_mm=pad), format_grid_label(y1, fmt)),
        LabelAnchor(anchor_padded_xy(tr, pad_y_mm=pad), format_grid_label(x1, fmt)),
        LabelAnchor(anchor_padded_xy(tl, pad_y_mm=pad), format_grid_label(x0, fmt)),
        LabelAnchor(anchor_padded_xy(tl, pad_x_mm=-pad), format_grid_label(y1, fmt)),
        LabelAnchor(anchor_padded_xy(bl, pad_x_mm=-pad), format_grid_label(y0, fmt)),
    )


def xy_tick_grid_lines_and_labels(
    frame: SceneFrame,
    *,
    tick_mm: float = SCENE_GRID_MINOR_TICK_MM,
    label_pad_mm: float = SCENE_GRID_LABEL_ENDPOINT_PAD_MM,
    label_format: str = SCENE_GRID_LABEL_FORMAT,
) -> tuple[tuple[AxisLineSpec, ...], tuple[LabelAnchor, ...]]:
    """Minor lines at ``±tick_mm``, ``±2*tick_mm``, … centered on 0, with endpoint labels."""
    z = float(frame.z_min)
    x0, x1, y0, y1 = xy_grid_extent(frame, tick_mm=tick_mm)
    fmt = label_format

    lines: list[AxisLineSpec] = []
    labels: list[LabelAnchor] = []

    for x_tick in tick_values_centered_on_zero(x0, x1, tick_mm):
        start = (x_tick, y0, z)
        end = (x_tick, y1, z)
        segment = AxisLineSpec(start=start, end=end)
        lines.append(segment)
        tick_text = format_grid_label(x_tick, fmt)
        labels.extend(
            segment_endpoint_labels(segment, tick_text, tick_text, label_pad_mm),
        )

    for y_tick in tick_values_centered_on_zero(y0, y1, tick_mm):
        start = (x0, y_tick, z)
        end = (x1, y_tick, z)
        segment = AxisLineSpec(start=start, end=end)
        lines.append(segment)
        tick_text = format_grid_label(y_tick, fmt)
        labels.extend(
            segment_endpoint_labels(segment, tick_text, tick_text, label_pad_mm),
        )

    return tuple(lines), tuple(labels)


def xy_zero_axes(
    frame: SceneFrame,
    *,
    tick_mm: float = SCENE_GRID_MINOR_TICK_MM,
    label_pad_mm: float = SCENE_GRID_LABEL_ENDPOINT_PAD_MM,
    label_format: str = SCENE_GRID_LABEL_FORMAT,
) -> GridPlan:
    """Major lines: x=0 and y=0 on the scene z-min plane with ``0`` at both ends."""
    z = float(frame.z_min)
    x0, x1, y0, y1 = xy_grid_extent(frame, tick_mm=tick_mm)

    x_start = (0.0, y0, z)
    x_end = (0.0, y1, z)
    y_start = (x0, 0.0, z)
    y_end = (x1, 0.0, z)

    x_zero = AxisLineSpec(start=x_start, end=x_end)
    y_zero = AxisLineSpec(start=y_start, end=y_end)
    zero_text = format_grid_label(0.0, label_format)

    labels = (
        LabelAnchor(
            position=anchor_beyond_line_end(x_start, x_end, label_pad_mm),
            text=zero_text,
        ),
        LabelAnchor(
            position=anchor_beyond_line_end(x_end, x_start, label_pad_mm),
            text=zero_text,
        ),
        LabelAnchor(
            position=anchor_beyond_line_end(y_start, y_end, label_pad_mm),
            text=zero_text,
        ),
        LabelAnchor(
            position=anchor_beyond_line_end(y_end, y_start, label_pad_mm),
            text=zero_text,
        ),
    )
    return GridPlan(major_lines=(x_zero, y_zero), labels=labels)


def plan_xy_grid(
    frame: SceneFrame,
    *,
    label_pad_mm: float = SCENE_GRID_LABEL_ENDPOINT_PAD_MM,
    label_format: str = SCENE_GRID_LABEL_FORMAT,
    minor_tick_mm: float = SCENE_GRID_MINOR_TICK_MM,
) -> GridPlan:
    """XY major zero axes, minor perimeter, and 10 mm tick grid centered on 0."""
    zero = xy_zero_axes(
        frame,
        tick_mm=minor_tick_mm,
        label_pad_mm=label_pad_mm,
        label_format=label_format,
    )
    edges = xy_boundary_perimeter(frame, tick_mm=minor_tick_mm)
    boundary_labels = xy_boundary_labels(
        frame,
        edges,
        tick_mm=minor_tick_mm,
        label_pad_mm=label_pad_mm,
        label_format=label_format,
    )
    tick_lines, tick_labels = xy_tick_grid_lines_and_labels(
        frame,
        tick_mm=minor_tick_mm,
        label_pad_mm=label_pad_mm,
        label_format=label_format,
    )
    return GridPlan(
        major_lines=zero.major_lines,
        minor_lines=edges + tick_lines,
        labels=zero.labels + boundary_labels + tick_labels,
    )
