"""Label placement relative to grid lines (pure geometry)."""

from __future__ import annotations

import math

from .types import AxisLineSpec, LabelAnchor


def anchor_beyond_line_end(
    endpoint: tuple[float, float, float],
    toward: tuple[float, float, float],
    pad_mm: float,
) -> tuple[float, float, float]:
    """Return a label anchor ``pad_mm`` past ``endpoint``, away from ``toward``.

    ``toward`` is the other end of the line segment (interior direction).
    """
    pad = float(pad_mm)
    if pad <= 0.0:
        return endpoint
    dx = float(endpoint[0]) - float(toward[0])
    dy = float(endpoint[1]) - float(toward[1])
    dz = float(endpoint[2]) - float(toward[2])
    length = math.hypot(dx, dy, dz)
    if length == 0.0:
        return endpoint
    scale = pad / length
    return (
        float(endpoint[0]) + dx * scale,
        float(endpoint[1]) + dy * scale,
        float(endpoint[2]) + dz * scale,
    )


def anchor_padded_xy(
    point: tuple[float, float, float],
    *,
    pad_x_mm: float = 0.0,
    pad_y_mm: float = 0.0,
) -> tuple[float, float, float]:
    """Offset a label anchor in scene XY (mm)."""
    return (
        float(point[0]) + float(pad_x_mm),
        float(point[1]) + float(pad_y_mm),
        float(point[2]),
    )


def segment_endpoint_labels(
    segment: AxisLineSpec,
    start_text: str,
    end_text: str,
    pad_mm: float,
) -> tuple[LabelAnchor, LabelAnchor]:
    """Label anchors just beyond both ends of a line segment."""
    return (
        LabelAnchor(
            position=anchor_beyond_line_end(segment.start, segment.end, pad_mm),
            text=start_text,
        ),
        LabelAnchor(
            position=anchor_beyond_line_end(segment.end, segment.start, pad_mm),
            text=end_text,
        ),
    )
