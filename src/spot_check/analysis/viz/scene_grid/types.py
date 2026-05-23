"""Immutable specs for custom scene grid rendering."""

from __future__ import annotations

from dataclasses import dataclass

from spot_check.constants import (
    SCENE_GRID_LABEL_BELOW_PLANE_MM,
    SCENE_GRID_LABEL_ENDPOINT_PAD_MM,
    SCENE_GRID_LABEL_FORMAT,
    SCENE_GRID_LABEL_OPACITY,
    SCENE_GRID_LINE_TUBE_RADIUS_MM,
    SCENE_GRID_MINOR_OPACITY_SCALE,
    SCENE_GRID_MINOR_TICK_MM,
)

Bounds6 = tuple[float, float, float, float, float, float]


@dataclass(frozen=True, slots=True)
class GridStyle:
    """Visual defaults for grid lines and numeric labels."""

    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    opacity: float = 0.5
    line_width: float = 1.0
    line_tube_radius_mm: float = SCENE_GRID_LINE_TUBE_RADIUS_MM
    label_font_size: int = 11
    label_endpoint_pad_mm: float = SCENE_GRID_LABEL_ENDPOINT_PAD_MM
    label_below_plane_mm: float = SCENE_GRID_LABEL_BELOW_PLANE_MM
    label_format: str = SCENE_GRID_LABEL_FORMAT
    label_opacity: float = SCENE_GRID_LABEL_OPACITY
    minor_tick_mm: float = SCENE_GRID_MINOR_TICK_MM
    minor_opacity_scale: float = SCENE_GRID_MINOR_OPACITY_SCALE

    @property
    def minor_opacity(self) -> float:
        return float(self.opacity) * float(self.minor_opacity_scale)


@dataclass(frozen=True, slots=True)
class SceneFrame:
    """Axis-aligned scene extent for grid planning."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    @classmethod
    def from_bounds6(cls, bounds: Bounds6) -> SceneFrame:
        return cls(
            x_min=float(bounds[0]),
            x_max=float(bounds[1]),
            y_min=float(bounds[2]),
            y_max=float(bounds[3]),
            z_min=float(bounds[4]),
            z_max=float(bounds[5]),
        )

    def as_bounds6(self) -> Bounds6:
        return (
            self.x_min,
            self.x_max,
            self.y_min,
            self.y_max,
            self.z_min,
            self.z_max,
        )


@dataclass(frozen=True, slots=True)
class AxisLineSpec:
    """Single world-space line segment."""

    start: tuple[float, float, float]
    end: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class LabelAnchor:
    """Billboard label at a world position."""

    position: tuple[float, float, float]
    text: str


@dataclass(frozen=True, slots=True)
class GridPlan:
    """Pure plan: major/minor line segments and label anchors to render."""

    major_lines: tuple[AxisLineSpec, ...]
    minor_lines: tuple[AxisLineSpec, ...] = ()
    labels: tuple[LabelAnchor, ...] = ()
