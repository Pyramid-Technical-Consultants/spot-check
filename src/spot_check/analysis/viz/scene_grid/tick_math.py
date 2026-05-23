"""Tick value planning for scene grid lines."""

from __future__ import annotations

import math


def bounds_expanded_to_tick_step(
    vmin: float,
    vmax: float,
    step_mm: float,
) -> tuple[float, float]:
    """Expand ``[vmin, vmax]`` outward to the nearest tick-step multiples."""
    step = float(step_mm)
    lo = float(min(vmin, vmax))
    hi = float(max(vmin, vmax))
    if step <= 0.0 or not math.isfinite(step):
        return lo, hi
    return math.floor(lo / step) * step, math.ceil(hi / step) * step


def tick_values_centered_on_zero(
    vmin: float,
    vmax: float,
    step_mm: float,
) -> tuple[float, ...]:
    """Return ``±step``, ``±2*step``, … strictly between bounds; never includes 0."""
    step = float(step_mm)
    if step <= 0.0 or not math.isfinite(step):
        return ()
    lo = float(min(vmin, vmax))
    hi = float(max(vmin, vmax))
    if lo == hi:
        return ()

    ticks: list[float] = []
    max_k = int(math.floor(max(abs(lo), abs(hi)) / step))
    for k in range(1, max_k + 1):
        for signed in (k * step, -k * step):
            if lo <= signed <= hi:
                ticks.append(float(signed))
    ticks.sort()
    return tuple(ticks)
