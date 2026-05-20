"""Sequential plan-order spot assignment for ``layer_mode='auto'``."""

from __future__ import annotations

import numpy as np

from spot_check.analysis.auto_columns import AutoFitColumns, position_fit_deadtime_mask
from spot_check.analysis.episodes import EpisodeSpan
from spot_check.constants import (
    AUTO_PLAN_SEQ_CLUSTER_RADIUS_MM_DEFAULT,
    AUTO_PLAN_SEQ_CLUSTER_RADIUS_SCALE,
)

PlanIndexArray = np.ndarray


def infer_plan_seq_cluster_radius_mm2(plan_xy: np.ndarray) -> float:
    """On-spot cluster radius from median consecutive plan XY spacing."""
    pts = np.asarray(plan_xy, dtype=np.float64)
    if pts.shape[0] < 2:
        r_mm = float(AUTO_PLAN_SEQ_CLUSTER_RADIUS_MM_DEFAULT)
        return r_mm * r_mm
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    d = d[(d > 1e-6) & np.isfinite(d)]
    step = float(np.median(d)) if d.size else float(AUTO_PLAN_SEQ_CLUSTER_RADIUS_MM_DEFAULT)
    r_mm = max(3.0, step * float(AUTO_PLAN_SEQ_CLUSTER_RADIUS_SCALE))
    return r_mm * r_mm


def assign_plan_indices_sequential(
    cols: AutoFitColumns,
    plan_xy: np.ndarray,
    *,
    min_rows_on_spot: int = 1,
    cluster_radius_mm2: float | None = None,
) -> PlanIndexArray:
    """Per-row plan spot index in delivery order; ``-1`` for deadtime rows.

    Deadtime = neither Fit Mean Position A nor B (see :func:`position_fit_deadtime_mask`).
    Walk acquisition time from the first plan spot (index 0 = first spot of the
    highest nominal-energy layer). After each deadtime break, advance by exactly one plan
    slot (``plan_i + 1`` only) once the current spot has at least ``min_rows_on_spot``
    assigned rows. Plan slots are never skipped and multi-spot jumps are not allowed.
    """
    n = len(cols)
    m = int(plan_xy.shape[0])
    del cluster_radius_mm2  # reserved for API; advance is break-driven only
    out = np.full(n, -1, dtype=np.int32)
    if n == 0 or m == 0:
        return out

    dead = position_fit_deadtime_mask(cols)
    plan_i = 0
    in_break = False
    rows_on_spot = 0
    min_on = max(1, int(min_rows_on_spot))

    for ri in range(n):
        if dead[ri]:
            in_break = True
            continue
        if plan_i >= m - 1:
            out[ri] = plan_i
            rows_on_spot += 1
            in_break = False
            continue
        if in_break and rows_on_spot >= min_on:
            plan_i += 1
            rows_on_spot = 0
            in_break = False
        out[ri] = plan_i
        rows_on_spot += 1
    return out


def sequential_spans_from_plan_indices(plan_idx: PlanIndexArray) -> list[EpisodeSpan]:
    """Contiguous on-spot runs sharing one plan index (deadtime rows omitted)."""
    n = int(plan_idx.size)
    spans: list[EpisodeSpan] = []
    i = 0
    while i < n:
        while i < n and int(plan_idx[i]) < 0:
            i += 1
        if i >= n:
            break
        pi = int(plan_idx[i])
        s = i
        i += 1
        while i < n and int(plan_idx[i]) == pi:
            i += 1
        spans.append((s, i))
    return spans


def plan_spot_index_per_span(
    plan_idx: PlanIndexArray, spans: list[EpisodeSpan]
) -> np.ndarray:
    """Plan delivery index for each span (``len(spans)``)."""
    if not spans:
        return np.zeros(0, dtype=np.int64)
    return np.fromiter((int(plan_idx[s]) for s, _ in spans), dtype=np.int64, count=len(spans))
