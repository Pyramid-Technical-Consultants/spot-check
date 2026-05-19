"""Infer ``layer_mode='auto'`` segmentation and Viterbi settings from plan + CSV."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from spot_check.analysis.auto_columns import AutoFitColumns
from spot_check.analysis.episodes import segment_into_episodes_cols
from spot_check.analysis.spatial import _plan_xy_by_energy_layer, nominal_layer_energies_mev
from spot_check.constants import (
    AUTO_EDGE_DEAD_RATIO_DEFAULT,
    AUTO_EDGE_TINY_MERGE_ROWS,
    AUTO_MIN_EPISODE_ROWS_DEFAULT,
    AUTO_MIN_ON_SPOT_WEIGHT_NA_DEFAULT,
    AUTO_SPOT_XY_JUMP_MM_DEFAULT,
    TIME_LAYER_GAP_S_DEFAULT,
    VITERBI_LAYER_ADVANCE_PENALTY_MM2_DEFAULT,
)

_last_params: AutoLayerParams | None = None


@dataclass(frozen=True)
class AutoLayerParams:
    """Tuning for signal episodes + plan spot-count alignment + Viterbi layers."""

    episode_gap_s: float
    spot_xy_jump_mm: float
    min_on_spot_weight_na: float
    min_episode_rows: int
    viterbi_advance_penalty_mm2: float
    dead_ratio: float
    tiny_merge_rows: int


def last_auto_layer_params() -> AutoLayerParams | None:
    return _last_params


def _set_last_params(p: AutoLayerParams | None) -> None:
    global _last_params
    _last_params = p


def _infer_episode_gap_s(t: np.ndarray, n_plan: int) -> float:
    t = t[np.isfinite(t)]
    if t.size < 2:
        return float(TIME_LAYER_GAP_S_DEFAULT)
    dt = np.diff(t)
    dt = dt[(dt > 0) & np.isfinite(dt)]
    if dt.size == 0:
        return float(TIME_LAYER_GAP_S_DEFAULT)

    median_dt = float(np.median(dt))
    spread = float(np.percentile(dt, 90) - np.percentile(dt, 10))
    if spread < max(median_dt * 0.05, 1e-9):
        gap = median_dt * 2.5
    else:
        sorted_dt = np.sort(dt)
        ratios = sorted_dt[1:] / np.maximum(sorted_dt[:-1], 1e-12)
        if ratios.size:
            ki = int(np.argmax(ratios))
            knee = float((sorted_dt[ki] + sorted_dt[ki + 1]) * 0.5)
        else:
            knee = median_dt * 1.5
        span = float(t[-1] - t[0])
        mean_inter = span / max(1, n_plan - 1) if n_plan > 1 else span
        p90 = float(np.percentile(dt, 90))
        gap = min(knee, mean_inter * 0.5, max(p90 * 1.35, median_dt * 3.0))
    gap = min(gap, median_dt * 8.0)
    return float(np.clip(gap, 0.005, 5.0))


def _calibrate_dead_ratio(
    cols: AutoFitColumns,
    n_plan: int,
    *,
    min_episode_rows: int,
    tiny_merge_rows: int,
) -> float:
    """Pick deadtime ratio so episode count approximates the plan spot count."""
    if n_plan <= 0 or len(cols) < 2:
        return float(AUTO_EDGE_DEAD_RATIO_DEFAULT)
    best_ratio = float(AUTO_EDGE_DEAD_RATIO_DEFAULT)
    best_err = float("inf")
    grid = np.linspace(0.52, 0.64, 49)
    counts: list[int] = []
    for ratio in grid:
        n_ep = len(
            segment_into_episodes_cols(
                cols,
                episode_gap_s=1.0,
                min_on_spot_weight_na=0.0,
                spot_xy_jump_mm=999.0,
                min_episode_rows=min_episode_rows,
                dead_ratio=float(ratio),
                tiny_merge_rows=tiny_merge_rows,
            )
        )
        counts.append(n_ep)
        err = abs(n_ep - n_plan)
        if err < best_err:
            best_err = err
            best_ratio = float(ratio)
    if best_err > 0:
        counts_arr = np.asarray(counts, dtype=np.int64)
        above = np.nonzero(counts_arr >= n_plan)[0]
        below = np.nonzero(counts_arr <= n_plan)[0]
        if above.size and below.size:
            ia = int(above[0])
            ib = int(below[-1])
            if ia > ib and counts_arr[ia] != counts_arr[ib]:
                t = (n_plan - counts_arr[ib]) / float(counts_arr[ia] - counts_arr[ib])
                best_ratio = float(grid[ib] + t * (grid[ia] - grid[ib]))
    return float(np.clip(best_ratio, 0.52, 0.64))


def _infer_min_on_spot_weight_na(weight: np.ndarray) -> float:
    w = weight[np.isfinite(weight) & (weight > 0)]
    if w.size == 0:
        return float(AUTO_MIN_ON_SPOT_WEIGHT_NA_DEFAULT)
    p1 = float(np.percentile(w, 1))
    p50 = float(np.percentile(w, 50))
    thr = max(1e-12, p1 * 0.05, p50 * 0.002)
    return float(min(thr, p50 * 0.02))


def _infer_min_episode_rows(n_rows: int, n_plan: int) -> int:
    if n_plan <= 0 or n_rows <= 0:
        return int(AUTO_MIN_EPISODE_ROWS_DEFAULT)
    rps = n_rows / float(n_plan)
    return max(1, min(20, int(max(1.0, round(rps * 0.03)))))


def _infer_viterbi_penalty_mm2(
    planned_xyz: Sequence[tuple[float, float, float]],
) -> float:
    energies = nominal_layer_energies_mev(planned_xyz)
    if not energies:
        return float(VITERBI_LAYER_ADVANCE_PENALTY_MM2_DEFAULT)
    layer_xy = _plan_xy_by_energy_layer(planned_xyz, energies)
    cents: list[np.ndarray] = []
    for arr in layer_xy:
        pts = np.asarray(arr, dtype=np.float64).reshape(-1, 2)
        if pts.size == 0:
            continue
        cents.append(pts.mean(axis=0))
    if len(cents) < 2:
        return float(VITERBI_LAYER_ADVANCE_PENALTY_MM2_DEFAULT)
    c = np.asarray(cents, dtype=np.float64)
    d2 = np.sum(np.diff(c, axis=0) ** 2, axis=1)
    d2 = d2[np.isfinite(d2) & (d2 > 0)]
    if d2.size == 0:
        return float(VITERBI_LAYER_ADVANCE_PENALTY_MM2_DEFAULT)
    from_plan = float(np.clip(float(np.median(d2)) * 1.25, 50.0, 2000.0))
    return max(from_plan, float(VITERBI_LAYER_ADVANCE_PENALTY_MM2_DEFAULT))


def infer_auto_layer_params(
    cols: AutoFitColumns,
    planned_xyz: Sequence[tuple[float, float, float]],
) -> AutoLayerParams:
    """Derive auto-mode thresholds from acquisition timing, weights, and plan geometry."""
    n_plan = len(planned_xyz)
    min_rows = _infer_min_episode_rows(len(cols), n_plan)
    tiny_merge = int(AUTO_EDGE_TINY_MERGE_ROWS)
    dead_ratio = _calibrate_dead_ratio(
        cols, n_plan, min_episode_rows=min_rows, tiny_merge_rows=tiny_merge
    )
    params = AutoLayerParams(
        episode_gap_s=_infer_episode_gap_s(cols.t, n_plan),
        spot_xy_jump_mm=float(AUTO_SPOT_XY_JUMP_MM_DEFAULT),
        min_on_spot_weight_na=_infer_min_on_spot_weight_na(cols.weight),
        min_episode_rows=min_rows,
        viterbi_advance_penalty_mm2=_infer_viterbi_penalty_mm2(planned_xyz),
        dead_ratio=dead_ratio,
        tiny_merge_rows=tiny_merge,
    )
    _set_last_params(params)
    return params
