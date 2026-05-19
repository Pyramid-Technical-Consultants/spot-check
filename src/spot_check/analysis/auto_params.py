"""Infer ``layer_mode='auto'`` segmentation and Viterbi settings from plan + CSV."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from spot_check.analysis.auto_columns import AutoFitColumns
from spot_check.analysis.episodes import segment_into_episodes_cols
from spot_check.analysis.spatial import _plan_xy_by_energy_layer, nominal_layer_energies_mev
from spot_check.constants import (
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


def last_auto_layer_params() -> AutoLayerParams | None:
    return _last_params


def _set_last_params(p: AutoLayerParams | None) -> None:
    global _last_params
    _last_params = p


def _median_consecutive_plan_step_mm(
    planned_xyz: Sequence[tuple[float, float, float]],
) -> float:
    if len(planned_xyz) < 2:
        return float(AUTO_SPOT_XY_JUMP_MM_DEFAULT)
    xy = np.asarray([(float(p[0]), float(p[1])) for p in planned_xyz], dtype=np.float64)
    d = np.sqrt(np.sum(np.diff(xy, axis=0) ** 2, axis=1))
    d = d[np.isfinite(d) & (d > 1e-6)]
    if d.size == 0:
        return float(AUTO_SPOT_XY_JUMP_MM_DEFAULT)
    return float(np.median(d))


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


def _calibrate_xy_jump_mm(
    cols: AutoFitColumns,
    n_plan: int,
    *,
    episode_gap_s: float,
    min_on_spot_weight_na: float,
    min_episode_rows: int,
) -> float:
    """Pick XY step threshold so signal episodes approximate the plan spot count."""
    if n_plan <= 0 or len(cols) < 2:
        return float(AUTO_SPOT_XY_JUMP_MM_DEFAULT)
    lo, hi = 1.0, 120.0
    best_xy = float(AUTO_SPOT_XY_JUMP_MM_DEFAULT)
    best_err = float("inf")
    for _ in range(22):
        mid = (lo + hi) * 0.5
        n_ep = len(
            segment_into_episodes_cols(
                cols,
                episode_gap_s=episode_gap_s,
                min_on_spot_weight_na=min_on_spot_weight_na,
                spot_xy_jump_mm=mid,
                min_episode_rows=min_episode_rows,
            )
        )
        err = abs(n_ep - n_plan)
        if err < best_err:
            best_err = err
            best_xy = mid
        if n_ep > n_plan:
            lo = mid
        else:
            hi = mid
    return float(np.clip(best_xy, 2.0, 80.0))


def _infer_spot_xy_jump_mm(
    cols: AutoFitColumns,
    planned_xyz: Sequence[tuple[float, float, float]],
    *,
    episode_gap_s: float,
    min_on_spot_weight_na: float,
    min_episode_rows: int,
) -> float:
    n_plan = len(planned_xyz)
    if n_plan > 0 and len(cols) >= 2:
        return _calibrate_xy_jump_mm(
            cols,
            n_plan,
            episode_gap_s=episode_gap_s,
            min_on_spot_weight_na=min_on_spot_weight_na,
            min_episode_rows=min_episode_rows,
        )
    step = _median_consecutive_plan_step_mm(planned_xyz)
    return float(np.clip(step * 0.42, 0.8, 15.0))


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
    min_w = _infer_min_on_spot_weight_na(cols.weight)
    gap_s = _infer_episode_gap_s(cols.t, n_plan)
    xy_jump = _infer_spot_xy_jump_mm(
        cols,
        planned_xyz,
        episode_gap_s=gap_s,
        min_on_spot_weight_na=min_w,
        min_episode_rows=min_rows,
    )
    params = AutoLayerParams(
        episode_gap_s=gap_s,
        spot_xy_jump_mm=xy_jump,
        min_on_spot_weight_na=min_w,
        min_episode_rows=min_rows,
        viterbi_advance_penalty_mm2=_infer_viterbi_penalty_mm2(planned_xyz),
    )
    _set_last_params(params)
    return params
