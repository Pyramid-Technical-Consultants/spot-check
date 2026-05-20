"""Clinic-scale regression tests: episode alignment must finish (no quadratic hangs)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spot_check import analysis
from spot_check.analysis.episodes import AutoFitRow, segment_align_auto_episodes
from tests.conftest import MINIMAL_PLANNED_XYZ, write_measured_csv


def _spot_pattern_rows(n_spots: int, *, spot_len: int = 1, dead_len: int = 0) -> list[AutoFitRow]:
    """Rows with high on-spot signal and optional deadtime gaps between spots."""
    rows: list[AutoFitRow] = []
    idx = 0
    high, low = 100.0, 5.0
    for s in range(n_spots):
        for _ in range(spot_len):
            x = float(s)
            rows.append(
                AutoFitRow(
                    t=float(idx) * 1e-3,
                    mx=x,
                    my=0.0,
                    a=x,
                    b=0.0,
                    mx_p=x,
                    my_p=0.0,
                    weight=high,
                    ch_n=high,
                    pcd=0,
                    sa=None,
                    sb=None,
                )
            )
            idx += 1
        for _ in range(dead_len):
            rows.append(
                AutoFitRow(
                    t=float(idx) * 1e-3,
                    mx=float(s),
                    my=0.0,
                    a=float(s),
                    b=0.0,
                    mx_p=float(s),
                    my_p=0.0,
                    weight=low,
                    ch_n=low,
                    pcd=0,
                    sa=None,
                    sb=None,
                )
            )
            idx += 1
    return rows


@pytest.mark.slow
@pytest.mark.timeout(180)
def test_align_merge_many_singleton_episodes_finishes() -> None:
    """Regression: merge scan must stay O(M²) compares with O(M) finalize updates (not O(M³))."""
    n_plan = 6200
    rows = _spot_pattern_rows(n_plan + 1800, spot_len=1, dead_len=1)
    groups, diag = segment_align_auto_episodes(
        rows,
        n_plan_spots=n_plan,
        episode_gap_s=0.0005,
        min_on_spot_weight_na=1e-12,
        spot_xy_jump_mm=500.0,
        min_episode_rows=1,
        dead_ratio=0.55,
    )
    assert len(groups) == n_plan
    assert diag.count_align_ok


@pytest.mark.slow
@pytest.mark.timeout(90)
def test_align_split_single_long_episode_finishes() -> None:
    """Many splits inside one segmented episode (scan-heavy inner loops)."""
    n_plan = 600
    rows = _spot_pattern_rows(1, spot_len=n_plan, dead_len=0)
    groups, diag = segment_align_auto_episodes(
        rows,
        n_plan_spots=n_plan,
        episode_gap_s=10.0,
        min_on_spot_weight_na=1e-12,
        spot_xy_jump_mm=500.0,
        min_episode_rows=1,
        dead_ratio=0.55,
    )
    assert len(groups) == n_plan
    assert diag.count_align_ok
    assert diag.n_raw_episodes == 1


@pytest.mark.slow
@pytest.mark.timeout(120)
def test_segment_align_million_dwell_rows_finishes() -> None:
    """Millions of sample rows grouped into plan-scale episodes (typical clinic CSV)."""
    from spot_check.analysis.auto_columns import AutoFitColumns
    from spot_check.analysis.episodes import segment_align_auto_columns

    n_plan = 4000
    rows_per_spot = 250
    dead_per_spot = 1
    high, low = 100.0, 5.0
    chunks = rows_per_spot + dead_per_spot
    n = n_plan * chunks
    spot_id = np.repeat(np.arange(n_plan), chunks)
    within = np.tile(
        np.concatenate(
            (
                np.arange(rows_per_spot, dtype=np.float64) * 1e-4,
                np.array([0.05], dtype=np.float64),
            )
        ),
        n_plan,
    )
    t = np.repeat(np.arange(n_plan, dtype=np.float64), chunks) * 0.1 + within
    xy = spot_id.astype(np.float64) * 0.05
    on = np.tile(
        np.concatenate((np.ones(rows_per_spot), np.zeros(dead_per_spot))),
        n_plan,
    ).astype(bool)
    sig = np.where(on, high, low)
    cols = AutoFitColumns(
        t=t,
        mx=xy,
        my=np.zeros(n, dtype=np.float64),
        a=xy,
        b=np.zeros(n, dtype=np.float64),
        mx_p=xy,
        my_p=np.zeros(n, dtype=np.float64),
        weight=sig,
        ch_n=sig,
        fit_a=sig * 0.1,
        pcd=np.zeros(n, dtype=np.int32),
        sa=np.full(n, np.nan, dtype=np.float64),
        sb=np.full(n, np.nan, dtype=np.float64),
    )
    spans, diag = segment_align_auto_columns(
        cols,
        n_plan_spots=n_plan,
        episode_gap_s=0.05,
        min_on_spot_weight_na=1e-12,
        spot_xy_jump_mm=500.0,
        min_episode_rows=1,
        dead_ratio=0.55,
    )
    assert len(spans) == n_plan
    assert diag.count_align_ok
    assert abs(diag.n_raw_episodes - n_plan) <= max(1, n_plan // 100)


def test_align_repartitions_by_weight_when_many_splits_needed() -> None:
    """When n_plan >> raw episodes, weight-quantile repartition reaches plan count."""
    from spot_check.analysis.auto_columns import AutoFitColumns
    from spot_check.analysis.episodes import (
        align_episode_spans_to_plan_count_cols,
        segment_into_episodes_cols,
    )

    n_rows = 4000
    sig = np.full(n_rows, 100.0, dtype=np.float64)
    cols = AutoFitColumns(
        t=np.arange(n_rows, dtype=np.float64) * 1e-3,
        mx=np.zeros(n_rows, dtype=np.float64),
        my=np.zeros(n_rows, dtype=np.float64),
        a=np.zeros(n_rows, dtype=np.float64),
        b=np.zeros(n_rows, dtype=np.float64),
        mx_p=np.zeros(n_rows, dtype=np.float64),
        my_p=np.zeros(n_rows, dtype=np.float64),
        weight=sig,
        ch_n=sig,
        fit_a=sig * 0.1,
        pcd=np.zeros(n_rows, dtype=np.int32),
        sa=np.full(n_rows, np.nan, dtype=np.float64),
        sb=np.full(n_rows, np.nan, dtype=np.float64),
    )
    spans = segment_into_episodes_cols(
        cols,
        episode_gap_s=0.05,
        min_on_spot_weight_na=1e-12,
        spot_xy_jump_mm=500.0,
        min_episode_rows=1,
        dead_ratio=0.55,
    )
    raw_n = len(spans)
    n_plan = 3000
    plan_xy = np.zeros((n_plan, 2), dtype=np.float64)
    work, diag = align_episode_spans_to_plan_count_cols(
        cols, spans, n_plan, plan_xy=plan_xy
    )
    assert diag.count_align_ok
    assert len(work) == n_plan
    assert diag.n_raw_episodes == raw_n


@pytest.mark.slow
@pytest.mark.timeout(120)
def test_measured_auto_many_rows_csv_finishes(tmp_path: Path) -> None:
    """End-to-end measured.load + Viterbi on thousands of synthesized spots."""
    n = 2000
    rows_csv = [
        {
            "time (s)": str(float(i) * 0.002),
            "IX512 Channel Sum (nA)": "1.0",
            "Fit Amplitude A (nA)": "0.5",
            "Fit Mean Position A (mm)": str(float(i % 200) * 0.02),
            "Fit Mean Position B (mm)": str(float(i % 151) * 0.03),
            "Gate Counter": str(2 * i + 1),
        }
        for i in range(n)
    ]
    csv_path = write_measured_csv(tmp_path / "bulk.csv", rows_csv)
    planned = list(MINIMAL_PLANNED_XYZ) * ((n // len(MINIMAL_PLANNED_XYZ)) + 1)
    planned = planned[:n]
    out = analysis.measured_spot_abc_from_csv(
        csv_path,
        planned_xyz=planned,
        layer_mode="auto",
        a_is_x=False,
        auto_infer_params=False,
        auto_episode_gap_s=0.001,
        auto_spot_xy_jump_mm=5.0,
        auto_min_on_spot_weight_na=1e-12,
        auto_min_episode_rows=1,
        viterbi_advance_penalty_mm2=400.0,
    )
    assert len(out) == n
    diag = analysis.last_auto_episode_diagnostics()
    assert diag is not None
