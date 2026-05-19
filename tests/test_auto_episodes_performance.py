"""Clinic-scale regression tests: episode alignment must finish (no quadratic hangs)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spot_check import analysis
from spot_check.analysis.episodes import AutoFitRow, segment_align_auto_episodes
from tests.conftest import MINIMAL_PLANNED_XYZ, write_measured_csv


def _singleton_rows(n: int, *, dt_s: float = 1e-3) -> list[AutoFitRow]:
    """Consecutive dt = dt_s; with episode_gap_s < dt_s each row becomes its own episode."""
    rows: list[AutoFitRow] = []
    for i in range(n):
        x = float(i % 997) * 0.1
        t = float(i) * dt_s
        rows.append(
            AutoFitRow(
                t=t,
                mx=x,
                my=0.0,
                a=x,
                b=0.0,
                mx_p=x,
                my_p=0.0,
                weight=1.0,
                ch_n=1.0,
                pcd=0,
                sa=None,
                sb=None,
            )
        )
    return rows


@pytest.mark.slow
@pytest.mark.timeout(180)
def test_align_merge_many_singleton_episodes_finishes() -> None:
    """Regression: merge scan must stay O(M²) compares with O(M) finalize updates (not O(M³))."""
    n_rows = 8000
    n_plan = 6200
    rows = _singleton_rows(n_rows)
    groups, diag = segment_align_auto_episodes(
        rows,
        n_plan_spots=n_plan,
        episode_gap_s=0.0005,
        min_on_spot_weight_na=1e-12,
        spot_xy_jump_mm=500.0,
        min_episode_rows=1,
    )
    assert len(groups) == n_plan
    assert diag.count_align_ok
    assert diag.n_raw_episodes == n_rows


@pytest.mark.slow
@pytest.mark.timeout(90)
def test_align_split_single_long_episode_finishes() -> None:
    """Many splits inside one segmented episode (scan-heavy inner loops)."""
    n_plan = 600
    rows = _singleton_rows(n_plan, dt_s=1e-6)
    groups, diag = segment_align_auto_episodes(
        rows,
        n_plan_spots=n_plan,
        episode_gap_s=10.0,
        min_on_spot_weight_na=1e-12,
        spot_xy_jump_mm=500.0,
        min_episode_rows=1,
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
    n = n_plan * rows_per_spot
    spot_id = np.repeat(np.arange(n_plan), rows_per_spot)
    within = np.tile(np.arange(rows_per_spot, dtype=np.float64) * 1e-4, n_plan)
    t = np.repeat(np.arange(n_plan, dtype=np.float64), rows_per_spot) * 0.1 + within
    xy = spot_id.astype(np.float64) * 0.05
    cols = AutoFitColumns(
        t=t,
        mx=xy,
        my=np.zeros(n, dtype=np.float64),
        a=xy,
        b=np.zeros(n, dtype=np.float64),
        mx_p=xy,
        my_p=np.zeros(n, dtype=np.float64),
        weight=np.ones(n, dtype=np.float64),
        ch_n=np.ones(n, dtype=np.float64),
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
    )
    assert len(spans) == n_plan
    assert diag.count_align_ok
    assert diag.n_raw_episodes == n_plan


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
