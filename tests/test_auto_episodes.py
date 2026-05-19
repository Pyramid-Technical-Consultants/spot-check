"""Tests for signal episode segmentation and plan-assisted count alignment."""

from __future__ import annotations

from pathlib import Path

import pytest

from spot_check import analysis
from spot_check.analysis.episodes import AutoFitRow, segment_align_auto_episodes
from tests.conftest import MINIMAL_PLANNED_XYZ, minimal_measured_rows, write_measured_csv


def _row(i: int, *, dt: float = 10.0) -> AutoFitRow:
    x = float(i)
    return AutoFitRow(
        t=float(i * dt),
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


def test_segment_align_preserves_count_when_raw_equals_plan() -> None:
    rows = [_row(i) for i in range(4)]
    groups, diag = segment_align_auto_episodes(
        rows,
        n_plan_spots=4,
        episode_gap_s=0.2,
        min_on_spot_weight_na=1e-12,
        spot_xy_jump_mm=50.0,
        min_episode_rows=1,
    )
    assert len(groups) == 4
    assert diag.count_align_ok
    assert diag.n_raw_episodes == 4


def test_merge_when_more_episodes_than_plan() -> None:
    rows = [_row(i, dt=10.0) for i in range(5)]
    groups, diag = segment_align_auto_episodes(
        rows,
        n_plan_spots=2,
        episode_gap_s=0.2,
        min_on_spot_weight_na=1e-12,
        spot_xy_jump_mm=50.0,
        min_episode_rows=1,
    )
    assert len(groups) == 2
    assert diag.count_align_ok
    assert diag.n_raw_episodes == 5


def test_auto_measured_inferred_params_match_plan(tmp_path: Path) -> None:
    """Default API infers episode settings from CSV + plan."""
    rows_csv = [
        {
            "time (s)": str(i * 2.0),
            "IX512 Channel Sum (nA)": "1.0",
            "Fit Amplitude A (nA)": "0.5",
            "Fit Mean Position A (mm)": str(float(i)),
            "Fit Mean Position B (mm)": "0.0",
            "Gate Counter": str(2 * i + 1),
        }
        for i in range(4)
    ]
    csv_path = write_measured_csv(tmp_path / "four_inferred.csv", rows_csv)
    out = analysis.measured_spot_abc_from_csv(
        csv_path,
        planned_xyz=list(MINIMAL_PLANNED_XYZ),
        layer_mode="auto",
        a_is_x=False,
    )
    assert len(out) == len(MINIMAL_PLANNED_XYZ)
    assert analysis.last_auto_layer_params() is not None


def test_auto_measured_row_count_matches_plan(tmp_path: Path) -> None:
    """Synthetic CSV with four spaced rows vs four plan spots."""
    rows_csv = [
        {
            "time (s)": str(i * 2.0),
            "IX512 Channel Sum (nA)": "1.0",
            "Fit Amplitude A (nA)": "0.5",
            "Fit Mean Position A (mm)": str(float(i)),
            "Fit Mean Position B (mm)": "0.0",
            "Gate Counter": str(2 * i + 1),
        }
        for i in range(4)
    ]
    csv_path = write_measured_csv(tmp_path / "four.csv", rows_csv)
    out = analysis.measured_spot_abc_from_csv(
        csv_path,
        planned_xyz=list(MINIMAL_PLANNED_XYZ),
        layer_mode="auto",
        a_is_x=False,
        auto_infer_params=False,
        auto_episode_gap_s=0.5,
        auto_spot_xy_jump_mm=50.0,
        auto_min_on_spot_weight_na=1e-12,
        auto_min_episode_rows=1,
        viterbi_advance_penalty_mm2=400.0,
    )
    assert len(out) == len(MINIMAL_PLANNED_XYZ)
    diag = analysis.last_auto_episode_diagnostics()
    assert diag is not None
    assert diag.count_align_ok


def test_auto_requires_planned_xyz(tmp_path: Path) -> None:
    csv_path = write_measured_csv(tmp_path / "m.csv", minimal_measured_rows())
    with pytest.raises(ValueError, match="auto requires"):
        analysis.measured_spot_abc_from_csv(
            csv_path,
            planned_xyz=None,
            layer_mode="auto",
            a_is_x=False,
        )
