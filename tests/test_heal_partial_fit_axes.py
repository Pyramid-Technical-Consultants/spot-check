"""Partial fit A/B row filtering and optional plan heal."""

from __future__ import annotations

from pathlib import Path

from spot_check import analysis
from spot_check.constants import CHANNEL_SUM_KEY, FIT_AMPLITUDE_A_KEY
from tests.conftest import MINIMAL_PLANNED_XYZ, write_measured_csv


def _row(
    t: str,
    a: str,
    b: str,
    *,
    gate: str = "1",
) -> dict[str, str]:
    return {
        "time (s)": t,
        CHANNEL_SUM_KEY: "1.0",
        FIT_AMPLITUDE_A_KEY: "0.5",
        "Fit Mean Position A (mm)": a,
        "Fit Mean Position B (mm)": b,
        "Gate Counter": gate,
    }


def test_partial_and_both_missing_dropped_by_default(tmp_path: Path) -> None:
    csv_path = write_measured_csv(
        tmp_path / "partial.csv",
        [
            _row("0.0", "0.0", "0.0"),
            _row("0.05", "5.0", ""),
            _row("0.10", "", ""),
        ],
    )
    rows = analysis.measured_spot_abc_from_csv(
        csv_path,
        planned_xyz=list(MINIMAL_PLANNED_XYZ),
        layer_mode="time_gap",
        a_is_x=False,
        aggregate_spots=False,
    )
    assert len(rows) == 1
    assert int(rows[0][4]) == 0


def test_heal_keeps_one_axis_missing_rows(tmp_path: Path) -> None:
    csv_path = write_measured_csv(
        tmp_path / "heal.csv",
        [
            _row("0.0", "0.0", "0.0"),
            _row("0.05", "5.0", ""),
        ],
    )
    rows = analysis.measured_spot_abc_from_csv(
        csv_path,
        planned_xyz=list(MINIMAL_PLANNED_XYZ),
        layer_mode="time_gap",
        a_is_x=False,
        aggregate_spots=False,
        heal_partial_fit_axes=True,
    )
    assert len(rows) == 2
    assert int(rows[0][4]) == 0
    assert int(rows[1][4]) > 0


def test_both_missing_never_kept_even_with_heal(tmp_path: Path) -> None:
    csv_path = write_measured_csv(
        tmp_path / "both.csv",
        [
            _row("0.0", "0.0", "0.0"),
            _row("0.05", "", ""),
        ],
    )
    rows = analysis.measured_spot_abc_from_csv(
        csv_path,
        planned_xyz=list(MINIMAL_PLANNED_XYZ),
        layer_mode="time_gap",
        a_is_x=False,
        aggregate_spots=False,
        heal_partial_fit_axes=True,
    )
    assert len(rows) == 1
