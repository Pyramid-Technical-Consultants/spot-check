"""Measured spot weight mode and CSV column probing."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from spot_check import analysis
from spot_check.analysis._core import _probe_csv_columns_for_measured_weights
from spot_check.constants import (
    CHANNEL_SUM_KEY,
    FIT_AMPLITUDE_A_KEY,
    FIT_AMPLITUDE_B_KEY,
    GATE_COUNTER_KEY,
)


def test_probe_allows_fit_amp_b_without_b_column_when_channel_sum_present(tmp_path: Path) -> None:
    p = tmp_path / "spots.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "time (s)",
                CHANNEL_SUM_KEY,
                FIT_AMPLITUDE_A_KEY,
                GATE_COUNTER_KEY,
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "time (s)": "0",
                CHANNEL_SUM_KEY: "1.0",
                FIT_AMPLITUDE_A_KEY: "0.5",
                GATE_COUNTER_KEY: "1",
            }
        )
    _probe_csv_columns_for_measured_weights(
        p, aggregate_spots=True, spot_weight_mode="fit_amplitude_b"
    )


def test_probe_rejects_fit_amp_b_without_b_or_channel_sum(tmp_path: Path) -> None:
    p = tmp_path / "spots.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[FIT_AMPLITUDE_A_KEY, GATE_COUNTER_KEY])
        w.writeheader()
        w.writerow({FIT_AMPLITUDE_A_KEY: "1", GATE_COUNTER_KEY: "1"})
    with pytest.raises(ValueError, match="Fit Amplitude B"):
        _probe_csv_columns_for_measured_weights(
            p, aggregate_spots=False, spot_weight_mode="fit_amplitude_b"
        )


def test_measured_spot_weight_from_row_falls_back_to_channel_sum() -> None:
    row = {CHANNEL_SUM_KEY: "2.5", FIT_AMPLITUDE_B_KEY: ""}
    assert analysis.measured_spot_weight_from_row(row, "fit_amplitude_b") == pytest.approx(2.5)
