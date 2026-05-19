"""Shared fixtures for measured / pipeline integration tests."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from pathlib import Path

import pytest

from spot_check.constants import (
    CHANNEL_SUM_KEY,
    FIT_AMPLITUDE_A_KEY,
    GATE_COUNTER_KEY,
)

# Two nominal energy layers with four plan spots (exercises layer bucketing).
MINIMAL_PLANNED_XYZ: list[tuple[float, float, float]] = [
    (0.0, 0.0, 100.0),
    (5.0, 0.0, 100.0),
    (0.0, 5.0, 120.0),
    (5.0, 5.0, 120.0),
]

MEASURED_CSV_FIELDNAMES: tuple[str, ...] = (
    "time (s)",
    CHANNEL_SUM_KEY,
    FIT_AMPLITUDE_A_KEY,
    "Fit Mean Position A (mm)",
    "Fit Mean Position B (mm)",
    GATE_COUNTER_KEY,
)


def write_measured_csv(
    path: Path,
    rows: Iterable[Mapping[str, str]],
    *,
    fieldnames: tuple[str, ...] = MEASURED_CSV_FIELDNAMES,
) -> Path:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    return path


def minimal_measured_rows() -> list[dict[str, str]]:
    """Rows on-plan with monotonic time (suitable for time-gap / Viterbi modes)."""
    return [
        {
            "time (s)": "0.0",
            CHANNEL_SUM_KEY: "1.0",
            FIT_AMPLITUDE_A_KEY: "0.5",
            "Fit Mean Position A (mm)": "0.0",
            "Fit Mean Position B (mm)": "0.0",
            GATE_COUNTER_KEY: "1",
        },
        {
            "time (s)": "0.05",
            CHANNEL_SUM_KEY: "1.0",
            FIT_AMPLITUDE_A_KEY: "0.5",
            "Fit Mean Position A (mm)": "5.0",
            "Fit Mean Position B (mm)": "0.0",
            GATE_COUNTER_KEY: "1",
        },
        {
            "time (s)": "0.5",
            CHANNEL_SUM_KEY: "1.0",
            FIT_AMPLITUDE_A_KEY: "0.5",
            "Fit Mean Position A (mm)": "0.0",
            "Fit Mean Position B (mm)": "5.0",
            GATE_COUNTER_KEY: "3",
        },
    ]


@pytest.fixture
def minimal_planned_xyz() -> list[tuple[float, float, float]]:
    return list(MINIMAL_PLANNED_XYZ)


@pytest.fixture
def measured_csv_writer(tmp_path: Path):
    def _write(
        rows: Iterable[Mapping[str, str]] | None = None,
        name: str = "measured.csv",
    ) -> Path:
        data = list(minimal_measured_rows() if rows is None else rows)
        return write_measured_csv(tmp_path / name, data)

    return _write
