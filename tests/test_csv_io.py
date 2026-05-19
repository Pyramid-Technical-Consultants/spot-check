"""Acquisition CSV I/O (.csv.gz)."""

from __future__ import annotations

import gzip
from pathlib import Path

from spot_check import analysis
from spot_check.analysis.csv_io import acquisition_csv_stem, open_acquisition_csv
from tests.conftest import MINIMAL_PLANNED_XYZ, minimal_measured_rows, write_measured_csv


def test_acquisition_csv_stem() -> None:
    assert acquisition_csv_stem(Path("foo.csv")) == "foo"
    assert acquisition_csv_stem(Path("foo.csv.gz")) == "foo"
    assert acquisition_csv_stem(Path("15186535_T0G40_data.csv.gz")) == "15186535_T0G40_data"


def test_measured_spot_abc_from_csv_gz_matches_plain(tmp_path: Path) -> None:
    plain = write_measured_csv(tmp_path / "spots.csv", minimal_measured_rows())
    gz_path = tmp_path / "spots.csv.gz"
    with gzip.open(gz_path, "wb") as zf:
        zf.write(plain.read_bytes())

    kwargs = dict(
        planned_xyz=list(MINIMAL_PLANNED_XYZ),
        layer_mode="time_gap",
        a_is_x=False,
        layer_gap_s=0.2,
        refill_same_spot_xy_tol_mm=3.0,
        refill_trust_time_gap_stay_dist_mm=35.0,
        viterbi_advance_penalty_mm2=400.0,
    )
    rows_plain = analysis.measured_spot_abc_from_csv(plain, **kwargs)
    rows_gz = analysis.measured_spot_abc_from_csv(gz_path, **kwargs)
    assert len(rows_gz) == len(rows_plain)
    for rg, rp in zip(rows_gz, rows_plain, strict=True):
        assert rg[:5] == rp[:5]
        assert rg[7] == rp[7]


def test_open_acquisition_csv_rejects_unknown_suffix(tmp_path: Path) -> None:
    bad = tmp_path / "spots.txt"
    bad.write_text("x", encoding="utf-8")
    try:
        with open_acquisition_csv(bad):
            pass
    except ValueError as exc:
        assert ".csv" in str(exc)
    else:
        raise AssertionError("expected ValueError")
