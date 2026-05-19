"""Pyramid plan CSV loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spot_check.exceptions import PlanDataError
from spot_check.plan import (
    is_pyramid_plan_csv,
    is_supported_plan_file,
    plan_label_from_pyramid_csv_stem,
    planned_spot_xyz_and_counts_from_plan,
    planned_spot_xyz_and_counts_from_pyramid_csv,
)

_FIXTURE = Path(__file__).resolve().parent.parent / "test_data" / "R20M10_cube_original.csv"


@pytest.mark.skipif(not _FIXTURE.is_file(), reason="Pyramid plan fixture missing")
def test_pyramid_plan_fixture_loads() -> None:
    assert is_pyramid_plan_csv(_FIXTURE)
    assert is_supported_plan_file(_FIXTURE)
    planned, fwhm, mu, n_kept, n_raw = planned_spot_xyz_and_counts_from_pyramid_csv(_FIXTURE)
    assert n_raw == 8921
    assert n_kept == n_raw
    assert len(planned) == n_kept
    assert planned[0][0] == pytest.approx(46.8011741638)
    assert planned[0][1] == pytest.approx(-51.0839118958)
    assert planned[0][2] == pytest.approx(177.5)
    assert mu[0] == pytest.approx(0.1668692751)
    assert fwhm is not None
    assert fwhm.shape == (n_kept, 2)
    assert fwhm[0, 0] == pytest.approx(3.6155875802)
    assert fwhm[0, 1] == pytest.approx(3.6155875802)


def test_plan_label_from_pyramid_stem() -> None:
    assert plan_label_from_pyramid_csv_stem("R20M10_cube_original") == "R20M10"
    assert plan_label_from_pyramid_csv_stem("custom_plan") == "custom_plan"


def test_pyramid_plan_drops_nonpositive_charge(tmp_path: Path) -> None:
    path = tmp_path / "tiny_cube.csv"
    path.write_text(
        "#NO,ENERGY(MeV),X_POSITION(mm),Y_POSITION(mm),CHARGE_REQ(MU),BEAM_SIZE(mm)\n"
        "1,100,0,0,1.0,2.0\n"
        "2,100,1,0,0,2.0\n"
        "3,100,2,0,-1,2.0\n",
        encoding="utf-8",
    )
    planned, _, mu, n_kept, n_raw = planned_spot_xyz_and_counts_from_pyramid_csv(path)
    assert n_raw == 3
    assert n_kept == 1
    assert len(planned) == 1
    assert mu.shape == (1,)
    assert mu[0] == pytest.approx(1.0)


def test_unsupported_plan_csv_raises(tmp_path: Path) -> None:
    path = tmp_path / "acq.csv"
    path.write_text(
        "time (s),IX512 Channel Sum (nA),Fit Amplitude A (nA)\n0,1,0.5\n",
        encoding="utf-8",
    )
    assert not is_pyramid_plan_csv(path)
    with pytest.raises(PlanDataError, match="Unsupported plan"):
        planned_spot_xyz_and_counts_from_plan(path)


@pytest.mark.skipif(not _FIXTURE.is_file(), reason="Pyramid plan fixture missing")
def test_dispatch_matches_pyramid_loader() -> None:
    a = planned_spot_xyz_and_counts_from_pyramid_csv(_FIXTURE)
    b = planned_spot_xyz_and_counts_from_plan(_FIXTURE)
    assert a[0] == b[0]
    assert a[3] == b[3]
    assert a[4] == b[4]
    np.testing.assert_array_equal(a[1], b[1])
    np.testing.assert_array_equal(a[2], b[2])
