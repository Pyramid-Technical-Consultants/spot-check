"""PSTAR CSDA range in water (mm) for proton display Z axis."""

from __future__ import annotations

import numpy as np
import pytest

from spot_check.geometry.proton_csda_water import (
    _PSTAR_ENERGY_MEV,
    _PSTAR_RANGE_MM,
    normalize_z_depth_metric,
    proton_csda_water_range_mm,
    proton_water_depth_mm,
)

# Spot-check tabulated knots (UCL / NIST PSTAR, liquid water).
_PSTAR_SPOTCHECK_MEV_MM: tuple[tuple[float, float], ...] = (
    (10.0, 1.23),
    (50.0, 22.27),
    (70.0, 40.80),
    (100.0, 77.18),
    (150.0, 157.70),
    (200.0, 259.60),
    (230.0, 329.50),
    (350.0, 662.80),
)


@pytest.mark.parametrize("energy_mev, range_mm", _PSTAR_SPOTCHECK_MEV_MM)
def test_pstar_table_knots(energy_mev: float, range_mm: float) -> None:
    got = float(proton_csda_water_range_mm(energy_mev))
    assert got == pytest.approx(range_mm, rel=0, abs=0.01)


def test_pstar_table_arrays_match_constants() -> None:
    assert len(_PSTAR_ENERGY_MEV) == len(_PSTAR_RANGE_MM)
    assert float(_PSTAR_ENERGY_MEV[0]) == 10.0
    i70 = int(np.searchsorted(_PSTAR_ENERGY_MEV, 70.0))
    assert float(_PSTAR_ENERGY_MEV[i70]) == 70.0
    assert float(_PSTAR_RANGE_MM[i70]) == pytest.approx(40.80, abs=0.01)


def test_interpolation_between_knots() -> None:
    r72 = float(proton_csda_water_range_mm(72.5))
    assert 40.80 < r72 < 46.18
    assert r72 == pytest.approx(43.49, abs=0.05)


def test_monotonic_in_therapeutic_band() -> None:
    e = np.linspace(50.0, 250.0, 41)
    r = proton_csda_water_range_mm(e)
    assert np.all(np.diff(r) > 0)


def test_r80_r90_shallower_than_csda_at_70mev() -> None:
    csda = float(proton_water_depth_mm(70.0, metric="csda"))
    r90 = float(proton_water_depth_mm(70.0, metric="r90"))
    r80 = float(proton_water_depth_mm(70.0, metric="r80"))
    assert csda == pytest.approx(40.80, abs=0.01)
    assert r80 < r90 < csda
    assert r90 == pytest.approx(39.18, abs=0.15)
    assert r80 == pytest.approx(35.99, abs=0.15)


def test_normalize_z_depth_metric() -> None:
    assert normalize_z_depth_metric("R90") == "r90"
    assert normalize_z_depth_metric("unknown") == "csda"


def test_vectorized_matches_scalar() -> None:
    e = np.array([70.0, 100.0, 177.5])
    rv = proton_csda_water_range_mm(e)
    for i, ei in enumerate(e):
        assert float(rv[i]) == pytest.approx(float(proton_csda_water_range_mm(ei)), rel=0, abs=1e-6)
