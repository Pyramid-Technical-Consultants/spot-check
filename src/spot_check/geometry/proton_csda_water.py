"""Proton depth in liquid water: PSTAR CSDA and empirical R90/R80 (display / QA only).

CSDA tabulated energies and ranges (mm) match the UCL PBT wiki table derived from PSTAR
(https://www.hep.ucl.ac.uk/pbt/wiki/Proton_ranges). R90/R80 are linear corrections to
CSDA (distal 90%/80% dose falloff depths in water, ~50–230 MeV).

**Not for clinical range verification.**
"""

from __future__ import annotations

import numpy as np

from spot_check.constants import (
    PROTON_R80_FROM_CSDA_OFFSET_MM,
    PROTON_R80_FROM_CSDA_SCALE,
    PROTON_R90_FROM_CSDA_OFFSET_MM,
    PROTON_R90_FROM_CSDA_SCALE,
    Z_DEPTH_METRIC_DEFAULT,
    Z_DEPTH_METRICS,
)

# NIST PSTAR — liquid water, CSDA range (mm); 5 MeV steps, 10–350 MeV.
_PSTAR_ENERGY_MEV: np.ndarray = np.arange(10.0, 351.0, 5.0, dtype=np.float64)
_PSTAR_RANGE_MM: np.ndarray = np.array(
    [
        1.23,
        2.54,
        4.26,
        6.37,
        8.85,
        11.70,
        14.89,
        18.41,
        22.27,
        26.44,
        30.93,
        35.72,
        40.80,
        46.18,
        51.84,
        57.77,
        63.98,
        70.45,
        77.18,
        84.16,
        91.40,
        98.88,
        106.60,
        114.60,
        122.80,
        131.20,
        139.80,
        148.70,
        157.70,
        167.00,
        176.50,
        186.20,
        196.10,
        206.20,
        216.50,
        227.00,
        237.70,
        248.50,
        259.60,
        270.80,
        282.20,
        293.80,
        305.50,
        317.40,
        329.50,
        341.70,
        354.10,
        366.70,
        379.40,
        392.20,
        405.30,
        418.40,
        431.70,
        445.20,
        458.80,
        472.50,
        486.40,
        500.40,
        514.50,
        528.80,
        543.20,
        557.70,
        572.40,
        587.10,
        602.00,
        617.00,
        632.20,
        647.40,
        662.80,
    ],
    dtype=np.float64,
)

_E_TAB_MIN = float(_PSTAR_ENERGY_MEV[0])
_E_TAB_MAX = float(_PSTAR_ENERGY_MEV[-1])
# Low-E extrapolation: R(mm) = _LOW_K * E^_LOW_P through (10 MeV, 1.23 mm) and (20 MeV, 4.26 mm).
_LOW_P = float(np.log(_PSTAR_RANGE_MM[2] / _PSTAR_RANGE_MM[0]) / np.log(20.0 / 10.0))
_LOW_K = float(_PSTAR_RANGE_MM[0] / (_E_TAB_MIN**_LOW_P))
# High-E extrapolation: same form anchored at (350 MeV, 662.8 mm) and (340 MeV, 632.2 mm).
_HIGH_P = float(np.log(_PSTAR_RANGE_MM[-1] / _PSTAR_RANGE_MM[-3]) / np.log(350.0 / 340.0))
_HIGH_K = float(_PSTAR_RANGE_MM[-1] / (_E_TAB_MAX**_HIGH_P))


def proton_csda_water_range_mm(energy_mev: np.ndarray | float) -> np.ndarray:
    """CSDA range in water (mm) for mono-energetic protons at ``energy_mev`` (MeV)."""
    e = np.maximum(np.asarray(energy_mev, dtype=np.float64), 1e-6)
    out = np.empty_like(e, dtype=np.float64)
    low = e < _E_TAB_MIN
    mid = (~low) & (e <= _E_TAB_MAX)
    high = e > _E_TAB_MAX
    if np.any(low):
        out[low] = _LOW_K * np.power(e[low], _LOW_P)
    if np.any(mid):
        out[mid] = np.interp(e[mid], _PSTAR_ENERGY_MEV, _PSTAR_RANGE_MM)
    if np.any(high):
        out[high] = _HIGH_K * np.power(e[high], _HIGH_P)
    return out


def normalize_z_depth_metric(metric: str) -> str:
    """Return ``csda``, ``r90``, or ``r80``; unknown values default to CSDA."""
    key = str(metric).strip().lower()
    if key in Z_DEPTH_METRICS:
        return key
    return Z_DEPTH_METRIC_DEFAULT


def proton_water_depth_mm(
    energy_mev: np.ndarray | float,
    *,
    metric: str = Z_DEPTH_METRIC_DEFAULT,
) -> np.ndarray:
    """Depth in water (mm) for mono-energetic protons at ``energy_mev`` (MeV).

    ``metric`` is ``csda`` (PSTAR CSDA range), ``r90``, or ``r80`` (empirical vs CSDA).
    """
    csda = proton_csda_water_range_mm(energy_mev)
    m = normalize_z_depth_metric(metric)
    if m == "csda":
        return csda
    if m == "r90":
        return np.maximum(
            0.0,
            PROTON_R90_FROM_CSDA_SCALE * csda - PROTON_R90_FROM_CSDA_OFFSET_MM,
        )
    if m == "r80":
        return np.maximum(
            0.0,
            PROTON_R80_FROM_CSDA_SCALE * csda - PROTON_R80_FROM_CSDA_OFFSET_MM,
        )
    return csda


# Backward-compatible alias (historical typo: cda vs csda).
proton_cda_water_range_mm = proton_csda_water_range_mm
