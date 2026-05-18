"""Z-axis mapping and cube-axes tick density for 3D views."""

from __future__ import annotations

import math

import numpy as np

from spot_check.constants import (
    _ENERGY_AXIS_VIEW_SCALE,
    BOUNDS_XY_LABELS_MAX,
    BOUNDS_Z_TICK_MEV_DEFAULT,
    BOUNDS_Z_TICK_MM_DEFAULT,
    PROTON_WATER_CSDA_RANGE_MM_COEFF,
    PROTON_WATER_CSDA_RANGE_MM_POW,
)
from spot_check.models import CubeZAxisSpec


def proton_cda_water_range_mm(energy_mev: np.ndarray | float) -> np.ndarray:
    e = np.maximum(np.asarray(energy_mev, dtype=np.float64), 0.05)
    return PROTON_WATER_CSDA_RANGE_MM_COEFF * np.power(e, PROTON_WATER_CSDA_RANGE_MM_POW)


def nominal_mev_to_plot_z(
    energy_mev: np.ndarray,
    *,
    use_proton_water_depth_mm: bool,
) -> np.ndarray:
    e = np.asarray(energy_mev, dtype=np.float64)
    if use_proton_water_depth_mm:
        return -proton_cda_water_range_mm(e)
    return -e * float(_ENERGY_AXIS_VIEW_SCALE)


def n_cube_axis_labels_for_mm_step(
    vmin: float,
    vmax: float,
    step_mm: float,
    *,
    max_n: int = BOUNDS_XY_LABELS_MAX,
) -> int:
    if step_mm <= 0 or not math.isfinite(step_mm):
        return 5
    lo, hi = float(min(vmin, vmax)), float(max(vmin, vmax))
    span = hi - lo
    if not math.isfinite(span) or span <= 0:
        return 5
    n = int(math.ceil(span / step_mm)) + 1
    return max(5, min(n, max_n))


def cube_z_axis_spec(
    z_scene: np.ndarray,
    *,
    use_proton_water_depth_mm: bool,
    tick_mm: float,
    tick_mev: float = BOUNDS_Z_TICK_MEV_DEFAULT,
) -> CubeZAxisSpec:
    z = np.asarray(z_scene, dtype=np.float64).reshape(-1)
    if z.size == 0:
        zmin_b, zmax_b = 0.0, 1.0
    else:
        zmin_b = float(np.min(z))
        zmax_b = float(np.max(z))
    z_span = zmax_b - zmin_b
    if use_proton_water_depth_mm:
        min_pad = 0.5
        if not math.isfinite(z_span) or z_span <= 0.0:
            z_span = min_pad * 2.0
            zmin_b -= min_pad
            zmax_b += min_pad
        z_pad = max(z_span * 0.06, min_pad)
        zmin_p = zmin_b - z_pad * 0.5
        zmax_p = zmax_b + z_pad * 0.5
        z_lbl_min = -zmin_p
        z_lbl_max = -zmax_p
        z_step = (
            float(tick_mm)
            if tick_mm > 0.0 and math.isfinite(tick_mm)
            else float(BOUNDS_Z_TICK_MM_DEFAULT)
        )
        n_z = n_cube_axis_labels_for_mm_step(z_lbl_min, z_lbl_max, z_step)
        return CubeZAxisSpec(zmin_p, zmax_p, z_lbl_min, z_lbl_max, n_z, "Water depth (mm)")
    s_view = float(_ENERGY_AXIS_VIEW_SCALE)
    min_pad = 0.5 * s_view
    if not math.isfinite(z_span) or z_span <= 0.0:
        z_span = min_pad * 2.0
        zmin_b -= min_pad
        zmax_b += min_pad
    z_pad = max(z_span * 0.06, min_pad)
    zmin_p = zmin_b - z_pad * 0.5
    zmax_p = zmax_b + z_pad * 0.5
    e_at_zmin = -zmin_p / s_view
    e_at_zmax = -zmax_p / s_view
    n_z = n_cube_axis_labels_for_mm_step(e_at_zmin, e_at_zmax, float(tick_mev))
    return CubeZAxisSpec(zmin_p, zmax_p, e_at_zmin, e_at_zmax, n_z, "Z (MeV)")
