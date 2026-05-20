"""Z-axis mapping and cube-axes tick density for 3D views."""

from __future__ import annotations

import math

import numpy as np

from spot_check.constants import (
    _ENERGY_AXIS_VIEW_SCALE,
    BOUNDS_XY_LABELS_MAX,
    BOUNDS_Z_TICK_MEV_DEFAULT,
    BOUNDS_Z_TICK_MM_DEFAULT,
)
from spot_check.geometry.proton_csda_water import (
    normalize_z_depth_metric,
    proton_water_depth_mm,
)
from spot_check.models import CubeZAxisSpec


def label_at_scene_z(actor: object, z_scene: float) -> float | None:
    """Interpolate the displayed Z tick value at a scene-Z position.

    VTK places tick index 0 at scene ``zmin`` and the last index at ``zmax``; labels
    are built from ``z_label_at_min`` to ``z_label_at_max`` along those bounds.
    """
    try:
        zl = [float(actor.z_labels[i]) for i in range(len(actor.z_labels))]  # type: ignore[attr-defined]
        bb = actor.bounds  # type: ignore[attr-defined]
        zmin, zmax = float(bb[4]), float(bb[5])
    except Exception:
        return None
    if len(zl) < 2 or zmin == zmax:
        return None
    lbl_at_zmin, lbl_at_zmax = float(zl[0]), float(zl[-1])
    frac = (float(z_scene) - zmin) / (zmax - zmin)
    frac = float(np.clip(frac, 0.0, 1.0))
    return float(lbl_at_zmin + (lbl_at_zmax - lbl_at_zmin) * frac)


def _label_at_scene_z(actor: object, z_scene: float) -> float | None:
    """Alias for :func:`label_at_scene_z` (tests and internal callers)."""
    return label_at_scene_z(actor, z_scene)


def cube_z_axis_label_endpoints(z_spec: CubeZAxisSpec) -> tuple[float, float]:
    """Tick label at scene ``zmin`` then ``zmax`` (VTK index 0 is at scene zmin)."""
    return float(z_spec.z_label_at_min), float(z_spec.z_label_at_max)


def nominal_mev_to_plot_z(
    energy_mev: np.ndarray,
    *,
    use_proton_water_depth_mm: bool,
    upstream_wet_mm: float = 0.0,
    z_depth_metric: str = "csda",
) -> np.ndarray:
    """Map nominal energy (MeV) to scene Z for 3D display (mm depth or scaled MeV).

    Scene Z is **negative** depth (shallow / low energy toward +Z, i.e. top of the view).
    Cube tick labels use positive mm via :func:`cube_z_axis_spec`.

    When ``use_proton_water_depth_mm`` is true, depth uses ``z_depth_metric`` (CSDA / R90 / R80)
    minus ``upstream_wet_mm`` water-equivalent shifter thickness (beam stops shallower).
    """
    e = np.asarray(energy_mev, dtype=np.float64)
    if use_proton_water_depth_mm:
        wet = max(0.0, float(upstream_wet_mm))
        metric = normalize_z_depth_metric(z_depth_metric)
        depth_mm = proton_water_depth_mm(e, metric=metric) - wet
        return -np.maximum(depth_mm, 0.0)
    return -e * float(_ENERGY_AXIS_VIEW_SCALE)


def nominal_mev_to_scene_z_mev_cube(
    energy_mev: np.ndarray,
    *,
    e_lo: float,
    e_hi: float,
) -> np.ndarray:
    """Positive scene Z: high nominal MeV (deep) at ``zmin`` (``bounds == axes_ranges`` ticks)."""
    e = np.asarray(energy_mev, dtype=np.float64)
    return float(e_hi) + float(e_lo) - e


def plan_depth_bounds_mm(
    e_lo_mev: float,
    e_hi_mev: float,
    *,
    upstream_wet_mm: float = 0.0,
    z_depth_metric: str = "csda",
) -> tuple[float, float]:
    """Shallow and deep plan water-depth bounds (mm) for cube scene-Z mapping."""
    wet = max(0.0, float(upstream_wet_mm))
    metric = normalize_z_depth_metric(z_depth_metric)
    depths = np.maximum(
        proton_water_depth_mm(
            np.array([float(e_lo_mev), float(e_hi_mev)], dtype=np.float64),
            metric=metric,
        )
        - wet,
        0.0,
    )
    return float(np.min(depths)), float(np.max(depths))


def nominal_depth_to_scene_z_cube(
    energy_mev: np.ndarray,
    *,
    upstream_wet_mm: float = 0.0,
    z_depth_metric: str = "csda",
    depth_lo_mm: float | None = None,
    depth_hi_mm: float | None = None,
) -> np.ndarray:
    """Positive scene Z: deep layers (large mm) at ``zmin``; ``bounds == axes_ranges`` ticks.

    When ``depth_lo_mm`` / ``depth_hi_mm`` are set, use those plan-wide bounds instead of
    min/max from ``energy_mev`` alone (required for per-spot QA error lines).
    """
    e = np.asarray(energy_mev, dtype=np.float64)
    wet = max(0.0, float(upstream_wet_mm))
    metric = normalize_z_depth_metric(z_depth_metric)
    depth = np.maximum(proton_water_depth_mm(e, metric=metric) - wet, 0.0)
    if depth.size == 0:
        return depth
    if depth_lo_mm is not None and depth_hi_mm is not None:
        d_lo = float(depth_lo_mm)
        d_hi = float(depth_hi_mm)
    else:
        d_lo = float(np.min(depth))
        d_hi = float(np.max(depth))
    if d_hi <= d_lo:
        d_hi = d_lo + 1.0
    return float(d_hi) + float(d_lo) - depth


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


def _tick_labels_at_scene_bounds(
    z_scene: np.ndarray,
    tick_values: np.ndarray,
    zmin_scene: float,
    zmax_scene: float,
) -> tuple[float, float]:
    """Tick labels at cube ``SetBounds`` Z endpoints (VTK index 0 is at ``zmin_scene``)."""
    z = np.asarray(z_scene, dtype=np.float64).reshape(-1)
    tv = np.asarray(tick_values, dtype=np.float64).reshape(-1)
    if z.size == 0 or tv.size == 0:
        return 0.0, 1.0
    if z.size != tv.size:
        return float(np.min(tv)), float(np.max(tv))

    order = np.argsort(z)
    zs = z[order]
    tvs = tv[order]
    # Plan spots often share layer Z; keep one tick value per scene Z for interp.
    uz: list[float] = []
    utv: list[float] = []
    for zi, ti in zip(zs, tvs, strict=True):
        if uz and float(zi) == uz[-1]:
            utv[-1] = float(ti)
        else:
            uz.append(float(zi))
            utv.append(float(ti))
    if len(uz) == 1:
        v = utv[0]
        return v, v

    zs_u = np.asarray(uz, dtype=np.float64)
    tvs_u = np.asarray(utv, dtype=np.float64)

    def _at(zq: float) -> float:
        zq = float(zq)
        if zq <= float(zs_u[0]):
            dz = float(zs_u[1] - zs_u[0])
            if abs(dz) < 1e-12:
                return float(tvs_u[0])
            slope = float((tvs_u[1] - tvs_u[0]) / dz)
            return float(tvs_u[0] + slope * (zq - zs_u[0]))
        if zq >= float(zs_u[-1]):
            dz = float(zs_u[-1] - zs_u[-2])
            if abs(dz) < 1e-12:
                return float(tvs_u[-1])
            slope = float((tvs_u[-1] - tvs_u[-2]) / dz)
            return float(tvs_u[-1] + slope * (zq - zs_u[-1]))
        return float(np.interp(zq, zs_u, tvs_u))

    return _at(zmin_scene), _at(zmax_scene)


def cube_z_axis_spec(
    z_scene: np.ndarray,
    *,
    use_proton_water_depth_mm: bool,
    tick_mm: float,
    tick_mev: float = BOUNDS_Z_TICK_MEV_DEFAULT,
    nominal_energy_mev: np.ndarray | None = None,
    upstream_wet_mm: float = 0.0,
    z_depth_metric: str = "csda",
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
        zmin_p = zmin_b
        zmax_p = zmax_b
        if nominal_energy_mev is not None:
            e_lbl = np.asarray(nominal_energy_mev, dtype=np.float64).reshape(-1)
        else:
            e_lbl = np.array([], dtype=np.float64)
        if e_lbl.size > 0:
            wet = max(0.0, float(upstream_wet_mm))
            metric = normalize_z_depth_metric(z_depth_metric)
            depth_mm = proton_water_depth_mm(e_lbl, metric=metric) - wet
            depth_mm = np.maximum(depth_mm, 0.0)
            z_lbl_min, z_lbl_max = _tick_labels_at_scene_bounds(z, depth_mm, zmin_p, zmax_p)
        else:
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
    zmin_p = zmin_b
    zmax_p = zmax_b
    if nominal_energy_mev is not None:
        e_lbl = np.asarray(nominal_energy_mev, dtype=np.float64).reshape(-1)
    else:
        e_lbl = np.array([], dtype=np.float64)
    if e_lbl.size > 0:
        z_lbl_min, z_lbl_max = _tick_labels_at_scene_bounds(z, e_lbl, zmin_p, zmax_p)
    else:
        z_lbl_min = -zmin_p / s_view
        z_lbl_max = -zmax_p / s_view
    n_z = n_cube_axis_labels_for_mm_step(z_lbl_min, z_lbl_max, float(tick_mev))
    return CubeZAxisSpec(zmin_p, zmax_p, z_lbl_min, z_lbl_max, n_z, "Z (MeV)")
