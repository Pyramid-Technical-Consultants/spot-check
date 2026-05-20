"""Z-axis mapping and cube-axes tick density for 3D views."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from spot_check.constants import (
    _ENERGY_AXIS_VIEW_SCALE,
    BOUNDS_XY_LABELS_MAX,
    BOUNDS_Z_TICK_MEV_DEFAULT,
    BOUNDS_Z_TICK_MM_DEFAULT,
)
from spot_check.geometry.cube_axes_style import (
    PYVISTA_CUBE_AXES_LABEL_OFFSET,
    PYVISTA_CUBE_Z_TICK_LABEL_ORIENTATION,
    apply_pyvista_cube_axes_style,
)
from spot_check.geometry.proton_csda_water import (
    normalize_z_depth_metric,
    proton_water_depth_mm,
)
from spot_check.models import CubeZAxisSpec


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
        z_pad = max(z_span * 0.06, min_pad)
        zmin_p = zmin_b - z_pad * 0.5
        zmax_p = zmax_b + z_pad * 0.5
        # Tick labels must be PSTAR depth (mm), not ``-scene_z`` (confused with raw MeV).
        if nominal_energy_mev is not None:
            e_lbl = np.asarray(nominal_energy_mev, dtype=np.float64).reshape(-1)
        else:
            e_lbl = np.array([], dtype=np.float64)
        if e_lbl.size > 0:
            wet = max(0.0, float(upstream_wet_mm))
            metric = normalize_z_depth_metric(z_depth_metric)
            depth_mm = proton_water_depth_mm(e_lbl, metric=metric) - wet
            depth_mm = np.maximum(depth_mm, 0.0)
            z_lbl_min = float(np.max(depth_mm))
            z_lbl_max = float(np.min(depth_mm))
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
    z_pad = max(z_span * 0.06, min_pad)
    zmin_p = zmin_b - z_pad * 0.5
    zmax_p = zmax_b + z_pad * 0.5
    e_at_zmin = -zmin_p / s_view
    e_at_zmax = -zmax_p / s_view
    n_z = n_cube_axis_labels_for_mm_step(e_at_zmin, e_at_zmax, float(tick_mev))
    return CubeZAxisSpec(zmin_p, zmax_p, e_at_zmin, e_at_zmax, n_z, "Z (MeV)")


def _cube_z_depth_label_endpoints(z_spec: CubeZAxisSpec) -> tuple[float, float]:
    """Positive depth (mm) at scene zmin (deep) and zmax (shallow).

    ``z_label_at_min`` is depth at the most-negative scene Z (deepest); it is the larger mm value.
    """
    deep = float(max(z_spec.z_label_at_min, z_spec.z_label_at_max))
    shallow = float(min(z_spec.z_label_at_min, z_spec.z_label_at_max))
    return deep, shallow


def _vtk_z_tick_labels(z_spec: CubeZAxisSpec, *, fmt: str = "%.4g") -> Any:
    """Build vtkStringArray for Z ticks (deep→shallow along scene zmin→zmax)."""
    from pyvista.plotting.cube_axes_actor import make_axis_labels

    deep, shallow = _cube_z_depth_label_endpoints(z_spec)
    # VTK maps tick index 0 to z_axis_range[0] at scene zmin (deepest); use descending labels.
    return make_axis_labels(
        vmin=deep,
        vmax=shallow,
        n=int(z_spec.n_zlabels),
        fmt=fmt,
    )


def apply_pyvista_cube_z_axis(
    actor: Any,
    z_spec: CubeZAxisSpec,
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> None:
    """Apply scene bounds and Z tick labels for water-depth or MeV cube axes.

    PyVista's ``actor.bounds`` setter copies scene Z into ``z_axis_range`` (negative mm).
    Use VTK ``SetBounds`` plus explicit ``z_axis_range`` so labels stay positive depth mm.

    Use ascending ``SetZAxisRange(shallow_mm, deep_mm)`` so VTK draws Z ticks, then
    ``SetAxisLabels`` with deep→shallow strings (index 0 at scene zmin = deepest).
    Do **not** assign ``actor.z_axis_range`` afterward — PyVista's setter calls
    ``SetZAxisRange`` again and ``_update_z_labels()``, which clears custom labels.
    Do not call ``SetRebuildAxes`` afterward.
    """
    apply_pyvista_cube_axes_style(actor)
    actor.SetBounds(
        float(x_min),
        float(x_max),
        float(y_min),
        float(y_max),
        float(z_spec.zmin_scene),
        float(z_spec.zmax_scene),
    )
    actor.x_axis_range = (float(x_min), float(x_max))
    actor.y_axis_range = (float(y_min), float(y_max))
    actor._n_zlabels = int(z_spec.n_zlabels)
    actor.SetZTitle(str(z_spec.ztitle))
    deep_mm, shallow_mm = _cube_z_depth_label_endpoints(z_spec)
    actor.SetZAxisRange(shallow_mm, deep_mm)
    actor.SetAxisLabels(2, _vtk_z_tick_labels(z_spec))
    actor._z_label_visibility = True
    actor.SetZAxisVisibility(True)
    actor.SetZAxisTickVisibility(True)
    try:
        actor.GetZAxesLabelProperty().SetOrientation(
            float(PYVISTA_CUBE_Z_TICK_LABEL_ORIENTATION)
        )
    except Exception:
        pass
    try:
        actor.SetLabelOffset(float(PYVISTA_CUBE_AXES_LABEL_OFFSET))
    except Exception:
        pass
    try:
        actor.SetUseTextActor3D(False)
    except Exception:
        pass
    try:
        actor.SetEnableViewAngleLOD(False)
        actor.SetEnableDistanceLOD(False)
    except Exception:
        pass
