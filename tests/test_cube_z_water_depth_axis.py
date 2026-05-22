"""Regression tests: water-depth Z axis, cube tick labels, and MeV vs mm confusion.

Guards against:
- Plotting raw nominal MeV on the scene Z axis while ticks show mm depth.
- Slice / ``update_bounds`` shrinking Z ticks to the visible layer band.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from spot_check.geometry import (
    cube_z_axis_label_endpoints,
    cube_z_axis_spec,
    cube_z_axis_spec_for_display,
    heal_plan_cube_axes,
    nominal_energy_to_scene_z,
    plan_cube_scene_bounds_and_axes_ranges,
    plan_depth_bounds_mm,
    refresh_pyvista_cube_axes,
)
from spot_check.geometry.proton_csda_water import proton_water_depth_mm
from spot_check.geometry.z_axis import label_at_scene_z
from spot_check.models import CubeZAxisSpec, ZAxisDisplayConfig

_T0G10_DCM = Path("test_data/RN.15186535.T0G10.dcm")
_T0G40_DCM = Path("test_data/RN.15186535.T0G40.dcm")

# MeV-like magnitudes on a depth (mm) axis — catches axis/spot unit mismatch.
_MEV_DEPTH_CONFUSION_MM = 12.0


def _pyvista_off_screen():
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    pv = pytest.importorskip("pyvista")
    pv.OFF_SCREEN = True
    return pv


def _tick_depths_mm(actor) -> list[float]:
    return [float(actor.z_labels[i]) for i in range(len(actor.z_labels))]


def _cube_spec_from_energies(
    energies_mev: np.ndarray,
    *,
    tick_mm: float = 5.0,
    tick_mev: float = 5.0,
    use_water_depth: bool = True,
    upstream_wet_mm: float = 0.0,
    z_depth_metric: str = "csda",
) -> CubeZAxisSpec:
    e = np.asarray(energies_mev, dtype=np.float64).reshape(-1)
    e_lo, e_hi = float(np.min(e)), float(np.max(e))
    cfg = ZAxisDisplayConfig(
        use_water_depth_mm=use_water_depth,
        upstream_wet_mm=upstream_wet_mm,
        z_depth_metric=z_depth_metric,
        tick_mm=tick_mm,
        tick_mev=tick_mev,
    )
    if use_water_depth:
        d_lo, d_hi = plan_depth_bounds_mm(
            e_lo,
            e_hi,
            upstream_wet_mm=upstream_wet_mm,
            z_depth_metric=z_depth_metric,
        )
        z = nominal_energy_to_scene_z(
            e,
            plan_e_lo=e_lo,
            plan_e_hi=e_hi,
            config=cfg,
            depth_lo_mm=d_lo,
            depth_hi_mm=d_hi,
        )
    else:
        z = nominal_energy_to_scene_z(e, plan_e_lo=e_lo, plan_e_hi=e_hi, config=cfg)
    return cube_z_axis_spec_for_display(z, e, cfg)


def _interpolate_tick_depth_mm(
    z_scene: float,
    spec: CubeZAxisSpec,
    tick_depth_mm: Sequence[float],
) -> float:
    """Depth (mm) implied by tick strings (deep at index 0, shallow at last)."""
    deep_lbl, shallow_lbl = cube_z_axis_label_endpoints(spec)
    zmin_s, zmax_s = float(spec.zmin_scene), float(spec.zmax_scene)
    span_s = zmax_s - zmin_s
    if span_s == 0.0:
        return float(tick_depth_mm[0])
    frac = (float(z_scene) - zmin_s) / span_s
    frac = float(np.clip(frac, 0.0, 1.0))
    return float(deep_lbl + frac * (shallow_lbl - deep_lbl))


def _cube_actor_like_plotter(
    spec: CubeZAxisSpec,
    *,
    x_min: float = -40.0,
    x_max: float = 40.0,
    y_min: float = -40.0,
    y_max: float = 40.0,
):
    """``show_bounds`` with ``bounds == axes_ranges``, then ``heal_plan_cube_axes``."""
    pv = _pyvista_off_screen()
    bounds, axes_ranges = plan_cube_scene_bounds_and_axes_ranges(
        float(x_min),
        float(x_max),
        float(y_min),
        float(y_max),
        spec,
    )
    pl = pv.Plotter(off_screen=True)
    actor = pl.show_bounds(
        bounds=bounds,
        axes_ranges=axes_ranges,
        grid="back",
        location="outer",
        ticks="inside",
        padding=0.0,
        n_zlabels=spec.n_zlabels,
        fmt="%.0f",
    )
    heal_plan_cube_axes(actor, bounds, z_spec=spec, apply_style=False)
    return pl, actor, bounds, axes_ranges


def _assert_cube_z_depth_mapping(actor, spec: CubeZAxisSpec) -> None:
    """Z ticks use scene-Z magnitudes; labels run high→low (large values near origin)."""
    ticks = _tick_depths_mm(actor)
    assert len(ticks) >= 2
    zmin_s, zmax_s = float(spec.zmin_scene), float(spec.zmax_scene)
    lo_scene, hi_scene = min(zmin_s, zmax_s), max(zmin_s, zmax_s)
    z_rng = actor.GetZAxisRange()
    z_lo, z_hi = float(z_rng[0]), float(z_rng[1])
    assert min(z_lo, z_hi) == pytest.approx(lo_scene, rel=0.02, abs=1.0)
    assert max(z_lo, z_hi) == pytest.approx(hi_scene, rel=0.02, abs=1.0)
    assert all(lo_scene - 2.0 <= t <= hi_scene + 2.0 for t in ticks)
    assert ticks[0] > ticks[-1]


def _assert_spot_depth_matches_ticks(
    energies_mev: np.ndarray,
    spec: CubeZAxisSpec,
    tick_depth_mm: Sequence[float],
    *,
    atol_mm: float = 1.5,
) -> None:
    e = np.asarray(energies_mev, dtype=np.float64).reshape(-1)
    e_lo, e_hi = float(np.min(e)), float(np.max(e))
    cfg = ZAxisDisplayConfig(use_water_depth_mm=True, tick_mm=5.0)
    d_lo, d_hi = plan_depth_bounds_mm(e_lo, e_hi)
    z_scene = nominal_energy_to_scene_z(
        e,
        plan_e_lo=e_lo,
        plan_e_hi=e_hi,
        config=cfg,
        depth_lo_mm=d_lo,
        depth_hi_mm=d_hi,
    )
    depth = proton_water_depth_mm(energies_mev, metric="csda")
    np.testing.assert_allclose(float(d_hi) + float(d_lo) - z_scene, depth, rtol=0, atol=0.05)
    for zs, d_mm in zip(z_scene, depth, strict=True):
        tick_lbl = _interpolate_tick_depth_mm(float(zs), spec, tick_depth_mm)
        assert tick_lbl == pytest.approx(float(d_mm), abs=atol_mm), (
            f"tick interpolation {tick_lbl:.2f} mm vs PSTAR depth {d_mm:.2f} mm at scene Z={zs:.2f}"
        )


@pytest.mark.parametrize(
    ("energies", "depth_atol"),
    [
        (np.array([78.5, 100.0, 134.4]), 1.5),
        (np.linspace(80.0, 130.0, 9), 1.5),
    ],
)
def test_cube_ticks_track_pstar_depth_for_synthetic_energies(
    energies: np.ndarray,
    depth_atol: float,
) -> None:
    spec = _cube_spec_from_energies(energies, tick_mm=5.0)
    _pl, actor, _b, _a = _cube_actor_like_plotter(spec)
    try:
        _assert_cube_z_depth_mapping(actor, spec)
        _assert_spot_depth_matches_ticks(energies, spec, _tick_depths_mm(actor), atol_mm=depth_atol)
    finally:
        _pl.close()


def test_regression_wrong_z_axis_range_inverts_depth_ticks() -> None:
    """Depth cube maps deep nominal MeV to deep mm at the scene-zmin tick."""
    energies = np.array([78.5, 134.4])
    e_lo, e_hi = float(np.min(energies)), float(np.max(energies))
    cfg = ZAxisDisplayConfig(use_water_depth_mm=True, tick_mm=5.0)
    d_lo, d_hi = plan_depth_bounds_mm(e_lo, e_hi)
    z = nominal_energy_to_scene_z(
        energies,
        plan_e_lo=e_lo,
        plan_e_hi=e_hi,
        config=cfg,
        depth_lo_mm=d_lo,
        depth_hi_mm=d_hi,
    )
    spec = _cube_spec_from_energies(energies, tick_mm=5.0)
    depth_hi = float(proton_water_depth_mm(134.4))

    _pl, actor, _b, _a = _cube_actor_like_plotter(spec)
    try:
        _assert_cube_z_depth_mapping(actor, spec)
        ticks_ok = _tick_depths_mm(actor)
        lbl_deep_ok = _interpolate_tick_depth_mm(float(z[1]), spec, ticks_ok)
        assert lbl_deep_ok == pytest.approx(depth_hi, abs=2.0)
    finally:
        _pl.close()


def test_refresh_after_update_bounds_axes() -> None:
    """After ``update_bounds_axes``, refresh restores full-plan scene Z (MeV mode)."""
    energies = np.array([78.5, 134.4])
    spec = _cube_spec_from_energies(energies, tick_mev=5.0, use_water_depth=False)
    x_min, x_max, y_min, y_max = -40.0, 40.0, -40.0, 40.0

    _pl, actor, bounds, axes_ranges = _cube_actor_like_plotter(
        spec, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max
    )
    try:
        _pl.renderer.update_bounds_axes()
        refresh_pyvista_cube_axes(actor, bounds, axes_ranges, z_spec=spec)
        zl = [float(actor.z_labels[i]) for i in range(len(actor.z_labels))]
        assert min(zl) == pytest.approx(78.5, abs=2.0)
        assert max(zl) == pytest.approx(134.4, abs=2.0)
    finally:
        _pl.close()


def test_refresh_pins_plan_z_not_visible_mesh_extent() -> None:
    """Tight ``update_bounds`` must not leave cube Z stuck on the visible mesh band."""
    energies = np.array([78.5, 134.4])
    spec = _cube_spec_from_energies(energies, tick_mev=5.0, use_water_depth=False)
    x_min, x_max, y_min, y_max = -40.0, 40.0, -40.0, 40.0
    tight_z = (-220.0, -180.0)

    _pl, actor, bounds, axes_ranges = _cube_actor_like_plotter(
        spec, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max
    )
    try:
        actor.update_bounds((x_min, x_max, y_min, y_max, *tight_z))
        refresh_pyvista_cube_axes(actor, bounds, axes_ranges, z_spec=spec)
        z_shallow_scene = float(spec.zmax_scene)
        shallow_lbl = label_at_scene_z(actor, z_shallow_scene)
        assert shallow_lbl == pytest.approx(78.5, abs=2.0)
        assert float(actor.bounds[4]) == pytest.approx(float(spec.zmin_scene), rel=0.01)
        assert float(actor.bounds[5]) == pytest.approx(float(spec.zmax_scene), rel=0.01)
    finally:
        _pl.close()


def test_refresh_after_bounds_setter() -> None:
    """``bounds`` assignment must not leave scene-Z tick strings (−E×scale) on the actor."""
    energies = np.array([78.5, 134.4])
    spec = _cube_spec_from_energies(energies, tick_mev=5.0, use_water_depth=False)
    x_min, x_max, y_min, y_max = -40.0, 40.0, -40.0, 40.0
    bounds = (x_min, x_max, y_min, y_max, spec.zmin_scene, spec.zmax_scene)

    _pl, actor, bounds, axes_ranges = _cube_actor_like_plotter(
        spec, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max
    )
    try:
        actor.bounds = bounds
        refresh_pyvista_cube_axes(actor, bounds, axes_ranges, z_spec=spec)
        zl = [float(actor.z_labels[i]) for i in range(len(actor.z_labels))]
        assert min(zl) == pytest.approx(78.5, abs=3.0)
        assert max(zl) == pytest.approx(134.4, abs=3.0)
        assert all(t > 0 for t in zl)
    finally:
        _pl.close()


def test_refresh_after_cube_axes_update_bounds() -> None:
    """``CubeAxesActor.update_bounds`` must not leave scene-Z ticks on the shallow corner."""
    energies = np.array([78.5, 134.4])
    spec = _cube_spec_from_energies(energies, tick_mev=5.0, use_water_depth=False)
    x_min, x_max, y_min, y_max = -40.0, 40.0, -40.0, 40.0

    _pl, actor, bounds, axes_ranges = _cube_actor_like_plotter(
        spec, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max
    )
    try:
        tight = (x_min, x_max, y_min, y_max, -220.0, -180.0)
        actor.update_bounds(tight)
        refresh_pyvista_cube_axes(actor, bounds, axes_ranges, z_spec=spec)
        zl = [float(actor.z_labels[i]) for i in range(len(actor.z_labels))]
        assert min(zl) == pytest.approx(78.5, abs=3.0)
        assert max(zl) == pytest.approx(134.4, abs=3.0)
    finally:
        _pl.close()


def test_regression_bounds_setter_then_refresh_restores_depth_mm() -> None:
    energies = np.linspace(100.0, 160.0, 12)
    spec = _cube_spec_from_energies(energies, tick_mm=5.0)
    x_min, x_max, y_min, y_max = -40.0, 40.0, -40.0, 40.0
    bounds = (x_min, x_max, y_min, y_max, spec.zmin_scene, spec.zmax_scene)

    _pl, actor, bounds, axes_ranges = _cube_actor_like_plotter(
        spec, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max
    )
    try:
        _assert_cube_z_depth_mapping(actor, spec)
        actor.bounds = bounds
        refresh_pyvista_cube_axes(actor, bounds, axes_ranges, z_spec=spec)
        _assert_cube_z_depth_mapping(actor, spec)
    finally:
        _pl.close()


def test_water_depth_labels_from_mev_not_negated_scene_z() -> None:
    """Using raw MeV as scene Z used to print MeV magnitudes on a depth axis."""
    energies = np.array([78.5, 134.4])
    z_mev_scene = energies.copy()
    spec_bad = cube_z_axis_spec(
        z_mev_scene,
        use_proton_water_depth_mm=True,
        tick_mm=5.0,
        nominal_energy_mev=None,
    )
    spec_ok = _cube_spec_from_energies(energies, tick_mm=5.0)
    deep_ok, shallow_ok = cube_z_axis_label_endpoints(spec_ok)
    deep_bad, shallow_bad = cube_z_axis_label_endpoints(spec_bad)
    assert shallow_ok == pytest.approx(float(proton_water_depth_mm(78.5)), abs=1.0)
    assert deep_ok == pytest.approx(float(proton_water_depth_mm(134.4)), abs=1.5)
    assert abs(shallow_bad - shallow_ok) > 15.0
    assert abs(deep_bad - deep_ok) > 15.0


def test_scene_z_positive_water_depth_matches_affine() -> None:
    e_lo, e_hi = 50.0, 230.0
    cfg_d = ZAxisDisplayConfig(use_water_depth_mm=True)
    cfg_m = ZAxisDisplayConfig(use_water_depth_mm=False)
    d_lo, d_hi = plan_depth_bounds_mm(e_lo, e_hi)
    for e in (78.5, 100.0, 134.4, 180.0):
        z_depth = float(
            nominal_energy_to_scene_z(
                np.array([e], dtype=np.float64),
                plan_e_lo=e_lo,
                plan_e_hi=e_hi,
                config=cfg_d,
                depth_lo_mm=d_lo,
                depth_hi_mm=d_hi,
            )[0]
        )
        z_mev_axis = float(
            nominal_energy_to_scene_z(
                np.array([e], dtype=np.float64),
                plan_e_lo=e_lo,
                plan_e_hi=e_hi,
                config=cfg_m,
            )[0]
        )
        depth = float(proton_water_depth_mm(e))
        assert z_depth == pytest.approx(float(d_hi) + float(d_lo) - depth, abs=0.05)
        assert z_mev_axis == pytest.approx(e_hi + e_lo - e, abs=0.01)
        recon = float(d_hi) + float(d_lo) - z_depth
        assert abs(recon - depth) < abs(recon - e)
        assert abs(z_depth - z_mev_axis) > _MEV_DEPTH_CONFUSION_MM


@pytest.mark.local_data
@pytest.mark.skipif(not _T0G10_DCM.is_file(), reason="T0G10 DICOM not under test_data/")
def test_t0g10_plan_spots_and_cube_ticks_use_csda_mm_not_mev() -> None:
    from spot_check.analysis.spatial import nominal_layer_energies_mev
    from spot_check.plan import planned_spot_xyz_and_counts_from_dicom

    planned, *_ = planned_spot_xyz_and_counts_from_dicom(_T0G10_DCM)
    layer_e = np.asarray(nominal_layer_energies_mev(list(planned)), dtype=np.float64)
    assert float(layer_e.min()) == pytest.approx(78.5, abs=0.05)
    assert float(layer_e.max()) == pytest.approx(134.4, abs=0.1)

    e_all = np.array([s[2] for s in planned], dtype=np.float64)
    spec = _cube_spec_from_energies(e_all, tick_mm=5.0)

    _pl, actor, _b, _a = _cube_actor_like_plotter(spec)
    try:
        _assert_cube_z_depth_mapping(actor, spec)
        _assert_spot_depth_matches_ticks(
            np.array([78.5, 134.4]), spec, _tick_depths_mm(actor), atol_mm=1.5
        )
        e_lo, e_hi = float(np.min(e_all)), float(np.max(e_all))
        cfg = ZAxisDisplayConfig(use_water_depth_mm=True)
        d_lo, d_hi = plan_depth_bounds_mm(e_lo, e_hi)
        for e_nom in layer_e[:: max(1, len(layer_e) // 8)]:
            zs = float(
                nominal_energy_to_scene_z(
                    np.array([e_nom], dtype=np.float64),
                    plan_e_lo=e_lo,
                    plan_e_hi=e_hi,
                    config=cfg,
                    depth_lo_mm=d_lo,
                    depth_hi_mm=d_hi,
                )[0]
            )
            d_nom = float(proton_water_depth_mm(e_nom))
            assert abs(zs - (d_hi + d_lo - d_nom)) < abs(zs - e_nom)
    finally:
        _pl.close()


@pytest.mark.local_data
@pytest.mark.skipif(not _T0G40_DCM.is_file(), reason="T0G40 DICOM not under test_data/")
def test_t0g40_deepest_layer_same_as_t0g10_shallowest_differs() -> None:
    from spot_check.analysis.spatial import nominal_layer_energies_mev
    from spot_check.plan import planned_spot_xyz_and_counts_from_dicom

    planned, *_ = planned_spot_xyz_and_counts_from_dicom(_T0G40_DCM)
    layer_e = nominal_layer_energies_mev(list(planned))
    assert float(min(layer_e)) == pytest.approx(78.5, abs=0.05)
    assert float(max(layer_e)) == pytest.approx(127.2, abs=0.1)

    e_all = np.array([s[2] for s in planned], dtype=np.float64)
    spec = _cube_spec_from_energies(e_all, tick_mm=5.0)
    deep, shallow = cube_z_axis_label_endpoints(spec)
    e_lo, e_hi = float(min(layer_e)), float(max(layer_e))
    assert deep == pytest.approx(float(proton_water_depth_mm(e_hi)), abs=2.5)
    assert shallow == pytest.approx(float(proton_water_depth_mm(e_lo)), abs=2.5)
    assert deep > 95.0
    assert shallow < 60.0


@pytest.mark.local_data
@pytest.mark.skipif(not _T0G10_DCM.is_file(), reason="T0G10 DICOM not under test_data/")
def test_t0g10_mev_axis_full_plan_range_with_slice_on() -> None:
    """Slice must not collapse Z; full-plan scene Z uses matching bounds and axes_ranges."""
    from spot_check import analysis
    from spot_check.plan import planned_spot_xyz_and_counts_from_dicom

    _pyvista_off_screen()
    planned, *_ = planned_spot_xyz_and_counts_from_dicom(_T0G10_DCM)
    pl = pytest.importorskip("pyvista").Plotter(off_screen=True)
    try:
        analysis.show_comparison_3d_pyvista(
            list(planned),
            [],
            title="T0G10 MeV slice",
            a_is_x=False,
            z_axis_use_proton_water_depth_mm=False,
            slice_band_init={"slice_on": True, "center_i": 20},
            reuse_plotter=pl,
            reembed_qt=False,
        )
        actor = pl.renderer.cube_axes_actor
        zl = [float(actor.z_labels[i]) for i in range(len(actor.z_labels))]
        assert 5 <= len(zl) <= 11
        assert all(math.isfinite(t) for t in zl)
        assert min(zl) < max(zl)
        assert all(t > 0 for t in zl)
        assert min(zl) == pytest.approx(78.5, abs=8.0)
        assert max(zl) == pytest.approx(134.4, abs=8.0)
        assert zl[0] > zl[-1]  # high MeV (scene zmin) shown as larger tick near origin
        assert float(actor.bounds[4]) > 70.0
        assert float(actor.bounds[5]) < 145.0
        assert actor.z_label_visibility is True
    finally:
        pl.close()


@pytest.mark.local_data
@pytest.mark.skipif(not _T0G10_DCM.is_file(), reason="T0G10 DICOM not under test_data/")
def test_show_comparison_3d_pyvista_water_depth_cube_axes() -> None:
    """End-to-end: plotter shows Z tick labels with ``bounds == axes_ranges`` (water depth)."""
    from spot_check import analysis
    from spot_check.plan import planned_spot_xyz_and_counts_from_dicom

    _pyvista_off_screen()
    planned, *_ = planned_spot_xyz_and_counts_from_dicom(_T0G10_DCM)
    pl = pytest.importorskip("pyvista").Plotter(off_screen=True)
    try:
        out = analysis.show_comparison_3d_pyvista(
            list(planned),
            [],
            title="T0G10 depth axis",
            a_is_x=False,
            z_axis_use_proton_water_depth_mm=True,
            reuse_plotter=pl,
            reembed_qt=False,
        )
        assert out is pl
        actor = pl.renderer.cube_axes_actor
        assert actor is not None
        zl = [float(actor.z_labels[i]) for i in range(len(actor.z_labels))]
        assert 5 <= len(zl) <= 11
        assert all(math.isfinite(t) for t in zl)
        zb0, zb1 = float(actor.bounds[4]), float(actor.bounds[5])
        z_lo, z_hi = min(zb0, zb1), max(zb0, zb1)
        assert min(zl) >= z_lo - 2.0
        assert max(zl) <= z_hi + 2.0
        assert zl[0] > zl[-1]
        assert actor.z_label_visibility is True
    finally:
        pl.close()
