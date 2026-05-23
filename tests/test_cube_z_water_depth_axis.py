"""Regression tests: water-depth Z axis, tick labels, and MeV vs mm confusion.

Guards against:
- Plotting raw nominal MeV on the scene Z axis while ticks show mm depth.
- Incorrect Z label interpolation for custom scene grid (future Z phase).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from spot_check.geometry import (
    cube_z_axis_label_endpoints,
    cube_z_axis_spec,
    cube_z_axis_spec_for_display,
    nominal_energy_to_scene_z,
    plan_depth_bounds_mm,
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
) -> float:
    """Depth (mm) implied by Z spec labels (deep at zmin, shallow at zmax)."""
    lbl = label_at_scene_z(z_scene, spec)
    assert lbl is not None
    return float(lbl)


def _assert_cube_z_depth_mapping(spec: CubeZAxisSpec) -> None:
    """Z labels run high→low (large values at scene zmin)."""
    deep_lbl, shallow_lbl = cube_z_axis_label_endpoints(spec)
    assert deep_lbl > shallow_lbl
    zmin_s, zmax_s = float(spec.zmin_scene), float(spec.zmax_scene)
    assert label_at_scene_z(zmin_s, spec) == pytest.approx(deep_lbl, abs=0.51)
    assert label_at_scene_z(zmax_s, spec) == pytest.approx(shallow_lbl, abs=0.51)


def _assert_spot_depth_matches_ticks(
    energies_mev: np.ndarray,
    spec: CubeZAxisSpec,
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
        tick_lbl = _interpolate_tick_depth_mm(float(zs), spec)
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
    _assert_cube_z_depth_mapping(spec)
    _assert_spot_depth_matches_ticks(energies, spec, atol_mm=depth_atol)


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

    _assert_cube_z_depth_mapping(spec)
    lbl_deep_ok = _interpolate_tick_depth_mm(float(z[1]), spec)
    assert lbl_deep_ok == pytest.approx(depth_hi, abs=2.0)


def test_label_at_scene_z_mev_mode_endpoints() -> None:
    energies = np.array([78.5, 134.4])
    spec = _cube_spec_from_energies(energies, tick_mev=5.0, use_water_depth=False)
    assert label_at_scene_z(float(spec.zmin_scene), spec) == pytest.approx(134.4, abs=2.0)
    assert label_at_scene_z(float(spec.zmax_scene), spec) == pytest.approx(78.5, abs=2.0)


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

    _assert_cube_z_depth_mapping(spec)
    _assert_spot_depth_matches_ticks(np.array([78.5, 134.4]), spec, atol_mm=1.5)
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
    """Slice must not collapse scene bounds used for camera reset."""
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
        assert pl.renderer.cube_axes_actor is None
        grid = getattr(pl, "_spot_check_scene_grid", None)
        assert grid is not None
        bounds = grid.camera_bounds()
        assert bounds is not None
        assert float(bounds[4]) > 70.0
        assert float(bounds[5]) < 145.0
        assert grid._render.major_line_actor is not None
        assert grid._render.minor_line_actor is not None
    finally:
        pl.close()


@pytest.mark.local_data
@pytest.mark.skipif(not _T0G10_DCM.is_file(), reason="T0G10 DICOM not under test_data/")
def test_show_comparison_3d_pyvista_water_depth_scene_grid() -> None:
    """End-to-end: plotter shows custom XY grid and retains full-plan scene bounds."""
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
        assert pl.renderer.cube_axes_actor is None
        grid = getattr(pl, "_spot_check_scene_grid", None)
        assert grid is not None
        assert grid.z_spec is not None
        assert grid._render.major_line_actor is not None
        assert grid._render.minor_line_actor is not None
        bounds = grid.camera_bounds()
        assert bounds is not None
        zb0, zb1 = float(bounds[4]), float(bounds[5])
        assert min(zb0, zb1) < max(zb0, zb1)
    finally:
        pl.close()
