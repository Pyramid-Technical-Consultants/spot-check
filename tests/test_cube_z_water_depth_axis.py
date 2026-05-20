"""Regression tests: water-depth Z axis, cube tick labels, and MeV vs mm confusion.

Guards against:
- Plotting raw nominal MeV on the scene Z axis while ticks show mm depth.
- ``SetZAxisRange`` ascending (shallow, deep) paired with descending tick strings.
- ``actor.bounds`` resetting ``z_axis_range`` to negative scene coordinates.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from spot_check.geometry import (
    apply_pyvista_cube_z_axis,
    cube_z_axis_spec,
    nominal_mev_to_plot_z,
)
from spot_check.geometry.proton_csda_water import proton_water_depth_mm
from spot_check.geometry.z_axis import _cube_z_depth_label_endpoints
from spot_check.models import CubeZAxisSpec

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
    upstream_wet_mm: float = 0.0,
    z_depth_metric: str = "csda",
) -> CubeZAxisSpec:
    e = np.asarray(energies_mev, dtype=np.float64).reshape(-1)
    z = nominal_mev_to_plot_z(
        e,
        use_proton_water_depth_mm=True,
        upstream_wet_mm=upstream_wet_mm,
        z_depth_metric=z_depth_metric,
    )
    return cube_z_axis_spec(
        z,
        use_proton_water_depth_mm=True,
        tick_mm=tick_mm,
        nominal_energy_mev=e,
        upstream_wet_mm=upstream_wet_mm,
        z_depth_metric=z_depth_metric,
    )


def _interpolate_tick_depth_mm(
    z_scene: float,
    spec: CubeZAxisSpec,
    tick_depth_mm: Sequence[float],
) -> float:
    """Depth (mm) implied by tick strings (deep at index 0, shallow at last)."""
    deep_lbl, shallow_lbl = _cube_z_depth_label_endpoints(spec)
    depth = -float(z_scene)
    span = shallow_lbl - deep_lbl
    if span == 0.0:
        return float(tick_depth_mm[0])
    frac = (depth - deep_lbl) / span
    frac = float(np.clip(frac, 0.0, 1.0))
    return float(tick_depth_mm[0]) + frac * (float(tick_depth_mm[-1]) - float(tick_depth_mm[0]))


def _cube_actor_like_plotter(
    spec: CubeZAxisSpec,
    *,
    x_min: float = -40.0,
    x_max: float = 40.0,
    y_min: float = -40.0,
    y_max: float = 40.0,
):
    """``show_bounds`` + ``apply_pyvista_cube_z_axis`` in the same order as ``plotter.py``."""
    pv = _pyvista_off_screen()
    z_deep, z_shallow = _cube_z_depth_label_endpoints(spec)
    bounds = (
        float(x_min),
        float(x_max),
        float(y_min),
        float(y_max),
        float(spec.zmin_scene),
        float(spec.zmax_scene),
    )
    axes_ranges = (
        float(x_min),
        float(x_max),
        float(y_min),
        float(y_max),
        z_deep,
        z_shallow,
    )
    pl = pv.Plotter(off_screen=True)
    actor = pl.show_bounds(
        bounds=bounds,
        axes_ranges=axes_ranges,
        grid="back",
        location="outer",
        ticks="inside",
        n_zlabels=spec.n_zlabels,
        fmt="%.4g",
    )
    apply_pyvista_cube_z_axis(
        actor,
        spec,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )
    return pl, actor


def _assert_cube_z_depth_mapping(actor, spec: CubeZAxisSpec) -> None:
    """Cube Z ticks must track PSTAR depth along scene Z (not MeV magnitudes)."""
    ticks = _tick_depths_mm(actor)
    assert len(ticks) >= 2
    assert ticks[0] > ticks[-1], "deep mm at scene zmin (tick index 0), shallow at zmax"
    deep, shallow = _cube_z_depth_label_endpoints(spec)
    z_lo, z_hi = actor.GetZAxisRange()
    assert z_lo == pytest.approx(shallow)
    assert z_hi == pytest.approx(deep)
    assert deep > shallow
    assert all(t > 0 for t in ticks), "water-depth ticks must be positive mm, not scene Z"


def _assert_spot_depth_matches_ticks(
    energies_mev: np.ndarray,
    spec: CubeZAxisSpec,
    tick_depth_mm: Sequence[float],
    *,
    atol_mm: float = 1.5,
) -> None:
    z_scene = nominal_mev_to_plot_z(energies_mev, use_proton_water_depth_mm=True)
    depth = proton_water_depth_mm(energies_mev, metric="csda")
    np.testing.assert_allclose(-z_scene, depth, rtol=0, atol=0.05)
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
    _pl, actor = _cube_actor_like_plotter(spec)
    try:
        _assert_cube_z_depth_mapping(actor, spec)
        _assert_spot_depth_matches_ticks(
            energies, spec, _tick_depths_mm(actor), atol_mm=depth_atol
        )
    finally:
        _pl.close()


def test_regression_ascending_labels_with_ascending_range_inverts_depth_ticks() -> None:
    """Regression: ascending range + ascending label strings swap deep/shallow (~50 vs ~130)."""
    from pyvista.plotting.cube_axes_actor import make_axis_labels

    energies = np.array([78.5, 134.4])
    z = nominal_mev_to_plot_z(energies, use_proton_water_depth_mm=True)
    spec = _cube_spec_from_energies(energies, tick_mm=5.0)
    deep, shallow = _cube_z_depth_label_endpoints(spec)
    depth_lo = float(proton_water_depth_mm(78.5))
    depth_hi = float(proton_water_depth_mm(134.4))

    _pl, actor = _cube_actor_like_plotter(spec)
    try:
        _assert_cube_z_depth_mapping(actor, spec)
        ticks_ok = _tick_depths_mm(actor)
        lbl_deep_ok = _interpolate_tick_depth_mm(float(z[1]), spec, ticks_ok)
        assert lbl_deep_ok == pytest.approx(depth_hi, abs=2.0)

        actor.SetZAxisRange(shallow, deep)
        actor.SetAxisLabels(
            2,
            make_axis_labels(vmin=shallow, vmax=deep, n=spec.n_zlabels, fmt="%.4g"),
        )
        ticks_bad = _tick_depths_mm(actor)
        lbl_deep_bad = _interpolate_tick_depth_mm(float(z[1]), spec, ticks_bad)
        lbl_shallow_bad = _interpolate_tick_depth_mm(float(z[0]), spec, ticks_bad)
        assert abs(lbl_deep_bad - depth_hi) > 20.0
        assert abs(lbl_shallow_bad - depth_lo) > 20.0
        assert lbl_deep_bad == pytest.approx(depth_lo, abs=2.0)
        assert lbl_shallow_bad == pytest.approx(depth_hi, abs=2.0)
    finally:
        _pl.close()


def test_apply_must_not_assign_z_axis_range_property() -> None:
    """PyVista ``z_axis_range`` setter re-runs ``_update_z_labels`` and drops custom Z ticks."""
    energies = np.array([78.5, 134.4])
    spec = _cube_spec_from_energies(energies, tick_mm=5.0)
    deep, shallow = _cube_z_depth_label_endpoints(spec)

    _pl, actor = _cube_actor_like_plotter(spec)
    try:
        ticks_ok = _tick_depths_mm(actor)
        assert len(ticks_ok) >= 2
        assert float(ticks_ok[0]) > float(ticks_ok[-1])

        actor.SetZAxisRange(shallow, deep)
        actor.z_axis_range = (deep, shallow)
        n_after_prop = len(_tick_depths_mm(actor))
        apply_pyvista_cube_z_axis(
            actor,
            spec,
            x_min=-40.0,
            x_max=40.0,
            y_min=-40.0,
            y_max=40.0,
        )
        ticks_fixed = _tick_depths_mm(actor)
        assert len(ticks_fixed) >= 2
        assert float(ticks_fixed[0]) > float(ticks_fixed[-1])
        assert float(ticks_fixed[0]) == pytest.approx(deep, rel=0.02)
        assert float(ticks_fixed[-1]) == pytest.approx(shallow, rel=0.05)
    finally:
        _pl.close()


def test_regression_bounds_setter_resets_z_axis_then_apply_restores_depth_mm() -> None:
    energies = np.linspace(100.0, 160.0, 12)
    spec = _cube_spec_from_energies(energies, tick_mm=5.0)
    x_min, x_max, y_min, y_max = -40.0, 40.0, -40.0, 40.0
    bounds = (x_min, x_max, y_min, y_max, spec.zmin_scene, spec.zmax_scene)

    _pl, actor = _cube_actor_like_plotter(spec)
    try:
        _assert_cube_z_depth_mapping(actor, spec)
        actor.bounds = bounds
        assert float(actor.z_labels[0]) < 0.0, "bounds setter copies negative scene Z into ticks"
        apply_pyvista_cube_z_axis(
            actor, spec, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max
        )
        _assert_cube_z_depth_mapping(actor, spec)
        assert float(actor.z_labels[0]) > float(actor.z_labels[-1])
    finally:
        _pl.close()


def test_water_depth_labels_from_mev_not_negated_scene_z() -> None:
    """Using ``-scene_z`` on raw MeV positions used to print ~78–134 on the axis."""
    energies = np.array([78.5, 134.4])
    z_mev_scene = -energies  # bug class: nominal MeV stored as scene Z
    spec_bad = cube_z_axis_spec(
        z_mev_scene,
        use_proton_water_depth_mm=True,
        tick_mm=5.0,
        nominal_energy_mev=None,
    )
    spec_ok = _cube_spec_from_energies(energies, tick_mm=5.0)
    deep_ok, shallow_ok = _cube_z_depth_label_endpoints(spec_ok)
    deep_bad, shallow_bad = _cube_z_depth_label_endpoints(spec_bad)
    assert shallow_ok == pytest.approx(float(proton_water_depth_mm(78.5)), abs=1.0)
    assert deep_ok == pytest.approx(float(proton_water_depth_mm(134.4)), abs=1.5)
    assert shallow_bad == pytest.approx(78.5, abs=2.5)
    assert deep_bad == pytest.approx(134.4, abs=2.5)
    assert abs(shallow_bad - shallow_ok) > 15.0


def test_scene_z_not_raw_negative_mev_when_water_depth_on() -> None:
    for e in (78.5, 100.0, 134.4, 70.0):
        z_depth = float(
            nominal_mev_to_plot_z(np.array([e]), use_proton_water_depth_mm=True)[0]
        )
        z_mev_axis = float(
            nominal_mev_to_plot_z(np.array([e]), use_proton_water_depth_mm=False)[0]
        )
        depth = float(proton_water_depth_mm(e))
        assert z_depth == pytest.approx(-depth, abs=0.05)
        assert z_mev_axis == pytest.approx(-2.0 * e, abs=0.01)
        assert abs(z_depth + depth) < abs(z_depth + e)
        assert abs(z_depth - z_mev_axis) > _MEV_DEPTH_CONFUSION_MM


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

    _pl, actor = _cube_actor_like_plotter(spec)
    try:
        _assert_cube_z_depth_mapping(actor, spec)
        _assert_spot_depth_matches_ticks(
            np.array([78.5, 134.4]), spec, _tick_depths_mm(actor), atol_mm=1.5
        )
        # Full layer grid: spot Z must not sit near -MeV (would look like MeV on a mm axis).
        for e_nom in layer_e[:: max(1, len(layer_e) // 8)]:
            zs = float(nominal_mev_to_plot_z(np.array([e_nom]), use_proton_water_depth_mm=True)[0])
            d_nom = float(proton_water_depth_mm(e_nom))
            assert abs(zs + d_nom) < abs(zs + e_nom)
    finally:
        _pl.close()


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
    deep, shallow = _cube_z_depth_label_endpoints(spec)
    e_lo, e_hi = float(min(layer_e)), float(max(layer_e))
    assert deep == pytest.approx(float(proton_water_depth_mm(e_hi)), abs=2.5)
    assert shallow == pytest.approx(float(proton_water_depth_mm(e_lo)), abs=2.5)
    assert deep > 95.0
    assert shallow < 60.0


@pytest.mark.skipif(not _T0G10_DCM.is_file(), reason="T0G10 DICOM not under test_data/")
def test_show_comparison_3d_pyvista_water_depth_cube_axes() -> None:
    """End-to-end: plotter path must leave cube axes in depth-mm convention."""
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
        e = np.array([78.5, 134.4])
        spec = _cube_spec_from_energies(e, tick_mm=5.0)
        _assert_cube_z_depth_mapping(actor, spec)
        _assert_spot_depth_matches_ticks(e, spec, _tick_depths_mm(actor), atol_mm=2.0)
    finally:
        pl.close()
