import numpy as np
import pytest

from spot_check.geometry import (
    PlanCubeAxesController,
    apply_z_display_to_comparison_clouds,
    cube_z_axis_spec,
    heal_plan_cube_axes,
    n_cube_axis_labels_for_mm_step,
    nominal_energy_to_scene_z,
    pin_pyvista_cube_bounds,
    pin_xy_cube_axis_tick_endpoints,
    plan_depth_bounds_mm,
    proton_csda_water_range_mm,
    proton_water_depth_mm,
    refresh_pyvista_cube_axes,
)
from spot_check.geometry.z_axis import label_at_scene_z
from spot_check.models import ZAxisDisplayConfig


def test_proton_csda_monotonic() -> None:
    e = np.array([50.0, 100.0, 150.0])
    r = proton_csda_water_range_mm(e)
    assert r.shape == (3,)
    assert r[0] < r[1] < r[2]


def test_proton_csda_70mev_pstar_table_mm() -> None:
    """PSTAR CSDA range in water at 70 MeV is 40.80 mm (4.08 cm)."""
    r70 = float(proton_csda_water_range_mm(70.0))
    assert r70 == pytest.approx(40.80, abs=0.01)


def test_n_cube_axis_labels_capped() -> None:
    n = n_cube_axis_labels_for_mm_step(0.0, 1000.0, 5.0, max_n=11)
    assert n == 11


def test_cube_z_padded_bounds_labels_at_box_corners() -> None:
    """``show_bounds`` with matching ``bounds`` / ``axes_ranges`` labels box corners."""
    import os

    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    pv = pytest.importorskip("pyvista")
    pv.OFF_SCREEN = True

    z = np.linspace(0.0, 10.0, 11)
    spec = cube_z_axis_spec(
        z,
        use_proton_water_depth_mm=False,
        tick_mm=1.0,
        tick_mev=1.0,
        nominal_energy_mev=np.linspace(0.0, 10.0, 11),
    )
    assert spec.zmin_scene == pytest.approx(0.0)
    assert spec.zmax_scene == pytest.approx(10.0)
    box = (0.0, 10.0, 0.0, 10.0, spec.zmin_scene, spec.zmax_scene)
    pl = pv.Plotter(off_screen=True)
    pl.add_mesh(pv.Box(bounds=(0, 10, 0, 10, 0, 10)))
    actor = pl.show_bounds(
        bounds=box,
        axes_ranges=box,
        padding=0.0,
        n_zlabels=spec.n_zlabels,
        fmt="%.0f",
    )
    assert label_at_scene_z(actor, 0.0) == pytest.approx(0.0, abs=0.51)
    assert label_at_scene_z(actor, 10.0) == pytest.approx(10.0, abs=0.51)
    pl.close()


def test_cube_axes_10_show_bounds_labels_at_scene_extents() -> None:
    """10³ cube: label 0 at scene zmin, 10 at zmax (VTK index 0 at zmin)."""
    import os

    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    pv = pytest.importorskip("pyvista")
    pv.OFF_SCREEN = True

    z = np.linspace(0.0, 10.0, 11)
    spec = cube_z_axis_spec(
        z,
        use_proton_water_depth_mm=False,
        tick_mm=1.0,
        tick_mev=1.0,
        nominal_energy_mev=np.linspace(0.0, 10.0, 11),
    )
    bounds = (0.0, 10.0, 0.0, 10.0, spec.zmin_scene, spec.zmax_scene)
    pl = pv.Plotter(off_screen=True)
    pl.add_mesh(pv.Box(bounds=bounds))
    actor = pl.show_bounds(
        bounds=bounds,
        axes_ranges=bounds,
        padding=0.0,
        n_zlabels=spec.n_zlabels,
        fmt="%.0f",
    )
    assert float(actor.z_labels[0]) == pytest.approx(spec.z_label_at_min, abs=0.51)
    assert float(actor.z_labels[-1]) == pytest.approx(spec.z_label_at_max, abs=0.51)
    assert label_at_scene_z(actor, 0.0) == pytest.approx(0.0, abs=0.51)
    assert label_at_scene_z(actor, 10.0) == pytest.approx(10.0, abs=0.51)
    pl.close()


def test_pin_xy_cube_axis_tick_endpoints_span_axis_range() -> None:
    """First/last XY labels match ``x_axis_range`` / ``y_axis_range`` (corner scale)."""
    import os

    from spot_check.geometry import disable_pyvista_cube_axes_label_lod

    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    pv = pytest.importorskip("pyvista")
    pv.OFF_SCREEN = True
    x0, x1, y0, y1, z0, z1 = -40.0, 40.0, -35.0, 35.0, 0.0, 100.0
    bounds = (x0, x1, y0, y1, z0, z1)
    pl = pv.Plotter(off_screen=True)
    actor = pl.show_bounds(
        bounds=bounds,
        axes_ranges=bounds,
        grid="back",
        location="outer",
        ticks="inside",
        padding=0.0,
        n_xlabels=6,
        n_ylabels=6,
        n_zlabels=5,
        fmt="%.0f",
    )
    disable_pyvista_cube_axes_label_lod(actor)
    pin_xy_cube_axis_tick_endpoints(actor)
    assert float(actor.x_labels[0]) == pytest.approx(x0, abs=0.01)
    assert float(actor.x_labels[-1]) == pytest.approx(x1, abs=0.01)
    assert float(actor.y_labels[0]) == pytest.approx(y0, abs=0.01)
    assert float(actor.y_labels[-1]) == pytest.approx(y1, abs=0.01)
    pl.close()


def test_cube_z_axis_spec_labels_match_scene_z_order() -> None:
    """VTK tick index 0 is at scene zmin — labels must follow scene Z, not max/min energy."""
    z = np.linspace(0.0, 10.0, 11)
    e = np.linspace(0.0, 10.0, 11)
    spec = cube_z_axis_spec(
        z,
        use_proton_water_depth_mm=False,
        tick_mm=1.0,
        tick_mev=1.0,
        nominal_energy_mev=e,
    )
    assert spec.z_label_at_min < spec.z_label_at_max
    assert spec.zmin_scene == pytest.approx(0.0)
    assert spec.zmax_scene == pytest.approx(10.0)
    assert spec.z_label_at_min == pytest.approx(0.0)
    assert spec.z_label_at_max == pytest.approx(10.0)


def test_refresh_pyvista_cube_axes_after_update_bounds() -> None:
    """After a tight ``update_bounds``, refresh restores full-plan scene Z on the axis."""
    import os

    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    pv = pytest.importorskip("pyvista")
    pv.OFF_SCREEN = True
    e = np.linspace(100.0, 160.0, 12)
    e_lo, e_hi = float(np.min(e)), float(np.max(e))
    cfg = ZAxisDisplayConfig(use_water_depth_mm=True, tick_mm=5.0)
    d_lo, d_hi = plan_depth_bounds_mm(e_lo, e_hi)
    z = nominal_energy_to_scene_z(
        e,
        plan_e_lo=e_lo,
        plan_e_hi=e_hi,
        config=cfg,
        depth_lo_mm=d_lo,
        depth_hi_mm=d_hi,
    )
    spec = cube_z_axis_spec(
        z,
        use_proton_water_depth_mm=True,
        tick_mm=5.0,
        nominal_energy_mev=e,
    )
    from spot_check.geometry import plan_cube_scene_bounds_and_axes_ranges

    x_min, x_max, y_min, y_max = -40.0, 40.0, -40.0, 40.0
    bounds, axes = plan_cube_scene_bounds_and_axes_ranges(
        x_min, x_max, y_min, y_max, spec
    )
    pl = pv.Plotter(off_screen=True)
    actor = pl.show_bounds(
        bounds=bounds,
        axes_ranges=axes,
        grid="back",
        location="outer",
        ticks="inside",
        padding=0.0,
        n_zlabels=spec.n_zlabels,
        fmt="%.0f",
    )
    heal_plan_cube_axes(actor, bounds, z_spec=spec, apply_style=False)
    assert float(actor.z_labels[0]) > 0.0
    actor.update_bounds((x_min, x_max, y_min, y_max, -220.0, -180.0))
    refresh_pyvista_cube_axes(actor, bounds, axes, z_spec=spec)
    assert actor.z_label_visibility is True
    z_rng = actor.GetZAxisRange()
    z_lo, z_hi = float(z_rng[0]), float(z_rng[1])
    assert z_lo == pytest.approx(float(spec.z_label_at_min), rel=0.02, abs=1.0)
    assert z_hi == pytest.approx(float(spec.z_label_at_max), rel=0.02, abs=1.0)
    zl0, zl1 = float(actor.z_labels[0]), float(actor.z_labels[-1])
    assert zl0 > zl1
    assert label_at_scene_z(actor, float(bounds[4])) == pytest.approx(
        float(spec.z_label_at_min), abs=1.0
    )
    assert label_at_scene_z(actor, float(bounds[5])) == pytest.approx(
        float(spec.z_label_at_max), abs=1.0
    )
    pl.close()


def _vtk_z_label_strings(actor: object) -> list[str]:
    arr = actor.GetAxisLabels(2)  # type: ignore[attr-defined]
    if arr is None:
        return []
    return [str(arr.GetValue(i)) for i in range(arr.GetNumberOfValues())]


def test_pin_pyvista_cube_bounds_restores_inverted_z_after_pin_xy() -> None:
    """``bounds`` assignment resets Z to scene; ``pin_pyvista_cube_bounds`` must re-invert."""
    import os

    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    pv = pytest.importorskip("pyvista")
    pv.OFF_SCREEN = True
    bounds = (-40.0, 40.0, -40.0, 40.0, 78.5, 134.4)
    pl = pv.Plotter(off_screen=True)
    actor = pl.show_bounds(
        bounds=bounds,
        axes_ranges=bounds,
        padding=0.0,
        n_zlabels=6,
        fmt="%.0f",
    )
    heal_plan_cube_axes(actor, bounds, apply_style=False)
    assert float(actor.z_labels[0]) > float(actor.z_labels[-1])
    actor.bounds = bounds
    assert float(actor.z_labels[0]) < float(actor.z_labels[-1])
    pin_pyvista_cube_bounds(actor, bounds)
    assert float(actor.z_labels[0]) > float(actor.z_labels[-1])
    pl.close()


def test_inverted_z_labels_use_fixed_width_integer_strings() -> None:
    """``PYVISTA_CUBE_Z_LABEL_FORMAT`` avoids variable-width overlap on the axis."""
    import os

    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    pv = pytest.importorskip("pyvista")
    pv.OFF_SCREEN = True
    bounds = (-40.0, 40.0, -40.0, 40.0, 50.14, 130.19)
    pl = pv.Plotter(off_screen=True)
    actor = pl.show_bounds(
        bounds=bounds,
        axes_ranges=bounds,
        padding=0.0,
        n_zlabels=11,
        fmt="%.4g",
    )
    heal_plan_cube_axes(actor, bounds, apply_style=True)
    strings = _vtk_z_label_strings(actor)
    assert len(strings) >= 2
    assert all("." not in s for s in strings)
    assert float(strings[0]) > float(strings[-1])
    pl.close()


def test_plan_cube_axes_controller_show_heals_when_bounds_unchanged() -> None:
    """``show()`` must re-invert Z even when plan bounds did not change (no early-return skip)."""
    import os

    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    pv = pytest.importorskip("pyvista")
    pv.OFF_SCREEN = True
    e = np.linspace(100.0, 160.0, 12)
    e_lo, e_hi = float(np.min(e)), float(np.max(e))
    cfg = ZAxisDisplayConfig(use_water_depth_mm=True, tick_mm=5.0)
    d_lo, d_hi = plan_depth_bounds_mm(e_lo, e_hi)
    z = nominal_energy_to_scene_z(
        e,
        plan_e_lo=e_lo,
        plan_e_hi=e_hi,
        config=cfg,
        depth_lo_mm=d_lo,
        depth_hi_mm=d_hi,
    )
    pl = pv.Plotter(off_screen=True)
    ctrl = PlanCubeAxesController(
        xlab="X",
        ylab="Y",
        x_min=-40.0,
        x_max=40.0,
        y_min=-40.0,
        y_max=40.0,
        z_display_cfg=cfg,
    )
    ctrl.ready = True
    ctrl.show(pl, z, e)
    actor = pl.renderer.cube_axes_actor
    assert float(actor.z_labels[0]) > float(actor.z_labels[-1])
    pin_xy_cube_axis_tick_endpoints(actor)
    ctrl.show(pl, z, e)
    assert float(actor.z_labels[0]) > float(actor.z_labels[-1])
    pl.close()


def test_scene_z_upstream_wet_shifter_subtracts_mm() -> None:
    e = np.array([70.0])
    e_lo, e_hi = 50.0, 90.0
    d_lo, d_hi = plan_depth_bounds_mm(e_lo, e_hi, upstream_wet_mm=0.0)
    cfg0 = ZAxisDisplayConfig(
        use_water_depth_mm=True, upstream_wet_mm=0.0, z_depth_metric="csda"
    )
    cfg10 = ZAxisDisplayConfig(
        use_water_depth_mm=True, upstream_wet_mm=10.0, z_depth_metric="csda"
    )
    z0 = nominal_energy_to_scene_z(
        e, plan_e_lo=e_lo, plan_e_hi=e_hi, config=cfg0, depth_lo_mm=d_lo, depth_hi_mm=d_hi
    )
    z10 = nominal_energy_to_scene_z(
        e, plan_e_lo=e_lo, plan_e_hi=e_hi, config=cfg10, depth_lo_mm=d_lo, depth_hi_mm=d_hi
    )
    assert float(z10[0]) > float(z0[0])
    assert float(z10[0] - z0[0]) == pytest.approx(10.0, abs=0.01)


def test_scene_z_upstream_wet_shifter_clamps_at_zero_depth() -> None:
    e = np.array([70.0])
    e_lo, e_hi = 60.0, 80.0
    wet_excess = float(proton_water_depth_mm(70.0, metric="csda")) + 1.0
    cfg = ZAxisDisplayConfig(use_water_depth_mm=True, upstream_wet_mm=wet_excess)
    d_lo, d_hi = plan_depth_bounds_mm(e_lo, e_hi, upstream_wet_mm=wet_excess)
    z = nominal_energy_to_scene_z(
        e, plan_e_lo=e_lo, plan_e_hi=e_hi, config=cfg, depth_lo_mm=d_lo, depth_hi_mm=d_hi
    )
    depth_after = max(
        0.0,
        float(proton_water_depth_mm(70.0, metric="csda")) - wet_excess,
    )
    assert depth_after == pytest.approx(0.0, abs=1e-9)
    assert float(np.asarray(z).reshape(-1)[0]) == pytest.approx(
        float(d_hi) + float(d_lo) - depth_after,
        abs=1e-9,
    )


def test_scene_z_upstream_wet_ignored_when_mev_axis() -> None:
    e = np.array([70.0])
    cfg_wet = ZAxisDisplayConfig(use_water_depth_mm=False, upstream_wet_mm=50.0)
    cfg_plain = ZAxisDisplayConfig(use_water_depth_mm=False, upstream_wet_mm=0.0)
    z0 = nominal_energy_to_scene_z(e, plan_e_lo=60.0, plan_e_hi=80.0, config=cfg_wet)
    z1 = nominal_energy_to_scene_z(e, plan_e_lo=60.0, plan_e_hi=80.0, config=cfg_plain)
    assert float(z0[0]) == float(z1[0])


def test_scene_z_shallow_high_and_depth_labels() -> None:
    e = np.array([100.0, 140.0, 180.0])
    e_lo, e_hi = float(np.min(e)), float(np.max(e))
    cfg = ZAxisDisplayConfig(use_water_depth_mm=True, tick_mm=5.0)
    d_lo, d_hi = plan_depth_bounds_mm(e_lo, e_hi)
    z = nominal_energy_to_scene_z(
        e, plan_e_lo=e_lo, plan_e_hi=e_hi, config=cfg, depth_lo_mm=d_lo, depth_hi_mm=d_hi
    )
    assert z[0] > z[-1]  # lower energy / shallower → higher scene Z
    spec = cube_z_axis_spec(
        z,
        use_proton_water_depth_mm=True,
        tick_mm=5.0,
        nominal_energy_mev=e,
    )
    assert spec.z_label_at_min > spec.z_label_at_max  # deep mm at zmin, shallow at zmax
    assert spec.zmin_scene < spec.zmax_scene
    assert spec.z_label_at_min > 0 and spec.z_label_at_max > 0


def test_cube_z_axis_spec_empty_scene_z() -> None:
    spec = cube_z_axis_spec(np.array([]), use_proton_water_depth_mm=True, tick_mm=5.0)
    assert spec.zmin_scene < spec.zmax_scene
    assert spec.z_label_at_min > spec.z_label_at_max
    assert spec.n_zlabels >= 5


def test_n_cube_axis_labels_invalid_step() -> None:
    assert n_cube_axis_labels_for_mm_step(0.0, 100.0, 0.0) == 5
    assert n_cube_axis_labels_for_mm_step(0.0, 100.0, float("nan")) == 5


def test_same_nominal_mev_same_scene_z_plan_and_measured() -> None:
    e_grid = np.array([90.0, 110.0, 125.0], dtype=np.float64)
    e_lo, e_hi = 80.0, 130.0
    cfg = ZAxisDisplayConfig(use_water_depth_mm=True)
    d_lo, d_hi = plan_depth_bounds_mm(e_lo, e_hi)
    z_plan_col = nominal_energy_to_scene_z(
        e_grid, plan_e_lo=e_lo, plan_e_hi=e_hi, config=cfg, depth_lo_mm=d_lo, depth_hi_mm=d_hi
    )
    assert z_plan_col.shape == (3,)
    plan_xyz = np.column_stack([np.zeros(3), np.zeros(3), e_grid])
    meas_xyz = np.column_stack([np.ones(3), np.ones(3), e_grid])
    plan_s, meas_s, _db = apply_z_display_to_comparison_clouds(
        plan_xyz,
        meas_xyz,
        plan_e_lo=e_lo,
        plan_e_hi=e_hi,
        config=cfg,
    )
    np.testing.assert_array_almost_equal(plan_s[:, 2], z_plan_col)
    np.testing.assert_array_almost_equal(meas_s[:, 2], z_plan_col)


def test_prepare_comparison_then_z_display_layers_align() -> None:
    from spot_check.analysis.viz.data import prepare_comparison_3d_data

    planned_xyz = [(0.0, 0.0, 100.0), (1.0, 0.0, 120.0), (2.0, 0.0, 140.0)]
    measured_abc = [(0.1, 0.0, 1.0, 1.0, 0)]
    prep = prepare_comparison_3d_data(
        planned_xyz,
        measured_abc,
        a_is_x=True,
    )
    cfg = ZAxisDisplayConfig(use_water_depth_mm=False)
    plan_s, meas_s, _ = apply_z_display_to_comparison_clouds(
        prep.plan_xyz,
        prep.meas_xyz,
        plan_e_lo=float(prep.e_lo),
        plan_e_hi=float(prep.e_hi),
        config=cfg,
    )
    e_match = float(prep.plan_xyz[1, 2])
    assert plan_s[1, 2] == pytest.approx(meas_s[0, 2])
    assert plan_s[1, 2] == pytest.approx(
        float(prep.e_hi) + float(prep.e_lo) - e_match,
    )


def test_plotter_cube_axes_sanity_0_to_10() -> None:
    """Plotter sanity mode: bare ``show_bounds`` with ``bounds == axes_ranges`` on 0..10."""
    import os

    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    pv = pytest.importorskip("pyvista")
    pv.OFF_SCREEN = True
    from spot_check.analysis.viz.plotter import show_comparison_3d_pyvista

    pl = pv.Plotter(off_screen=True)
    show_comparison_3d_pyvista(
        [(5.0, 5.0, 100.0)],
        [],
        title="sanity",
        a_is_x=False,
        z_axis_use_proton_water_depth_mm=False,
        cube_axes_sanity=True,
        reuse_plotter=pl,
        reembed_qt=False,
    )
    actor = pl.renderer.cube_axes_actor
    assert actor is not None
    assert actor.GetDrawXGridlines()
    assert actor.GetDrawYGridlines()
    assert actor.GetDrawZGridlines()
    zl = [float(actor.z_labels[i]) for i in range(len(actor.z_labels))]
    assert min(zl) == pytest.approx(0.0, abs=0.51)
    assert max(zl) == pytest.approx(10.0, abs=0.51)
    pl.close()
