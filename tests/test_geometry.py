import numpy as np
import pytest

from spot_check.geometry import (
    cube_axes_ranges,
    cube_z_axis_spec,
    n_cube_axis_labels_for_mm_step,
    nominal_mev_to_plot_z,
    proton_cda_water_range_mm,
    refresh_pyvista_cube_axes,
)
from spot_check.geometry.z_axis import label_at_scene_z


def test_proton_cda_monotonic() -> None:
    e = np.array([50.0, 100.0, 150.0])
    r = proton_cda_water_range_mm(e)
    assert r.shape == (3,)
    assert r[0] < r[1] < r[2]


def test_proton_cda_70mev_pstar_table_mm() -> None:
    """PSTAR CSDA range in water at 70 MeV is 40.80 mm (4.08 cm)."""
    r70 = float(proton_cda_water_range_mm(70.0))
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
    """Public ``*_axis_range`` refresh restores depth-mm ticks after ``update_bounds``."""
    import os

    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    pv = pytest.importorskip("pyvista")
    pv.OFF_SCREEN = True
    e = np.linspace(100.0, 160.0, 12)
    z = nominal_mev_to_plot_z(e, use_proton_water_depth_mm=True)
    spec = cube_z_axis_spec(
        z,
        use_proton_water_depth_mm=True,
        tick_mm=5.0,
        nominal_energy_mev=e,
    )
    x_min, x_max, y_min, y_max = -40.0, 40.0, -40.0, 40.0
    bounds = (x_min, x_max, y_min, y_max, spec.zmin_scene, spec.zmax_scene)
    axes = cube_axes_ranges(x_min, x_max, y_min, y_max, spec)
    pl = pv.Plotter(off_screen=True)
    actor = pl.show_bounds(
        bounds=bounds,
        axes_ranges=axes,
        grid="back",
        location="outer",
        ticks="inside",
        padding=0.0,
        n_zlabels=spec.n_zlabels,
        fmt="%.4g",
    )
    assert float(actor.z_labels[0]) > 0.0
    actor.update_bounds((x_min, x_max, y_min, y_max, -220.0, -180.0))
    refresh_pyvista_cube_axes(actor, bounds, axes)
    assert actor.z_label_visibility is True
    assert spec.z_label_at_min > spec.z_label_at_max
    assert float(actor.z_labels[0]) == pytest.approx(spec.z_label_at_min, rel=0.02)
    assert float(actor.z_labels[-1]) == pytest.approx(spec.z_label_at_max, rel=0.02)
    z_lo, z_hi = actor.GetZAxisRange()
    assert z_lo == pytest.approx(spec.z_label_at_min)
    assert z_hi == pytest.approx(spec.z_label_at_max)
    pl.close()


def test_plot_z_upstream_wet_shifter_subtracts_mm() -> None:
    e = np.array([70.0])
    z0 = nominal_mev_to_plot_z(e, use_proton_water_depth_mm=True, upstream_wet_mm=0.0)
    z10 = nominal_mev_to_plot_z(e, use_proton_water_depth_mm=True, upstream_wet_mm=10.0)
    assert float(z10[0]) > float(z0[0])  # shallower scene Z (less negative)
    assert float(z10[0] - z0[0]) == pytest.approx(10.0, abs=0.01)


def test_plot_z_upstream_wet_shifter_clamps_at_zero_depth() -> None:
    e = np.array([10.0])
    z = nominal_mev_to_plot_z(e, use_proton_water_depth_mm=True, upstream_wet_mm=500.0)
    assert float(np.asarray(z).reshape(-1)[0]) == pytest.approx(0.0, abs=1e-9)


def test_plot_z_upstream_wet_ignored_when_mev_axis() -> None:
    e = np.array([70.0])
    z0 = nominal_mev_to_plot_z(e, use_proton_water_depth_mm=False, upstream_wet_mm=50.0)
    z1 = nominal_mev_to_plot_z(e, use_proton_water_depth_mm=False, upstream_wet_mm=0.0)
    assert float(z0[0]) == float(z1[0])


def test_plot_z_shallow_toward_top_and_depth_labels() -> None:
    e = np.array([100.0, 140.0, 180.0])
    z = nominal_mev_to_plot_z(e, use_proton_water_depth_mm=True)
    assert np.all(z < 0)
    assert z[0] > z[-1]  # lower energy / shallower → higher scene Z (top)
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
    zl = [float(actor.z_labels[i]) for i in range(len(actor.z_labels))]
    assert min(zl) == pytest.approx(0.0, abs=0.51)
    assert max(zl) == pytest.approx(10.0, abs=0.51)
    pl.close()
