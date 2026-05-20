import numpy as np
import pytest

from spot_check.geometry import (
    apply_pyvista_cube_z_axis,
    cube_z_axis_spec,
    n_cube_axis_labels_for_mm_step,
    nominal_mev_to_plot_z,
    proton_cda_water_range_mm,
)
from spot_check.geometry.cube_axes_style import PYVISTA_CUBE_AXES_LABEL_OFFSET


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


def test_pyvista_bounds_property_resets_z_labels() -> None:
    """Regression: actor.bounds copies scene Z into z_axis_range (negative mm)."""
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
    axes = (x_min, x_max, y_min, y_max, spec.z_label_at_min, spec.z_label_at_max)
    pl = pv.Plotter(off_screen=True)
    actor = pl.show_bounds(
        bounds=bounds,
        axes_ranges=axes,
        grid="back",
        location="outer",
        ticks="inside",
        n_zlabels=spec.n_zlabels,
        fmt="%.4g",
    )
    assert float(actor.z_labels[0]) > 0.0
    actor.bounds = bounds
    assert float(actor.z_labels[0]) < 0.0
    apply_pyvista_cube_z_axis(
        actor, spec, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max
    )
    assert actor.z_label_visibility is True
    assert spec.z_label_at_min > spec.z_label_at_max
    assert float(actor.z_labels[0]) > float(actor.z_labels[-1])
    assert actor.GetZAxesLabelProperty().GetOrientation() == pytest.approx(90.0)
    assert actor.GetLabelOffset() == pytest.approx(PYVISTA_CUBE_AXES_LABEL_OFFSET)
    deep = float(max(spec.z_label_at_min, spec.z_label_at_max))
    shallow = float(min(spec.z_label_at_min, spec.z_label_at_max))
    z_lo, z_hi = actor.GetZAxisRange()
    assert z_lo == pytest.approx(shallow)
    assert z_hi == pytest.approx(deep)


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
