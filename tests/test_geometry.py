import numpy as np

from spot_check.geometry import (
    apply_pyvista_cube_z_axis,
    cube_z_axis_spec,
    n_cube_axis_labels_for_mm_step,
    nominal_mev_to_plot_z,
    proton_cda_water_range_mm,
)


def test_proton_cda_monotonic() -> None:
    e = np.array([50.0, 100.0, 150.0])
    r = proton_cda_water_range_mm(e)
    assert r.shape == (3,)
    assert r[0] < r[1] < r[2]


def test_n_cube_axis_labels_capped() -> None:
    n = n_cube_axis_labels_for_mm_step(0.0, 1000.0, 5.0, max_n=11)
    assert n == 11


def test_pyvista_bounds_property_resets_z_labels() -> None:
    """Regression: actor.bounds copies scene Z into z_axis_range (negative mm)."""
    import os

    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    import pyvista as pv

    pv.OFF_SCREEN = True
    e = np.linspace(100.0, 160.0, 12)
    z = nominal_mev_to_plot_z(e, use_proton_water_depth_mm=True)
    spec = cube_z_axis_spec(z, use_proton_water_depth_mm=True, tick_mm=5.0)
    x_min, x_max, y_min, y_max = -40.0, 40.0, -40.0, 40.0
    bounds = (x_min, x_max, y_min, y_max, spec.zmin_scene, spec.zmax_scene)
    axes = (x_min, x_max, y_min, y_max, spec.z_label_at_min, spec.z_label_at_max)
    pl = pv.Plotter(off_screen=True)
    actor = pl.show_bounds(
        bounds=bounds,
        axes_ranges=axes,
        grid="back",
        location="closest",
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


def test_plot_z_shallow_toward_top_and_depth_labels() -> None:
    e = np.array([100.0, 140.0, 180.0])
    z = nominal_mev_to_plot_z(e, use_proton_water_depth_mm=True)
    assert np.all(z < 0)
    assert z[0] > z[-1]  # lower energy / shallower → higher scene Z (top)
    spec = cube_z_axis_spec(z, use_proton_water_depth_mm=True, tick_mm=5.0)
    assert spec.z_label_at_min > spec.z_label_at_max  # deep mm at zmin, shallow at zmax
    assert spec.zmin_scene < spec.zmax_scene
    assert spec.z_label_at_min > 0 and spec.z_label_at_max > 0
