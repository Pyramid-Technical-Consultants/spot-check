import numpy as np

from spot_check.geometry import (
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


def test_plot_z_shallow_toward_top_and_depth_labels() -> None:
    e = np.array([100.0, 140.0, 180.0])
    z = nominal_mev_to_plot_z(e, use_proton_water_depth_mm=True)
    assert np.all(z < 0)
    assert z[0] > z[-1]  # lower energy / shallower → higher scene Z (top)
    spec = cube_z_axis_spec(z, use_proton_water_depth_mm=True, tick_mm=5.0)
    assert spec.z_label_at_min > spec.z_label_at_max  # deep mm at zmin, shallow at zmax
    assert spec.zmin_scene < spec.zmax_scene
    assert spec.z_label_at_min > 0 and spec.z_label_at_max > 0
