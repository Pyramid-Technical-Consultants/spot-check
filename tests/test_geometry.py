import numpy as np

from spot_check.geometry import n_cube_axis_labels_for_mm_step, proton_cda_water_range_mm


def test_proton_cda_monotonic() -> None:
    e = np.array([50.0, 100.0, 150.0])
    r = proton_cda_water_range_mm(e)
    assert r.shape == (3,)
    assert r[0] < r[1] < r[2]


def test_n_cube_axis_labels_capped() -> None:
    n = n_cube_axis_labels_for_mm_step(0.0, 1000.0, 5.0, max_n=11)
    assert n == 11
