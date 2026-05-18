"""Layer-relative plan MU vs measured charge dose QA."""

from __future__ import annotations

import numpy as np

from spot_check import analysis
from spot_check.analysis._core import _hex_to_rgb_u8
from spot_check.constants import (
    _PLAN_QA_DOSE_UNDER_FAIL_HEX,
    _PLAN_QA_DOSE_UNDER_WARN_HEX,
    _PLAN_QA_FAIL_HEX,
    _PLAN_QA_PASS_HEX,
    _PLAN_QA_WARN_HEX,
)


def _rgb(hex_s: str) -> tuple[int, int, int]:
    r, g, b = _hex_to_rgb_u8(hex_s)
    return (int(r), int(g), int(b))


def test_measured_charge_na_from_tuple_uses_row_weight() -> None:
    tup = (0.0, 0.0, 0.0, 11.0, 0, 0.0, 0.0, 999.0)
    assert analysis.measured_charge_na_from_tuple(tup) == 11.0


def test_plan_dose_fraction_deviation_pp_single_layer() -> None:
    planned = [(0.0, 0.0, 100.0), (10.0, 0.0, 100.0), (20.0, 0.0, 100.0)]
    plan_mu = np.array([10.0, 20.0, 70.0])
    # Fit A=x, Fit B=y when a_is_x=True; layer index 0.
    measured = [
        (0.0, 0.0, 0.0, 11.0, 0, 0.0, 0.0, 11.0),
        (10.0, 0.0, 0.0, 19.0, 0, 0.0, 0.0, 19.0),
        (20.0, 0.0, 0.0, 70.0, 0, 0.0, 0.0, 70.0),
    ]
    dev_pp, plan_frac, meas_frac, _dist = analysis.plan_dose_fraction_deviation_pp(
        planned, plan_mu, measured, a_is_x=True
    )
    assert dev_pp.shape == (3,)
    np.testing.assert_allclose(plan_frac, [0.1, 0.2, 0.7], rtol=0, atol=1e-12)
    np.testing.assert_allclose(meas_frac, [0.11, 0.19, 0.70], rtol=0, atol=1e-12)
    np.testing.assert_allclose(dev_pp, [1.0, 1.0, 0.0], rtol=0, atol=1e-9)


def test_measured_rgba_by_plan_dose_qa_directional_colors() -> None:
    signed = np.array([-4.0, -2.0, 0.0, 2.0, 4.0])
    rgba = analysis.measured_rgba_by_plan_dose_qa(signed, pass_pp=1.0, warn_pp=3.0)
    assert rgba.shape == (5, 4)
    assert tuple(rgba[0, :3]) == _rgb(_PLAN_QA_DOSE_UNDER_FAIL_HEX)
    assert tuple(rgba[1, :3]) == _rgb(_PLAN_QA_DOSE_UNDER_WARN_HEX)
    assert tuple(rgba[2, :3]) == _rgb(_PLAN_QA_PASS_HEX)
    assert tuple(rgba[3, :3]) == _rgb(_PLAN_QA_WARN_HEX)
    assert tuple(rgba[4, :3]) == _rgb(_PLAN_QA_FAIL_HEX)


def test_plan_dose_qa_tier_counts() -> None:
    signed = np.array([-4.0, -2.0, 0.0, 2.0, 4.0, np.nan])
    counts = analysis.plan_dose_qa_tier_counts(signed, pass_pp=1.0, warn_pp=3.0)
    assert counts == (1, 1, 1, 1, 1)


def test_plan_dose_fraction_deviation_pp_uses_row_weight_not_channel_sum() -> None:
    planned = [(0.0, 0.0, 100.0), (10.0, 0.0, 100.0)]
    plan_mu = np.array([30.0, 70.0])
    # Index 3 = fit-style weights; index 7 = channel sums that would imply 50/50 if used.
    measured = [
        (0.0, 0.0, 0.0, 30.0, 0, 0.0, 0.0, 50.0),
        (10.0, 0.0, 0.0, 70.0, 0, 0.0, 0.0, 50.0),
    ]
    _dev, _pf, meas_frac, _dist = analysis.plan_dose_fraction_deviation_pp(
        planned, plan_mu, measured, a_is_x=True
    )
    np.testing.assert_allclose(meas_frac, [0.3, 0.7], rtol=0, atol=1e-12)


def test_plan_dose_fraction_deviation_pp_without_plan_mu_is_nan() -> None:
    planned = [(0.0, 0.0, 100.0)]
    measured = [(0.0, 0.0, 0.0, 1.0, 0, 0.0, 0.0, 1.0)]
    dev_pp, plan_frac, meas_frac, _dist = analysis.plan_dose_fraction_deviation_pp(
        planned, None, measured, a_is_x=True
    )
    assert np.isnan(dev_pp[0])
    assert np.isnan(plan_frac[0])
    assert np.isnan(meas_frac[0])
