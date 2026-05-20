"""Exercise 3D plotter and plan-QA code paths that broke after the analysis split."""

from __future__ import annotations

import math

import numpy as np
import pytest

from spot_check import analysis
from spot_check.analysis.plan_qa import _plan_qa_error_line_polylines
from tests.conftest import MINIMAL_PLANNED_XYZ

pytest.importorskip("pyvista")


def test_format_plan_dose_qa_caption_uses_weight_label() -> None:
    cap = analysis.format_plan_dose_qa_caption(
        pass_pp=1.0,
        warn_pp=3.0,
        n_pass=10,
        n_over_warn=2,
        n_over_fail=1,
        n_under_warn=0,
        n_under_fail=0,
        spot_weight_mode="channel_sum",
    )
    assert "Dose QA" in cap
    assert "IX512" in cap or "Channel Sum" in cap or "channel" in cap.lower()


def test_format_plan_qa_caption_position() -> None:
    cap = analysis.format_plan_qa_caption(
        pass_mm=1.0,
        warn_mm=3.0,
        n_pass=5,
        n_warn=2,
        n_fail=1,
    )
    assert "Position QA" in cap or "≤" in cap


def test_layer_nn_plan_xy_skips_nonfinite_measured_xy() -> None:
    planned = list(MINIMAL_PLANNED_XYZ)
    measured = [
        (1.0, 2.0, 0.0, 1.0, 0, float("nan"), float("nan"), 1.0),
        (3.0, 4.0, 0.0, 1.0, 0, float("nan"), float("nan"), 1.0),
    ]
    dist, _exp = analysis.layer_nn_plan_xy_distances_and_expected_xyz(
        planned, measured, a_is_x=False
    )
    assert dist.shape[0] == 2
    assert math.isfinite(float(dist[1]))


def test_plan_qa_error_line_polylines_builds_with_pyvista(measured_csv_writer) -> None:
    planned = list(MINIMAL_PLANNED_XYZ)
    measured = analysis.measured_spot_abc_from_csv(
        measured_csv_writer(name="err_lines.csv"),
        planned_xyz=planned,
        layer_mode="gate_counter",
        a_is_x=False,
    )
    dist, exp = analysis.layer_nn_plan_xy_distances_and_expected_xyz(
        planned, measured, a_is_x=False
    )
    meas_pts = np.asarray([[r[0], r[1], planned[0][2]] for r in measured], dtype=np.float64)
    warn_lines, fail_lines = _plan_qa_error_line_polylines(
        meas_pts,
        exp,
        dist,
        pass_mm=1.0,
        warn_mm=3.0,
        plan_e_lo_mev=100.0,
        plan_e_hi_mev=120.0,
    )
    # May be None when all pass; with our fixture at least one point may warn/fail.
    assert warn_lines is None or hasattr(warn_lines, "n_points")
    assert fail_lines is None or hasattr(fail_lines, "n_points")


def test_plan_qa_error_line_target_z_uses_plan_depth_bounds() -> None:
    """Per-spot depth mapping must use plan-wide bounds (same as measured/plan clouds)."""
    from spot_check.geometry.z_axis import (
        nominal_depth_to_scene_z_cube,
        plan_depth_bounds_mm,
    )

    e_lo, e_hi = 100.0, 120.0
    d_lo, d_hi = plan_depth_bounds_mm(e_lo, e_hi)
    z_plan = float(
        nominal_depth_to_scene_z_cube(
            np.array([e_lo, e_hi], dtype=np.float64),
            depth_lo_mm=d_lo,
            depth_hi_mm=d_hi,
        )[0]
    )
    z_single_bug = float(nominal_depth_to_scene_z_cube(np.array([e_lo], dtype=np.float64))[0])
    assert z_single_bug != pytest.approx(z_plan)

    meas_pts = np.array([[1.0, 2.0, 0.0]], dtype=np.float64)
    exp_xyz = np.array([[3.0, 4.0, e_lo]], dtype=np.float64)
    warn_lines, _fail = _plan_qa_error_line_polylines(
        meas_pts,
        exp_xyz,
        np.array([2.0], dtype=np.float64),
        pass_mm=1.0,
        warn_mm=3.0,
        use_proton_water_depth_mm=True,
        plan_e_lo_mev=e_lo,
        plan_e_hi_mev=e_hi,
    )
    assert warn_lines is not None
    pts = np.asarray(warn_lines.points, dtype=np.float64)
    assert pts.shape == (2, 3)
    assert pts[1, 0] == pytest.approx(3.0)
    assert pts[1, 1] == pytest.approx(4.0)
    assert pts[1, 2] == pytest.approx(z_plan)


def test_show_comparison_3d_pyvista_plan_only_no_measured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plan without acquisition CSV must render (no channel-weight percentile on empty)."""
    import pyvista as pv

    pl = pv.Plotter(off_screen=True)
    monkeypatch.setattr(pl, "show", lambda *args, **kwargs: None)
    try:
        out = analysis.show_comparison_3d_pyvista(
            list(MINIMAL_PLANNED_XYZ),
            [],
            title="plan only",
            a_is_x=False,
            weight_measured_by_channel=True,
            plan_qa_coloring=True,
            slice_band_init={"slice_on": True, "center_i": 0},
            reuse_plotter=pl,
            reembed_qt=False,
        )
        assert out is pl
    finally:
        pl.close()


def test_show_comparison_3d_pyvista_dose_qa_coloring(
    measured_csv_writer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: dose QA caption called measured_spot_weight_caption without import."""
    import pyvista as pv

    csv_path = measured_csv_writer()
    measured = analysis.measured_spot_abc_from_csv(
        csv_path,
        planned_xyz=list(MINIMAL_PLANNED_XYZ),
        layer_mode="gate_counter",
        a_is_x=False,
    )
    plan_mu = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    pl = pv.Plotter(off_screen=True)
    monkeypatch.setattr(pl, "show", lambda *args, **kwargs: None)
    try:
        out = analysis.show_comparison_3d_pyvista(
            list(MINIMAL_PLANNED_XYZ),
            measured,
            title="test dose QA",
            a_is_x=False,
            plan_qa_coloring=True,
            plan_qa_mode="dose",
            plan_mu=plan_mu,
            plan_qa_pass_pp=1.0,
            plan_qa_warn_pp=3.0,
            reuse_plotter=pl,
            reembed_qt=False,
        )
        assert out is pl
    finally:
        pl.close()


def test_show_comparison_3d_pyvista_position_qa_and_error_lines(
    measured_csv_writer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pyvista as pv

    csv_path = measured_csv_writer()
    measured = analysis.measured_spot_abc_from_csv(
        csv_path,
        planned_xyz=list(MINIMAL_PLANNED_XYZ),
        layer_mode="gate_counter",
        a_is_x=False,
    )
    pl = pv.Plotter(off_screen=True)
    monkeypatch.setattr(pl, "show", lambda *args, **kwargs: None)
    try:
        analysis.show_comparison_3d_pyvista(
            list(MINIMAL_PLANNED_XYZ),
            measured,
            title="test position QA",
            a_is_x=False,
            plan_qa_coloring=True,
            plan_qa_mode="position",
            plan_qa_draw_error_lines=True,
            plan_qa_pass_mm=0.01,
            plan_qa_warn_mm=100.0,
            reuse_plotter=pl,
            reembed_qt=False,
        )
    finally:
        pl.close()
