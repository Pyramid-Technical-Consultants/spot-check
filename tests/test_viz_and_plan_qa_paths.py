"""Exercise 3D plotter and plan-QA code paths that broke after the analysis split."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from spot_check import analysis
from spot_check.analysis.measured import (
    MeasuredAssignResult,
    assign_measured_from_csv,
    finalize_measured_assign_coverage,
)
from spot_check.analysis.plan_qa import _plan_qa_error_line_polylines
from spot_check.analysis.viz.glyphs import _plan_spot_cross_mesh
from spot_check.constants import _PLAN_MISSING_CROSS_HALF_ARM_MM, _PLAN_QA_FAIL_HEX
from spot_check.models import ZAxisDisplayConfig
from tests.conftest import MINIMAL_PLANNED_XYZ, minimal_measured_rows, write_measured_csv

pytest.importorskip("pyvista")


def test_finalize_measured_assign_coverage_plan_sequential() -> None:
    result = MeasuredAssignResult(
        rows=[(0.0, 0.0, 0.0, 1.0, 0), (1.0, 1.0, 0.0, 1.0, 0)],
        spot_ids=[0, 2],
        layer_mode="auto",
        assign_method="plan_sequential",
        n_plan_spots=4,
    )
    planned = [(0.0, 0.0, 100.0)] * 4
    out = finalize_measured_assign_coverage(result, planned_xyz=planned)
    assert out.plan_spots_no_data is not None
    assert out.plan_spots_no_data.tolist() == [False, True, False, True]


def test_finalize_measured_assign_coverage_gate_counter(tmp_path) -> None:
    csv_path = write_measured_csv(tmp_path / "gc.csv", minimal_measured_rows())
    assigned = assign_measured_from_csv(
        csv_path,
        planned_xyz=list(MINIMAL_PLANNED_XYZ),
        layer_mode="gate_counter",
        a_is_x=False,
    )
    out = finalize_measured_assign_coverage(assigned, planned_xyz=list(MINIMAL_PLANNED_XYZ))
    assert out.plan_spots_no_data is not None
    # minimal rows: gate 1 -> spot 0, gate 3 -> spot 1; plan spots 2–3 missing
    assert out.plan_spots_no_data.tolist() == [False, False, True, True]


def _plan_cross_actor_mesh(pl: Any) -> Any | None:
    """Return the missing-plan-spot cross mesh, or None if not found."""
    for actor in pl.renderer.actors.values():
        mapper = getattr(actor, "mapper", None)
        if mapper is None:
            continue
        ds = getattr(mapper, "dataset", None)
        if ds is None or int(getattr(ds, "n_lines", 0)) <= 0:
            continue
        if "rgba" in ds.point_data:
            return ds
    return None


def test_plan_spot_cross_mesh_uses_fixed_arm_length() -> None:
    plan_pts = np.array([[0.0, 0.0, 100.0], [5.0, 5.0, 100.0]], dtype=np.float64)
    mask = np.array([True, False], dtype=bool)
    arm = float(_PLAN_MISSING_CROSS_HALF_ARM_MM)
    cross = _plan_spot_cross_mesh(plan_pts, spot_mask=mask)
    pts = np.asarray(cross.points, dtype=np.float64)
    assert pts.shape == (4, 3)
    assert np.allclose(pts[0], (-arm, 0.0, 100.0))
    assert np.allclose(pts[1], (arm, 0.0, 100.0))
    assert np.allclose(pts[2], (0.0, -arm, 100.0))
    assert np.allclose(pts[3], (0.0, arm, 100.0))


def test_show_comparison_3d_pyvista_missing_plan_spots_render_red_crosses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plan-sequential spots with no assigned rows render as red crosses, not circles."""
    import pyvista as pv

    from spot_check.analysis.colors import _hex_to_rgb_u8

    planned = [(0.0, 0.0, 100.0), (1.0, 0.0, 100.0), (2.0, 0.0, 100.0)]
    measured = [(0.1, 0.0, 0.0, 1.0, 0), (2.1, 0.0, 0.0, 1.0, 0)]
    no_data = np.array([False, True, False], dtype=bool)
    pl = pv.Plotter(off_screen=True)
    monkeypatch.setattr(pl, "show", lambda *args, **kwargs: None)
    try:
        analysis.show_comparison_3d_pyvista(
            planned,
            measured,
            title="missing plan crosses",
            a_is_x=False,
            scale_plan_spots_by_dicom_fwhm=False,
            plan_spots_no_data=no_data,
            reuse_plotter=pl,
            reembed_qt=False,
        )
        cross = _plan_cross_actor_mesh(pl)
        assert cross is not None
        assert int(cross.n_lines) == 2
        rgba = np.asarray(cross["rgba"], dtype=np.uint8)
        rf, gf, bf = _hex_to_rgb_u8(_PLAN_QA_FAIL_HEX)
        assert int(rgba[0, 0]) == rf
        assert int(rgba[0, 1]) == gf
        assert int(rgba[0, 2]) == bf
    finally:
        pl.close()


def test_apply_comparison_3d_projection_view_toggles_camera() -> None:
    import pyvista as pv

    pl = pv.Plotter(off_screen=True)
    try:
        pl.add_mesh(pv.Sphere())
        analysis.apply_comparison_3d_projection_view(pl, perspective=True, render=False)
        assert pl.camera.parallel_projection is False
        analysis.apply_comparison_3d_projection_view(pl, perspective=False, render=False)
        assert pl.camera.parallel_projection is True
    finally:
        pl.close()


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
    z_cfg = ZAxisDisplayConfig(use_water_depth_mm=False)
    warn_lines, fail_lines = _plan_qa_error_line_polylines(
        meas_pts,
        exp,
        dist,
        pass_mm=1.0,
        warn_mm=3.0,
        z_display_cfg=z_cfg,
        plan_e_lo_mev=100.0,
        plan_e_hi_mev=120.0,
    )
    # May be None when all pass; with our fixture at least one point may warn/fail.
    assert warn_lines is None or hasattr(warn_lines, "n_points")
    assert fail_lines is None or hasattr(fail_lines, "n_points")


def test_plan_qa_error_line_target_z_uses_plan_depth_bounds() -> None:
    """Per-spot depth mapping must use plan-wide bounds (same as measured/plan clouds)."""
    from spot_check.geometry.z_axis import (
        nominal_energy_to_scene_z,
        nominal_mev_column_to_scene_z,
    )

    e_lo, e_hi = 100.0, 120.0
    cfg = ZAxisDisplayConfig(use_water_depth_mm=True)
    z_plan = float(
        nominal_mev_column_to_scene_z(
            np.array([e_lo], dtype=np.float64),
            plan_e_lo=e_lo,
            plan_e_hi=e_hi,
            config=cfg,
        )[0]
    )
    # Per-spot min/max affine (no plan-wide depth bounds) mis-maps a lone energy.
    z_single_bug = float(
        nominal_energy_to_scene_z(
            np.array([e_lo], dtype=np.float64),
            plan_e_lo=e_lo,
            plan_e_hi=e_lo,
            config=cfg,
        )[0]
    )
    assert z_single_bug != pytest.approx(z_plan)

    meas_pts = np.array([[1.0, 2.0, 0.0]], dtype=np.float64)
    exp_xyz = np.array([[3.0, 4.0, e_lo]], dtype=np.float64)
    warn_lines, _fail = _plan_qa_error_line_polylines(
        meas_pts,
        exp_xyz,
        np.array([2.0], dtype=np.float64),
        pass_mm=1.0,
        warn_mm=3.0,
        z_display_cfg=cfg,
        plan_e_lo_mev=e_lo,
        plan_e_hi_mev=e_hi,
    )
    assert warn_lines is not None
    pts = np.asarray(warn_lines.points, dtype=np.float64)
    assert pts.shape == (2, 3)
    assert pts[1, 0] == pytest.approx(3.0)
    assert pts[1, 1] == pytest.approx(4.0)
    assert pts[1, 2] == pytest.approx(z_plan)


def test_build_spot_display_meshes_measured_z_matches_plan_scene_z() -> None:
    """Measured draw positions must use the same scene-Z map as plan spots."""
    from spot_check.analysis.viz.data import prepare_comparison_3d_data
    from spot_check.analysis.viz.display_meshes import build_spot_display_meshes
    from spot_check.geometry.z_axis import apply_z_display_to_comparison_clouds
    from spot_check.models import ZAxisDisplayConfig

    planned_xyz = [(0.0, 0.0, 100.0), (1.0, 0.0, 120.0), (2.0, 0.0, 140.0)]
    measured_abc = [(0.1, 0.0, 1.0, 1.0, 0)]
    prep = prepare_comparison_3d_data(planned_xyz, measured_abc, a_is_x=True)
    cfg = ZAxisDisplayConfig(use_water_depth_mm=True, upstream_wet_mm=5.0)
    state = build_spot_display_meshes(
        planned_xyz,
        measured_abc,
        prep,
        z_display_cfg=cfg,
        a_is_x=True,
        weight_measured_by_channel=False,
        plan_qa_coloring=False,
        qa_mode="position",
        plan_qa_pass_mm=1.0,
        plan_qa_warn_mm=3.0,
        plan_qa_pass_pp=1.0,
        plan_qa_warn_pp=3.0,
        plan_mu=None,
        plan_qa_hide_pass_spots=False,
        plan_qa_draw_error_lines=False,
        scale_plan_spots_by_dicom_fwhm=False,
        measured_spots_sigma_world_mm=False,
        measured_sigma_glyph_scale=None,
        spot_weight_mode="channel_sum",
    )
    plan_pts = np.asarray(state["plan_pts"], dtype=np.float64)
    meas_pts_final = np.asarray(state["meas_pts_final"], dtype=np.float64)
    _, meas_scene, _ = apply_z_display_to_comparison_clouds(
        prep.plan_xyz,
        prep.meas_xyz,
        plan_e_lo=float(prep.e_lo),
        plan_e_hi=float(prep.e_hi),
        config=cfg,
    )
    assert meas_pts_final.shape[0] == 1
    assert meas_pts_final[0, 2] == pytest.approx(float(meas_scene[0, 2]))
    plan_layer_idx = 1
    assert meas_pts_final[0, 2] == pytest.approx(float(plan_pts[plan_layer_idx, 2]))


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


def _plan_actor_mesh(pl: Any) -> Any | None:
    """Return the plan spot mesh (full stack), or None if not found."""
    for actor in pl.renderer.actors.values():
        mapper = getattr(actor, "mapper", None)
        if mapper is None:
            continue
        ds = getattr(mapper, "dataset", None)
        if ds is None or "rgba" not in ds.point_data:
            continue
        return ds
    return None


def test_refresh_comparison_3d_display_hides_pass_spots_in_place(
    measured_csv_writer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Display-only refresh must filter pass-tier spots without rebuilding the plotter."""
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
            title="qa hide refresh",
            a_is_x=False,
            plan_qa_coloring=True,
            plan_qa_mode="position",
            plan_qa_pass_mm=0.01,
            plan_qa_warn_mm=100.0,
            plan_qa_hide_pass_spots=False,
            reuse_plotter=pl,
            reembed_qt=False,
        )
        n_before = _meas_actor_point_count(pl)
        assert n_before is not None and n_before > 0
        analysis.refresh_comparison_3d_display(
            pl,
            list(MINIMAL_PLANNED_XYZ),
            measured,
            a_is_x=False,
            plan_qa_coloring=True,
            plan_qa_mode="position",
            plan_qa_pass_mm=0.01,
            plan_qa_warn_mm=100.0,
            plan_qa_hide_pass_spots=True,
        )
        n_after = _meas_actor_point_count(pl)
        assert n_after is not None
        assert n_after <= n_before
    finally:
        pl.close()


def _meas_actor_point_count(pl: Any) -> int | None:
    for actor in pl.renderer.actors.values():
        mapper = getattr(actor, "mapper", None)
        if mapper is None:
            continue
        ds = getattr(mapper, "dataset", None)
        if ds is None:
            continue
        n = int(getattr(ds, "n_points", 0))
        if n > 0:
            return n
    return None


def test_slice_band_hides_plan_spots_invisibly(monkeypatch: pytest.MonkeyPatch) -> None:
    """5-layer slice keeps all plan geometry but hides out-of-band spots (alpha=0)."""
    import pyvista as pv

    from spot_check.analysis.spatial import nominal_layer_energies_mev
    from spot_check.analysis.viz.data import _energy_slice_mask, _nominal_layer_index_band_mev

    planned = [(0.0, 0.0, e) for e in (70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0)]
    layer_e = nominal_layer_energies_mev(planned)
    center_i = 3
    lo_m, hi_m = _nominal_layer_index_band_mev(layer_e, center_i, half_width=2)
    plan_e = np.array([s[2] for s in planned], dtype=np.float64)
    n_in_band = int(np.count_nonzero(_energy_slice_mask(plan_e, lo_m, hi_m)))
    assert n_in_band == 5
    assert len(planned) == 7

    pl = pv.Plotter(off_screen=True)
    monkeypatch.setattr(pl, "show", lambda *args, **kwargs: None)
    try:
        analysis.show_comparison_3d_pyvista(
            planned,
            [],
            title="slice plan filter",
            a_is_x=False,
            slice_band_init={"slice_on": True, "center_i": center_i},
            reuse_plotter=pl,
            reembed_qt=False,
        )
        mesh = _plan_actor_mesh(pl)
        assert mesh is not None
        assert int(mesh.n_points) == len(planned)
        rgba = np.asarray(mesh["rgba"], dtype=np.uint8)
        assert int(np.count_nonzero(rgba[:, 3] > 0)) == n_in_band
        assert int(np.count_nonzero(rgba[:, 3] == 0)) == len(planned) - n_in_band
    finally:
        pl.close()


def test_time_slice_mask_window() -> None:
    from spot_check.analysis.viz.data import _time_slice_mask
    from spot_check.constants import TIME_SLICE_WINDOW_FULL

    t = np.array([0.0, 0.4, 1.1, 1.9, 2.0], dtype=np.float64)
    m = _time_slice_mask(t, 0.0, window_s=1.0)
    assert list(m) == [True, True, False, False, False]
    m1 = _time_slice_mask(t, 1.0, window_s=1.0)
    assert list(m1) == [False, False, True, True, True]
    m_all = _time_slice_mask(
        t, 1.0, window_s=TIME_SLICE_WINDOW_FULL, t_min=0.0, t_max=2.0
    )
    assert list(m_all) == [True, True, False, False, False]
    m_all_end = _time_slice_mask(
        t, 2.0, window_s=TIME_SLICE_WINDOW_FULL, t_min=0.0, t_max=2.0
    )
    assert list(m_all_end) == [True, True, True, True, True]


def test_time_slice_hides_out_of_window_measured(
    measured_csv_writer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """1 s time window filters measured spots outside the window."""
    import pyvista as pv

    from spot_check.analysis.viz.data import _time_slice_mask

    csv_path = measured_csv_writer(
        [
            {
                "time (s)": "0.0",
                "IX512 Channel Sum (nA)": "1.0",
                "Fit Amplitude A (nA)": "0.5",
                "Fit Mean Position A (mm)": "0.0",
                "Fit Mean Position B (mm)": "0.0",
                "Gate Counter": "1",
            },
            {
                "time (s)": "0.4",
                "IX512 Channel Sum (nA)": "1.0",
                "Fit Amplitude A (nA)": "0.5",
                "Fit Mean Position A (mm)": "5.0",
                "Fit Mean Position B (mm)": "0.0",
                "Gate Counter": "1",
            },
            {
                "time (s)": "1.5",
                "IX512 Channel Sum (nA)": "1.0",
                "Fit Amplitude A (nA)": "0.5",
                "Fit Mean Position A (mm)": "0.0",
                "Fit Mean Position B (mm)": "5.0",
                "Gate Counter": "3",
            },
        ]
    )
    measured = analysis.measured_spot_abc_from_csv(
        csv_path,
        planned_xyz=list(MINIMAL_PLANNED_XYZ),
        layer_mode="gate_counter",
        a_is_x=False,
        aggregate_spots=False,
    )
    times = np.array([analysis.measured_row_time_s(r) for r in measured], dtype=np.float64)
    assert int(np.count_nonzero(_time_slice_mask(times, 0.0, window_s=1.0))) == 2

    pl = pv.Plotter(off_screen=True)
    monkeypatch.setattr(pl, "show", lambda *args, **kwargs: None)
    try:
        analysis.show_comparison_3d_pyvista(
            [],
            measured,
            title="time slice",
            a_is_x=False,
            time_slice_init={"slice_on": True, "start_ms": 0},
            reuse_plotter=pl,
            reembed_qt=False,
        )
        n_meas_drawn = _meas_actor_point_count(pl)
        assert n_meas_drawn == 2
    finally:
        pl.close()


def test_format_timeline_seconds() -> None:
    from spot_check.gui.timeline_playback import format_timeline_seconds

    assert format_timeline_seconds(4.2) == "0:04.2"
    assert format_timeline_seconds(64.5) == "1:04.5"
    assert format_timeline_seconds(float("nan")) == "—:—"


def test_advance_playback_start_ms() -> None:
    from spot_check.gui.timeline_playback import advance_playback_start_ms

    nxt, at_end = advance_playback_start_ms(
        0, elapsed_ms=500.0, speed=2.0, start_min_ms=0, start_max_ms=5000
    )
    assert nxt == 1000
    assert not at_end
    nxt2, at_end2 = advance_playback_start_ms(
        4900, elapsed_ms=200.0, speed=1.0, start_min_ms=0, start_max_ms=5000
    )
    assert nxt2 == 5000
    assert at_end2


def test_build_plan_spot_delivery_times_s() -> None:
    from spot_check.analysis.viz.data import build_plan_spot_delivery_times_s

    # (a, b, c, weight, ..., time_s at index 8)
    rows = [
        (0.0, 0.0, 70.0, 2.0, 0, 0, 0, 0, 1.0),
        (0.0, 0.0, 70.0, 1.0, 0, 0, 0, 0, 3.0),
        (0.0, 0.0, 80.0, 1.0, 0, 0, 0, 0, 5.0),
    ]
    plan_idx = [0, 0, 1]
    t = build_plan_spot_delivery_times_s(3, rows, plan_idx)
    assert t.shape == (3,)
    assert abs(float(t[0]) - (2.0 * 1.0 + 1.0 * 3.0) / 3.0) < 1e-9
    assert float(t[1]) == 5.0
    assert math.isnan(float(t[2]))


def test_time_slice_hides_out_of_window_plan_spots(monkeypatch: pytest.MonkeyPatch) -> None:
    """1 s time window hides out-of-band plan spots (alpha=0), same as layer band."""
    import pyvista as pv

    from spot_check.analysis.viz.data import _time_slice_mask

    planned = [(0.0, 0.0, 70.0 + i) for i in range(4)]
    plan_times = np.array([0.0, 0.4, 1.5, 2.0], dtype=np.float64)
    n_in_window = int(np.count_nonzero(_time_slice_mask(plan_times, 0.0, window_s=1.0)))
    assert n_in_window == 2

    pl = pv.Plotter(off_screen=True)
    monkeypatch.setattr(pl, "show", lambda *args, **kwargs: None)
    try:
        analysis.show_comparison_3d_pyvista(
            planned,
            [],
            title="time slice plan",
            a_is_x=False,
            time_slice_init={"slice_on": True, "start_ms": 0},
            plan_spot_time_s=plan_times,
            reuse_plotter=pl,
            reembed_qt=False,
        )
        mesh = _plan_actor_mesh(pl)
        assert mesh is not None
        assert int(mesh.n_points) == len(planned)
        rgba = np.asarray(mesh["rgba"], dtype=np.uint8)
        assert int(np.count_nonzero(rgba[:, 3] > 0)) == n_in_window
        assert int(np.count_nonzero(rgba[:, 3] == 0)) == len(planned) - n_in_window
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


def test_spot_index_on_plan_point_and_glyph_meshes() -> None:
    from spot_check.analysis.viz.glyphs import (
        _instanced_axis_aligned_ellipsoids,
        _plan_spot_point_mesh,
    )

    plan_pts = np.array([[0.0, 0.0, 100.0], [1.0, 2.0, 100.0]], dtype=np.float64)
    pts = _plan_spot_point_mesh(plan_pts)
    assert np.array_equal(pts["spot_index"], np.array([0, 1], dtype=np.int32))

    semi = np.array([[1.0, 1.0, 0.1], [2.0, 2.0, 0.1]], dtype=np.float64)
    glyphs = _instanced_axis_aligned_ellipsoids(plan_pts, semi)
    gidx = np.asarray(glyphs["spot_index"], dtype=np.int32)
    assert gidx.shape[0] == int(glyphs.n_points)
    assert set(np.unique(gidx).tolist()) == {0, 1}


def test_format_spot_info_plan_and_measured() -> None:
    from spot_check.analysis.viz.spot_info import format_spot_info

    planned = [(0.0, 0.0, 100.0), (1.0, 0.0, 100.0)]
    measured = [(0.1, 0.2, 0.0, 1.5, 0, 0.3, 0.4)]
    plan_rows = format_spot_info(
        "plan",
        0,
        planned_xyz=planned,
        measured_rows=measured,
        xlab="Fit B (mm)",
        ylab="Fit A (mm)",
        a_is_x=False,
    )
    labels = [r.label for r in plan_rows]
    assert "Plan spot" in [r.value for r in plan_rows]
    assert "Index" in labels
    assert "Nominal energy" in labels

    meas_rows = format_spot_info(
        "measured",
        0,
        planned_xyz=planned,
        measured_rows=measured,
        xlab="Fit B (mm)",
        ylab="Fit A (mm)",
        a_is_x=False,
    )
    meas_labels = [r.label for r in meas_rows]
    assert "Measured spot" in [r.value for r in meas_rows]
    assert "Fit A" in meas_labels
    assert "Plan XY distance" in meas_labels


def test_wire_spot_double_click_pick_registers_observer(monkeypatch: pytest.MonkeyPatch) -> None:
    import pyvista as pv

    pl = pv.Plotter(off_screen=True)
    monkeypatch.setattr(pl, "show", lambda *args, **kwargs: None)
    picked: list[object] = []

    def _cb(ev: object) -> None:
        picked.append(ev)

    try:
        measured = [(0.0, 0.0, 0.0, 1.0, 0), (5.0, 0.0, 0.0, 1.0, 0)]
        analysis.show_comparison_3d_pyvista(
            list(MINIMAL_PLANNED_XYZ),
            measured,
            title="pick wire smoke",
            a_is_x=False,
            scale_plan_spots_by_dicom_fwhm=False,
            reuse_plotter=pl,
            reembed_qt=False,
            on_spot_picked=_cb,
        )
        ctx = getattr(pl, "_spot_check_pick", None)
        assert ctx is not None
        assert ctx.get("observer_id") is not None
    finally:
        pl.close()
