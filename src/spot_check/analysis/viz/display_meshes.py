"""Measured/plan spot mesh prep for 3D comparison (full build and fast display refresh)."""

from __future__ import annotations

import math
from typing import Any

from spot_check.analysis._imports import *  # noqa: F403
from spot_check.analysis.colors import (
    _hex_to_rgb_u8,
    _measured_alpha_u8_from_channel_weights,
    measured_rgba_by_channel_weight,
)
from spot_check.analysis.plan_qa import (
    format_plan_dose_qa_caption,
    format_plan_qa_caption,
    measured_rgba_by_plan_dose_qa,
    measured_rgba_by_plan_qa,
    plan_dose_fraction_deviation_pp,
    plan_dose_qa_tier_counts,
    plan_qa_pass_warn_fail_counts,
)
from spot_check.analysis.pyvista_backend import pv, require_pyvista
from spot_check.analysis.spatial import layer_nn_plan_xy_distances_and_expected_xyz
from spot_check.analysis.viz.glyphs import (
    _measured_spot_sigma_glyph_mesh,
    _plan_spot_cross_mesh,
    _plan_spot_fwhm_glyph_mesh,
)


def build_spot_display_meshes(
    planned_xyz: list[tuple[float, float, float]],
    measured_abc: list[tuple[float, ...]],
    prep: Comparison3DData,
    *,
    z_display_cfg: ZAxisDisplayConfig,
    a_is_x: bool,
    weight_measured_by_channel: bool,
    plan_qa_coloring: bool,
    qa_mode: str,
    plan_qa_pass_mm: float,
    plan_qa_warn_mm: float,
    plan_qa_pass_pp: float,
    plan_qa_warn_pp: float,
    plan_mu: np.ndarray | None,
    plan_qa_hide_pass_spots: bool,
    plan_qa_draw_error_lines: bool,
    scale_plan_spots_by_dicom_fwhm: bool,
    measured_spots_sigma_world_mm: bool,
    measured_sigma_glyph_scale: float | None,
    spot_weight_mode: str,
    plan_spots_no_data: np.ndarray | None = None,
    plan_spot_time_s: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute measured/plan display meshes and QA draw arrays (no plotter I/O)."""
    require_pyvista()
    qa_hide_pass_spots = bool(plan_qa_hide_pass_spots) and bool(plan_qa_coloring)
    qa_draw_lines = bool(plan_qa_draw_error_lines) and qa_mode == "position"

    plan_pts, meas_pts, _depth_bounds = apply_z_display_to_comparison_clouds(
        prep.plan_xyz,
        prep.meas_xyz,
        plan_e_lo=float(prep.e_lo),
        plan_e_hi=float(prep.e_hi),
        config=z_display_cfg,
    )
    n_m = int(meas_pts.shape[0])
    meas_cloud = pv.PolyData(np.asarray(meas_pts, dtype=np.float64).reshape(-1, 3))
    gold_mask: np.ndarray | None = None
    if prep.meas_partial_raw is not None and prep.meas_partial_raw.shape[0] == n_m:
        gold_mask = prep.meas_partial_raw > 0

    rows_for_qa = list(measured_abc)[:n_m]
    plan_qa_caption_extra = ""
    dist_qa: np.ndarray | None = None
    exp_xyz_qa: np.ndarray | None = None
    weight_alpha_u8: np.ndarray | None = None
    if (
        n_m > 0
        and weight_measured_by_channel
        and prep.meas_weight is not None
        and int(prep.meas_weight.shape[0]) == n_m
    ):
        weight_alpha_u8 = _measured_alpha_u8_from_channel_weights(prep.meas_weight)
    qa_metric: np.ndarray | None = None
    if plan_qa_coloring and n_m > 0 and planned_xyz:
        if qa_mode == "dose":
            dev_pp, plan_frac_qa, meas_frac_qa, dist_qa = plan_dose_fraction_deviation_pp(
                planned_xyz, plan_mu, rows_for_qa, a_is_x=a_is_x
            )
            qa_metric = dev_pp
            signed_pp = np.where(
                np.isfinite(plan_frac_qa) & np.isfinite(meas_frac_qa),
                (meas_frac_qa - plan_frac_qa) * 100.0,
                np.nan,
            )
            _dist_nn, exp_xyz_qa = layer_nn_plan_xy_distances_and_expected_xyz(
                planned_xyz, rows_for_qa, a_is_x=a_is_x
            )
            meas_cloud["rgba"] = measured_rgba_by_plan_dose_qa(
                signed_pp,
                pass_pp=float(plan_qa_pass_pp),
                warn_pp=float(plan_qa_warn_pp),
                alpha_u8=weight_alpha_u8,
            )
            npass, now, nof, nuw, nuf = plan_dose_qa_tier_counts(
                signed_pp, pass_pp=float(plan_qa_pass_pp), warn_pp=float(plan_qa_warn_pp)
            )
            plan_qa_caption_extra = format_plan_dose_qa_caption(
                pass_pp=float(plan_qa_pass_pp),
                warn_pp=float(plan_qa_warn_pp),
                n_pass=npass,
                n_over_warn=now,
                n_over_fail=nof,
                n_under_warn=nuw,
                n_under_fail=nuf,
                spot_weight_mode=spot_weight_mode,
            )
        else:
            dist_qa, exp_xyz_qa = layer_nn_plan_xy_distances_and_expected_xyz(
                planned_xyz, rows_for_qa, a_is_x=a_is_x
            )
            qa_metric = dist_qa
            meas_cloud["rgba"] = measured_rgba_by_plan_qa(
                dist_qa,
                pass_mm=plan_qa_pass_mm,
                warn_mm=plan_qa_warn_mm,
                alpha_u8=weight_alpha_u8,
            )
            npass, nwarn, nfail = plan_qa_pass_warn_fail_counts(
                dist_qa, pass_mm=plan_qa_pass_mm, warn_mm=plan_qa_warn_mm
            )
            plan_qa_caption_extra = format_plan_qa_caption(
                pass_mm=plan_qa_pass_mm,
                warn_mm=plan_qa_warn_mm,
                n_pass=npass,
                n_warn=nwarn,
                n_fail=nfail,
            )
            if qa_draw_lines:
                plan_qa_caption_extra += " Lines: warn+fail → NN plan spot."
    elif (
        n_m > 0
        and weight_measured_by_channel
        and prep.meas_weight is not None
        and int(prep.meas_weight.shape[0]) == n_m
    ):
        meas_cloud["rgba"] = measured_rgba_by_channel_weight(
            prep.meas_weight,
            gold_mask=gold_mask,
        )

    if n_m > 0:
        use_rgba = plan_qa_coloring or (gold_mask is not None and bool(np.any(gold_mask)))
        if use_rgba and "rgba" not in meas_cloud.point_data and gold_mask is not None:
            r0, g0, b0 = _hex_to_rgb_u8(_MEASURED_COLOR_3D)
            r1, g1, b1 = _hex_to_rgb_u8(_PARTIAL_AXIS_MEAS_COLOR_3D)
            gm = gold_mask
            rgba = np.zeros((n_m, 4), dtype=np.uint8)
            rgba[:, 0] = np.where(gm, np.uint8(r1), np.uint8(r0))
            rgba[:, 1] = np.where(gm, np.uint8(g1), np.uint8(g0))
            rgba[:, 2] = np.where(gm, np.uint8(b1), np.uint8(b0))
            rgba[:, 3] = 255
            meas_cloud["rgba"] = rgba

    meas_idx = np.arange(int(n_m), dtype=np.int64)
    if qa_hide_pass_spots and qa_metric is not None:
        d_q = np.asarray(qa_metric, dtype=np.float64).reshape(-1)
        if int(d_q.shape[0]) != int(n_m):
            raise ValueError(
                "plan_qa_hide_pass_spots: QA metric length does not match measured count"
            )
        pass_thr = float(plan_qa_pass_pp) if qa_mode == "dose" else float(plan_qa_pass_mm)
        keep = ~(np.isfinite(d_q) & (d_q <= pass_thr))
        n_pass_pts = int(np.count_nonzero(np.isfinite(d_q) & (d_q <= pass_thr)))
        idx = np.flatnonzero(keep)
        meas_idx = idx.astype(np.int64)
        if idx.size == 0:
            meas_cloud = pv.PolyData(np.empty((0, 3), dtype=np.float64))
            n_m = 0
        else:
            meas_cloud = meas_cloud.extract_points(idx)
            n_m = int(meas_cloud.n_points)
        if n_pass_pts > 0:
            plan_qa_caption_extra += (
                f" Omitting {n_pass_pts} pass-tier measured spot(s); {n_m} warn/fail drawn."
            )

    meas_pts_final = np.asarray(meas_cloud.points, dtype=np.float64).copy()
    meas_e_final = np.asarray(prep.meas_xyz[meas_idx, 2], dtype=np.float64).reshape(-1)
    n_meas_src = int(prep.meas_xyz.shape[0])
    if prep.meas_time_s is not None and int(prep.meas_time_s.shape[0]) == n_meas_src:
        meas_time_final = np.asarray(prep.meas_time_s, dtype=np.float64).reshape(-1)[meas_idx]
    else:
        meas_time_final = np.full(int(meas_idx.shape[0]), np.nan, dtype=np.float64)
    dist_qa_draw: np.ndarray | None = None
    exp_xyz_qa_draw: np.ndarray | None = None
    if dist_qa is not None:
        dist_qa_draw = np.asarray(dist_qa, dtype=np.float64).reshape(-1)[meas_idx]
    if exp_xyz_qa is not None:
        exp_xyz_qa_draw = np.asarray(exp_xyz_qa, dtype=np.float64).reshape(-1, 3)[meas_idx]
    meas_rgba_final: np.ndarray | None = None
    if "rgba" in meas_cloud.point_data:
        meas_rgba_final = np.asarray(meas_cloud.point_data["rgba"]).copy()

    sig_scale_eff = float(
        MEASURED_SIGMA_GLYPH_SCALE_DEFAULT
        if measured_sigma_glyph_scale is None
        else measured_sigma_glyph_scale
    )
    n_sig_src = int(prep.meas_xyz.shape[0])
    if prep.meas_sigma_xy_mm is None or int(prep.meas_sigma_xy_mm.shape[0]) != n_sig_src:
        meas_sigma_all = np.full((n_sig_src, 2), np.nan, dtype=np.float64)
    else:
        meas_sigma_all = np.asarray(prep.meas_sigma_xy_mm, dtype=np.float64).reshape(n_sig_src, 2)
    meas_sigma_final = (
        meas_sigma_all[meas_idx] if n_sig_src > 0 else np.zeros((0, 2), dtype=np.float64)
    )
    display_perf_note = ""
    n_m_pre_display = int(n_m)
    if plan_qa_coloring and n_m_pre_display > 80_000 and _cKDTree is None:
        logger.warning(
            "Install scipy for much faster plan QA on large acquisitions (%s measured rows).",
            n_m_pre_display,
        )

    want_sigma_glyphs = bool(measured_spots_sigma_world_mm) and n_m_pre_display > 0

    if want_sigma_glyphs and n_m_pre_display > DISPLAY_GLYPH_INSTANCE_CAP:
        step = int(math.ceil(n_m_pre_display / DISPLAY_GLYPH_INSTANCE_CAP))
        sub = np.arange(0, n_m_pre_display, step, dtype=np.intp)
        display_perf_note += (
            f" Measured σ ellipsoid stride {step} "
            f"(~{sub.size} of {n_m_pre_display} spots for display)."
        )
        meas_pts_final = meas_pts_final[sub]
        meas_e_final = meas_e_final[sub]
        meas_time_final = meas_time_final[sub]
        if dist_qa_draw is not None:
            dist_qa_draw = dist_qa_draw[sub]
        if exp_xyz_qa_draw is not None:
            exp_xyz_qa_draw = exp_xyz_qa_draw[sub]
        if meas_rgba_final is not None:
            meas_rgba_final = meas_rgba_final[sub]
        meas_sigma_final = meas_sigma_final[sub]
        n_m = int(sub.size)
    elif not want_sigma_glyphs and n_m_pre_display > DISPLAY_POINT_MESH_TARGET:
        step = int(math.ceil(n_m_pre_display / DISPLAY_POINT_MESH_TARGET))
        sub = np.arange(0, n_m_pre_display, step, dtype=np.intp)
        display_perf_note += (
            f" Measured mesh stride {step} (~{sub.size} of {n_m_pre_display} points for display)."
        )
        meas_pts_final = meas_pts_final[sub]
        meas_e_final = meas_e_final[sub]
        meas_time_final = meas_time_final[sub]
        if dist_qa_draw is not None:
            dist_qa_draw = dist_qa_draw[sub]
        if exp_xyz_qa_draw is not None:
            exp_xyz_qa_draw = exp_xyz_qa_draw[sub]
        if meas_rgba_final is not None:
            meas_rgba_final = meas_rgba_final[sub]
        meas_sigma_final = meas_sigma_final[sub]
        n_m = int(sub.size)

    meas_sigma_glyphs = want_sigma_glyphs and n_m > 0

    def _make_measured_view_mesh(
        pts: np.ndarray,
        sig_xy: np.ndarray,
        rgba: np.ndarray | None,
    ) -> Any:
        if pts.shape[0] == 0:
            return pv.PolyData(np.zeros((0, 3), dtype=np.float64))
        if meas_sigma_glyphs:
            return _measured_spot_sigma_glyph_mesh(
                pts,
                sig_xy,
                sigma_scale=sig_scale_eff,
                rgba=rgba,
            )
        m = pv.PolyData(pts)
        if rgba is not None:
            m["rgba"] = rgba
        return m

    meas_view0 = _make_measured_view_mesh(meas_pts_final, meas_sigma_final, meas_rgba_final)

    plan_cloud = pv.PolyData(plan_pts)
    n_plan = int(plan_pts.shape[0])
    if plan_spots_no_data is not None:
        no_data = np.asarray(plan_spots_no_data, dtype=bool).reshape(-1)
        if int(no_data.shape[0]) != n_plan:
            raise ValueError("plan_spots_no_data length must match plan spot count")
    else:
        no_data = np.zeros(n_plan, dtype=bool)
    has_data_mask = ~no_data
    plan_allow_fwhm_glyphs = bool(scale_plan_spots_by_dicom_fwhm) and n_plan <= int(
        DISPLAY_GLYPH_INSTANCE_CAP
    )
    if bool(scale_plan_spots_by_dicom_fwhm) and n_plan > int(
        DISPLAY_GLYPH_INSTANCE_CAP
    ):
        display_perf_note += (
            f" Plan FWHM ellipsoids disabled ({n_plan} spots > "
            f"{DISPLAY_GLYPH_INSTANCE_CAP} glyph budget — use points."
        )

    plan_rendered_fwhm_glyphs = False
    plan_glyphs: Any = None
    if (
        plan_allow_fwhm_glyphs
        and prep.plan_fwhm_xy_mm is not None
        and prep.plan_fwhm_xy_mm.shape[0] == n_plan
        and bool(np.any(np.isfinite(prep.plan_fwhm_xy_mm)))
    ):
        try:
            plan_glyphs = _plan_spot_fwhm_glyph_mesh(plan_pts, prep.plan_fwhm_xy_mm)
            plan_rendered_fwhm_glyphs = True
        except Exception:
            plan_rendered_fwhm_glyphs = False
            plan_glyphs = None

    plan_cross_mesh: Any = None
    if bool(np.any(no_data)):
        plan_cross_mesh = _plan_spot_cross_mesh(plan_pts, spot_mask=no_data)

    n_plan = int(plan_pts.shape[0])
    if plan_spot_time_s is not None:
        pt = np.asarray(plan_spot_time_s, dtype=np.float64).reshape(-1)
        if int(pt.shape[0]) != n_plan:
            raise ValueError("plan_spot_time_s length must match plan spot count")
        plan_time_final = pt.copy()
    else:
        plan_time_final = np.full(n_plan, np.nan, dtype=np.float64)

    return {
        "plan_pts": plan_pts,
        "meas_pts": meas_pts,
        "meas_view0": meas_view0,
        "meas_pts_final": meas_pts_final,
        "meas_e_final": meas_e_final,
        "meas_time_final": meas_time_final,
        "plan_time_final": plan_time_final,
        "meas_sigma_final": meas_sigma_final,
        "meas_rgba_final": meas_rgba_final,
        "dist_qa_draw": dist_qa_draw,
        "exp_xyz_qa_draw": exp_xyz_qa_draw,
        "plan_qa_caption_extra": plan_qa_caption_extra,
        "plan_cloud": plan_cloud,
        "plan_glyphs": plan_glyphs,
        "plan_cross_mesh": plan_cross_mesh,
        "plan_spots_no_data": no_data,
        "plan_has_data_mask": has_data_mask,
        "plan_rendered_fwhm_glyphs": plan_rendered_fwhm_glyphs,
        "plan_allow_fwhm_glyphs": plan_allow_fwhm_glyphs,
        "meas_sigma_glyphs": meas_sigma_glyphs,
        "sig_scale_eff": sig_scale_eff,
        "n_m": n_m,
        "display_perf_note": display_perf_note,
        "plan_qa_coloring": bool(plan_qa_coloring),
        "qa_draw_lines": bool(qa_draw_lines),
        "scale_plan_spots_by_dicom_fwhm": bool(scale_plan_spots_by_dicom_fwhm),
    }
