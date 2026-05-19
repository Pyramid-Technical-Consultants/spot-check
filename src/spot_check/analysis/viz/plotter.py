"""Plotter."""

from __future__ import annotations

from spot_check.analysis._imports import *  # noqa: F403
from spot_check.analysis.colors import (
    _hex_to_rgb_u8,
    _measured_alpha_u8_from_channel_weights,
    measured_rgba_by_channel_weight,
)
from spot_check.analysis.measured import measured_spot_weight_caption
from spot_check.analysis.plan_qa import (
    _plan_qa_error_line_polylines,
    format_plan_dose_qa_caption,
    format_plan_qa_caption,
    measured_rgba_by_plan_dose_qa,
    measured_rgba_by_plan_qa,
    plan_dose_fraction_deviation_pp,
    plan_dose_qa_tier_counts,
    plan_qa_pass_warn_fail_counts,
)
from spot_check.analysis.pyvista_backend import pv, require_pyvista
from spot_check.analysis.spatial import (
    layer_nn_plan_xy_distances_and_expected_xyz,
    nominal_layer_energies_mev,
)
from spot_check.analysis.viz.data import (
    _energy_slice_mask,
    _nominal_layer_index_band_mev,
    prepare_comparison_3d_data,
)
from spot_check.analysis.viz.embed import (
    _embed_pyvista_plotter_in_qt,
    _embed_pyvista_plotter_in_tk,
    _ensure_pyvista_iren_initialized,
    _show_tk_vtk_fallback_panel,
    _start_tk_vtk_event_pump,
    _stop_tk_vtk_event_pump,
    _vtk_rendering_tk_dll_present,
    _wire_slice_band_controls,
    _wire_slice_band_controls_qt,
    idle_slice_band_controls,
    idle_slice_band_controls_qt,
)
from spot_check.analysis.viz.glyphs import (
    _disc_point_add_mesh_kwargs,
    _measured_spot_sigma_glyph_mesh,
    _plan_spot_fwhm_glyph_mesh,
)


def show_comparison_3d_pyvista(
    planned_xyz: list[tuple[float, float, float]],
    measured_abc: list[tuple[float, ...]],
    *,
    title: str,
    a_is_x: bool,
    max_measured_draw: int | None = None,
    layer_mode: str | None = None,
    layer_gap_s: float | None = None,
    refill_same_spot_xy_tol_mm: float | None = None,
    refill_trust_time_gap_stay_dist_mm: float | None = None,
    viterbi_advance_penalty_mm2: float | None = None,
    weight_measured_by_channel: bool = True,
    aggregate_spots: bool = False,
    aggregate_even_rows_after_odd: int = 0,
    spot_weight_mode: str = SPOT_WEIGHT_MODE_DEFAULT,
    detector_align_caption: str | None = None,
    bounds_xy_tick_mm: float | None = None,
    plan_qa_coloring: bool = False,
    plan_qa_mode: str = "position",
    plan_qa_pass_mm: float = PLAN_QA_PASS_MM_DEFAULT,
    plan_qa_warn_mm: float = PLAN_QA_WARN_MM_DEFAULT,
    plan_qa_pass_pp: float = PLAN_QA_DOSE_PASS_PP_DEFAULT,
    plan_qa_warn_pp: float = PLAN_QA_DOSE_WARN_PP_DEFAULT,
    plan_mu: np.ndarray | None = None,
    plan_qa_draw_error_lines: bool = False,
    plan_qa_hide_pass_spots: bool = False,
    plan_fwhm_xy_mm: np.ndarray | None = None,
    scale_plan_spots_by_dicom_fwhm: bool = True,
    measured_spots_sigma_world_mm: bool = False,
    measured_sigma_glyph_scale: float | None = None,
    reuse_plotter: Any | None = None,
    reuse_camera: bool = False,
    reembed_qt: bool = True,
    embed_parent: Any | None = None,
    slice_tk: dict[str, Any] | None = None,
    embed_qt: Any | None = None,
    slice_qt: dict[str, Any] | None = None,
    slice_band_init: dict[str, bool | int] | None = None,
    z_axis_use_proton_water_depth_mm: bool = True,
    view_projection_perspective: bool = True,
) -> Any:
    require_pyvista()

    qa_mode = str(plan_qa_mode).strip().lower().replace("-", "_")
    if qa_mode not in ("position", "dose"):
        raise GeometryConfigError("plan_qa_mode must be 'position' or 'dose'")
    if plan_qa_coloring:
        if qa_mode == "dose":
            if float(plan_qa_warn_pp) <= float(plan_qa_pass_pp):
                raise GeometryConfigError("plan_qa_warn_pp must be greater than plan_qa_pass_pp")
        elif float(plan_qa_warn_mm) <= float(plan_qa_pass_mm):
            raise GeometryConfigError("plan_qa_warn_mm must be greater than plan_qa_pass_mm")
    if plan_qa_draw_error_lines and not plan_qa_coloring:
        raise GeometryConfigError("plan_qa_draw_error_lines requires plan_qa_coloring")
    if plan_qa_draw_error_lines and qa_mode == "dose":
        raise GeometryConfigError("plan_qa_draw_error_lines applies to position QA only")
    qa_hide_pass_spots = bool(plan_qa_hide_pass_spots) and bool(plan_qa_coloring)
    qa_draw_lines = bool(plan_qa_draw_error_lines) and qa_mode == "position"

    prep = prepare_comparison_3d_data(
        planned_xyz,
        measured_abc,
        a_is_x=a_is_x,
        max_measured_draw=max_measured_draw,
        plan_fwhm_xy_mm=plan_fwhm_xy_mm,
    )

    plan_pts = prep.plan_xyz.copy()
    meas_pts = prep.meas_xyz.copy()
    use_depth_z = bool(z_axis_use_proton_water_depth_mm)
    plan_pts[:, 2] = nominal_mev_to_plot_z(plan_pts[:, 2], use_proton_water_depth_mm=use_depth_z)
    meas_pts[:, 2] = nominal_mev_to_plot_z(meas_pts[:, 2], use_proton_water_depth_mm=use_depth_z)

    n_m = meas_pts.shape[0]
    _POINT_SIZE_3D = 9
    meas_cloud = pv.PolyData(meas_pts)
    gold_mask: np.ndarray | None = None
    if prep.meas_partial_raw is not None and prep.meas_partial_raw.shape[0] == n_m:
        gold_mask = prep.meas_partial_raw > 0

    rows_for_qa = list(measured_abc)[:n_m]
    plan_qa_caption_extra = ""
    dist_qa: np.ndarray | None = None
    exp_xyz_qa: np.ndarray | None = None
    weight_alpha_u8: np.ndarray | None = None
    if (
        weight_measured_by_channel
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
        weight_measured_by_channel
        and prep.meas_weight is not None
        and int(prep.meas_weight.shape[0]) == n_m
    ):
        meas_cloud["rgba"] = measured_rgba_by_channel_weight(
            prep.meas_weight,
            gold_mask=gold_mask,
        )

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
            meas_pts = np.empty((0, 3), dtype=np.float64)
            n_m = 0
        else:
            meas_cloud = meas_cloud.extract_points(idx)
            meas_pts = np.asarray(meas_cloud.points, dtype=np.float64)
            n_m = int(meas_pts.shape[0])
        if n_pass_pts > 0:
            plan_qa_caption_extra += (
                f" Omitting {n_pass_pts} pass-tier measured spot(s); {n_m} warn/fail drawn."
            )

    meas_pts_final = np.asarray(meas_cloud.points, dtype=np.float64).copy()
    meas_e_final = np.asarray(prep.meas_xyz[meas_idx, 2], dtype=np.float64).reshape(-1)
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

    plan_e_mev = np.asarray(prep.plan_xyz[:, 2], dtype=np.float64).reshape(-1)
    if plan_pts.shape[0] and meas_pts.shape[0]:
        x_all = np.r_[plan_pts[:, 0], meas_pts[:, 0]]
        y_all = np.r_[plan_pts[:, 1], meas_pts[:, 1]]
    elif plan_pts.shape[0]:
        x_all = plan_pts[:, 0]
        y_all = plan_pts[:, 1]
    else:
        x_all = meas_pts[:, 0]
        y_all = meas_pts[:, 1]
    x_min, x_max = float(np.min(x_all)), float(np.max(x_all))
    y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
    eff_tick = (
        float(BOUNDS_XY_TICK_MM_DEFAULT) if bounds_xy_tick_mm is None else float(bounds_xy_tick_mm)
    )
    if eff_tick > 0.0 and math.isfinite(eff_tick):
        n_xlabels = n_cube_axis_labels_for_mm_step(x_min, x_max, eff_tick)
        n_ylabels = n_cube_axis_labels_for_mm_step(y_min, y_max, eff_tick)
    else:
        n_xlabels = 5
        n_ylabels = 5

    plan_cloud = pv.PolyData(plan_pts)

    plan_allow_fwhm_glyphs = bool(scale_plan_spots_by_dicom_fwhm) and int(plan_pts.shape[0]) <= int(
        DISPLAY_GLYPH_INSTANCE_CAP
    )
    if bool(scale_plan_spots_by_dicom_fwhm) and int(plan_pts.shape[0]) > int(
        DISPLAY_GLYPH_INSTANCE_CAP
    ):
        display_perf_note += (
            f" Plan FWHM ellipsoids disabled ({plan_pts.shape[0]} spots > "
            f"{DISPLAY_GLYPH_INSTANCE_CAP} glyph budget — use points."
        )

    # Sharp circular disc sprites (VTK sphere impostors — not square GL points or gaussian blur).
    point_r = _POINT_SIZE_3D
    point_kw = _disc_point_add_mesh_kwargs(point_size=point_r)

    plan_rendered_fwhm_glyphs = False
    plan_glyphs: Any = None
    if (
        plan_allow_fwhm_glyphs
        and prep.plan_fwhm_xy_mm is not None
        and prep.plan_fwhm_xy_mm.shape[0] == plan_pts.shape[0]
        and bool(np.any(np.isfinite(prep.plan_fwhm_xy_mm)))
    ):
        try:
            plan_glyphs = _plan_spot_fwhm_glyph_mesh(plan_pts, prep.plan_fwhm_xy_mm)
            plan_rendered_fwhm_glyphs = True
        except Exception:
            plan_rendered_fwhm_glyphs = False
            plan_glyphs = None
    else:
        plan_glyphs = None

    pl = reuse_plotter
    saved_camera_position: Any = None
    if pl is not None:
        if reuse_camera:
            try:
                saved_camera_position = pl.camera_position
            except Exception:
                saved_camera_position = None
        # pl.clear() does not remove vtkCubeAxesActor; stale axes keep wrong Z ticks.
        try:
            pl.remove_bounds_axes()
        except Exception:
            pass
        pl.clear()
        try:
            if pl.renderer.cube_axes_actor is not None:
                pl.remove_bounds_axes()
        except Exception:
            pass
    if pl is None:
        pl = pv.Plotter(window_size=(1440, 960), title="Plan vs measured (PyVista)")
    pl.set_background("#0d1117")
    try:
        pl.enable_anti_aliasing("msaa")
    except (TypeError, ValueError):
        try:
            pl.enable_anti_aliasing()
        except Exception:
            pass

    plan_actor: Any | None = None
    if plan_pts.shape[0] > 0:
        if plan_rendered_fwhm_glyphs and plan_glyphs is not None:
            plan_actor = pl.add_mesh(
                plan_glyphs,
                color=_PLAN_COLOR_3D,
                opacity=0.45,
                pickable=True,
                smooth_shading=False,
                lighting=False,
            )
        else:
            plan_actor = pl.add_mesh(
                plan_cloud,
                color=_PLAN_COLOR_3D,
                opacity=0.45,
                pickable=True,
                **point_kw,
            )

    line_warn_actor: Any | None = None
    line_fail_actor: Any | None = None
    if (
        plan_qa_coloring
        and qa_draw_lines
        and dist_qa_draw is not None
        and exp_xyz_qa_draw is not None
    ):
        lines_warn, lines_fail = _plan_qa_error_line_polylines(
            meas_pts_final,
            exp_xyz_qa_draw,
            dist_qa_draw,
            pass_mm=plan_qa_pass_mm,
            warn_mm=plan_qa_warn_mm,
            use_proton_water_depth_mm=use_depth_z,
        )
        if lines_warn is not None:
            line_warn_actor = pl.add_mesh(
                lines_warn,
                color=_PLAN_QA_WARN_HEX,
                line_width=2,
                opacity=0.7,
                pickable=False,
            )
        if lines_fail is not None:
            line_fail_actor = pl.add_mesh(
                lines_fail,
                color=_PLAN_QA_FAIL_HEX,
                line_width=2,
                opacity=0.7,
                pickable=False,
            )

    meas_view0 = _make_measured_view_mesh(meas_pts_final, meas_sigma_final, meas_rgba_final)
    meas_actor: Any | None = None
    if n_m > 0:
        if meas_sigma_glyphs:
            has_rgba = meas_rgba_final is not None and "rgba" in meas_view0.point_data
            if has_rgba:
                meas_actor = pl.add_mesh(
                    meas_view0,
                    scalars="rgba",
                    rgba=True,
                    smooth_shading=False,
                    lighting=False,
                    opacity=1.0,
                    pickable=True,
                )
            else:
                meas_actor = pl.add_mesh(
                    meas_view0,
                    color=_MEASURED_COLOR_3D,
                    smooth_shading=False,
                    lighting=False,
                    opacity=1.0,
                    pickable=True,
                )
        elif meas_rgba_final is not None:
            meas_actor = pl.add_mesh(
                meas_view0,
                scalars="rgba",
                rgba=True,
                opacity=1.0,
                pickable=True,
                **point_kw,
            )
        else:
            meas_actor = pl.add_mesh(
                meas_view0,
                color=_MEASURED_COLOR_3D,
                opacity=1.0,
                pickable=True,
                **point_kw,
            )

    e_rng_lo = float(prep.e_lo)
    e_rng_hi = float(prep.e_hi)
    if e_rng_hi <= e_rng_lo:
        e_rng_lo -= 0.5
        e_rng_hi += 0.5

    layer_energies_plan = nominal_layer_energies_mev(planned_xyz)
    n_plan_layers = len(layer_energies_plan)
    _center_default = (
        int(np.clip(n_plan_layers // 2, 0, max(0, n_plan_layers - 1))) if n_plan_layers else 0
    )
    slice_cfg: dict[str, bool | int] = {
        "slice_on": False,
        "center_i": _center_default,
    }
    if slice_band_init:
        if "slice_on" in slice_band_init:
            slice_cfg["slice_on"] = bool(slice_band_init["slice_on"])
        if "center_i" in slice_band_init:
            ci0 = int(slice_band_init["center_i"])
            slice_cfg["center_i"] = int(np.clip(ci0, 0, max(0, n_plan_layers - 1)))
    # Filled when embedding in Qt so slice callback can repaint the QVTK widget after updates.
    _qt_vtk_embed: dict[str, Any] = {"widget": None}
    _cube_axes: dict[str, Any] = {"actor": None, "z_spec": None}

    def _empty_poly() -> Any:
        return pv.PolyData(np.zeros((0, 3), dtype=np.float64))

    def _slice_lo_hi_mev() -> tuple[float, float]:
        if not bool(slice_cfg["slice_on"]) or n_plan_layers == 0:
            return e_rng_lo, e_rng_hi
        return _nominal_layer_index_band_mev(
            layer_energies_plan, int(slice_cfg["center_i"]), half_width=2
        )

    def _scene_z_for_cube_axes(
        pm: np.ndarray,
        mm: np.ndarray,
    ) -> np.ndarray:
        if bool(slice_cfg["slice_on"]):
            z_parts: list[np.ndarray] = []
            if np.any(pm):
                z_parts.append(np.asarray(plan_pts[pm, 2], dtype=np.float64))
            if np.any(mm):
                z_parts.append(np.asarray(meas_pts_final[mm, 2], dtype=np.float64))
            if z_parts:
                return np.concatenate(z_parts)
        return np.r_[plan_pts[:, 2], meas_pts[:, 2]]

    def _apply_cube_z_axis(actor: Any, z_spec: _CubeZAxisSpec) -> None:
        apply_pyvista_cube_z_axis(
            actor,
            z_spec,
            x_min=float(x_min),
            x_max=float(x_max),
            y_min=float(y_min),
            y_max=float(y_max),
        )

    def _sync_cube_z_axis(
        pm: np.ndarray | None = None,
        mm: np.ndarray | None = None,
    ) -> None:
        actor = _cube_axes.get("actor")
        z_spec0 = _cube_axes.get("z_spec")
        if actor is None or z_spec0 is None:
            return
        if pm is None or mm is None:
            lo_m, hi_m = _slice_lo_hi_mev()
            pm = _energy_slice_mask(plan_e_mev, lo_m, hi_m)
            mm = (
                _energy_slice_mask(meas_e_final, lo_m, hi_m)
                if meas_e_final.size > 0
                else np.zeros(0, dtype=bool)
            )
        z_spec = _cube_z_axis_spec(
            _scene_z_for_cube_axes(pm, mm),
            use_proton_water_depth_mm=use_depth_z,
            tick_mm=eff_tick,
        )
        _cube_axes["z_spec"] = z_spec
        try:
            _apply_cube_z_axis(actor, z_spec)
        except Exception as exc:
            logger.warning("Cube Z-axis refresh failed: %s", exc)

    def _apply_nominal_energy_slice() -> None:
        nonlocal line_warn_actor, line_fail_actor
        lo_m, hi_m = _slice_lo_hi_mev()
        pm = _energy_slice_mask(plan_e_mev, lo_m, hi_m)
        mm = (
            _energy_slice_mask(meas_e_final, lo_m, hi_m)
            if meas_e_final.size > 0
            else np.zeros(0, dtype=bool)
        )

        if plan_rendered_fwhm_glyphs and prep.plan_fwhm_xy_mm is not None:
            if not np.any(pm):
                plan_actor.mapper.dataset = _empty_poly()
            else:
                plan_actor.mapper.dataset = _plan_spot_fwhm_glyph_mesh(
                    plan_pts[pm], prep.plan_fwhm_xy_mm[pm]
                )
        else:
            if not np.any(pm):
                plan_actor.mapper.dataset = _empty_poly()
            else:
                plan_actor.mapper.dataset = pv.PolyData(plan_pts[pm])

        if meas_actor is not None:
            if not np.any(mm):
                meas_actor.mapper.dataset = _empty_poly()
            else:
                sub_pts = meas_pts_final[mm]
                sub_sig = meas_sigma_final[mm]
                sub_rgba = meas_rgba_final[mm] if meas_rgba_final is not None else None
                meas_actor.mapper.dataset = _make_measured_view_mesh(sub_pts, sub_sig, sub_rgba)

        if line_warn_actor is not None:
            pl.remove_actor(line_warn_actor)
            line_warn_actor = None
        if line_fail_actor is not None:
            pl.remove_actor(line_fail_actor)
            line_fail_actor = None
        if (
            plan_qa_coloring
            and qa_draw_lines
            and dist_qa_draw is not None
            and exp_xyz_qa_draw is not None
            and np.any(mm)
        ):
            lw, lf = _plan_qa_error_line_polylines(
                meas_pts_final[mm],
                exp_xyz_qa_draw[mm],
                dist_qa_draw[mm],
                pass_mm=plan_qa_pass_mm,
                warn_mm=plan_qa_warn_mm,
                use_proton_water_depth_mm=use_depth_z,
            )
            if lw is not None:
                line_warn_actor = pl.add_mesh(
                    lw,
                    color=_PLAN_QA_WARN_HEX,
                    line_width=2,
                    opacity=0.7,
                    pickable=False,
                )
            if lf is not None:
                line_fail_actor = pl.add_mesh(
                    lf,
                    color=_PLAN_QA_FAIL_HEX,
                    line_width=2,
                    opacity=0.7,
                    pickable=False,
                )
        _sync_cube_z_axis(pm, mm)
        pl.render()
        qw = None
        if slice_qt is not None:
            qw = slice_qt.get("_qt_vtk_widget")
        if qw is None:
            qw = _qt_vtk_embed.get("widget")
        if qw is not None:
            try:
                qw.update()
            except Exception:
                pass

    if n_plan_layers > 0 and embed_parent is None and embed_qt is None:

        def _on_center_layer(value: float) -> None:
            ci = int(round(float(value)))
            slice_cfg["center_i"] = int(np.clip(ci, 0, n_plan_layers - 1))
            _apply_nominal_energy_slice()

        def _on_slice_mode_checkbox(checked: bool) -> None:
            slice_cfg["slice_on"] = bool(checked)
            _apply_nominal_energy_slice()

        _slider_rng = (0.0, float(max(0, n_plan_layers - 1)))
        _slider_val = float(slice_cfg["center_i"])
        try:
            pl.add_slider_widget(
                _on_center_layer,
                rng=_slider_rng,
                value=_slider_val,
                title="center plan layer (5 around)",
                pointa=(0.02, 0.92),
                pointb=(0.40, 0.92),
                fmt="%.0f",
                style="modern",
                interaction_event="always",
            )
        except (TypeError, ValueError):
            try:
                pl.add_slider_widget(
                    _on_center_layer,
                    rng=_slider_rng,
                    value=_slider_val,
                    title="center plan layer (5 around)",
                    pointa=(0.02, 0.92),
                    pointb=(0.40, 0.92),
                    fmt="%.0f",
                    interaction_event="always",
                )
            except TypeError:
                pl.add_slider_widget(
                    _on_center_layer,
                    rng=_slider_rng,
                    value=_slider_val,
                    title="center plan layer (5 around)",
                    pointa=(0.02, 0.92),
                    pointb=(0.40, 0.92),
                    fmt="%.0f",
                )
        pl.add_checkbox_button_widget(
            _on_slice_mode_checkbox,
            value=False,
            position=(14, 118),
            size=22,
            border_size=4,
        )
        pl.add_text(
            "5-layer slice",
            position=(42, 121),
            font_size=11,
            color="#c9d1d9",
        )

    pl.add_axes(
        line_width=4,
        x_color="#79c0ff",
        y_color="#56d364",
        z_color="#d2a8ff",
        cone_radius=0.4,
        shaft_length=0.7,
    )

    _lm = (layer_mode or "time_gap").strip().lower().replace("-", "_")
    if _lm == "plan_viterbi":
        _vp = (
            VITERBI_LAYER_ADVANCE_PENALTY_MM2_DEFAULT
            if viterbi_advance_penalty_mm2 is None
            else viterbi_advance_penalty_mm2
        )
        caption = (
            f"n_plan={plan_pts.shape[0]}, n_meas={n_m}, MeV [{prep.e_lo:.1f}, {prep.e_hi:.1f}]. "
            f"Layers: Viterbi vs plan, advance penalty {_vp:g} mm^2."
        )
    elif _lm == "auto":
        _vp = (
            VITERBI_LAYER_ADVANCE_PENALTY_MM2_DEFAULT
            if viterbi_advance_penalty_mm2 is None
            else viterbi_advance_penalty_mm2
        )
        _gap = TIME_LAYER_GAP_S_DEFAULT if layer_gap_s is None else layer_gap_s
        _xytol = (
            REFILL_SAME_SPOT_XY_TOLERANCE_MM
            if refill_same_spot_xy_tol_mm is None
            else refill_same_spot_xy_tol_mm
        )
        caption = (
            f"n_plan={plan_pts.shape[0]}, n_meas={n_m}, MeV [{prep.e_lo:.1f}, {prep.e_hi:.1f}]. "
            f"Layers: auto episodes (Δt≥{_gap:g} s or XY step>{_xytol:g} mm starts new episode); "
            f"Viterbi advance {_vp:g} mm^2."
        )
    elif _lm == "gate_counter":
        caption = (
            f"n_plan={plan_pts.shape[0]}, n_meas={n_m}, MeV [{prep.e_lo:.1f}, {prep.e_hi:.1f}]. "
            f"Layers: CSV Gate Counter — odd bucket=spot, even=deadtime; "
            f"spot index advances when count changes; layer from plan order ({GATE_COUNTER_KEY})."
        )
    else:
        _gap = TIME_LAYER_GAP_S_DEFAULT if layer_gap_s is None else layer_gap_s
        _xytol = (
            REFILL_SAME_SPOT_XY_TOLERANCE_MM
            if refill_same_spot_xy_tol_mm is None
            else refill_same_spot_xy_tol_mm
        )
        _trust = (
            REFILL_TRUST_TIME_GAP_STAY_DIST_MM
            if refill_trust_time_gap_stay_dist_mm is None
            else refill_trust_time_gap_stay_dist_mm
        )
        caption = (
            f"n_plan={plan_pts.shape[0]}, n_meas={n_m}, MeV [{prep.e_lo:.1f}, {prep.e_hi:.1f}]. "
            f"Layers: time gap | dt>={_gap:g} s, same-spot<={_xytol:g} mm, far-slice>{_trust:g} mm."
        )
    if prep.meas_partial_raw is not None:
        n_gold = int(np.count_nonzero(prep.meas_partial_raw))
        if n_gold:
            caption += (
                f" Gold: {n_gold} one-axis row(s); missing fit axis from plan at nominal layer."
            )
    if aggregate_spots and _lm != "auto":
        _sw = measured_spot_weight_caption(spot_weight_mode)
        caption += (
            f" Measured spots aggregated: {_sw}-weighted mean XY + σ per odd "
            f"{GATE_COUNTER_KEY} phase."
        )
        if aggregate_even_rows_after_odd > 0:
            caption += (
                f" Gate-counter: up to {aggregate_even_rows_after_odd} even-phase row(s) "
                "with good fits merged after each odd→even switch."
            )
    if plan_rendered_fwhm_glyphs:
        caption += (
            " Plan: FWHM ellipsoids from DICOM Scanning Spot Size (300A,0398); "
            "semiaxis = FWHM/2 in X/Y, thin along scene Z."
        )
    if meas_sigma_glyphs:
        caption += (
            " Measured: world-space σ ellipsoids — X/Y diameter (mm) = "
            f"{2.0 * sig_scale_eff:g}× fit σ ({SIGMA_A_KEY} / {SIGMA_B_KEY} axes mapped like A/B); "
            "thin disk along scene Z."
        )
    if weight_measured_by_channel and prep.meas_weight is not None:
        caption += f" Measured opacity ∝ {measured_spot_weight_caption(spot_weight_mode)}."
    if detector_align_caption:
        caption += " " + detector_align_caption.strip()
    if plan_qa_caption_extra:
        caption += " " + plan_qa_caption_extra
    if n_plan_layers > 0:
        if embed_qt is not None:
            caption += (
                " Right-hand panel: toggle 5-layer slice and drag the center layer slider "
                "(unchecked = full MeV range)."
            )
        elif embed_parent is None:
            caption += (
                " Upper left: enable 5-layer slice (checkbox) to show up to five consecutive "
                "nominal-energy layers (DICOM order) around the center layer index on the "
                "slider; leave the checkbox off to view the full MeV range."
            )
        elif _vtk_rendering_tk_dll_present():
            caption += (
                " Right-hand panel: toggle 5-layer slice and drag the center layer slider "
                "(unchecked = full MeV range)."
            )
        else:
            caption += (
                " Separate 3D window: 5-layer slice uses the right-hand panel "
                "(pip VTK omits Tk embedding)."
            )
    if display_perf_note:
        caption += display_perf_note
    if use_depth_z:
        caption += (
            " Z axis: proton CSDA water-equivalent depth (mm); tick step follows XY bounds (mm)."
        )
    pl.add_text(title, position="upper_left", font_size=11, color="#f0f6fc", shadow=True)
    pl.add_text(caption, position="lower_left", font_size=9, color="#8b949e")

    # Scene Z: negative depth/mm or −E×view_scale (shallow toward top); see nominal_mev_to_plot_z.
    # ``axes_ranges`` maps bounding-box corners to tick labels (mm or MeV).
    lo_m0, hi_m0 = _slice_lo_hi_mev()
    pm0 = _energy_slice_mask(plan_e_mev, lo_m0, hi_m0)
    mm0 = (
        _energy_slice_mask(meas_e_final, lo_m0, hi_m0)
        if meas_e_final.size > 0
        else np.zeros(0, dtype=bool)
    )
    z_all = _scene_z_for_cube_axes(pm0, mm0)
    z_spec_init = _cube_z_axis_spec(
        z_all,
        use_proton_water_depth_mm=use_depth_z,
        tick_mm=eff_tick,
    )
    _cube_axes["z_spec"] = z_spec_init
    bounds_axes = (
        float(x_min),
        float(x_max),
        float(y_min),
        float(y_max),
        z_spec_init.zmin_scene,
        z_spec_init.zmax_scene,
    )
    axes_ranges_scene = (
        float(x_min),
        float(x_max),
        float(y_min),
        float(y_max),
        z_spec_init.z_label_at_min,
        z_spec_init.z_label_at_max,
    )
    _cube_axes["actor"] = pl.show_bounds(
        bounds=bounds_axes,
        axes_ranges=axes_ranges_scene,
        all_edges=False,
        color="#8b949e",
        xtitle=prep.xlab,
        ytitle=prep.ylab,
        ztitle=z_spec_init.ztitle,
        n_xlabels=n_xlabels,
        n_ylabels=n_ylabels,
        n_zlabels=z_spec_init.n_zlabels,
        show_xaxis=True,
        show_yaxis=True,
        show_zaxis=True,
        show_xlabels=True,
        show_ylabels=True,
        show_zlabels=True,
        fmt="%.4g",
        **pyvista_show_bounds_kwargs(),
    )
    try:
        _apply_cube_z_axis(_cube_axes["actor"], z_spec_init)
    except Exception as exc:
        logger.warning("Initial cube Z-axis apply failed: %s", exc)

    if reuse_plotter is None or not reuse_camera:
        pl.reset_camera()
        pl.camera.zoom(1.05)
    elif saved_camera_position is not None:
        try:
            pl.camera_position = saved_camera_position
        except Exception:
            pl.reset_camera()
            pl.camera.zoom(1.05)
    else:
        pl.reset_camera()
        pl.camera.zoom(1.05)

    try:
        pl.camera.parallel_projection = not bool(view_projection_perspective)
    except Exception:
        pass

    _sync_cube_z_axis()
    _apply_nominal_energy_slice()

    if embed_qt is not None:
        try:
            if reembed_qt:
                w = _embed_pyvista_plotter_in_qt(embed_qt, pl)
                _qt_vtk_embed["widget"] = w
                if slice_qt is not None:
                    slice_qt["_qt_vtk_widget"] = w
            else:
                pl.render()
                qw = slice_qt.get("_qt_vtk_widget") if slice_qt is not None else None
                if qw is None:
                    qw = _qt_vtk_embed.get("widget")
                if qw is not None:
                    try:
                        qw.update()
                    except Exception:
                        pass
            _sync_cube_z_axis()
            pl.render()
        except Exception:
            idle_slice_band_controls_qt(slice_qt)
            raise
        if slice_qt is not None:
            if n_plan_layers > 0:
                _wire_slice_band_controls_qt(
                    slice_qt,
                    slice_cfg,
                    layer_energies_plan,
                    n_plan_layers,
                    _apply_nominal_energy_slice,
                )
            else:
                idle_slice_band_controls_qt(slice_qt)
    elif embed_parent is not None:
        if tk is None:
            pl.show()
            idle_slice_band_controls(slice_tk)
        else:
            for _child in embed_parent.winfo_children():
                _child.destroy()
            tk_top = embed_parent.winfo_toplevel()
            _stop_tk_vtk_event_pump(tk_top)
            embedded = False
            if _vtk_rendering_tk_dll_present():
                try:
                    _embed_pyvista_plotter_in_tk(embed_parent, pl)
                    embedded = True
                except Exception:
                    embedded = False
            if not embedded:
                _show_tk_vtk_fallback_panel(embed_parent)
                pl.show(interactive_update=True, auto_close=False, interactive=True)
                _ensure_pyvista_iren_initialized(pl)
                _start_tk_vtk_event_pump(tk_top, pl)
            if slice_tk is not None:
                if n_plan_layers > 0:
                    _wire_slice_band_controls(
                        slice_tk,
                        slice_cfg,
                        layer_energies_plan,
                        n_plan_layers,
                        _apply_nominal_energy_slice,
                    )
                else:
                    idle_slice_band_controls(slice_tk)
    else:
        pl.show()
    return pl
