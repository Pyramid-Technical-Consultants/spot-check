"""Plotter."""

from __future__ import annotations

import os

from spot_check.analysis._imports import *  # noqa: F403
from spot_check.analysis.measured import measured_spot_weight_caption
from spot_check.analysis.plan_qa import _plan_qa_error_line_polylines
from spot_check.analysis.pyvista_backend import pv, require_pyvista
from spot_check.analysis.spatial import nominal_layer_energies_mev
from spot_check.analysis.viz.data import (
    _energy_slice_mask,
    _nominal_layer_index_band_mev,
    _time_slice_mask,
    _timeline_range_ms,
    is_full_time_slice_window,
    prepare_comparison_3d_data,
)
from spot_check.analysis.viz.display_meshes import build_spot_display_meshes
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
    _wire_time_slice_controls,
    _wire_time_slice_controls_qt,
    disconnect_slice_band_controls_qt,
    disconnect_time_slice_controls_qt,
    idle_slice_band_controls,
    idle_slice_band_controls_qt,
    idle_time_slice_controls,
    idle_time_slice_controls_qt,
)
from spot_check.analysis.viz.glyphs import (
    _attach_spot_index,
    _disc_point_add_mesh_kwargs,
    _measured_spot_sigma_glyph_mesh,
    _plan_spot_cross_mesh,
    _plan_spot_fwhm_glyph_mesh,
    _plan_spot_point_mesh,
)
from spot_check.analysis.viz.scene_grid import PlanSceneGridController
from spot_check.analysis.viz.spot_pick import (
    SpotPickCallback,
    disconnect_spot_pick,
    update_spot_pick_plan_visibility,
    wire_spot_double_click_pick,
)
from spot_check.constants import TIME_SLICE_WINDOW_S_DEFAULT


def _set_actor_polydata(actor: Any | None, mesh: Any) -> None:
    """Update a PyVista actor mesh; no-op when the actor or mapper is missing."""
    if actor is None:
        return
    mapper = getattr(actor, "mapper", None)
    if mapper is None:
        return
    mapper.dataset = mesh


def _maybe_show_pyvista_plotter(pl: Any) -> None:
    """Interactive ``show()`` only when a display is expected; headless CI segfaults otherwise."""
    off = bool(getattr(pl, "off_screen", False))
    if not off:
        env = os.environ.get("PYVISTA_OFF_SCREEN", "").strip().lower()
        off = env in ("1", "true", "yes", "on")
    if off:
        try:
            pl.render()
        except Exception:
            pass
        return
    pl.show()


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
    spot_weight_mode: str = SPOT_WEIGHT_MODE_DEFAULT,
    detector_align_caption: str | None = None,
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
    time_slice_qt: dict[str, Any] | None = None,
    time_slice_tk: dict[str, Any] | None = None,
    time_slice_init: dict[str, bool | int] | None = None,
    z_axis_use_proton_water_depth_mm: bool = True,
    upstream_wet_shifter_mm: float = 0.0,
    z_depth_metric: str = "csda",
    view_projection_perspective: bool = True,
    cube_axes_sanity: bool | None = None,
    display_only: bool = False,
    plan_spots_no_data: np.ndarray | None = None,
    plan_spot_time_s: np.ndarray | None = None,
    time_slice_speed: float = 1.0,
    on_spot_picked: SpotPickCallback | None = None,
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

    if display_only and reuse_plotter is not None:
        _apply_display = getattr(reuse_plotter, "_spot_check_apply_display", None)
        if _apply_display is not None:
            _apply_display(
                planned_xyz,
                measured_abc,
                a_is_x=a_is_x,
                max_measured_draw=max_measured_draw,
                plan_fwhm_xy_mm=plan_fwhm_xy_mm,
                weight_measured_by_channel=weight_measured_by_channel,
                plan_qa_coloring=plan_qa_coloring,
                qa_mode=qa_mode,
                plan_qa_pass_mm=plan_qa_pass_mm,
                plan_qa_warn_mm=plan_qa_warn_mm,
                plan_qa_pass_pp=plan_qa_pass_pp,
                plan_qa_warn_pp=plan_qa_warn_pp,
                plan_mu=plan_mu,
                plan_qa_hide_pass_spots=plan_qa_hide_pass_spots,
                plan_qa_draw_error_lines=plan_qa_draw_error_lines,
                scale_plan_spots_by_dicom_fwhm=scale_plan_spots_by_dicom_fwhm,
                measured_spots_sigma_world_mm=measured_spots_sigma_world_mm,
                measured_sigma_glyph_scale=measured_sigma_glyph_scale,
                spot_weight_mode=spot_weight_mode,
                z_axis_use_proton_water_depth_mm=z_axis_use_proton_water_depth_mm,
                upstream_wet_shifter_mm=upstream_wet_shifter_mm,
                z_depth_metric=z_depth_metric,
                plan_spots_no_data=plan_spots_no_data,
                plan_spot_time_s=plan_spot_time_s,
            )
            return reuse_plotter

    qa_draw_lines = bool(plan_qa_draw_error_lines) and qa_mode == "position"

    prep = prepare_comparison_3d_data(
        planned_xyz,
        measured_abc,
        a_is_x=a_is_x,
        max_measured_draw=max_measured_draw,
        plan_fwhm_xy_mm=plan_fwhm_xy_mm,
    )

    z_display_cfg = z_display_config_for_plotter(
        use_water_depth=z_axis_use_proton_water_depth_mm,
        upstream_wet_shifter_mm=upstream_wet_shifter_mm,
        z_depth_metric=z_depth_metric,
    )

    _POINT_SIZE_3D = 9
    _display_state: dict[str, Any] = build_spot_display_meshes(
        planned_xyz,
        measured_abc,
        prep,
        z_display_cfg=z_display_cfg,
        a_is_x=a_is_x,
        weight_measured_by_channel=weight_measured_by_channel,
        plan_qa_coloring=plan_qa_coloring,
        qa_mode=qa_mode,
        plan_qa_pass_mm=plan_qa_pass_mm,
        plan_qa_warn_mm=plan_qa_warn_mm,
        plan_qa_pass_pp=plan_qa_pass_pp,
        plan_qa_warn_pp=plan_qa_warn_pp,
        plan_mu=plan_mu,
        plan_qa_hide_pass_spots=plan_qa_hide_pass_spots,
        plan_qa_draw_error_lines=plan_qa_draw_error_lines,
        scale_plan_spots_by_dicom_fwhm=scale_plan_spots_by_dicom_fwhm,
        measured_spots_sigma_world_mm=measured_spots_sigma_world_mm,
        measured_sigma_glyph_scale=measured_sigma_glyph_scale,
        spot_weight_mode=spot_weight_mode,
        plan_spots_no_data=plan_spots_no_data,
        plan_spot_time_s=plan_spot_time_s,
    )
    plan_pts = _display_state["plan_pts"]
    meas_pts = _display_state["meas_pts"]
    meas_view0 = _display_state["meas_view0"]
    meas_pts_final = _display_state["meas_pts_final"]
    meas_e_final = _display_state["meas_e_final"]
    meas_time_final = np.asarray(
        _display_state.get("meas_time_final", np.zeros(0, dtype=np.float64)),
        dtype=np.float64,
    ).reshape(-1)
    meas_sigma_final = _display_state["meas_sigma_final"]
    meas_rgba_final = _display_state["meas_rgba_final"]
    dist_qa_draw = _display_state["dist_qa_draw"]
    exp_xyz_qa_draw = _display_state["exp_xyz_qa_draw"]
    plan_qa_caption_extra = str(_display_state["plan_qa_caption_extra"])
    plan_cloud = _display_state["plan_cloud"]
    plan_glyphs = _display_state["plan_glyphs"]
    plan_rendered_fwhm_glyphs = bool(_display_state["plan_rendered_fwhm_glyphs"])
    plan_allow_fwhm_glyphs = bool(_display_state["plan_allow_fwhm_glyphs"])
    meas_sigma_glyphs = bool(_display_state["meas_sigma_glyphs"])
    sig_scale_eff = float(_display_state.get("sig_scale_eff", MEASURED_SIGMA_GLYPH_SCALE_DEFAULT))
    display_perf_note = str(_display_state.get("display_perf_note", ""))
    n_m = int(_display_state["n_m"])

    def _make_measured_view_mesh(
        pts: np.ndarray,
        sig_xy: np.ndarray,
        rgba: np.ndarray | None,
        spot_idx: np.ndarray | None = None,
    ) -> Any:
        if pts.shape[0] == 0:
            return pv.PolyData(np.zeros((0, 3), dtype=np.float64))
        if spot_idx is None:
            src = _display_state.get("meas_src_idx")
            if src is not None and int(np.asarray(src).shape[0]) == int(pts.shape[0]):
                spot_idx = np.asarray(src, dtype=np.int32).reshape(-1)
            else:
                spot_idx = np.arange(int(pts.shape[0]), dtype=np.int32)
        if _display_state["meas_sigma_glyphs"]:
            return _measured_spot_sigma_glyph_mesh(
                pts,
                sig_xy,
                sigma_scale=float(_display_state["sig_scale_eff"]),
                rgba=rgba,
                spot_indices=spot_idx,
            )
        m = pv.PolyData(pts)
        if rgba is not None:
            m["rgba"] = rgba
        _attach_spot_index(m, spot_idx)
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

    scene_grid = PlanSceneGridController(
        xlab=prep.xlab,
        ylab=prep.ylab,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        z_display_cfg=z_display_cfg,
    )

    # Sharp circular disc sprites (VTK sphere impostors — not square GL points or gaussian blur).
    point_r = _POINT_SIZE_3D
    point_kw = _disc_point_add_mesh_kwargs(point_size=point_r)

    layer_energies_plan = nominal_layer_energies_mev(planned_xyz)
    n_plan_layers = len(layer_energies_plan)
    plan_actor_uses_slice_visibility = n_plan_layers > 0

    pl = reuse_plotter
    saved_camera_position: Any = None
    if pl is not None:
        disconnect_spot_pick(pl)
        PlanSceneGridController.clear_on_plotter(pl)
        if reuse_camera:
            try:
                saved_camera_position = pl.camera_position
            except Exception:
                saved_camera_position = None
        pl.clear()
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
    plan_missing_actor: Any | None = None
    plan_actor_uses_fwhm_glyphs = False
    plan_has_data_mask = np.asarray(
        _display_state.get("plan_has_data_mask", np.ones(plan_pts.shape[0], dtype=bool)),
        dtype=bool,
    )
    _plan_vis_ref: dict[str, np.ndarray | None] = {
        "mask": (
            np.ones(int(plan_pts.shape[0]), dtype=bool)
            if int(plan_pts.shape[0]) > 0
            else None
        )
    }

    def _rewire_spot_pick() -> None:
        wire_spot_double_click_pick(
            pl,
            plan_actor=plan_actor,
            meas_actor=meas_actor,
            plan_missing_actor=plan_missing_actor,
            plan_visible_mask=_plan_vis_ref.get("mask"),
            plan_has_data_mask=plan_has_data_mask if int(plan_pts.shape[0]) else None,
            on_picked=on_spot_picked,
        )
    if plan_pts.shape[0] > 0:
        circle_vis0 = plan_has_data_mask
        if plan_rendered_fwhm_glyphs and plan_glyphs is not None:
            if plan_actor_uses_slice_visibility and prep.plan_fwhm_xy_mm is not None:
                plan_glyphs = _plan_spot_fwhm_glyph_mesh(
                    plan_pts,
                    prep.plan_fwhm_xy_mm,
                    visible_mask=circle_vis0,
                )
                plan_actor = pl.add_mesh(
                    plan_glyphs,
                    scalars="rgba",
                    rgba=True,
                    pickable=True,
                    smooth_shading=False,
                    lighting=False,
                )
            else:
                plan_actor = pl.add_mesh(
                    plan_glyphs,
                    color=_PLAN_COLOR_3D,
                    opacity=0.45,
                    pickable=True,
                    smooth_shading=False,
                    lighting=False,
                )
            plan_actor_uses_fwhm_glyphs = True
        elif plan_actor_uses_slice_visibility:
            plan_actor = pl.add_mesh(
                _plan_spot_point_mesh(plan_pts, visible_mask=circle_vis0),
                scalars="rgba",
                rgba=True,
                pickable=True,
                **point_kw,
            )
        else:
            if bool(np.all(circle_vis0)):
                plan_actor = pl.add_mesh(
                    plan_cloud,
                    color=_PLAN_COLOR_3D,
                    opacity=0.45,
                    pickable=True,
                    **point_kw,
                )
            elif bool(np.any(circle_vis0)):
                plan_actor = pl.add_mesh(
                    _plan_spot_point_mesh(plan_pts, visible_mask=circle_vis0),
                    scalars="rgba",
                    rgba=True,
                    pickable=True,
                    **point_kw,
                )
        cross0 = _display_state.get("plan_cross_mesh")
        if cross0 is not None:
            plan_missing_actor = pl.add_mesh(
                cross0,
                scalars="rgba",
                rgba=True,
                line_width=2,
                pickable=True,
                lighting=False,
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
            z_display_cfg=z_display_cfg,
            plan_e_lo_mev=float(prep.e_lo),
            plan_e_hi_mev=float(prep.e_hi),
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
    _plan_time_for_rng = np.asarray(
        _display_state.get("plan_time_final", np.zeros(0, dtype=np.float64)),
        dtype=np.float64,
    ).reshape(-1)
    _timeline_parts: list[np.ndarray] = []
    if meas_time_final.size:
        _timeline_parts.append(meas_time_final.reshape(-1))
    if _plan_time_for_rng.size:
        _timeline_parts.append(_plan_time_for_rng.reshape(-1))
    _timeline_time_s = (
        np.concatenate(_timeline_parts) if _timeline_parts else np.zeros(0, dtype=np.float64)
    )
    _time_start_default_ms = 0
    time_slice_cfg: dict[str, bool | int | float] = {
        "slice_on": False,
        "start_ms": _time_start_default_ms,
        "window_s": float(TIME_SLICE_WINDOW_S_DEFAULT),
    }
    if time_slice_init:
        if "slice_on" in time_slice_init:
            time_slice_cfg["slice_on"] = bool(time_slice_init["slice_on"])
        if "start_ms" in time_slice_init:
            time_slice_cfg["start_ms"] = int(time_slice_init["start_ms"])
        if "window_s" in time_slice_init:
            time_slice_cfg["window_s"] = float(time_slice_init["window_s"])

    def _cfg_window_s() -> float:
        return float(time_slice_cfg.get("window_s", TIME_SLICE_WINDOW_S_DEFAULT))

    _time_rng = _timeline_range_ms(
        meas_time_final,
        _plan_time_for_rng if _plan_time_for_rng.size else None,
        window_s=_cfg_window_s(),
    )
    if _time_rng is not None:
        _time_start_default_ms = int(_time_rng[0])
        sm0 = int(time_slice_cfg["start_ms"])
        time_slice_cfg["start_ms"] = int(np.clip(sm0, _time_rng[0], _time_rng[1]))
    else:
        time_slice_cfg["start_ms"] = int(time_slice_cfg.get("start_ms", 0))

    def _timeline_bounds_s() -> tuple[float, float]:
        if _time_rng is None:
            return 0.0, 0.0
        return float(_time_rng[2]), float(_time_rng[3])
    # Filled when embedding in Qt so slice callback can repaint the QVTK widget after updates.
    _qt_vtk_embed: dict[str, Any] = {"widget": None}

    def _plan_energies_mev_for_cube_axis() -> np.ndarray:
        return np.asarray(plan_e_mev, dtype=np.float64).reshape(-1)

    def _plan_scene_z_for_cube_axis() -> np.ndarray:
        if plan_pts.shape[0]:
            return np.asarray(plan_pts[:, 2], dtype=np.float64).reshape(-1)
        return np.zeros(0, dtype=np.float64)

    def _reset_camera_full_plan_bounds() -> None:
        sb = scene_grid.camera_bounds()
        try:
            if sb is not None:
                pl.reset_camera(bounds=sb)
            else:
                pl.reset_camera()
            pl.camera.zoom(1.05)
        except Exception:
            pass
    if slice_qt is not None:
        disconnect_slice_band_controls_qt(slice_qt)
    if time_slice_qt is not None:
        disconnect_time_slice_controls_qt(time_slice_qt)

    def _empty_poly() -> Any:
        return pv.PolyData(np.zeros((0, 3), dtype=np.float64))

    def _slice_lo_hi_mev() -> tuple[float, float]:
        if not bool(slice_cfg["slice_on"]) or n_plan_layers == 0:
            return e_rng_lo, e_rng_hi
        return _nominal_layer_index_band_mev(
            layer_energies_plan, int(slice_cfg["center_i"]), half_width=2
        )

    def _refresh_plan_actor(visible_mask: np.ndarray) -> None:
        """Keep the full plan mesh; hide out-of-band spots with alpha=0 for stable bounds."""
        if plan_pts.shape[0] <= 0:
            return
        vis = np.asarray(visible_mask, dtype=bool).reshape(-1)
        if int(vis.shape[0]) != int(plan_pts.shape[0]):
            raise ValueError("plan visibility mask length must match plan spot count")
        has_data = np.asarray(
            _display_state.get(
                "plan_has_data_mask",
                np.ones(plan_pts.shape[0], dtype=bool),
            ),
            dtype=bool,
        ).reshape(-1)
        circle_vis = vis & has_data
        cross_vis = vis & ~has_data
        if plan_actor is not None:
            if plan_actor_uses_slice_visibility:
                if plan_actor_uses_fwhm_glyphs and prep.plan_fwhm_xy_mm is not None:
                    mesh = _plan_spot_fwhm_glyph_mesh(
                        plan_pts,
                        prep.plan_fwhm_xy_mm,
                        visible_mask=circle_vis,
                    )
                else:
                    mesh = _plan_spot_point_mesh(plan_pts, visible_mask=circle_vis)
                _set_actor_polydata(plan_actor, mesh)
            elif plan_actor_uses_fwhm_glyphs and prep.plan_fwhm_xy_mm is not None:
                _set_actor_polydata(
                    plan_actor,
                    _plan_spot_fwhm_glyph_mesh(plan_pts, prep.plan_fwhm_xy_mm),
                )
            elif bool(np.all(circle_vis)):
                _set_actor_polydata(plan_actor, plan_cloud)
            else:
                _set_actor_polydata(
                    plan_actor,
                    _plan_spot_point_mesh(plan_pts, visible_mask=circle_vis),
                )
        no_data = _display_state.get("plan_spots_no_data")
        if plan_missing_actor is not None and no_data is not None:
            nd = np.asarray(no_data, dtype=bool).reshape(-1)
            if bool(np.any(nd)):
                cross = _plan_spot_cross_mesh(
                    plan_pts,
                    spot_mask=nd,
                    visible_mask=cross_vis if plan_actor_uses_slice_visibility else None,
                )
                _set_actor_polydata(plan_missing_actor, cross)

    def _update_slice_meshes() -> np.ndarray:
        """Show/hide spot actors for energy and/or time slices (does not touch cube axes)."""
        layer_slice_active = bool(slice_cfg["slice_on"]) and n_plan_layers > 0
        time_slice_active = bool(time_slice_cfg["slice_on"]) and _time_rng is not None
        n_plan = int(plan_pts.shape[0])
        all_plan_vis = np.ones(n_plan, dtype=bool) if n_plan else np.zeros(0, dtype=bool)
        pm_full = all_plan_vis
        if layer_slice_active:
            lo_m, hi_m = _slice_lo_hi_mev()
            pm_full = (
                _energy_slice_mask(plan_e_mev, lo_m, hi_m)
                if plan_e_mev.size > 0
                else np.zeros(0, dtype=bool)
            )
        if time_slice_active:
            plan_t = np.asarray(
                _display_state.get("plan_time_final", np.zeros(0, dtype=np.float64)),
                dtype=np.float64,
            ).reshape(-1)
            if plan_t.size == n_plan:
                start_s = float(int(time_slice_cfg["start_ms"])) / 1000.0
                t_lo, t_hi = _timeline_bounds_s()
                win = _cfg_window_s()
                t_mask = _time_slice_mask(
                    plan_t, start_s, window_s=win, t_min=t_lo, t_max=t_hi
                )
                pm_full = pm_full & t_mask if pm_full.size else t_mask
        _refresh_plan_actor(pm_full)

        n_meas_draw = int(_display_state["meas_pts_final"].shape[0])
        if n_meas_draw <= 0:
            mm_full = np.zeros(0, dtype=bool)
            if meas_actor is not None:
                _set_actor_polydata(meas_actor, _empty_poly())
        elif not layer_slice_active and not time_slice_active:
            mm_full = np.ones(n_meas_draw, dtype=bool)
            if meas_actor is not None:
                _set_actor_polydata(meas_actor, _display_state["meas_view0"])
        else:
            mm_full = np.ones(n_meas_draw, dtype=bool)
            if layer_slice_active:
                lo_m, hi_m = _slice_lo_hi_mev()
                mm_full &= (
                    _energy_slice_mask(_display_state["meas_e_final"], lo_m, hi_m)
                    if _display_state["meas_e_final"].size > 0
                    else np.zeros(0, dtype=bool)
                )
            if time_slice_active:
                start_s = float(int(time_slice_cfg["start_ms"])) / 1000.0
                t_lo, t_hi = _timeline_bounds_s()
                win = _cfg_window_s()
                mm_full &= _time_slice_mask(
                    _display_state["meas_time_final"],
                    start_s,
                    window_s=win,
                    t_min=t_lo,
                    t_max=t_hi,
                )

            if meas_actor is not None:
                if not np.any(mm_full):
                    _set_actor_polydata(meas_actor, _empty_poly())
                else:
                    sub_pts = _display_state["meas_pts_final"][mm_full]
                    sub_sig = _display_state["meas_sigma_final"][mm_full]
                    sub_rgba = (
                        _display_state["meas_rgba_final"][mm_full]
                        if _display_state["meas_rgba_final"] is not None
                        else None
                    )
                    sub_idx = np.asarray(
                        _display_state.get("meas_src_idx", np.arange(n_meas_draw, dtype=np.int64)),
                        dtype=np.int32,
                    ).reshape(-1)[mm_full]
                    _set_actor_polydata(
                        meas_actor,
                        _make_measured_view_mesh(sub_pts, sub_sig, sub_rgba, sub_idx),
                    )

        update_spot_pick_plan_visibility(pl, pm_full)
        _plan_vis_ref["mask"] = pm_full.copy() if pm_full.size else pm_full
        return mm_full

    def _apply_nominal_energy_slice() -> None:
        nonlocal line_warn_actor, line_fail_actor
        if not scene_grid.ready:
            return
        mm_full = _update_slice_meshes()
        if line_warn_actor is not None:
            pl.remove_actor(line_warn_actor)
            line_warn_actor = None
        if line_fail_actor is not None:
            pl.remove_actor(line_fail_actor)
            line_fail_actor = None
        if (
            bool(_display_state["plan_qa_coloring"])
            and bool(_display_state["qa_draw_lines"])
            and _display_state["dist_qa_draw"] is not None
            and _display_state["exp_xyz_qa_draw"] is not None
            and np.any(mm_full)
        ):
            lw, lf = _plan_qa_error_line_polylines(
                _display_state["meas_pts_final"][mm_full],
                _display_state["exp_xyz_qa_draw"][mm_full],
                _display_state["dist_qa_draw"][mm_full],
                pass_mm=plan_qa_pass_mm,
                warn_mm=plan_qa_warn_mm,
                z_display_cfg=z_display_cfg,
                plan_e_lo_mev=float(prep.e_lo),
                plan_e_hi_mev=float(prep.e_hi),
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
        pl.render()

    def _refresh_slice_view() -> None:
        _apply_nominal_energy_slice()

    if n_plan_layers > 0 and embed_parent is None and embed_qt is None:

        def _on_center_layer(value: float) -> None:
            ci = int(round(float(value)))
            slice_cfg["center_i"] = int(np.clip(ci, 0, n_plan_layers - 1))
            _refresh_slice_view()

        def _on_slice_mode_checkbox(checked: bool) -> None:
            slice_cfg["slice_on"] = bool(checked)
            _refresh_slice_view()

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
            value=bool(slice_cfg["slice_on"]),
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

    if _time_rng is not None and embed_parent is None and embed_qt is None:
        _t_start_min_ms, _t_start_max_ms, _, _ = _time_rng

        def _on_time_window_start(value: float) -> None:
            sm = int(round(float(value)))
            time_slice_cfg["start_ms"] = int(np.clip(sm, _t_start_min_ms, _t_start_max_ms))
            _refresh_slice_view()

        def _on_time_slice_checkbox(checked: bool) -> None:
            time_slice_cfg["slice_on"] = bool(checked)
            _refresh_slice_view()

        _t_slider_rng = (float(_t_start_min_ms), float(_t_start_max_ms))
        _t_slider_val = float(int(time_slice_cfg["start_ms"]))
        try:
            pl.add_slider_widget(
                _on_time_window_start,
                rng=_t_slider_rng,
                value=_t_slider_val,
                title="time window start (ms, 1 s wide)",
                pointa=(0.02, 0.86),
                pointb=(0.40, 0.86),
                fmt="%.0f",
                style="modern",
                interaction_event="always",
            )
        except (TypeError, ValueError):
            try:
                pl.add_slider_widget(
                    _on_time_window_start,
                    rng=_t_slider_rng,
                    value=_t_slider_val,
                    title="time window start (ms, 1 s wide)",
                    pointa=(0.02, 0.86),
                    pointb=(0.40, 0.86),
                    fmt="%.0f",
                    interaction_event="always",
                )
            except TypeError:
                pl.add_slider_widget(
                    _on_time_window_start,
                    rng=_t_slider_rng,
                    value=_t_slider_val,
                    title="time window start (ms, 1 s wide)",
                    pointa=(0.02, 0.86),
                    pointb=(0.40, 0.86),
                    fmt="%.0f",
                )
        pl.add_checkbox_button_widget(
            _on_time_slice_checkbox,
            value=bool(time_slice_cfg["slice_on"]),
            position=(14, 78),
            size=22,
            border_size=4,
        )
        pl.add_text(
            "1 s time slice",
            position=(42, 81),
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
        _gap = TIME_LAYER_GAP_S_DEFAULT if layer_gap_s is None else layer_gap_s
        _xytol = (
            REFILL_SAME_SPOT_XY_TOLERANCE_MM
            if refill_same_spot_xy_tol_mm is None
            else refill_same_spot_xy_tol_mm
        )
        caption = (
            f"n_plan={plan_pts.shape[0]}, n_meas={n_m}, MeV [{prep.e_lo:.1f}, {prep.e_hi:.1f}]. "
            f"Layers: auto deadtime episodes; acquisition time, layer 0 = highest energy "
            f"(Δt≥{_gap:g} s segment gaps)."
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
    if aggregate_spots:
        _sw = measured_spot_weight_caption(spot_weight_mode)
        caption += (
            f" Measured spots aggregated: {_sw}-weighted mean XY + σ per assigned plan spot."
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
        if bool(slice_cfg["slice_on"]):
            _slo, _shi = _nominal_layer_index_band_mev(
                layer_energies_plan, int(slice_cfg["center_i"]), half_width=2
            )
            if z_display_cfg.use_water_depth_mm:
                caption += (
                    " Z axis ticks: full plan water-depth range (PSTAR);"
                    f" 5-layer band ≈ {_slo:.1f}–{_shi:.1f} MeV"
                    " (plan spots outside band hidden; full stack kept for stable axes)."
                )
            else:
                caption += (
                    f" Z axis ticks: full plan {e_rng_lo:.1f}–{e_rng_hi:.1f} MeV"
                    f" (not the slice band); 5-layer band ≈ {_slo:.1f}–{_shi:.1f} MeV"
                    " (plan spots outside band hidden; full stack kept for stable axes)."
                    " High energy (deep) toward axis origin."
                )
        elif not z_display_cfg.use_water_depth_mm:
            caption += " Z: high energy (deep) toward axis origin; ticks are nominal MeV."
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
    if _time_rng is not None:
        if bool(time_slice_cfg["slice_on"]):
            _t0 = float(int(time_slice_cfg["start_ms"])) / 1000.0
            _win = _cfg_window_s()
            if is_full_time_slice_window(_win):
                _t_lo, _ = _timeline_bounds_s()
                caption += (
                    f" Time window: all since start [{_t_lo:.3f}, {_t0:.3f}] s"
                    " on plan and measured spots."
                )
            else:
                _t1 = _t0 + _win
                caption += (
                    f" Time window: [{_t0:.3f}, {_t1:.3f}] s on plan and measured spots"
                    " (combines with layer band when both are on)."
                )
        if embed_qt is not None:
            caption += (
                " Timeline bar (bottom): toggle Time, pick window width, play/scrub."
            )
        elif embed_parent is None and embed_qt is None:
            caption += (
                " Upper left (below layer band): 1 s time window on measured acquisition order."
            )
    if display_perf_note:
        caption += display_perf_note
    if z_display_cfg.use_water_depth_mm:
        metric_lbl = str(z_display_cfg.z_depth_metric).upper()
        caption += (
            f" Z axis: {metric_lbl} depth in water (mm, PSTAR CSDA base for R90/R80); "
            "tick step follows XY bounds (mm)."
        )
        if z_display_cfg.upstream_wet_mm > 0.0:
            caption += f" Upstream WET shifter −{z_display_cfg.upstream_wet_mm:g} mm from depth."
    pl.add_text(title, position="upper_left", font_size=11, color="#f0f6fc", shadow=True)
    pl.add_text(caption, position="lower_left", font_size=9, color="#8b949e")

    # Scene grid last so it sits above spot meshes; camera uses the same scene bounds.
    _sanity_env = os.environ.get("SPOT_CHECK_CUBE_AXES_SANITY", "").strip().lower()
    use_cube_sanity = (
        bool(cube_axes_sanity)
        if cube_axes_sanity is not None
        else _sanity_env in ("1", "true", "yes", "on")
    )
    scene_grid.sanity = use_cube_sanity
    scene_grid.ready = True
    _apply_nominal_energy_slice()
    scene_grid.show(pl, _plan_scene_z_for_cube_axis(), _plan_energies_mev_for_cube_axis())
    pl._spot_check_scene_grid = scene_grid  # noqa: SLF001
    sb = scene_grid.camera_bounds()
    if sb is not None:
        try:
            pl.reset_camera(bounds=sb)
            pl.camera.zoom(1.05)
        except Exception:
            pass

    if reuse_plotter is None or not reuse_camera:
        _reset_camera_full_plan_bounds()
    elif saved_camera_position is not None:
        try:
            pl.camera_position = saved_camera_position
        except Exception:
            _reset_camera_full_plan_bounds()
    else:
        _reset_camera_full_plan_bounds()

    try:
        pl.camera.parallel_projection = not bool(view_projection_perspective)
    except Exception:
        pass

    if embed_qt is not None:
        try:
            if reembed_qt:
                w = _embed_pyvista_plotter_in_qt(embed_qt, pl)
                _qt_vtk_embed["widget"] = w
                if slice_qt is not None:
                    slice_qt["_qt_vtk_widget"] = w
                if time_slice_qt is not None:
                    time_slice_qt["_qt_vtk_widget"] = w
            else:
                pl.render()
                qw = slice_qt.get("_qt_vtk_widget") if slice_qt is not None else None
                if qw is None and time_slice_qt is not None:
                    qw = time_slice_qt.get("_qt_vtk_widget")
                if qw is None:
                    qw = _qt_vtk_embed.get("widget")
                if qw is not None:
                    try:
                        qw.update()
                    except Exception:
                        pass
            pl.render()
        except Exception:
            idle_slice_band_controls_qt(slice_qt)
            idle_time_slice_controls_qt(time_slice_qt)
            raise
        if slice_qt is not None:
            if n_plan_layers > 0:
                _wire_slice_band_controls_qt(
                    slice_qt,
                    slice_cfg,
                    layer_energies_plan,
                    n_plan_layers,
                    _refresh_slice_view,
                )
            else:
                idle_slice_band_controls_qt(slice_qt)
        if time_slice_qt is not None:
            if _time_rng is not None:
                _wire_time_slice_controls_qt(
                    time_slice_qt,
                    time_slice_cfg,
                    _timeline_time_s,
                    window_s=_cfg_window_s(),
                    apply_slice=_refresh_slice_view,
                    saved_speed=float(time_slice_speed),
                )
            else:
                idle_time_slice_controls_qt(time_slice_qt)
    elif embed_parent is not None:
        if tk is None:
            _maybe_show_pyvista_plotter(pl)
            idle_slice_band_controls(slice_tk)
            idle_time_slice_controls(time_slice_tk)
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
                        _refresh_slice_view,
                    )
                else:
                    idle_slice_band_controls(slice_tk)
            if time_slice_tk is not None:
                if _time_rng is not None:
                    _wire_time_slice_controls(
                        time_slice_tk,
                        time_slice_cfg,
                        _timeline_time_s,
                        window_s=_cfg_window_s(),
                        apply_slice=_refresh_slice_view,
                    )
                else:
                    idle_time_slice_controls(time_slice_tk)
    else:
        _maybe_show_pyvista_plotter(pl)

    def _add_plan_actor_from_display() -> None:
        nonlocal plan_actor, plan_missing_actor, plan_actor_uses_fwhm_glyphs
        if plan_pts.shape[0] <= 0:
            plan_actor = None
            plan_missing_actor = None
            plan_actor_uses_fwhm_glyphs = False
            return
        has_data = np.asarray(
            _display_state.get(
                "plan_has_data_mask",
                np.ones(plan_pts.shape[0], dtype=bool),
            ),
            dtype=bool,
        )
        circle_vis0 = has_data
        prfg = bool(_display_state["plan_rendered_fwhm_glyphs"])
        pg = _display_state["plan_glyphs"]
        if prfg and pg is not None:
            if plan_actor_uses_slice_visibility and prep.plan_fwhm_xy_mm is not None:
                pg = _plan_spot_fwhm_glyph_mesh(
                    plan_pts,
                    prep.plan_fwhm_xy_mm,
                    visible_mask=circle_vis0,
                )
                plan_actor = pl.add_mesh(
                    pg,
                    scalars="rgba",
                    rgba=True,
                    pickable=True,
                    smooth_shading=False,
                    lighting=False,
                )
            else:
                plan_actor = pl.add_mesh(
                    pg,
                    color=_PLAN_COLOR_3D,
                    opacity=0.45,
                    pickable=True,
                    smooth_shading=False,
                    lighting=False,
                )
            plan_actor_uses_fwhm_glyphs = True
        elif plan_actor_uses_slice_visibility:
            plan_actor = pl.add_mesh(
                _plan_spot_point_mesh(plan_pts, visible_mask=circle_vis0),
                scalars="rgba",
                rgba=True,
                pickable=True,
                **point_kw,
            )
            plan_actor_uses_fwhm_glyphs = False
        elif bool(np.all(circle_vis0)):
            plan_actor = pl.add_mesh(
                _display_state["plan_cloud"],
                color=_PLAN_COLOR_3D,
                opacity=0.45,
                pickable=True,
                **point_kw,
            )
            plan_actor_uses_fwhm_glyphs = False
        elif bool(np.any(circle_vis0)):
            plan_actor = pl.add_mesh(
                _plan_spot_point_mesh(plan_pts, visible_mask=circle_vis0),
                scalars="rgba",
                rgba=True,
                pickable=True,
                **point_kw,
            )
            plan_actor_uses_fwhm_glyphs = False
        else:
            plan_actor = None
            plan_actor_uses_fwhm_glyphs = False
        cross0 = _display_state.get("plan_cross_mesh")
        if cross0 is not None:
            plan_missing_actor = pl.add_mesh(
                cross0,
                scalars="rgba",
                rgba=True,
                line_width=2,
                pickable=True,
                lighting=False,
            )
        else:
            plan_missing_actor = None

    def _add_meas_actor_from_display() -> None:
        nonlocal meas_actor
        view0 = _display_state["meas_view0"]
        if int(_display_state["n_m"]) <= 0:
            meas_actor = None
            return
        rgba_final = _display_state["meas_rgba_final"]
        if _display_state["meas_sigma_glyphs"]:
            has_rgba = rgba_final is not None and "rgba" in view0.point_data
            if has_rgba:
                meas_actor = pl.add_mesh(
                    view0,
                    scalars="rgba",
                    rgba=True,
                    smooth_shading=False,
                    lighting=False,
                    opacity=1.0,
                    pickable=True,
                )
            else:
                meas_actor = pl.add_mesh(
                    view0,
                    color=_MEASURED_COLOR_3D,
                    smooth_shading=False,
                    lighting=False,
                    opacity=1.0,
                    pickable=True,
                )
        elif rgba_final is not None:
            meas_actor = pl.add_mesh(
                view0,
                scalars="rgba",
                rgba=True,
                opacity=1.0,
                pickable=True,
                **point_kw,
            )
        else:
            meas_actor = pl.add_mesh(
                view0,
                color=_MEASURED_COLOR_3D,
                opacity=1.0,
                pickable=True,
                **point_kw,
            )

    def _sync_plan_actor_for_display() -> None:
        nonlocal plan_actor, plan_missing_actor, plan_actor_uses_fwhm_glyphs
        want_fwhm = bool(_display_state["plan_rendered_fwhm_glyphs"])
        want_cross = _display_state.get("plan_cross_mesh") is not None
        if (
            plan_actor is not None
            and plan_actor_uses_fwhm_glyphs == want_fwhm
            and (plan_missing_actor is not None) == want_cross
        ):
            return
        if plan_actor is not None:
            pl.remove_actor(plan_actor)
            plan_actor = None
        if plan_missing_actor is not None:
            pl.remove_actor(plan_missing_actor)
            plan_missing_actor = None
        _add_plan_actor_from_display()

    def _sync_meas_actor_for_display() -> None:
        nonlocal meas_actor
        mode = (
            bool(_display_state["meas_sigma_glyphs"]),
            _display_state["meas_rgba_final"] is not None,
        )
        if meas_actor is not None and _display_state.get("_meas_actor_mode") == mode:
            return
        if meas_actor is not None:
            pl.remove_actor(meas_actor)
            meas_actor = None
        _add_meas_actor_from_display()
        _display_state["_meas_actor_mode"] = mode

    def _spot_check_apply_display(
        planned_xyz_in: list[tuple[float, float, float]],
        measured_abc_in: list[tuple[float, ...]],
        *,
        a_is_x: bool,
        max_measured_draw: int | None,
        plan_fwhm_xy_mm: np.ndarray | None,
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
        z_axis_use_proton_water_depth_mm: bool,
        upstream_wet_shifter_mm: float,
        z_depth_metric: str,
        plan_spots_no_data: np.ndarray | None = None,
        plan_spot_time_s: np.ndarray | None = None,
    ) -> None:
        nonlocal prep, plan_pts, plan_qa_caption_extra, display_perf_note
        nonlocal plan_rendered_fwhm_glyphs, plan_glyphs, plan_cloud, plan_allow_fwhm_glyphs
        nonlocal meas_sigma_glyphs, n_m, meas_view0, meas_pts_final, meas_e_final
        nonlocal meas_sigma_final, meas_rgba_final, dist_qa_draw, exp_xyz_qa_draw
        prep = prepare_comparison_3d_data(
            planned_xyz_in,
            measured_abc_in,
            a_is_x=a_is_x,
            max_measured_draw=max_measured_draw,
            plan_fwhm_xy_mm=plan_fwhm_xy_mm,
        )
        z_display_cfg_l = z_display_config_for_plotter(
            use_water_depth=z_axis_use_proton_water_depth_mm,
            upstream_wet_shifter_mm=upstream_wet_shifter_mm,
            z_depth_metric=z_depth_metric,
        )
        bundle = build_spot_display_meshes(
            planned_xyz_in,
            measured_abc_in,
            prep,
            z_display_cfg=z_display_cfg_l,
            a_is_x=a_is_x,
            weight_measured_by_channel=weight_measured_by_channel,
            plan_qa_coloring=plan_qa_coloring,
            qa_mode=qa_mode,
            plan_qa_pass_mm=plan_qa_pass_mm,
            plan_qa_warn_mm=plan_qa_warn_mm,
            plan_qa_pass_pp=plan_qa_pass_pp,
            plan_qa_warn_pp=plan_qa_warn_pp,
            plan_mu=plan_mu,
            plan_qa_hide_pass_spots=plan_qa_hide_pass_spots,
            plan_qa_draw_error_lines=plan_qa_draw_error_lines,
            scale_plan_spots_by_dicom_fwhm=scale_plan_spots_by_dicom_fwhm,
            measured_spots_sigma_world_mm=measured_spots_sigma_world_mm,
            measured_sigma_glyph_scale=measured_sigma_glyph_scale,
            spot_weight_mode=spot_weight_mode,
            plan_spots_no_data=plan_spots_no_data,
            plan_spot_time_s=plan_spot_time_s,
        )
        _display_state.update(bundle)
        plan_pts = _display_state["plan_pts"]
        meas_view0 = _display_state["meas_view0"]
        meas_pts_final = _display_state["meas_pts_final"]
        meas_e_final = _display_state["meas_e_final"]
        meas_sigma_final = _display_state["meas_sigma_final"]
        meas_rgba_final = _display_state["meas_rgba_final"]
        dist_qa_draw = _display_state["dist_qa_draw"]
        exp_xyz_qa_draw = _display_state["exp_xyz_qa_draw"]
        plan_qa_caption_extra = str(_display_state["plan_qa_caption_extra"])
        plan_cloud = _display_state["plan_cloud"]
        plan_glyphs = _display_state["plan_glyphs"]
        plan_rendered_fwhm_glyphs = bool(_display_state["plan_rendered_fwhm_glyphs"])
        plan_allow_fwhm_glyphs = bool(_display_state["plan_allow_fwhm_glyphs"])
        meas_sigma_glyphs = bool(_display_state["meas_sigma_glyphs"])
        display_perf_note = str(_display_state.get("display_perf_note", ""))
        n_m = int(_display_state["n_m"])
        _sync_plan_actor_for_display()
        _sync_meas_actor_for_display()
        _apply_nominal_energy_slice()
        _rewire_spot_pick()

    pl._spot_check_apply_display = _spot_check_apply_display
    pl._spot_check_display_state = _display_state

    _rewire_spot_pick()

    return pl


def refresh_comparison_3d_display(
    plotter: Any,
    planned_xyz: list[tuple[float, float, float]],
    measured_abc: list[tuple[float, ...]],
    *,
    title: str = "",
    **kwargs: Any,
) -> Any:
    """Fast in-place spot color/mesh refresh (no pipeline reload or camera reset)."""
    return show_comparison_3d_pyvista(
        planned_xyz,
        measured_abc,
        title=title,
        display_only=True,
        reuse_plotter=plotter,
        reuse_camera=True,
        reembed_qt=False,
        **kwargs,
    )
