"""Default GUI settings (factory state)."""

from __future__ import annotations

from spot_check import constants as sc_const

DEFAULT_GUI_STATE: dict[str, object] = {
    "dcm_path": "",
    "csv_path": "",
    "window_geometry": "1400x900",
    "window_maximized": False,
    "layer_assign_mode": "gate_counter",
    "weight_measured_by_channel_sum": True,
    "spot_weight_mode": sc_const.SPOT_WEIGHT_MODE_DEFAULT,
    "aggregate_spots_by_gate": True,
    "heal_partial_fit_axes": False,
    "coarse_flat_align": True,
    "fine_align_xy": True,
    "fine_align_rotation": True,
    "fine_align_scale": True,
    "filter_xy_fliers": False,
    "filter_xy_flier_sigma": sc_const.FILTER_XY_FLIER_SIGMA_DEFAULT,
    "plan_qa_coloring": True,
    "plan_qa_mode": "position",
    "plan_qa_pass_mm": sc_const.PLAN_QA_PASS_MM_DEFAULT,
    "plan_qa_warn_mm": sc_const.PLAN_QA_WARN_MM_DEFAULT,
    "plan_qa_pass_pp": sc_const.PLAN_QA_DOSE_PASS_PP_DEFAULT,
    "plan_qa_warn_pp": sc_const.PLAN_QA_DOSE_WARN_PP_DEFAULT,
    "plan_qa_draw_error_lines": False,
    "plan_qa_hide_pass_spots": False,
    "scale_plan_spots_by_dicom_fwhm": False,
    "measured_spots_sigma_world_mm": False,
    "z_axis_proton_water_depth_mm": True,
    "upstream_wet_shifter_mm": sc_const.UPSTREAM_WET_SHIFTER_MM_DEFAULT,
    "z_depth_metric": sc_const.Z_DEPTH_METRIC_DEFAULT,
    "view_projection_perspective": True,
    "slice_band_on": False,
    "slice_band_center_i": 0,
    "time_slice_on": False,
    "time_slice_start_ms": 0,
    "time_slice_speed": 1.0,
    "gui_state_schema_version": 1,
}
