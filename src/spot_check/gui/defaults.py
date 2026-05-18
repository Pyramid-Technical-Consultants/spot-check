"""Default GUI settings (factory state)."""

from __future__ import annotations

from spot_check import constants as sc_const

DEFAULT_GUI_STATE: dict[str, object] = {
    "dcm_path": "",
    "csv_path": "",
    "window_geometry": "1400x900",
    "layer_assign_mode": "gate_counter",
    "layer_gap_s": sc_const.TIME_LAYER_GAP_S_DEFAULT,
    "refill_same_spot_xy_tol_mm": sc_const.REFILL_SAME_SPOT_XY_TOLERANCE_MM,
    "viterbi_advance_penalty_mm2": sc_const.VITERBI_LAYER_ADVANCE_PENALTY_MM2_DEFAULT,
    "weight_measured_by_channel_sum": True,
    "spot_weight_mode": sc_const.SPOT_WEIGHT_MODE_DEFAULT,
    "aggregate_spots_by_gate": True,
    "aggregate_even_rows_after_odd": sc_const.AGGREGATE_EVEN_ROWS_AFTER_ODD_DEFAULT,
    "auto_align_detector_xy": True,
    "bounds_xy_tick_mm": sc_const.BOUNDS_XY_TICK_MM_DEFAULT,
    "plan_qa_coloring": True,
    "plan_qa_pass_mm": sc_const.PLAN_QA_PASS_MM_DEFAULT,
    "plan_qa_warn_mm": sc_const.PLAN_QA_WARN_MM_DEFAULT,
    "plan_qa_draw_error_lines": False,
    "plan_qa_hide_pass_spots": False,
    "scale_plan_spots_by_dicom_fwhm": False,
    "measured_spots_sigma_world_mm": False,
    "z_axis_proton_water_depth_mm": True,
    "view_projection_perspective": True,
    "slice_band_on": False,
    "slice_band_center_i": 0,
    "gui_state_schema_version": 1,
}
