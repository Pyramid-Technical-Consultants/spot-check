"""Persist GUI layout and control values."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from PySide6.QtWidgets import QMainWindow

from spot_check.constants import project_root
from spot_check.gui.defaults import DEFAULT_GUI_STATE

logger = logging.getLogger(__name__)


def gui_state_file() -> Path:
    return project_root() / ".spot_check_gui_state.json"


def legacy_gui_state_file() -> Path:
    return project_root() / ".dicom_run_gui_state.json"


def load_gui_state() -> dict[str, object]:
    data: dict[str, object] = dict(DEFAULT_GUI_STATE)
    state_file = gui_state_file()
    if not state_file.is_file() and legacy_gui_state_file().is_file():
        state_file = legacy_gui_state_file()
    try:
        if state_file.is_file():
            loaded = json.loads(state_file.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data.update(loaded)
    except json.JSONDecodeError as exc:
        logger.warning(
            "GUI state file %s is not valid JSON (%s); using defaults.",
            state_file,
            exc,
        )
    except (OSError, TypeError) as exc:
        logger.warning("Could not load GUI state from %s: %s", state_file, exc)
    return data


def save_gui_state(
    *,
    dcm_path: str,
    csv_path: str,
    window_geometry: str,
    layer_assign_mode: str,
    layer_gap_s: float,
    refill_same_spot_xy_tol_mm: float,
    viterbi_advance_penalty_mm2: float,
    weight_measured_by_channel_sum: bool,
    spot_weight_mode: str,
    aggregate_spots_by_gate: bool,
    aggregate_even_rows_after_odd: int,
    auto_align_detector_xy: bool,
    bounds_xy_tick_mm: float,
    plan_qa_coloring: bool,
    plan_qa_mode: str,
    plan_qa_pass_mm: float,
    plan_qa_warn_mm: float,
    plan_qa_pass_pp: float,
    plan_qa_warn_pp: float,
    plan_qa_draw_error_lines: bool,
    plan_qa_hide_pass_spots: bool,
    scale_plan_spots_by_dicom_fwhm: bool,
    measured_spots_sigma_world_mm: bool,
    z_axis_proton_water_depth_mm: bool,
    view_projection_perspective: bool,
    slice_band_on: bool,
    slice_band_center_i: int,
) -> None:
    path = gui_state_file()
    try:
        path.write_text(
            json.dumps(
                {
                    "dcm_path": dcm_path,
                    "csv_path": csv_path,
                    "window_geometry": window_geometry,
                    "layer_assign_mode": layer_assign_mode,
                    "layer_gap_s": layer_gap_s,
                    "refill_same_spot_xy_tol_mm": refill_same_spot_xy_tol_mm,
                    "viterbi_advance_penalty_mm2": viterbi_advance_penalty_mm2,
                    "weight_measured_by_channel_sum": weight_measured_by_channel_sum,
                    "spot_weight_mode": spot_weight_mode,
                    "aggregate_spots_by_gate": aggregate_spots_by_gate,
                    "aggregate_even_rows_after_odd": aggregate_even_rows_after_odd,
                    "auto_align_detector_xy": auto_align_detector_xy,
                    "bounds_xy_tick_mm": bounds_xy_tick_mm,
                    "plan_qa_coloring": plan_qa_coloring,
                    "plan_qa_mode": plan_qa_mode,
                    "plan_qa_pass_mm": plan_qa_pass_mm,
                    "plan_qa_warn_mm": plan_qa_warn_mm,
                    "plan_qa_pass_pp": plan_qa_pass_pp,
                    "plan_qa_warn_pp": plan_qa_warn_pp,
                    "plan_qa_draw_error_lines": plan_qa_draw_error_lines,
                    "plan_qa_hide_pass_spots": plan_qa_hide_pass_spots,
                    "scale_plan_spots_by_dicom_fwhm": scale_plan_spots_by_dicom_fwhm,
                    "measured_spots_sigma_world_mm": measured_spots_sigma_world_mm,
                    "z_axis_proton_water_depth_mm": z_axis_proton_water_depth_mm,
                    "view_projection_perspective": view_projection_perspective,
                    "slice_band_on": slice_band_on,
                    "slice_band_center_i": slice_band_center_i,
                    "gui_state_schema_version": int(DEFAULT_GUI_STATE["gui_state_schema_version"]),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("Could not write GUI state file %s: %s", path, exc)


def sanitize_geometry(raw: object, *, default: str = "1400x900") -> str:
    s = str(raw or "").strip()
    if not s:
        return default
    try:
        wh = s.split("+", 1)[0]
        parts = wh.lower().replace("x", " ").split()
        if len(parts) != 2:
            return default
        w, h = int(float(parts[0])), int(float(parts[1]))
        if w < 200 or h < 200 or w > 16_000 or h > 16_000:
            return default
    except (ValueError, TypeError):
        return default
    return s


def apply_saved_geometry(win: QMainWindow, raw: object) -> None:
    s = sanitize_geometry(raw, default="1400x900")
    try:
        tokens = s.replace("+", " ").split()
        wh = tokens[0].lower().split("x")
        w, h = int(float(wh[0])), int(float(wh[1]))
        win.resize(max(w, 400), max(h, 400))
        if len(tokens) >= 3:
            win.move(int(float(tokens[1])), int(float(tokens[2])))
    except (ValueError, IndexError, TypeError):
        win.resize(1400, 900)


def geom_from_win(win: QMainWindow) -> str:
    g = win.geometry()
    return f"{g.width()}x{g.height()}+{g.x()}+{g.y()}"
