"""SpotCheck GUI controller — builds the main window and 3D refresh pipeline."""

from __future__ import annotations

import importlib.util
import logging
import math
import sys
from pathlib import Path

if importlib.util.find_spec("pydicom") is None:  # pragma: no cover
    raise ImportError("RT Ion GUI requires pydicom. Install with: pip install pydicom")

try:
    from PySide6.QtCore import QEvent, QObject, Qt, QThreadPool, QTimer
    from PySide6.QtWidgets import (
        QApplication,
        QButtonGroup,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QPushButton,
        QRadioButton,
        QScrollArea,
        QSizePolicy,
        QSlider,
        QSplitter,
        QVBoxLayout,
        QWidget,
    )
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "RT Ion GUI requires PySide6 for single-window 3D. Install with: pip install PySide6"
    ) from e


from spot_check import analysis
from spot_check import constants as sc_const
from spot_check._version import __version__
from spot_check.analysis.csv_io import acquisition_csv_stem
from spot_check.constants import project_root
from spot_check.export import build_combined_export_rows, write_combined_export_csv
from spot_check.gui.layer_assign import (
    normalize_layer_assign_mode,
    resolve_layer_assign_mode,
)
from spot_check.gui.load_overlay import LoadOverlayPanel
from spot_check.gui.panels.status_format import (
    append_auto_meas_lines,
    format_auto_tuning_label,
    format_qa_counts_label,
)
from spot_check.gui.parsers import (
    filter_xy_flier_sigma_input_in_progress,
    normalize_z_depth_metric,
    parse_filter_xy_flier_sigma,
    parse_plan_qa_thresholds,
    parse_upstream_wet_shifter_mm,
    plan_qa_thresholds_input_in_progress,
    spot_weight_mode_from_saved,
)
from spot_check.gui.pipeline import (
    GuiRefreshContext,
    PipelineLoadOK,
    file_mtime,
    is_acquisition_csv_file,
    pipeline_load_job,
)
from spot_check.gui.spot_info_popup import SpotInfoPopup, install_spot_popup_dismiss_filter
from spot_check.gui.state import (
    apply_saved_window_layout,
    finish_saved_window_layout,
    geom_from_win,
    load_gui_state,
    save_gui_state,
    win_is_maximized,
)
from spot_check.gui.theme import MUTED_BODY, MUTED_HELP, MUTED_HINT
from spot_check.gui.timeline_playback import TimelinePlaybackBar
from spot_check.gui.workers import PipelineLoaderSignals, PipelineLoadRunnable
from spot_check.pipeline import CallbackProgressSink, PipelineConfig
from spot_check.pipeline.export_job import pipeline_export_load
from spot_check.pipeline.phases.qa import run_qa_phase
from spot_check.pipeline.progress import NullProgressSink, ProgressEvent

FOLDER = project_root()
# Short pause after typing in numeric fields before recomputing the 3D view (ms).
_REFRESH_DEBOUNCE_MS = 380
_SPOT_WEIGHT_COMBO_LABELS: tuple[tuple[str, str], ...] = (
    ("channel_sum", "Channel sum"),
    ("fit_amplitude_a", "Fit amp A"),
    ("fit_amplitude_b", "Fit amp B"),
)
_SW_LABEL_BY_MODE: dict[str, str] = {m: lbl for m, lbl in _SPOT_WEIGHT_COMBO_LABELS}
_SW_MODE_BY_LABEL: dict[str, str] = {lbl: m for m, lbl in _SPOT_WEIGHT_COMBO_LABELS}

logger = logging.getLogger(__name__)



class SpotCheckController:
    """Builds and runs the SpotCheck Qt main window."""

    def run(self) -> None:
        saved = load_gui_state()
        app = QApplication.instance() or QApplication(sys.argv)

        win = QMainWindow()
        win.setWindowTitle(f"SpotCheck v{__version__} — Plan vs acquisition")
        win.setMinimumSize(1040, 640)
        restore_maximized = apply_saved_window_layout(
            win,
            saved.get("window_geometry"),
            maximized=saved.get("window_maximized"),
        )

        central = QWidget()
        win.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        vtk_host = QFrame()
        vtk_host.setMinimumWidth(320)
        vtk_host.setStyleSheet("background-color: #0d1117;")
        vtk_layout = QVBoxLayout(vtk_host)
        vtk_layout.setContentsMargins(0, 0, 0, 0)
        vtk_layout.setSpacing(0)
        vtk_view_pane = QFrame()
        vtk_view_pane.setStyleSheet("background-color: #0d1117;")
        vtk_view_layout = QVBoxLayout(vtk_view_pane)
        vtk_view_layout.setContentsMargins(0, 0, 0, 0)
        vtk_placeholder = QLabel(
            "3D view — pick a plan (.dcm or Pyramid .csv) and/or acquisition .csv; "
            "updates when inputs validate."
        )
        vtk_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vtk_placeholder.setWordWrap(True)
        vtk_placeholder.setStyleSheet("color: #8b949e; font-size: 11pt; padding: 24px;")
        vtk_view_layout.addWidget(vtk_placeholder)
        vtk_layout.addWidget(vtk_view_pane, 1)

        spot_info_popup = SpotInfoPopup(vtk_view_pane)
        install_spot_popup_dismiss_filter(vtk_view_pane, spot_info_popup)

        timeline_bar = TimelinePlaybackBar(central)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setMinimumWidth(400)
        drawer = QWidget()
        drawer_layout = QVBoxLayout(drawer)
        drawer_layout.setSpacing(6)
        drawer_layout.setContentsMargins(4, 6, 4, 6)
        drawer_layout.addStretch(0)

        def _group_layout(gb: QGroupBox) -> QVBoxLayout:
            lay = QVBoxLayout(gb)
            lay.setSpacing(4)
            lay.setContentsMargins(8, 8, 8, 6)
            return lay

        def _inline_row(*widgets: QWidget, stretch_last: bool = False) -> QWidget:
            wrap = QWidget()
            row = QHBoxLayout(wrap)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(6)
            n = len(widgets)
            for i, w in enumerate(widgets):
                if stretch_last and i == n - 1:
                    row.addWidget(w, 1)
                else:
                    row.addWidget(w)
            if not stretch_last:
                row.addStretch(1)
            return wrap

        def _add_hint_lbl(text: str) -> QLabel:
            hint = QLabel(text)
            hint.setWordWrap(True)
            hint.setStyleSheet(f"color: {MUTED_HINT};")
            hint.setMinimumWidth(0)
            hint.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            hint.setMaximumWidth(800)
            return hint

        mode0 = normalize_layer_assign_mode(
            str(saved.get("layer_assign_mode") or "gate_counter")
        )

        e_dcm = QLineEdit(str(saved.get("dcm_path") or ""))
        e_csv = QLineEdit(str(saved.get("csv_path") or ""))

        def _qa_thr_pair(
            pass_key: str,
            warn_key: str,
            pass_def: float,
            warn_def: float,
        ) -> tuple[float, float]:
            try:
                qp = float(saved.get(pass_key, pass_def))
                qw = float(saved.get(warn_key, warn_def))
                if 0.0 < qp < qw <= 500.0:
                    return qp, qw
            except (TypeError, ValueError):
                pass
            return float(pass_def), float(warn_def)

        _qa_thr_by_mode: dict[str, tuple[float, float]] = {
            "position": _qa_thr_pair(
                "plan_qa_pass_mm",
                "plan_qa_warn_mm",
                sc_const.PLAN_QA_PASS_MM_DEFAULT,
                sc_const.PLAN_QA_WARN_MM_DEFAULT,
            ),
            "dose": _qa_thr_pair(
                "plan_qa_pass_pp",
                "plan_qa_warn_pp",
                sc_const.PLAN_QA_DOSE_PASS_PP_DEFAULT,
                sc_const.PLAN_QA_DOSE_WARN_PP_DEFAULT,
            ),
        }
        qa_mode0 = str(saved.get("plan_qa_mode", "position")).strip().lower()
        if qa_mode0 not in ("position", "dose"):
            qa_mode0 = "position"
        qp0, qw0 = _qa_thr_by_mode[qa_mode0]
        e_qa_pass = QLineEdit(f"{qp0:g}")
        e_qa_warn = QLineEdit(f"{qw0:g}")

        def _bool_saved(key: str, default: bool = False) -> bool:
            v = saved.get(key, default)
            if isinstance(v, bool):
                return v
            return str(v).strip().lower() in ("1", "true", "yes")

        cb_weight_ch = QCheckBox("Tint measured by spot weight (opacity)")
        cb_weight_ch.setChecked(_bool_saved("weight_measured_by_channel_sum", True))
        cb_weight_ch.setToolTip(
            "Low weight → fainter measured markers. Source is the Weight dropdown "
            "(channel sum, Fit Amplitude A, or Fit Amplitude B)."
        )
        swm0 = spot_weight_mode_from_saved(saved.get("spot_weight_mode"))
        combo_sw = QComboBox()
        for _m, lbl in _SPOT_WEIGHT_COMBO_LABELS:
            combo_sw.addItem(lbl, _m)
        combo_sw.setCurrentText(_SW_LABEL_BY_MODE[swm0])
        cb_plan_fwhm = QCheckBox("Plan: FWHM ellipses (DICOM 300A,0398)")
        cb_plan_fwhm.setChecked(_bool_saved("scale_plan_spots_by_dicom_fwhm", False))
        cb_meas_sigma = QCheckBox("Measured: σ ellipsoids (fit σ → XY mm; B→X, A→Y)")
        cb_meas_sigma.setChecked(_bool_saved("measured_spots_sigma_world_mm", False))
        cb_z_water = QCheckBox("Z: water depth (mm)")
        cb_z_water.setChecked(_bool_saved("z_axis_proton_water_depth_mm", True))
        _Z_DEPTH_METRIC_ITEMS: tuple[tuple[str, str], ...] = (
            ("csda", "CSDA"),
            ("r90", "R90"),
            ("r80", "R80"),
        )
        metric0 = normalize_z_depth_metric(
            saved.get("z_depth_metric", sc_const.Z_DEPTH_METRIC_DEFAULT)
        )
        combo_z_depth = QComboBox()
        for key, label in _Z_DEPTH_METRIC_ITEMS:
            combo_z_depth.addItem(label, key)
        for i in range(combo_z_depth.count()):
            if combo_z_depth.itemData(i) == metric0:
                combo_z_depth.setCurrentIndex(i)
                break
        combo_z_depth.setToolTip(
            "Depth model in water for the Z axis: PSTAR CSDA range, or empirical R90/R80 "
            "(distal 90%/80% falloff vs CSDA)."
        )
        raw_wet = saved.get(
            "upstream_wet_shifter_mm", sc_const.UPSTREAM_WET_SHIFTER_MM_DEFAULT
        )
        try:
            wet0 = float(raw_wet)
            if wet0 < 0.0 or wet0 > float(sc_const.UPSTREAM_WET_SHIFTER_MM_MAX):
                wet0 = float(sc_const.UPSTREAM_WET_SHIFTER_MM_DEFAULT)
        except (TypeError, ValueError):
            wet0 = float(sc_const.UPSTREAM_WET_SHIFTER_MM_DEFAULT)
        e_wet = QLineEdit(f"{wet0:g}")
        e_wet.setFixedWidth(72)
        e_wet.setToolTip(
            "Water-equivalent thickness (mm) of upstream range shifter / WET material. "
            "Subtracted from the selected Z depth (shallower spots) when water depth is enabled."
        )
        cb_align = QCheckBox("Coarse flat 2D align (before assignment)")
        cb_align.setChecked(_bool_saved("coarse_flat_align", True))
        cb_agg = QCheckBox("Aggregate rows per assigned spot (weighted mean)")
        cb_agg.setChecked(_bool_saved("aggregate_spots_by_gate", True))
        cb_agg.setToolTip(
            "On: merge all CSV rows assigned to the same plan spot into one weighted-mean point. "
            "Off: keep every assigned row as its own point."
        )
        cb_heal_partial = QCheckBox("Heal one-axis fits from plan (keep partial A/B rows)")
        cb_heal_partial.setChecked(_bool_saved("heal_partial_fit_axes", False))
        cb_heal_partial.setToolTip(
            "Off (default): drop rows missing Fit Mean Position A or B. "
            "On: keep rows with exactly one missing axis; fill the gap from the nearest plan "
            "spot at the assigned layer (shown gold in 3D when QA coloring is off)."
        )
        cb_filter_xy = QCheckBox("Remove XY fliers vs expected plan (σ-normalized)")
        cb_filter_xy.setChecked(_bool_saved("filter_xy_fliers", False))
        cb_filter_xy.setToolTip(
            "After spot assignment, drop rows whose plan-frame offset to the nearest plan spot "
            "on the row's layer exceeds the σ limit on each fit axis. Requires a plan."
        )
        raw_filter_sigma = saved.get(
            "filter_xy_flier_sigma", sc_const.FILTER_XY_FLIER_SIGMA_DEFAULT
        )
        try:
            filter_sigma0 = float(raw_filter_sigma)
            if (
                filter_sigma0 < sc_const.FILTER_XY_FLIER_SIGMA_MIN
                or filter_sigma0 > sc_const.FILTER_XY_FLIER_SIGMA_MAX
            ):
                filter_sigma0 = float(sc_const.FILTER_XY_FLIER_SIGMA_DEFAULT)
        except (TypeError, ValueError):
            filter_sigma0 = float(sc_const.FILTER_XY_FLIER_SIGMA_DEFAULT)
        e_filter_sigma = QLineEdit(f"{filter_sigma0:g}")
        e_filter_sigma.setFixedWidth(56)
        e_filter_sigma.setToolTip(
            f"Keep rows with sqrt((dx/σ_x)² + (dy/σ_y)²) ≤ limit "
            f"({sc_const.FILTER_XY_FLIER_SIGMA_MIN:g}–{sc_const.FILTER_XY_FLIER_SIGMA_MAX:g})."
        )
        cb_pqa = QCheckBox("Color measured spots by plan QA (pass / warn / fail)")
        cb_pqa.setChecked(_bool_saved("plan_qa_coloring", True))
        rb_qa_pos = QRadioButton("Position (XY mm vs plan)")
        rb_qa_dose = QRadioButton("Dose (layer MU % vs measured weight %)")
        qa_mode_grp = QButtonGroup(win)
        qa_mode_grp.addButton(rb_qa_pos)
        qa_mode_grp.addButton(rb_qa_dose)
        if qa_mode0 == "dose":
            rb_qa_dose.setChecked(True)
        else:
            rb_qa_pos.setChecked(True)
        cb_qa_lines = QCheckBox("QA lines: warn/fail → plan (position only)")
        cb_qa_lines.setChecked(_bool_saved("plan_qa_draw_error_lines", False))
        cb_qa_hide = QCheckBox("Hide pass-tier in 3D (warn+fail only)")
        cb_qa_hide.setChecked(_bool_saved("plan_qa_hide_pass_spots", False))
        cb_view_proj = QCheckBox("Projection view")
        cb_view_proj.setChecked(_bool_saved("view_projection_perspective", True))
        cb_view_proj.setToolTip("On: perspective (default). Off: orthogonal / parallel projection.")

        rb_auto_episodes = QRadioButton("Auto: signal episodes + plan spot count")
        rb_auto_seq = QRadioButton("Auto: plan order (break + XY cluster)")
        rb_gate = QRadioButton("Gate counter (odd=spot, even=deadtime)")
        layer_grp = QButtonGroup(win)
        layer_grp.addButton(rb_auto_episodes)
        layer_grp.addButton(rb_auto_seq)
        layer_grp.addButton(rb_gate)
        if mode0 == "auto_plan_sequential":
            rb_auto_seq.setChecked(True)
        elif mode0 == "auto":
            rb_auto_episodes.setChecked(True)
        else:
            rb_gate.setChecked(True)

        def _layer_assign_mode_from_ui() -> str:
            if rb_gate.isChecked():
                return "gate_counter"
            if rb_auto_seq.isChecked():
                return "auto_plan_sequential"
            return "auto"

        help_lbl = QLabel()
        help_lbl.setWordWrap(True)
        help_lbl.setStyleSheet(f"color: {MUTED_HELP}; font-size: 9pt;")
        help_lbl.setMinimumWidth(0)
        help_lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

        agg_intro_lbl = QLabel()
        agg_intro_lbl.setWordWrap(True)
        agg_intro_lbl.setStyleSheet(f"color: {MUTED_HELP}; font-size: 9pt;")
        agg_intro_lbl.setMinimumWidth(0)
        agg_intro_lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

        slice_chk = QCheckBox("5-layer energy band (off = full stack)")
        slice_chk.setEnabled(False)
        slice_sli = QSlider(Qt.Orientation.Horizontal)
        slice_sli.setTracking(True)
        slice_sli.setEnabled(False)
        slice_sli.setMinimum(0)
        slice_sli.setMaximum(1)
        slice_chk.setChecked(_bool_saved("slice_band_on", False))
        try:
            _slice_ci0 = int(saved.get("slice_band_center_i", 0))
            if _slice_ci0 < 0:
                _slice_ci0 = 0
        except (TypeError, ValueError):
            _slice_ci0 = 0
        slice_sli.setValue(min(_slice_ci0, slice_sli.maximum()))
        slice_status = QLabel("Band slider enables after first good plot.")
        slice_status.setWordWrap(True)
        slice_status.setStyleSheet(f"color: {MUTED_HINT}; font-size: 9pt;")
        slice_status.setMinimumWidth(0)
        slice_status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

        slice_qt_bindings: dict[str, object] = {
            "check": slice_chk,
            "slider": slice_sli,
            "status": slice_status,
        }
        analysis.idle_slice_band_controls_qt(slice_qt_bindings)

        try:
            _time_start0 = int(saved.get("time_slice_start_ms", 0))
            if _time_start0 < 0:
                _time_start0 = 0
        except (TypeError, ValueError):
            _time_start0 = 0
        try:
            _time_speed0 = float(saved.get("time_slice_speed", 1.0))
            if not math.isfinite(_time_speed0) or _time_speed0 <= 0.0:
                _time_speed0 = 1.0
        except (TypeError, ValueError):
            _time_speed0 = 1.0
        try:
            _time_window0 = float(
                saved.get("time_slice_window_s", sc_const.TIME_SLICE_WINDOW_S_DEFAULT)
            )
            if not math.isfinite(_time_window0):
                _time_window0 = float(sc_const.TIME_SLICE_WINDOW_S_DEFAULT)
        except (TypeError, ValueError):
            _time_window0 = float(sc_const.TIME_SLICE_WINDOW_S_DEFAULT)
        _time_on0 = _bool_saved("time_slice_on", False)

        time_slice_qt_bindings: dict[str, object] = timeline_bar.bindings_dict()
        analysis.idle_time_slice_controls_qt(time_slice_qt_bindings)

        def _idle_view_slice_controls_qt() -> None:
            analysis.idle_slice_band_controls_qt(slice_qt_bindings)
            analysis.idle_time_slice_controls_qt(time_slice_qt_bindings)

        status_lbl = QLabel("Browse or paste paths. Summary here after each update.")
        status_lbl.setWordWrap(True)
        status_lbl.setStyleSheet(f"color: {MUTED_BODY};")
        status_lbl.setMinimumWidth(0)
        status_lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

        def _current_plan_qa_mode() -> str:
            return "dose" if rb_qa_dose.isChecked() else "position"

        def _stash_qa_thresholds() -> None:
            thr = parse_plan_qa_thresholds(e_qa_pass.text(), e_qa_warn.text())
            if thr is not None:
                _qa_thr_by_mode[_current_plan_qa_mode()] = thr

        def _apply_qa_mode_ui(*, refresh_fields: bool = True) -> None:
            mode = _current_plan_qa_mode()
            if mode == "dose":
                lbl_qa_pass.setText("Pass ≤ (pp)")
                lbl_qa_warn.setText("Warn ≤ (pp)")
                qa_hint_lbl.setText(
                    "Layer % vs plan MU using the Weight column (channel sum or fit amp A/B). "
                    "Green pass; yellow/red over-dose; cyan/violet under-dose."
                )
            else:
                lbl_qa_pass.setText("Pass ≤ (mm)")
                lbl_qa_warn.setText("Warn ≤ (mm)")
                qa_hint_lbl.setText("Need 0 < pass < warn.")
            if refresh_fields:
                qp, qw = _qa_thr_by_mode[mode]
                e_qa_pass.setText(f"{qp:g}")
                e_qa_warn.setText(f"{qw:g}")
            _sync_qa_lines()

        def persist() -> None:
            mode = _layer_assign_mode_from_ui()
            wet_save = parse_upstream_wet_shifter_mm(e_wet.text())
            if wet_save is None:
                wet_save = float(sc_const.UPSTREAM_WET_SHIFTER_MM_DEFAULT)
                e_wet.setText(f"{wet_save:g}")
            filter_sigma_save = parse_filter_xy_flier_sigma(e_filter_sigma.text())
            if filter_sigma_save is None:
                filter_sigma_save = float(sc_const.FILTER_XY_FLIER_SIGMA_DEFAULT)
                if not filter_xy_flier_sigma_input_in_progress(e_filter_sigma.text()):
                    e_filter_sigma.setText(f"{filter_sigma_save:g}")
            _stash_qa_thresholds()
            qa_thr = parse_plan_qa_thresholds(e_qa_pass.text(), e_qa_warn.text())
            if qa_thr is not None:
                qa_pass_sv, qa_warn_sv = qa_thr
                _qa_thr_by_mode[_current_plan_qa_mode()] = (float(qa_pass_sv), float(qa_warn_sv))
            else:
                qa_pass_sv, qa_warn_sv = _qa_thr_by_mode[_current_plan_qa_mode()]
                if not plan_qa_thresholds_input_in_progress(e_qa_pass.text(), e_qa_warn.text()):
                    e_qa_pass.setText(f"{qa_pass_sv:g}")
                    e_qa_warn.setText(f"{qa_warn_sv:g}")
            qa_mode_sv = _current_plan_qa_mode()
            pos_thr = _qa_thr_by_mode["position"]
            dose_thr = _qa_thr_by_mode["dose"]
            sw_lbl = combo_sw.currentText().strip()
            sw_internal = _SW_MODE_BY_LABEL.get(sw_lbl)
            if sw_internal is None:
                sw_internal = sc_const.SPOT_WEIGHT_MODE_DEFAULT
                combo_sw.setCurrentText(_SW_LABEL_BY_MODE[sw_internal])
            try:
                sw_mode_norm = analysis.normalize_measured_spot_weight_mode(sw_internal)
            except ValueError:
                sw_mode_norm = sc_const.SPOT_WEIGHT_MODE_DEFAULT
                combo_sw.setCurrentText(_SW_LABEL_BY_MODE[sw_mode_norm])
            try:
                _sb_ci_persist = int(slice_sli.value())
            except (AttributeError, TypeError, ValueError):
                _sb_ci_persist = 0
            try:
                _ts_start_persist = int(timeline_bar.start_ms())
            except (AttributeError, TypeError, ValueError):
                _ts_start_persist = _time_start0
            try:
                _ts_speed_persist = float(timeline_bar.speed_multiplier())
            except (AttributeError, TypeError, ValueError):
                _ts_speed_persist = _time_speed0
            try:
                _ts_window_persist = float(timeline_bar.window_seconds())
            except (AttributeError, TypeError, ValueError):
                _ts_window_persist = _time_window0
            try:
                _ts_on_persist = bool(timeline_bar.slice_enabled())
            except (AttributeError, TypeError, ValueError):
                _ts_on_persist = _time_on0
            save_gui_state(
                dcm_path=e_dcm.text().strip(),
                csv_path=e_csv.text().strip(),
                window_geometry=geom_from_win(win),
                window_maximized=win_is_maximized(win),
                layer_assign_mode=mode,
                weight_measured_by_channel_sum=cb_weight_ch.isChecked(),
                spot_weight_mode=sw_mode_norm,
                aggregate_spots_by_gate=cb_agg.isChecked(),
                heal_partial_fit_axes=cb_heal_partial.isChecked(),
                coarse_flat_align=cb_align.isChecked(),
                fine_align_xy=cb_fine_xy.isChecked(),
                fine_align_rotation=cb_fine_rot.isChecked(),
                fine_align_scale=cb_fine_scale.isChecked(),
                filter_xy_fliers=cb_filter_xy.isChecked(),
                filter_xy_flier_sigma=float(filter_sigma_save),
                plan_qa_coloring=cb_pqa.isChecked(),
                plan_qa_mode=qa_mode_sv,
                plan_qa_pass_mm=float(pos_thr[0]),
                plan_qa_warn_mm=float(pos_thr[1]),
                plan_qa_pass_pp=float(dose_thr[0]),
                plan_qa_warn_pp=float(dose_thr[1]),
                plan_qa_draw_error_lines=cb_qa_lines.isChecked(),
                plan_qa_hide_pass_spots=cb_qa_hide.isChecked(),
                scale_plan_spots_by_dicom_fwhm=cb_plan_fwhm.isChecked(),
                measured_spots_sigma_world_mm=cb_meas_sigma.isChecked(),
                z_axis_proton_water_depth_mm=cb_z_water.isChecked(),
                upstream_wet_shifter_mm=float(wet_save),
                z_depth_metric=_current_z_depth_metric(),
                view_projection_perspective=cb_view_proj.isChecked(),
                slice_band_on=bool(slice_chk.isChecked()),
                slice_band_center_i=_sb_ci_persist,
                time_slice_on=_ts_on_persist,
                time_slice_start_ms=_ts_start_persist,
                time_slice_speed=_ts_speed_persist,
                time_slice_window_s=_ts_window_persist,
            )

        def browse_dcm() -> None:
            init = str(Path(e_dcm.text().strip()).parent) if e_dcm.text().strip() else str(FOLDER)
            p, _ = QFileDialog.getOpenFileName(
                win,
                "Plan (DICOM or Pyramid CSV)",
                init if Path(init).is_dir() else str(FOLDER),
                (
                    "Plan files (*.dcm *.csv);;DICOM (*.dcm);;"
                    "Pyramid plan CSV (*.csv);;All files (*.*)"
                ),
            )
            if p:
                e_dcm.setText(p)
                _do_refresh()

        def browse_csv() -> None:
            init = str(Path(e_csv.text().strip()).parent) if e_csv.text().strip() else str(FOLDER)
            p, _ = QFileDialog.getOpenFileName(
                win,
                "CSV",
                init if Path(init).is_dir() else str(FOLDER),
                "CSV (*.csv *.csv.gz);;All files (*.*)",
            )
            if p:
                e_csv.setText(p)
                _do_refresh()

        def _sync_wet_shifter_ui() -> None:
            on = cb_z_water.isChecked()
            e_wet.setEnabled(on)
            combo_z_depth.setEnabled(on)

        def _sync_filter_xy_ui() -> None:
            on = cb_filter_xy.isChecked()
            e_filter_sigma.setEnabled(on)

        def _current_z_depth_metric() -> str:
            key = combo_z_depth.currentData()
            if key is None:
                return normalize_z_depth_metric(combo_z_depth.currentText())
            return normalize_z_depth_metric(str(key))

        def _sync_qa_lines() -> None:
            en = cb_pqa.isChecked()
            dose = _current_plan_qa_mode() == "dose"
            if not en:
                cb_qa_lines.setChecked(False)
                cb_qa_hide.setChecked(False)
            if dose:
                cb_qa_lines.setChecked(False)
            cb_qa_lines.setEnabled(en and not dose)
            cb_qa_hide.setEnabled(en)

        def _update_help() -> None:
            lam = _layer_assign_mode_from_ui()
            if lam == "auto_plan_sequential":
                help_lbl.setText(
                    "Auto plan-sequential: assign from the first plan spot (highest energy "
                    "layer). Deadtime = no fit on Fit Mean Position A or B. Spans are aligned "
                    "to the plan spot count and boundaries refined with plan XY (+1 only). "
                    "Gate Counter ignored."
                )
                lbl_auto_tuning.setText(
                    "Position-fit spans merged to plan count; plan XY refines boundaries."
                )
                cb_agg.setEnabled(True)
                agg_intro_lbl.setText(
                    "Checked: one weighted-mean row per plan-sequential span (assigned plan spot). "
                    "Unchecked: every on-spot CSV row in each span is kept."
                )
            elif lam == "auto":
                cb_agg.setEnabled(True)
                help_lbl.setText(
                    "Auto episodes: segment from timing, weight, and XY "
                    "(Gate Counter and Gate Signal ignored). "
                    "Thresholds are inferred and episodes aligned to the plan spot count."
                )
                lbl_auto_tuning.setText(
                    "Tuning: inferred per load — see status line after refresh."
                )
                agg_intro_lbl.setText(
                    "Checked: one weighted-mean row per signal episode (assigned plan spot). "
                    "Unchecked: every on-spot CSV row is kept."
                )
            else:
                cb_agg.setEnabled(True)
                help_lbl.setText(
                    f'Gate: DICOM order; "{sc_const.GATE_COUNTER_KEY}" odd=spot, even=deadtime; '
                    "new odd advances."
                )
                agg_intro_lbl.setText(
                    f"Checked: one weighted-mean row per odd {sc_const.GATE_COUNTER_KEY} phase "
                    "(assigned plan spot). Unchecked: every on-spot CSV row is kept."
                )

        def _path_clear_button(edit: QLineEdit) -> QPushButton:
            btn = QPushButton("×")
            btn.setFixedWidth(28)
            btn.setToolTip("Clear file path")
            btn.setEnabled(bool(edit.text().strip()))

            def _sync_clear_enabled(_text: str = "") -> None:
                btn.setEnabled(bool(edit.text().strip()))

            edit.textChanged.connect(_sync_clear_enabled)

            def _clear_path() -> None:
                edit.clear()
                _do_refresh()

            btn.clicked.connect(_clear_path)
            return btn

        cb_fine_xy = QCheckBox("Fine XY")
        cb_fine_xy.setChecked(_bool_saved("fine_align_xy", True))
        cb_fine_rot = QCheckBox("Fine rotation")
        cb_fine_rot.setChecked(_bool_saved("fine_align_rotation", True))
        cb_fine_scale = QCheckBox("Fine scale X/Y")
        cb_fine_scale.setChecked(_bool_saved("fine_align_scale", True))
        fine_tip = (
            "Optional refinement after aggregation: weighted least-squares adjustment vs plan XY "
            "(nearest plan spot on each row's nominal layer — same pairing as plan QA). "
            "Independent scale absorbs source-to-detector distance error; optional SAD entry for "
            "constrained magnification may follow."
        )
        for _cb_fin in (cb_fine_xy, cb_fine_rot, cb_fine_scale):
            _cb_fin.setToolTip(fine_tip)

        e_qa_pass.setFixedWidth(56)
        e_qa_warn.setFixedWidth(56)

        # --- drawer sections ---
        gb_files = QGroupBox("Files")
        fl = QFormLayout(gb_files)
        fl.setSpacing(4)
        fl.setContentsMargins(8, 8, 8, 6)
        row_dcm = QWidget()
        h_dcm = QHBoxLayout(row_dcm)
        h_dcm.setContentsMargins(0, 0, 0, 0)
        h_dcm.addWidget(e_dcm, 1)
        b_dcm = QPushButton("Browse…")
        b_dcm.clicked.connect(browse_dcm)
        h_dcm.addWidget(b_dcm)
        h_dcm.addWidget(_path_clear_button(e_dcm))
        row_csv = QWidget()
        h_csv = QHBoxLayout(row_csv)
        h_csv.setContentsMargins(0, 0, 0, 0)
        h_csv.addWidget(e_csv, 1)
        b_csv = QPushButton("Browse…")
        b_csv.clicked.connect(browse_csv)
        h_csv.addWidget(b_csv)
        h_csv.addWidget(_path_clear_button(e_csv))
        fl.addRow("Plan", row_dcm)
        fl.addRow("CSV", row_csv)
        btn_export_csv = QPushButton("Export combined CSV…")
        btn_export_csv.setToolTip(
            "Save one row per measured spot (aggregated or per-row per Aggregation), "
            "with nearest plan XY, energy, and meterset weight (MU) on that layer."
        )
        fl.addRow("", btn_export_csv)

        gb_filter = QGroupBox("Filtering")
        vl_filter = _group_layout(gb_filter)
        vl_filter.addWidget(cb_filter_xy)
        vl_filter.addWidget(
            _inline_row(QLabel("σ limit:"), e_filter_sigma, stretch_last=False)
        )
        vl_filter.addWidget(
            _add_hint_lbl(
                "Runs after assignment: nearest plan spot on the row's layer; "
                "fit σ A/B mapped like A/B axes."
            )
        )

        gb_assign = QGroupBox("Spot assignment & aggregation")
        vl_assign = _group_layout(gb_assign)
        vl_assign.addWidget(rb_auto_episodes)
        vl_assign.addWidget(rb_auto_seq)
        vl_assign.addWidget(rb_gate)
        lbl_auto_tuning = QLabel(
            "Tuning: inferred from plan + CSV when Auto is selected."
        )
        lbl_auto_tuning.setWordWrap(True)
        lbl_auto_tuning.setStyleSheet(f"color: {MUTED_HINT}; font-size: 9pt;")
        vl_assign.addWidget(lbl_auto_tuning)
        vl_assign.addWidget(help_lbl)
        vl_assign.addWidget(cb_heal_partial)
        vl_assign.addWidget(agg_intro_lbl)
        vl_assign.addWidget(cb_agg)

        gb_align = QGroupBox("Detector alignment")
        vl_align = _group_layout(gb_align)
        vl_align.addWidget(cb_align)
        fine_grid = QGridLayout()
        fine_grid.setContentsMargins(0, 0, 0, 0)
        fine_grid.setHorizontalSpacing(8)
        fine_grid.setVerticalSpacing(2)
        fine_grid.addWidget(cb_fine_xy, 0, 0)
        fine_grid.addWidget(cb_fine_rot, 0, 1)
        fine_grid.addWidget(cb_fine_scale, 1, 0, 1, 2)
        w_fine = QWidget()
        w_fine.setLayout(fine_grid)
        vl_align.addWidget(w_fine)

        gb_qa = QGroupBox("Plan QA")
        vqa = _group_layout(gb_qa)
        vqa.addWidget(cb_pqa)
        vqa.addWidget(_inline_row(rb_qa_pos, rb_qa_dose))
        vqa.addWidget(_inline_row(cb_qa_lines, cb_qa_hide))
        lbl_qa_pass = QLabel("Pass ≤ (mm)")
        lbl_qa_warn = QLabel("Warn ≤ (mm)")
        vqa.addWidget(_inline_row(lbl_qa_pass, e_qa_pass, lbl_qa_warn, e_qa_warn))
        qa_hint_lbl = _add_hint_lbl("Need 0 < pass < warn.")
        vqa.addWidget(qa_hint_lbl)
        lbl_qa_counts = QLabel("Counts: —")
        lbl_qa_counts.setWordWrap(True)
        lbl_qa_counts.setStyleSheet(f"color: {MUTED_BODY}; font-size: 9pt;")
        lbl_qa_counts.setMinimumWidth(0)
        lbl_qa_counts.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        vqa.addWidget(lbl_qa_counts)
        _apply_qa_mode_ui(refresh_fields=False)

        gb_disp = QGroupBox("Display & view")
        vdisp = _group_layout(gb_disp)
        vdisp.addWidget(cb_weight_ch)
        vdisp.addWidget(_inline_row(QLabel("Weight:"), combo_sw, stretch_last=True))
        vdisp.addWidget(_inline_row(cb_plan_fwhm, cb_meas_sigma))
        z_wet_row = QHBoxLayout()
        z_wet_row.setContentsMargins(0, 0, 0, 0)
        z_wet_row.setSpacing(6)
        z_wet_row.addWidget(cb_z_water)
        z_wet_row.addWidget(QLabel("WET (mm)"))
        z_wet_row.addWidget(e_wet)
        z_wet_row.addWidget(QLabel("Depth"))
        z_wet_row.addWidget(combo_z_depth, 1)
        w_z_wet = QWidget()
        w_z_wet.setLayout(z_wet_row)
        vdisp.addWidget(w_z_wet)
        vdisp.addWidget(_inline_row(slice_chk, slice_sli, stretch_last=True))
        vdisp.addWidget(slice_status)
        view_proj_row = QHBoxLayout()
        view_proj_row.setContentsMargins(0, 0, 0, 0)
        view_proj_row.setSpacing(6)
        view_proj_row.addWidget(cb_view_proj)
        btn_view_top = QPushButton("Top")
        btn_view_left = QPushButton("Left")
        btn_view_right = QPushButton("Right")
        for _vb in (btn_view_top, btn_view_left, btn_view_right):
            _vb.setFixedHeight(26)
        btn_view_top.setToolTip("Top down: look along scene Z (detector XY plane).")
        btn_view_left.setToolTip("From −X (Fit B) side — YZ plane.")
        btn_view_right.setToolTip("From +X (Fit B) side — YZ plane.")
        view_proj_row.addWidget(btn_view_top)
        view_proj_row.addWidget(btn_view_left)
        view_proj_row.addWidget(btn_view_right)
        view_proj_row.addStretch(1)
        vdisp.addLayout(view_proj_row)

        def _update_plan_qa_counts_label(
            planned: list[tuple[float, float, float]] | None,
            measured: list[tuple[float, ...]] | None,
            *,
            qa_mode: str,
            qa_pass_f: float,
            qa_warn_f: float,
            plan_mu: object,
        ) -> None:
            from spot_check.pipeline.types import PipelineState

            qa_state = PipelineState()
            qa = run_qa_phase(
                qa_state,
                NullProgressSink(),
                planned=list(planned or []),
                measured=list(measured or []),
                qa_mode=qa_mode,
                pass_thr=float(qa_pass_f),
                warn_thr=float(qa_warn_f),
                plan_mu=plan_mu,
                enabled=bool(cb_pqa.isChecked()),
            )
            _plot_cache["qa_result"] = qa
            lbl_qa_counts.setText(
                format_qa_counts_label(qa, enabled=bool(cb_pqa.isChecked()))
            )

        drawer_layout.addWidget(gb_files)
        drawer_layout.addWidget(gb_filter)
        drawer_layout.addWidget(gb_assign)
        drawer_layout.addWidget(gb_align)
        drawer_layout.addWidget(gb_qa)
        drawer_layout.addWidget(gb_disp)

        auto_hint = QLabel(
            "3D refreshes when inputs validate. Numbers debounce briefly after typing."
        )
        auto_hint.setWordWrap(True)
        auto_hint.setStyleSheet(f"color: {MUTED_HINT}; font-size: 9pt;")
        auto_hint.setMinimumWidth(0)
        auto_hint.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        drawer_layout.addWidget(auto_hint)

        drawer_layout.addWidget(status_lbl)
        foot = QLabel("pydicom · PyVista · PySide6. Engineering use only.")
        foot.setStyleSheet(f"color: {MUTED_HINT}; font-size: 8pt;")
        foot.setWordWrap(True)
        foot.setMinimumWidth(0)
        foot.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        drawer_layout.addWidget(foot)
        drawer_layout.addStretch(1)

        scroll.setWidget(drawer)
        splitter.addWidget(vtk_host)
        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setCollapsible(0, True)
        splitter.setCollapsible(1, False)
        splitter.setSizes([1000, 480])
        outer.addWidget(splitter, 1)
        outer.addWidget(timeline_bar)

        _sync_wet_shifter_ui()
        _sync_qa_lines()
        _update_help()

        _debounce = QTimer(win)
        _debounce.setSingleShot(True)
        _debounce.setInterval(_REFRESH_DEBOUNCE_MS)

        _plot_cache: dict[str, object] = {
            "pipeline_key": None,
            "planned": None,
            "plan_fwhm_xy": None,
            "plan_mu": None,
            "n_plan_kept": 0,
            "n_plan_raw": 0,
            "measured_unaligned": None,
            "measured_fine_aligned": None,
            "coarse_flat_align_info": None,
            "fine_align_info": None,
            "label": "",
            "csv_display_name": "",
            "layer_mode_run": "",
            "auto_assign_method": "episodes",
            "aggregate_run": False,
            "assign_diagnostics": None,
            "plan_spots_no_data": None,
            "plan_spot_time_s": None,
            "qa_result": None,
            "plotter": None,
            "coarse_flat": None,
            "fine_align_ui": None,
            "z_water_depth": False,
            "upstream_wet_mm": 0.0,
            "z_depth_metric": sc_const.Z_DEPTH_METRIC_DEFAULT,
            "spot_weight_mode_run": sc_const.SPOT_WEIGHT_MODE_DEFAULT,
            "slice_on": bool(slice_chk.isChecked()),
            "slice_center_i": int(slice_sli.value()),
            "time_slice_start_ms": _time_start0,
            "time_slice_speed": _time_speed0,
            "time_slice_window_s": _time_window0,
            "time_slice_on": _time_on0,
        }

        def _apply_quick_view(view: str) -> None:
            pl = _plot_cache.get("plotter")
            if pl is None:
                return
            try:
                analysis.apply_comparison_3d_camera_view(pl, view)
                qw = slice_qt_bindings.get("_qt_vtk_widget")
                if qw is not None:
                    try:
                        qw.update()
                    except Exception:
                        pass
            except Exception as exc:
                logger.warning("Could not set 3D camera view %r: %s", view, exc)

        def _apply_projection_view() -> None:
            pl = _plot_cache.get("plotter")
            if pl is not None:
                try:
                    analysis.apply_comparison_3d_projection_view(
                        pl,
                        perspective=cb_view_proj.isChecked(),
                    )
                    qw = slice_qt_bindings.get("_qt_vtk_widget")
                    if qw is not None:
                        try:
                            qw.update()
                        except Exception:
                            pass
                except Exception as exc:
                    logger.warning("Could not set 3D projection view: %s", exc)
            persist()

        def _apply_display_refresh() -> None:
            """Recolor/re-mesh spots in-place (QA, weight tint, glyphs) without pipeline reload."""
            pl = _plot_cache.get("plotter")
            planned_raw = _plot_cache.get("planned")
            measured_raw = _plot_cache.get("measured_unaligned")
            if pl is None or getattr(pl, "_spot_check_apply_display", None) is None:
                _do_refresh()
                return
            if planned_raw is None and measured_raw is None:
                _do_refresh()
                return
            planned = list(planned_raw) if planned_raw is not None else []
            measured_fine = _plot_cache.get("measured_fine_aligned")
            measured = list(measured_fine) if measured_fine else list(measured_raw or [])
            qa_mode_run = _current_plan_qa_mode()
            qa_pass_f = float(
                sc_const.PLAN_QA_DOSE_PASS_PP_DEFAULT
                if qa_mode_run == "dose"
                else sc_const.PLAN_QA_PASS_MM_DEFAULT
            )
            qa_warn_f = float(
                sc_const.PLAN_QA_DOSE_WARN_PP_DEFAULT
                if qa_mode_run == "dose"
                else sc_const.PLAN_QA_WARN_MM_DEFAULT
            )
            if cb_pqa.isChecked():
                qa_pair = parse_plan_qa_thresholds(e_qa_pass.text(), e_qa_warn.text())
                if qa_pair is None:
                    if plan_qa_thresholds_input_in_progress(
                        e_qa_pass.text(), e_qa_warn.text()
                    ):
                        qa_pass_f, qa_warn_f = _qa_thr_by_mode[qa_mode_run]
                    else:
                        return
                else:
                    qa_pass_f, qa_warn_f = qa_pair
                    _qa_thr_by_mode[qa_mode_run] = (float(qa_pass_f), float(qa_warn_f))
            sw_lbl = combo_sw.currentText().strip()
            sw_internal = _SW_MODE_BY_LABEL.get(sw_lbl, sc_const.SPOT_WEIGHT_MODE_DEFAULT)
            try:
                spot_weight_mode_run = analysis.normalize_measured_spot_weight_mode(sw_internal)
            except ValueError:
                spot_weight_mode_run = sc_const.SPOT_WEIGHT_MODE_DEFAULT
            try:
                analysis.refresh_comparison_3d_display(
                    pl,
                    planned,
                    measured,
                    a_is_x=False,
                    weight_measured_by_channel=cb_weight_ch.isChecked(),
                    spot_weight_mode=spot_weight_mode_run,
                    plan_qa_coloring=cb_pqa.isChecked(),
                    plan_qa_mode=qa_mode_run,
                    plan_qa_pass_mm=qa_pass_f
                    if qa_mode_run == "position"
                    else float(sc_const.PLAN_QA_PASS_MM_DEFAULT),
                    plan_qa_warn_mm=qa_warn_f
                    if qa_mode_run == "position"
                    else float(sc_const.PLAN_QA_WARN_MM_DEFAULT),
                    plan_qa_pass_pp=qa_pass_f
                    if qa_mode_run == "dose"
                    else float(sc_const.PLAN_QA_DOSE_PASS_PP_DEFAULT),
                    plan_qa_warn_pp=qa_warn_f
                    if qa_mode_run == "dose"
                    else float(sc_const.PLAN_QA_DOSE_WARN_PP_DEFAULT),
                    plan_mu=_plot_cache.get("plan_mu"),
                    plan_qa_draw_error_lines=cb_qa_lines.isChecked(),
                    plan_qa_hide_pass_spots=cb_qa_hide.isChecked(),
                    plan_fwhm_xy_mm=_plot_cache.get("plan_fwhm_xy"),
                    scale_plan_spots_by_dicom_fwhm=cb_plan_fwhm.isChecked(),
                    measured_spots_sigma_world_mm=cb_meas_sigma.isChecked(),
                    z_axis_use_proton_water_depth_mm=bool(_plot_cache.get("z_water_depth", False)),
                    upstream_wet_shifter_mm=float(_plot_cache.get("upstream_wet_mm", 0.0)),
                    z_depth_metric=str(
                        _plot_cache.get("z_depth_metric", sc_const.Z_DEPTH_METRIC_DEFAULT)
                    ),
                    plan_spots_no_data=_plot_cache.get("plan_spots_no_data"),
                    plan_spot_time_s=_plot_cache.get("plan_spot_time_s"),
                )
            except Exception as exc:
                logger.warning("Display refresh failed, running full replot: %s", exc)
                _do_refresh()
                return
            _update_plan_qa_counts_label(
                planned,
                measured,
                qa_mode=qa_mode_run,
                qa_pass_f=float(qa_pass_f),
                qa_warn_f=float(qa_warn_f),
                plan_mu=_plot_cache.get("plan_mu"),
            )
            qw = slice_qt_bindings.get("_qt_vtk_widget")
            if qw is not None:
                try:
                    qw.update()
                except Exception:
                    pass
            persist()

        load_generation = 0
        load_pool = QThreadPool(win)
        load_pool.setMaxThreadCount(1)
        load_signals = PipelineLoaderSignals(win)
        pending_ctx: GuiRefreshContext | None = None
        _loading_gen: int | None = None

        load_overlay = LoadOverlayPanel(vtk_host)

        def _sync_load_overlay_geometry() -> None:
            load_overlay.sync_geometry()
            if _loading_gen is not None:
                _hide_vtk_host_content()

        def _hide_vtk_host_content() -> None:
            vtk_view_pane.hide()

        def _show_vtk_host_content() -> None:
            vtk_view_pane.show()

        def _pin_load_overlay_on_top() -> None:
            _hide_vtk_host_content()
            load_overlay.sync_geometry()
            load_overlay.raise_()

        class _LoadOverlayResizeFilter(QObject):
            def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
                if event.type() == QEvent.Type.Resize:
                    _sync_load_overlay_geometry()
                return False

        _load_overlay_filter = _LoadOverlayResizeFilter(win)
        vtk_host.installEventFilter(_load_overlay_filter)

        def _show_loading(generation: int, message: str) -> None:
            nonlocal _loading_gen
            _loading_gen = int(generation)
            load_overlay.reset(message=message)
            _pin_load_overlay_on_top()
            load_overlay.show_loading()
            QApplication.processEvents()

        def _hide_loading(generation: int) -> None:
            nonlocal _loading_gen
            if _loading_gen is None or int(generation) != int(_loading_gen):
                return
            _loading_gen = None
            load_overlay.mark_complete()
            load_overlay.hide_loading()
            _show_vtk_host_content()

        def _on_pipeline_progress(event: object, generation: int) -> None:
            if generation != load_generation:
                return
            if isinstance(event, ProgressEvent):
                load_overlay.apply_event(event)

        def _vtk_placeholder_message(text: str, *, error: bool = False) -> None:
            clr = "#f85149" if error else "#8b949e"
            fn = getattr(analysis, "_clear_qt_layout_items", None)
            if fn is not None:
                fn(vtk_view_pane)
            lay = vtk_view_pane.layout()
            if lay is None:
                lay = QVBoxLayout(vtk_view_pane)
                lay.setContentsMargins(0, 0, 0, 0)
                vtk_view_pane.setLayout(lay)
            lbl = QLabel(text)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setWordWrap(True)
            lbl.setStyleSheet(f"color: {clr}; font-size: 11pt; padding: 24px;")
            lay.addWidget(lbl)

        def _schedule_refresh() -> None:
            _debounce.start()

        def _finalize_refresh(ctx: GuiRefreshContext, *, need_data: bool) -> None:
            nonlocal pending_ctx
            p = analysis
            pipeline_key = ctx.pipeline_key
            planned_raw = _plot_cache.get("planned")
            plan_fwhm_xy = _plot_cache.get("plan_fwhm_xy")
            n_plan_kept = int(_plot_cache.get("n_plan_kept", 0))
            n_plan_raw = int(_plot_cache.get("n_plan_raw", 0))
            measured_raw = _plot_cache.get("measured_unaligned")
            label = str(_plot_cache.get("label", ""))
            csv_display_name = str(_plot_cache.get("csv_display_name", ""))
            if planned_raw is None and measured_raw is None:
                logger.error(
                    "Plot cache inconsistency: pipeline_key matched but no data cached"
                )
                _plot_cache["pipeline_key"] = None
                _plot_cache["plotter"] = None
                _idle_view_slice_controls_qt()
                _vtk_placeholder_message(
                    "Display state reset — tweak an option or re-pick files.",
                    error=True,
                )
                status_lbl.setText("State error — change option or re-select files.")
                return

            planned = list(planned_raw) if planned_raw is not None else []
            measured_unaligned = list(measured_raw) if measured_raw is not None else []
            if not planned and not measured_unaligned:
                _plot_cache["pipeline_key"] = None
                _plot_cache["plotter"] = None
                _idle_view_slice_controls_qt()
                _vtk_placeholder_message(
                    "Select a plan (.dcm or Pyramid .csv) and/or acquisition .csv.",
                    error=False,
                )
                status_lbl.setText("Ready — pick plan and/or CSV.")
                return

            measured_fine = _plot_cache.get("measured_fine_aligned")
            measured = list(measured_fine) if measured_fine else list(measured_unaligned)
            coarse_info = _plot_cache.get("coarse_flat_align_info")
            fine_info = _plot_cache.get("fine_align_info")
            align_caption = p.format_total_detector_align_caption(
                coarse=coarse_info,
                fine=fine_info,
                measured_base=measured_unaligned if measured_unaligned else None,
                measured_final=measured if measured else None,
                a_is_x=False,
            )

            QApplication.processEvents()
            n_meas = len(measured)

            _si_on = bool(_plot_cache.get("slice_on", False))
            _si_ci = int(_plot_cache.get("slice_center_i", 0))
            try:
                _si_on = bool(slice_chk.isChecked())
                if slice_sli.isEnabled():
                    _si_ci = int(slice_sli.value())
            except (AttributeError, TypeError, ValueError):
                pass
            slice_band_init = {"slice_on": _si_on, "center_i": _si_ci}

            _ts_start = int(_plot_cache.get("time_slice_start_ms", _time_start0))
            _ts_speed = float(_plot_cache.get("time_slice_speed", _time_speed0))
            _ts_window = float(_plot_cache.get("time_slice_window_s", _time_window0))
            _ts_on = bool(_plot_cache.get("time_slice_on", _time_on0))
            try:
                if timeline_bar._slider.isEnabled():
                    _ts_start = int(timeline_bar.start_ms())
                    _ts_speed = float(timeline_bar.speed_multiplier())
                    _ts_window = float(timeline_bar.window_seconds())
                    _ts_on = bool(timeline_bar.slice_enabled())
            except (AttributeError, TypeError, ValueError):
                pass
            time_slice_init = {
                "slice_on": _ts_on,
                "start_ms": _ts_start,
                "window_s": _ts_window,
            }

            layer_mode_req, _, _ = resolve_layer_assign_mode(ctx.layer_assign_mode)
            layer_mode_plot = str(_plot_cache.get("layer_mode_run") or layer_mode_req)
            aggregate_plot = bool(_plot_cache.get("aggregate_run", False))

            auto_gap: float | None = None
            auto_xy: float | None = None
            auto_vp: float | None = None
            assign_plot = str(_plot_cache.get("auto_assign_method") or "episodes")
            assign_diag = _plot_cache.get("assign_diagnostics")
            if layer_mode_plot == "auto" and assign_plot == "episodes":
                auto_p = (
                    assign_diag.auto_layer_params
                    if assign_diag is not None
                    else analysis.last_auto_layer_params()
                )
                if auto_p is not None:
                    auto_gap = auto_p.episode_gap_s
                    auto_xy = auto_p.spot_xy_jump_mm
                    auto_vp = auto_p.viterbi_advance_penalty_mm2

            reuse_pl = _plot_cache.get("plotter")
            wet_use = float(sc_const.UPSTREAM_WET_SHIFTER_MM_DEFAULT)
            metric_use = _current_z_depth_metric()
            fine_xy_on = cb_fine_xy.isChecked()
            fine_rot_on = cb_fine_rot.isChecked()
            fine_scale_on = cb_fine_scale.isChecked()
            fine_tuple_ui = (fine_xy_on, fine_rot_on, fine_scale_on)
            if cb_z_water.isChecked():
                wet_parsed = parse_upstream_wet_shifter_mm(e_wet.text())
                if wet_parsed is None:
                    _bump_load_generation_invalidate_async()
                    _idle_view_slice_controls_qt()
                    _vtk_placeholder_message(
                        f"Upstream WET: 0–{sc_const.UPSTREAM_WET_SHIFTER_MM_MAX:g} mm.",
                        error=True,
                    )
                    status_lbl.setText("Fix upstream WET shifter (mm).")
                    return
                wet_use = float(wet_parsed)
            preserve_cam = (
                reuse_pl is not None
                and _plot_cache["pipeline_key"] == pipeline_key
                and _plot_cache.get("coarse_flat") == cb_align.isChecked()
                and tuple(_plot_cache.get("fine_align_ui") or ()) == fine_tuple_ui
                and _plot_cache.get("z_water_depth") == cb_z_water.isChecked()
                and float(_plot_cache.get("upstream_wet_mm", 0.0)) == wet_use
                and str(_plot_cache.get("z_depth_metric", "csda")) == metric_use
            )
            if label and csv_display_name:
                view_title = f"{label} — plan vs {csv_display_name}"
            elif label:
                view_title = f"{label} — plan"
            elif csv_display_name:
                view_title = csv_display_name
            else:
                view_title = "SpotCheck"

            def _on_spot_picked(ev: object) -> None:
                from spot_check.analysis.viz.spot_info import format_spot_info
                from spot_check.analysis.viz.spot_pick import SpotPickEvent

                if not isinstance(ev, SpotPickEvent):
                    return
                plotter = _plot_cache.get("plotter")
                disp = getattr(plotter, "_spot_check_display_state", None) if plotter else None
                rows = format_spot_info(
                    ev.kind,
                    ev.spot_index,
                    planned_xyz=planned,
                    measured_rows=measured,
                    xlab="Fit B (mm)",
                    ylab="Fit A (mm)",
                    a_is_x=False,
                    plan_mu=_plot_cache.get("plan_mu"),
                    plan_fwhm_xy_mm=plan_fwhm_xy,
                    plan_time_s=_plot_cache.get("plan_spot_time_s"),
                    plan_spots_no_data=_plot_cache.get("plan_spots_no_data"),
                    qa_mode=ctx.qa_mode,
                    display_state=disp,
                )
                title = (
                    f"Plan spot {ev.spot_index + 1}"
                    if ev.kind == "plan"
                    else f"Measured spot {ev.spot_index + 1}"
                )
                spot_info_popup.show_spot(
                    title=title,
                    rows=rows,
                    local_x=ev.display_x,
                    local_y=ev.display_y,
                )

            pl = p.show_comparison_3d_pyvista(
                planned,
                measured,
                title=view_title,
                a_is_x=False,
                layer_mode=layer_mode_plot,
                layer_gap_s=auto_gap,
                refill_same_spot_xy_tol_mm=auto_xy,
                viterbi_advance_penalty_mm2=auto_vp,
                weight_measured_by_channel=cb_weight_ch.isChecked(),
                aggregate_spots=aggregate_plot,
                spot_weight_mode=ctx.spot_weight_mode_run,
                detector_align_caption=align_caption,
                plan_qa_coloring=cb_pqa.isChecked(),
                plan_qa_mode=ctx.qa_mode,
                plan_qa_pass_mm=ctx.qa_pass_f
                if ctx.qa_mode == "position"
                else float(sc_const.PLAN_QA_PASS_MM_DEFAULT),
                plan_qa_warn_mm=ctx.qa_warn_f
                if ctx.qa_mode == "position"
                else float(sc_const.PLAN_QA_WARN_MM_DEFAULT),
                plan_qa_pass_pp=ctx.qa_pass_f
                if ctx.qa_mode == "dose"
                else float(sc_const.PLAN_QA_DOSE_PASS_PP_DEFAULT),
                plan_qa_warn_pp=ctx.qa_warn_f
                if ctx.qa_mode == "dose"
                else float(sc_const.PLAN_QA_DOSE_WARN_PP_DEFAULT),
                plan_mu=_plot_cache.get("plan_mu"),
                plan_qa_draw_error_lines=cb_qa_lines.isChecked(),
                plan_qa_hide_pass_spots=cb_qa_hide.isChecked(),
                plan_fwhm_xy_mm=plan_fwhm_xy,
                scale_plan_spots_by_dicom_fwhm=cb_plan_fwhm.isChecked(),
                measured_spots_sigma_world_mm=cb_meas_sigma.isChecked(),
                z_axis_use_proton_water_depth_mm=cb_z_water.isChecked(),
                upstream_wet_shifter_mm=wet_use,
                z_depth_metric=metric_use,
                view_projection_perspective=cb_view_proj.isChecked(),
                reuse_plotter=reuse_pl if reuse_pl is not None else None,
                reuse_camera=preserve_cam,
                reembed_qt=reuse_pl is None,
                embed_qt=vtk_view_pane,
                slice_qt=slice_qt_bindings,
                slice_band_init=slice_band_init,
                time_slice_qt=time_slice_qt_bindings,
                time_slice_init=time_slice_init,
                time_slice_speed=_ts_speed,
                plan_spots_no_data=_plot_cache.get("plan_spots_no_data"),
                plan_spot_time_s=_plot_cache.get("plan_spot_time_s"),
                on_spot_picked=_on_spot_picked,
            )
            if _loading_gen is not None:
                _pin_load_overlay_on_top()
            _plot_cache["plotter"] = pl
            _plot_cache["coarse_flat"] = cb_align.isChecked()
            _plot_cache["fine_align_ui"] = fine_tuple_ui
            _plot_cache["z_water_depth"] = cb_z_water.isChecked()
            _plot_cache["upstream_wet_mm"] = wet_use
            _plot_cache["z_depth_metric"] = metric_use
            _plot_cache["spot_weight_mode_run"] = str(ctx.spot_weight_mode_run)
            try:
                _plot_cache["slice_on"] = bool(slice_chk.isChecked())
                if slice_sli.isEnabled():
                    _plot_cache["slice_center_i"] = int(slice_sli.value())
            except (AttributeError, TypeError, ValueError):
                _plot_cache["slice_on"] = bool(slice_band_init["slice_on"])
                _plot_cache["slice_center_i"] = int(slice_band_init["center_i"])
            try:
                if timeline_bar._slider.isEnabled():
                    _plot_cache["time_slice_start_ms"] = int(timeline_bar.start_ms())
                    _plot_cache["time_slice_speed"] = float(timeline_bar.speed_multiplier())
                    _plot_cache["time_slice_window_s"] = float(timeline_bar.window_seconds())
                    _plot_cache["time_slice_on"] = bool(timeline_bar.slice_enabled())
            except (AttributeError, TypeError, ValueError):
                _plot_cache["time_slice_start_ms"] = int(time_slice_init["start_ms"])
                _plot_cache["time_slice_speed"] = float(_ts_speed)
                _plot_cache["time_slice_window_s"] = float(_ts_window)
                _plot_cache["time_slice_on"] = bool(time_slice_init["slice_on"])
            cache_note = (
                ""
                if need_data
                else " (reused plan/CSV; camera kept when only display options changed)."
            )
            if planned:
                plan_line = f"Plan spots: {n_plan_kept} after meterset filter (weight>0"
                if n_plan_raw != n_plan_kept:
                    plan_line += f"; {n_plan_raw} raw slots in CP maps before that filter"
                plan_line += ")"
            else:
                plan_line = "Plan: (none)"
            if measured:
                meas_line = f"Measured points: {n_meas} built and plotted"
            else:
                meas_line = "Measured: (none)"
            if measured and layer_mode_plot == "auto":
                assign_diag = _plot_cache.get("assign_diagnostics")
                lbl_auto_tuning.setText(
                    format_auto_tuning_label(
                        assign_diag,
                        assign_method=assign_plot,
                    )
                )
                meas_line = append_auto_meas_lines(
                    meas_line,
                    assign_diag,
                    assign_method=assign_plot,
                    n_meas=n_meas,
                    n_plan_kept=n_plan_kept,
                )
            if measured and aggregate_plot:
                _cap = p.measured_spot_weight_caption(ctx.spot_weight_mode_run)
                meas_line += f" (one {_cap}-weighted mean per assigned plan spot)"
            elif measured and ctx.aggregate_spots and not aggregate_plot:
                meas_line += " (aggregation off for this load)"
            if align_caption:
                meas_line += f". {align_caption}"
            if cb_pqa.isChecked():
                if ctx.qa_mode == "dose":
                    meas_line += (
                        f". Dose QA: |Δ|≤{ctx.qa_pass_f:g} pp pass, "
                        f"{ctx.qa_pass_f:g}<|Δ|≤{ctx.qa_warn_f:g} pp warn, "
                        f"|Δ|>{ctx.qa_warn_f:g} pp fail"
                    )
                else:
                    meas_line += (
                        f". Position QA: d≤{ctx.qa_pass_f:g} mm pass, "
                        f"{ctx.qa_pass_f:g}<d≤{ctx.qa_warn_f:g} mm warn, "
                        f"d>{ctx.qa_warn_f:g} mm fail"
                    )
                if cb_qa_hide.isChecked():
                    meas_line += "; pass-tier spots hidden in 3D"
                if cb_qa_lines.isChecked() and ctx.qa_mode == "position":
                    meas_line += "; error lines for warn+fail"
            if cb_meas_sigma.isChecked():
                meas_line += ". Measured: σ-sized ellipsoids in world XY (mm; see 3D caption)"
            parts = [plan_line, meas_line]
            status_lbl.setText(f"Updated. {'. '.join(parts)}.{cache_note}")
            _update_plan_qa_counts_label(
                planned,
                measured,
                qa_mode=ctx.qa_mode,
                qa_pass_f=ctx.qa_pass_f,
                qa_warn_f=ctx.qa_warn_f,
                plan_mu=_plot_cache.get("plan_mu"),
            )
            persist()
            pending_ctx = None

        def _on_pipeline_load_finished(res: object, generation: int) -> None:
            nonlocal pending_ctx, load_generation
            if generation != load_generation:
                return
            if not isinstance(res, PipelineLoadOK):
                _hide_loading(generation)
                return
            ctx0 = pending_ctx
            if ctx0 is None:
                _hide_loading(generation)
                return
            ld = res
            _plot_cache["pipeline_key"] = ld.pipeline_key
            _plot_cache["planned"] = ld.planned if ld.planned else []
            _plot_cache["plan_fwhm_xy"] = ld.plan_fwhm_xy
            _plot_cache["plan_mu"] = ld.plan_mu
            _plot_cache["n_plan_kept"] = ld.n_plan_kept
            _plot_cache["n_plan_raw"] = ld.n_plan_raw
            _plot_cache["measured_unaligned"] = (
                ld.measured_unaligned if ld.measured_unaligned else []
            )
            _plot_cache["measured_fine_aligned"] = (
                ld.measured_fine_aligned if ld.measured_fine_aligned else None
            )
            _plot_cache["coarse_flat_align_info"] = ld.coarse_flat_align_info
            _plot_cache["fine_align_info"] = ld.fine_align_info
            _plot_cache["label"] = ld.label
            _plot_cache["csv_display_name"] = ld.csv_display_name
            _plot_cache["layer_mode_run"] = ld.layer_mode_run
            _plot_cache["auto_assign_method"] = ld.auto_assign_method
            _plot_cache["aggregate_run"] = ld.aggregate_run
            _plot_cache["assign_diagnostics"] = ld.assign_diagnostics
            _plot_cache["plan_spots_no_data"] = ld.plan_spots_no_data
            _plot_cache["plan_spot_time_s"] = ld.plan_spot_time_s
            build_target = ld.csv_display_name or ld.label or "plan"
            build_note = f"Building 3D view — {build_target}…"
            load_overlay.set_visualize_phase(message=build_note)
            status_lbl.setText(build_note)
            try:
                _finalize_refresh(ctx0, need_data=True)
            except RuntimeError as e:
                _plot_cache["plotter"] = None
                _idle_view_slice_controls_qt()
                _vtk_placeholder_message(f"PyVista: {e}", error=True)
                status_lbl.setText(f"PyVista: {e}")
                logger.warning("PyVista runtime error during 3D refresh: %s", e)
            except Exception as e:
                _plot_cache["plotter"] = None
                _idle_view_slice_controls_qt()
                _vtk_placeholder_message(f"Error: {e}", error=True)
                status_lbl.setText(f"Error: {e}")
                logger.exception("Unexpected error during 3D refresh")
            finally:
                _hide_loading(generation)

        def _on_pipeline_load_failed(msg: str, generation: int) -> None:
            nonlocal pending_ctx, load_generation
            if generation != load_generation:
                return
            pending_ctx = None
            _plot_cache["pipeline_key"] = None
            _plot_cache["measured_unaligned"] = None
            _plot_cache["measured_fine_aligned"] = None
            _plot_cache["coarse_flat_align_info"] = None
            _plot_cache["fine_align_info"] = None
            _plot_cache["planned"] = None
            _plot_cache["plan_fwhm_xy"] = None
            _plot_cache["plan_mu"] = None
            _plot_cache["layer_mode_run"] = ""
            _plot_cache["auto_assign_method"] = "episodes"
            _plot_cache["aggregate_run"] = False
            _plot_cache["assign_diagnostics"] = None
            _plot_cache["plan_spots_no_data"] = None
            _plot_cache["plan_spot_time_s"] = None
            _plot_cache["qa_result"] = None
            _plot_cache["plotter"] = None
            _plot_cache["fine_align_ui"] = None
            _idle_view_slice_controls_qt()
            short = msg if len(msg) < 160 else f"{msg[:157]}…"
            _vtk_placeholder_message(short, error=True)
            status_lbl.setText(short)
            _hide_loading(generation)

        def export_combined_csv() -> None:
            dcm_s = e_dcm.text().strip()
            csv_s = e_csv.text().strip()
            if not dcm_s or not csv_s:
                status_lbl.setText("Export: set plan and acquisition CSV paths first.")
                return
            dcm = Path(dcm_s)
            csv_path = Path(csv_s)
            if not dcm.is_file():
                status_lbl.setText(f"Export: plan not found: {dcm}")
                return
            if not csv_path.is_file():
                status_lbl.setText(f"Export: CSV not found: {csv_path}")
                return

            layer_assign_mode = _layer_assign_mode_from_ui()
            sw_lbl_run = combo_sw.currentText().strip()
            sw_internal_run = _SW_MODE_BY_LABEL.get(sw_lbl_run, sc_const.SPOT_WEIGHT_MODE_DEFAULT)
            try:
                spot_weight_mode_run = analysis.normalize_measured_spot_weight_mode(
                    sw_internal_run
                )
            except ValueError as ex:
                status_lbl.setText(f"Export: {ex}")
                return

            aggregate_spots = bool(cb_agg.isChecked())
            try:
                config = PipelineConfig(
                    plan_path=dcm,
                    csv_path=csv_path,
                    layer_assign_mode=layer_assign_mode,
                    aggregate_spots=aggregate_spots,
                    spot_weight_mode=spot_weight_mode_run,
                    coarse_flat_align=bool(cb_align.isChecked()),
                    heal_partial_fit_axes=bool(cb_heal_partial.isChecked()),
                    fine_align_xy=bool(cb_fine_xy.isChecked()),
                    fine_align_rotation=bool(cb_fine_rot.isChecked()),
                    fine_align_scale=bool(cb_fine_scale.isChecked()),
                    filter_xy_fliers=bool(cb_filter_xy.isChecked()),
                    filter_xy_flier_sigma=float(
                        parse_filter_xy_flier_sigma(e_filter_sigma.text())
                        or sc_const.FILTER_XY_FLIER_SIGMA_DEFAULT
                    ),
                )
                ok, measured = pipeline_export_load(config)
                planned = ok.planned
                plan_mu = ok.plan_mu
                label = ok.label
                layer_mode_run = ok.layer_mode_run
                aggregate_run = ok.aggregate_run
                aligned_to_plan = bool(
                    planned
                    and (
                        ok.measured_fine_aligned is not None
                        or ok.coarse_flat_align_info is not None
                    )
                )
                suffix = "-spots-agg.csv" if aggregate_run else "-spots.csv"
                default_out = csv_path.with_name(acquisition_csv_stem(csv_path) + suffix)
                out_path, _ = QFileDialog.getSaveFileName(
                    win,
                    "Export combined CSV",
                    str(default_out),
                    "CSV (*.csv)",
                )
                if not out_path:
                    return
                rows = build_combined_export_rows(
                    planned,
                    plan_mu,
                    measured,
                    aggregated=aggregate_run,
                    positions_aligned_to_plan=aligned_to_plan,
                )
                write_combined_export_csv(
                    Path(out_path),
                    rows,
                    metadata={
                        "rt_plan_label": label,
                        "acquisition_csv": csv_path.name,
                        "layer_mode": layer_mode_run,
                        "aggregate_spots": "yes" if aggregate_run else "no",
                        "spot_weight_mode": spot_weight_mode_run,
                        "aligned_measured_xy": "yes" if aligned_to_plan else "no",
                    },
                )
                status_lbl.setText(f"Exported {len(rows)} spot row(s) to {out_path}")
            except Exception as ex:
                status_lbl.setText(f"Export failed: {ex}")
                logger.exception("Combined CSV export failed")

        btn_export_csv.clicked.connect(export_combined_csv)

        def _bump_load_generation_invalidate_async() -> None:
            nonlocal load_generation, pending_ctx
            load_generation += 1
            pending_ctx = None

        def _do_refresh() -> None:
            nonlocal pending_ctx, load_generation
            persist()
            p = analysis
            plan_text = e_dcm.text().strip()
            csv_text = e_csv.text().strip()
            plan_path = Path(plan_text) if plan_text else None
            csv_path = Path(csv_text) if csv_text else None
            has_plan = plan_path is not None and analysis.is_supported_plan_file(plan_path)
            has_csv = csv_path is not None and is_acquisition_csv_file(csv_path)

            if not has_plan and not has_csv:
                _bump_load_generation_invalidate_async()
                _plot_cache["plotter"] = None
                _plot_cache["pipeline_key"] = None
                _plot_cache["planned"] = None
                _plot_cache["measured_unaligned"] = None
                _plot_cache["measured_fine_aligned"] = None
                _plot_cache["coarse_flat_align_info"] = None
                _plot_cache["fine_align_info"] = None
                _idle_view_slice_controls_qt()
                _vtk_placeholder_message(
                    "Select a plan (.dcm or Pyramid .csv) and/or acquisition .csv.",
                    error=False,
                )
                status_lbl.setText("Ready — pick plan and/or CSV.")
                return
            filter_sigma_use = parse_filter_xy_flier_sigma(e_filter_sigma.text())
            if cb_filter_xy.isChecked() and filter_sigma_use is None:
                if filter_xy_flier_sigma_input_in_progress(e_filter_sigma.text()):
                    filter_sigma_use = float(sc_const.FILTER_XY_FLIER_SIGMA_DEFAULT)
                else:
                    _bump_load_generation_invalidate_async()
                    _idle_view_slice_controls_qt()
                    _vtk_placeholder_message(
                        f"XY flier σ: {sc_const.FILTER_XY_FLIER_SIGMA_MIN:g}–"
                        f"{sc_const.FILTER_XY_FLIER_SIGMA_MAX:g}.",
                        error=True,
                    )
                    status_lbl.setText("Fix XY flier σ limit.")
                    return
            elif filter_sigma_use is None:
                filter_sigma_use = float(sc_const.FILTER_XY_FLIER_SIGMA_DEFAULT)
            qa_mode_run = _current_plan_qa_mode()
            qa_pass_f = float(
                sc_const.PLAN_QA_DOSE_PASS_PP_DEFAULT
                if qa_mode_run == "dose"
                else sc_const.PLAN_QA_PASS_MM_DEFAULT
            )
            qa_warn_f = float(
                sc_const.PLAN_QA_DOSE_WARN_PP_DEFAULT
                if qa_mode_run == "dose"
                else sc_const.PLAN_QA_WARN_MM_DEFAULT
            )
            if cb_pqa.isChecked():
                qa_pair = parse_plan_qa_thresholds(e_qa_pass.text(), e_qa_warn.text())
                if qa_pair is None:
                    if plan_qa_thresholds_input_in_progress(
                        e_qa_pass.text(), e_qa_warn.text()
                    ):
                        qa_pass_f, qa_warn_f = _qa_thr_by_mode[qa_mode_run]
                    else:
                        _bump_load_generation_invalidate_async()
                        _idle_view_slice_controls_qt()
                        unit = "pp" if qa_mode_run == "dose" else "mm"
                        _vtk_placeholder_message(
                            f"QA: 0 < pass < warn ≤ 500 ({unit}).",
                            error=True,
                        )
                        status_lbl.setText(f"Fix QA pass/warn ({unit}).")
                        return
                else:
                    qa_pass_f, qa_warn_f = qa_pair
                    _qa_thr_by_mode[qa_mode_run] = (float(qa_pass_f), float(qa_warn_f))
            layer_assign_mode = _layer_assign_mode_from_ui()
            try:
                sw_lbl_run = combo_sw.currentText().strip()
                sw_internal_run = _SW_MODE_BY_LABEL.get(
                    sw_lbl_run, sc_const.SPOT_WEIGHT_MODE_DEFAULT
                )
                spot_weight_mode_run = p.normalize_measured_spot_weight_mode(sw_internal_run)

                pipeline_key = (
                    str(plan_path.resolve()) if has_plan and plan_path else "",
                    file_mtime(plan_path) if has_plan and plan_path else -1.0,
                    str(csv_path.resolve()) if has_csv and csv_path else "",
                    file_mtime(csv_path) if has_csv and csv_path else -1.0,
                    layer_assign_mode,
                    bool(cb_agg.isChecked()),
                    spot_weight_mode_run,
                    bool(cb_heal_partial.isChecked()),
                    bool(cb_filter_xy.isChecked()),
                    float(filter_sigma_use),
                )
                ctx = GuiRefreshContext(
                    plan_path=plan_path if has_plan else None,
                    csv_path=csv_path if has_csv else None,
                    qa_mode=qa_mode_run,
                    qa_pass_f=qa_pass_f,
                    qa_warn_f=qa_warn_f,
                    layer_assign_mode=layer_assign_mode,
                    aggregate_spots=bool(cb_agg.isChecked()),
                    spot_weight_mode_run=spot_weight_mode_run,
                    pipeline_key=pipeline_key,
                )
                need_data = (
                    pipeline_key != _plot_cache["pipeline_key"]
                    or _plot_cache["planned"] is None
                    or (
                        has_csv
                        and _plot_cache["measured_unaligned"] is None
                    )
                )

                if need_data:
                    load_generation += 1
                    gen = load_generation
                    pending_ctx = ctx
                    load_parts: list[str] = []
                    if has_plan and plan_path is not None:
                        load_parts.append(plan_path.name)
                    if has_csv and csv_path is not None:
                        load_parts.append(csv_path.name)
                    load_note = f"Loading {' + '.join(load_parts)}…"
                    if cb_align.isChecked() and has_plan and has_csv:
                        load_note = f"Loading + aligning {' + '.join(load_parts)}…"
                    status_lbl.setText(load_note)
                    _show_loading(gen, load_note)

                    def _job() -> PipelineLoadOK:
                        gen_local = gen

                        def _on_progress(event: ProgressEvent) -> None:
                            PipelineLoadRunnable.emit_progress(
                                load_signals, gen_local, event
                            )

                        sink = CallbackProgressSink(_on_progress)
                        return pipeline_load_job(
                            ctx.plan_path,
                            ctx.csv_path,
                            layer_assign_mode=ctx.layer_assign_mode,
                            aggregate_spots=ctx.aggregate_spots,
                            spot_weight_mode=ctx.spot_weight_mode_run,
                            coarse_flat_align=bool(cb_align.isChecked()),
                            heal_partial_fit_axes=bool(cb_heal_partial.isChecked()),
                            fine_align_xy=bool(cb_fine_xy.isChecked()),
                            fine_align_rotation=bool(cb_fine_rot.isChecked()),
                            fine_align_scale=bool(cb_fine_scale.isChecked()),
                            filter_xy_fliers=bool(cb_filter_xy.isChecked()),
                            filter_xy_flier_sigma=float(filter_sigma_use),
                            progress=sink,
                        )

                    load_pool.start(PipelineLoadRunnable(_job, load_signals, gen))
                    return

                planned = _plot_cache["planned"]
                measured_unaligned = _plot_cache["measured_unaligned"]
                if planned is None and measured_unaligned is None:
                    logger.error(
                        "Plot cache inconsistency: pipeline_key matched but no data cached"
                    )
                    _plot_cache["pipeline_key"] = None
                    _plot_cache["plotter"] = None
                    _idle_view_slice_controls_qt()
                    _vtk_placeholder_message(
                        "Display state reset — tweak an option or re-pick files.",
                        error=True,
                    )
                    status_lbl.setText("State error — change option or re-select files.")
                    return
                try:
                    _finalize_refresh(ctx, need_data=False)
                except RuntimeError as e:
                    _plot_cache["plotter"] = None
                    _idle_view_slice_controls_qt()
                    _vtk_placeholder_message(f"PyVista: {e}", error=True)
                    status_lbl.setText(f"PyVista: {e}")
                    logger.warning("PyVista runtime error during 3D refresh: %s", e)
                except Exception as e:
                    _plot_cache["plotter"] = None
                    _idle_view_slice_controls_qt()
                    _vtk_placeholder_message(f"Error: {e}", error=True)
                    status_lbl.setText(f"Error: {e}")
                    logger.exception("Unexpected error during 3D refresh")
            except Exception as e:
                _plot_cache["plotter"] = None
                _idle_view_slice_controls_qt()
                _vtk_placeholder_message(f"Error: {e}", error=True)
                status_lbl.setText(f"Error: {e}")
                logger.exception("Unexpected error during 3D refresh")

        load_signals.finished.connect(_on_pipeline_load_finished)
        load_signals.failed.connect(_on_pipeline_load_failed)
        load_signals.progress.connect(_on_pipeline_progress)
        _debounce.timeout.connect(_do_refresh)

        def _commit_field_refresh() -> None:
            _debounce.stop()
            _do_refresh()

        for w in (
            e_dcm,
            e_csv,
            e_wet,
        ):
            w.textChanged.connect(_schedule_refresh)
            w.editingFinished.connect(_commit_field_refresh)
        # QA thresholds: refresh on commit only so values like 0.5 can be typed without
        # debounced validation firing on intermediate "0" or "0.".
        for w in (e_qa_pass, e_qa_warn):
            w.editingFinished.connect(_apply_display_refresh)
        combo_sw.currentIndexChanged.connect(_apply_display_refresh)
        combo_z_depth.currentIndexChanged.connect(_do_refresh)
        cb_weight_ch.toggled.connect(_apply_display_refresh)
        cb_plan_fwhm.toggled.connect(_apply_display_refresh)
        cb_meas_sigma.toggled.connect(_apply_display_refresh)
        cb_z_water.toggled.connect(lambda _c: (_sync_wet_shifter_ui(), _do_refresh()))
        cb_align.toggled.connect(_do_refresh)
        cb_fine_xy.toggled.connect(_do_refresh)
        cb_fine_rot.toggled.connect(_do_refresh)
        cb_fine_scale.toggled.connect(_do_refresh)
        cb_agg.toggled.connect(lambda _c: (_update_help(), _do_refresh()))
        cb_heal_partial.toggled.connect(_do_refresh)
        cb_filter_xy.toggled.connect(lambda _c: (_sync_filter_xy_ui(), _do_refresh()))
        e_filter_sigma.editingFinished.connect(_do_refresh)
        cb_pqa.toggled.connect(lambda _c: (_sync_qa_lines(), _apply_display_refresh()))

        def _on_qa_mode_changed() -> None:
            _stash_qa_thresholds()
            _apply_qa_mode_ui(refresh_fields=True)
            _apply_display_refresh()

        rb_qa_pos.toggled.connect(lambda _c: _on_qa_mode_changed() if _c else None)
        rb_qa_dose.toggled.connect(lambda _c: _on_qa_mode_changed() if _c else None)
        cb_qa_lines.toggled.connect(_apply_display_refresh)
        cb_qa_hide.toggled.connect(_apply_display_refresh)
        cb_view_proj.toggled.connect(_apply_projection_view)
        btn_view_top.clicked.connect(lambda: _apply_quick_view("top"))
        btn_view_left.clicked.connect(lambda: _apply_quick_view("left"))
        btn_view_right.clicked.connect(lambda: _apply_quick_view("right"))
        rb_auto_episodes.toggled.connect(lambda _c: (_update_help(), _do_refresh()))
        rb_auto_seq.toggled.connect(lambda _c: (_update_help(), _do_refresh()))
        rb_gate.toggled.connect(lambda _c: (_update_help(), _do_refresh()))
        slice_chk.toggled.connect(lambda _c: persist())
        slice_sli.sliderReleased.connect(persist)
        timeline_bar._slider.sliderReleased.connect(persist)
        timeline_bar._combo_speed.currentIndexChanged.connect(lambda _i: persist())
        timeline_bar._combo_window.currentIndexChanged.connect(lambda _i: persist())
        timeline_bar._chk_slice.toggled.connect(lambda _c: persist())

        app.aboutToQuit.connect(persist)

        win.show()
        finish_saved_window_layout(win, maximized=restore_maximized)
        _sync_filter_xy_ui()
        QTimer.singleShot(0, _do_refresh)
        sys.exit(app.exec())
