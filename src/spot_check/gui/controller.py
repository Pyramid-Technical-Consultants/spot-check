"""SpotCheck GUI controller — builds the main window and 3D refresh pipeline."""

from __future__ import annotations

import importlib.util
import logging
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
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QProgressBar,
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
from spot_check.gui.parsers import (
    normalize_z_depth_metric,
    parse_aggregate_even_tail_n,
    parse_bounds_xy_tick_mm,
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
    resolve_csv_load_layer_mode,
)
from spot_check.gui.state import (
    apply_saved_window_layout,
    finish_saved_window_layout,
    geom_from_win,
    load_gui_state,
    save_gui_state,
    win_is_maximized,
)
from spot_check.gui.theme import MUTED_BODY, MUTED_HELP, MUTED_HINT
from spot_check.gui.workers import PipelineLoaderSignals, PipelineLoadRunnable

FOLDER = project_root()
# Short pause after typing in numeric fields before recomputing the 3D view (ms).
_REFRESH_DEBOUNCE_MS = 380
_LOAD_SPINNER_FRAMES: tuple[str, ...] = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
_LOAD_SPINNER_MS = 90
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
        outer = QHBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(splitter)

        vtk_host = QFrame()
        vtk_host.setMinimumWidth(320)
        vtk_host.setStyleSheet("background-color: #0d1117;")
        vtk_layout = QVBoxLayout(vtk_host)
        vtk_layout.setContentsMargins(0, 0, 0, 0)
        vtk_placeholder = QLabel(
            "3D view — pick a plan (.dcm or Pyramid .csv) and/or acquisition .csv; "
            "updates when inputs validate."
        )
        vtk_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vtk_placeholder.setWordWrap(True)
        vtk_placeholder.setStyleSheet("color: #8b949e; font-size: 11pt; padding: 24px;")
        vtk_layout.addWidget(vtk_placeholder)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setMinimumWidth(360)
        drawer = QWidget()
        drawer_layout = QVBoxLayout(drawer)
        drawer_layout.addStretch(0)

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
        e_agg_even = QLineEdit(
            str(
                int(
                    saved.get(
                        "aggregate_even_rows_after_odd",
                        sc_const.AGGREGATE_EVEN_ROWS_AFTER_ODD_DEFAULT,
                    )
                )
            )
        )
        raw_bxy = saved.get("bounds_xy_tick_mm", sc_const.BOUNDS_XY_TICK_MM_DEFAULT)
        try:
            bxy0 = float(raw_bxy)
            if bxy0 != 0.0 and (bxy0 < 0.05 or bxy0 > 500.0):
                bxy0 = float(sc_const.BOUNDS_XY_TICK_MM_DEFAULT)
        except (TypeError, ValueError):
            bxy0 = float(sc_const.BOUNDS_XY_TICK_MM_DEFAULT)
        e_bxy = QLineEdit(f"{bxy0:g}")
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
        cb_align = QCheckBox("Rigid XY align measured → plan (any rotation / A↔B)")
        cb_align.setChecked(_bool_saved("auto_align_detector_xy", True))
        cb_agg = QCheckBox("One measured point per odd gate phase (weighted mean)")
        cb_agg.setChecked(_bool_saved("aggregate_spots_by_gate", True))
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
            xy_tick_save = parse_bounds_xy_tick_mm(e_bxy.text())
            if xy_tick_save is None:
                xy_tick_save = float(sc_const.BOUNDS_XY_TICK_MM_DEFAULT)
                e_bxy.setText(f"{xy_tick_save:g}")
            wet_save = parse_upstream_wet_shifter_mm(e_wet.text())
            if wet_save is None:
                wet_save = float(sc_const.UPSTREAM_WET_SHIFTER_MM_DEFAULT)
                e_wet.setText(f"{wet_save:g}")
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
            tail_n = parse_aggregate_even_tail_n(e_agg_even.text())
            if tail_n is None:
                tail_n = int(sc_const.AGGREGATE_EVEN_ROWS_AFTER_ODD_DEFAULT)
                e_agg_even.setText(str(tail_n))
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
            save_gui_state(
                dcm_path=e_dcm.text().strip(),
                csv_path=e_csv.text().strip(),
                window_geometry=geom_from_win(win),
                window_maximized=win_is_maximized(win),
                layer_assign_mode=mode,
                weight_measured_by_channel_sum=cb_weight_ch.isChecked(),
                spot_weight_mode=sw_mode_norm,
                aggregate_spots_by_gate=cb_agg.isChecked(),
                aggregate_even_rows_after_odd=int(tail_n),
                auto_align_detector_xy=cb_align.isChecked(),
                bounds_xy_tick_mm=float(xy_tick_save),
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

        def _sync_agg_even() -> None:
            gate_only = rb_gate.isChecked()
            e_agg_even.setEnabled(gate_only and cb_agg.isChecked())

        def _sync_layer_mode_ui() -> None:
            _sync_agg_even()

        def _update_help() -> None:
            lam = _layer_assign_mode_from_ui()
            _sync_layer_mode_ui()
            if lam == "auto_plan_sequential":
                help_lbl.setText(
                    "Auto plan-sequential: assign from the first plan spot (highest energy "
                    "layer). Deadtime = no fit on Fit Mean Position A or B. After each gap, "
                    "advance exactly one plan slot (never skip). Gate Counter ignored."
                )
                lbl_auto_tuning.setText(
                    "Advance one plan slot per deadtime break after rows on the current spot."
                )
                agg_intro_lbl.setText(
                    "Merge rows within each assigned plan spot to one weighted-mean row "
                    "(per-row when unchecked)."
                )
            elif lam == "auto":
                help_lbl.setText(
                    "Auto episodes: segment from timing, weight, and XY "
                    "(Gate Counter and Gate Signal ignored). "
                    "Thresholds are inferred and episodes aligned to the plan spot count."
                )
                lbl_auto_tuning.setText(
                    "Tuning: inferred per load — see status line after refresh."
                )
                agg_intro_lbl.setText(
                    "Merge rows within each signal episode to one weighted-mean spot "
                    "(per-row when unchecked)."
                )
            else:
                help_lbl.setText(
                    f'Gate: DICOM order; "{sc_const.GATE_COUNTER_KEY}" odd=spot, even=deadtime; '
                    "new odd advances."
                )
                agg_intro_lbl.setText(
                    f"Many rows per counter value — merge below for one XY per odd phase "
                    f"({sc_const.GATE_COUNTER_KEY})."
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

        # --- drawer sections ---
        gb_files = QGroupBox("Files")
        fl = QFormLayout(gb_files)
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

        gb_layer = QGroupBox("Layer assignment")
        vl_layer = QVBoxLayout(gb_layer)
        vl_layer.addWidget(rb_auto_episodes)
        vl_layer.addWidget(rb_auto_seq)
        vl_layer.addWidget(rb_gate)
        lbl_auto_tuning = QLabel(
            "Tuning: inferred from plan + CSV when Auto is selected."
        )
        lbl_auto_tuning.setWordWrap(True)
        lbl_auto_tuning.setStyleSheet(f"color: {MUTED_HINT}; font-size: 9pt;")
        vl_layer.addWidget(lbl_auto_tuning)
        vl_layer.addWidget(help_lbl)

        gb_disp = QGroupBox("Display")
        vdisp = QVBoxLayout(gb_disp)
        vdisp.addWidget(cb_weight_ch)
        sw_row = QHBoxLayout()
        sw_row.addWidget(QLabel("Weight:"))
        sw_row.addWidget(combo_sw, 1)
        wsw = QWidget()
        wsw.setLayout(sw_row)
        vdisp.addWidget(wsw)
        vdisp.addWidget(cb_plan_fwhm)
        vdisp.addWidget(cb_meas_sigma)
        z_wet_row = QHBoxLayout()
        z_wet_row.addWidget(cb_z_water)
        z_wet_row.addWidget(QLabel("Upstream WET (mm)"))
        z_wet_row.addWidget(e_wet)
        z_wet_row.addWidget(QLabel("Depth"))
        z_wet_row.addWidget(combo_z_depth)
        z_wet_row.addStretch(1)
        w_z_wet = QWidget()
        w_z_wet.setLayout(z_wet_row)
        vdisp.addWidget(w_z_wet)
        vdisp.addWidget(cb_align)
        rt = QHBoxLayout()
        rt.addWidget(QLabel("XY ticks (mm)"))
        rt.addWidget(e_bxy)
        rt.addWidget(_add_hint_lbl("0 = coarse; ~5 common"), 1)
        wtick = QWidget()
        wtick.setLayout(rt)
        vdisp.addWidget(wtick)

        gb_qa = QGroupBox("Plan QA")
        vqa = QVBoxLayout(gb_qa)
        vqa.addWidget(cb_pqa)
        qa_mode_row = QHBoxLayout()
        qa_mode_row.addWidget(rb_qa_pos)
        qa_mode_row.addWidget(rb_qa_dose)
        wqa_mode = QWidget()
        wqa_mode.setLayout(qa_mode_row)
        vqa.addWidget(wqa_mode)
        vqa.addWidget(cb_qa_lines)
        vqa.addWidget(cb_qa_hide)
        lbl_qa_pass = QLabel("Pass ≤ (mm)")
        lbl_qa_warn = QLabel("Warn ≤ (mm)")
        qa_th = QHBoxLayout()
        qa_th.addWidget(lbl_qa_pass)
        qa_th.addWidget(e_qa_pass)
        qa_th.addWidget(lbl_qa_warn)
        qa_th.addWidget(e_qa_warn)
        wqa = QWidget()
        wqa.setLayout(qa_th)
        vqa.addWidget(wqa)
        qa_hint_lbl = _add_hint_lbl("Need 0 < pass < warn.")
        vqa.addWidget(qa_hint_lbl)
        lbl_qa_counts = QLabel("Counts: —")
        lbl_qa_counts.setWordWrap(True)
        lbl_qa_counts.setStyleSheet(f"color: {MUTED_BODY}; font-size: 9pt;")
        lbl_qa_counts.setMinimumWidth(0)
        lbl_qa_counts.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        vqa.addWidget(lbl_qa_counts)
        _apply_qa_mode_ui(refresh_fields=False)

        def _update_plan_qa_counts_label(
            planned: list[tuple[float, float, float]] | None,
            measured: list[tuple[float, ...]] | None,
            *,
            qa_mode: str,
            qa_pass_f: float,
            qa_warn_f: float,
            plan_mu: object,
        ) -> None:
            if not cb_pqa.isChecked():
                lbl_qa_counts.setText(
                    "Counts: enable “Color measured spots by plan QA” above."
                )
                return
            if not planned or not measured:
                lbl_qa_counts.setText("Counts: — (need plan and CSV)")
                return
            try:
                n_pass, n_warn, n_fail = analysis.plan_qa_measured_spot_pass_warn_fail(
                    planned,
                    measured,
                    qa_mode=qa_mode,
                    pass_thr=float(qa_pass_f),
                    warn_thr=float(qa_warn_f),
                    plan_mu=plan_mu,  # type: ignore[arg-type]
                    a_is_x=False,
                )
            except ValueError:
                lbl_qa_counts.setText("Counts: — (check pass / warn thresholds)")
                return
            unit = "pp" if qa_mode == "dose" else "mm"
            lbl_qa_counts.setText(
                f"Measured spots: {n_pass} pass · {n_warn} warn · {n_fail} fail "
                f"({unit} thresholds)"
            )

        gb_slice = QGroupBox("5-layer band")
        vsl = QVBoxLayout(gb_slice)
        vsl.addWidget(slice_chk)
        vsl.addWidget(slice_sli)
        vsl.addWidget(slice_status)

        gb_view3d = QGroupBox("3D view")
        vview3d = QVBoxLayout(gb_view3d)
        view_proj_row = QHBoxLayout()
        view_proj_row.addWidget(cb_view_proj, 1)
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
        vview3d.addLayout(view_proj_row)

        gb_agg = QGroupBox("Aggregation")
        vag = QVBoxLayout(gb_agg)
        vag.addWidget(agg_intro_lbl)
        vag.addWidget(cb_agg)
        erow = QHBoxLayout()
        erow.addWidget(QLabel("Merge ≤ even rows"))
        erow.addWidget(e_agg_even)
        erow.addWidget(
            _add_hint_lbl(
                "Even rows after odd→even, good fits; 0=off; max "
                f"{sc_const.AGGREGATE_EVEN_TAIL_MAX}"
            ),
            1,
        )
        we = QWidget()
        we.setLayout(erow)
        vag.addWidget(we)

        drawer_layout.addWidget(gb_files)
        drawer_layout.addWidget(gb_layer)
        drawer_layout.addWidget(gb_disp)
        drawer_layout.addWidget(gb_qa)
        drawer_layout.addWidget(gb_slice)
        drawer_layout.addWidget(gb_view3d)
        drawer_layout.addWidget(gb_agg)

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
            "measured_aligned": None,
            "align_info": None,
            "align_cache_key": None,
            "label": "",
            "csv_display_name": "",
            "layer_mode_run": "",
            "auto_assign_method": "episodes",
            "plotter": None,
            "aligned": None,
            "z_water_depth": False,
            "upstream_wet_mm": 0.0,
            "z_depth_metric": sc_const.Z_DEPTH_METRIC_DEFAULT,
            "slice_on": bool(slice_chk.isChecked()),
            "slice_center_i": int(slice_sli.value()),
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

        load_generation = 0
        load_pool = QThreadPool(win)
        load_pool.setMaxThreadCount(1)
        load_signals = PipelineLoaderSignals(win)
        pending_ctx: GuiRefreshContext | None = None
        _loading_gen: int | None = None
        _spinner_frame = 0

        load_overlay = QFrame(vtk_host)
        load_overlay.setObjectName("loadOverlay")
        load_overlay.setStyleSheet("#loadOverlay { background-color: rgba(13, 17, 23, 0.45); }")
        load_overlay.setVisible(False)
        load_overlay_lay = QVBoxLayout(load_overlay)
        load_overlay_lay.setContentsMargins(0, 0, 0, 0)
        load_overlay_lay.addStretch(1)
        load_overlay_row = QHBoxLayout()
        load_overlay_row.addStretch(1)
        load_overlay_card = QFrame()
        load_overlay_card.setObjectName("loadOverlayCard")
        load_overlay_card.setStyleSheet(
            "#loadOverlayCard { background-color: rgba(22, 27, 34, 0.96); "
            "border: 1px solid #30363d; border-radius: 8px; }"
        )
        load_overlay_card_lay = QVBoxLayout(load_overlay_card)
        load_overlay_card_lay.setContentsMargins(20, 16, 20, 16)
        load_overlay_card_lay.setSpacing(10)
        load_overlay_spinner = QLabel(_LOAD_SPINNER_FRAMES[0])
        load_overlay_spinner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        load_overlay_spinner.setStyleSheet("color: #58a6ff; font-size: 36pt;")
        load_overlay_msg = QLabel("Loading plan/CSV…")
        load_overlay_msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        load_overlay_msg.setWordWrap(True)
        load_overlay_msg.setStyleSheet("color: #c9d1d9; font-size: 12pt; font-weight: 600;")
        load_overlay_bar = QProgressBar()
        load_overlay_bar.setRange(0, 0)
        load_overlay_bar.setFixedHeight(6)
        load_overlay_bar.setTextVisible(False)
        load_overlay_bar.setStyleSheet(
            "QProgressBar { background-color: #21262d; border: none; border-radius: 3px; }"
            "QProgressBar::chunk { background-color: #58a6ff; border-radius: 3px; }"
        )
        load_overlay_card_lay.addWidget(load_overlay_spinner)
        load_overlay_card_lay.addWidget(load_overlay_msg)
        load_overlay_card_lay.addWidget(load_overlay_bar)
        load_overlay_row.addWidget(load_overlay_card)
        load_overlay_row.addStretch(1)
        load_overlay_lay.addLayout(load_overlay_row)
        load_overlay_lay.addStretch(1)

        def _sync_load_overlay_geometry() -> None:
            host_w = max(1, int(vtk_host.width()))
            host_h = max(1, int(vtk_host.height()))
            load_overlay.setGeometry(0, 0, host_w, host_h)
            card_w = max(160, int(host_w * 0.30))
            load_overlay_card.setFixedWidth(card_w)
            load_overlay.raise_()

        class _LoadOverlayResizeFilter(QObject):
            def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
                if event.type() == QEvent.Type.Resize:
                    _sync_load_overlay_geometry()
                return False

        _load_overlay_filter = _LoadOverlayResizeFilter(win)
        vtk_host.installEventFilter(_load_overlay_filter)

        _load_spinner_timer = QTimer(win)
        _load_spinner_timer.setInterval(_LOAD_SPINNER_MS)

        def _tick_load_spinner() -> None:
            nonlocal _spinner_frame
            _spinner_frame = (_spinner_frame + 1) % len(_LOAD_SPINNER_FRAMES)
            load_overlay_spinner.setText(_LOAD_SPINNER_FRAMES[_spinner_frame])

        _load_spinner_timer.timeout.connect(_tick_load_spinner)

        def _show_loading(generation: int, message: str) -> None:
            nonlocal _loading_gen, _spinner_frame
            _loading_gen = int(generation)
            _spinner_frame = 0
            load_overlay_spinner.setText(_LOAD_SPINNER_FRAMES[0])
            load_overlay_msg.setText(message)
            _sync_load_overlay_geometry()
            load_overlay.show()
            load_overlay.raise_()
            if not _load_spinner_timer.isActive():
                _load_spinner_timer.start()
            QApplication.processEvents()

        def _hide_loading(generation: int) -> None:
            nonlocal _loading_gen
            if _loading_gen is None or int(generation) != int(_loading_gen):
                return
            _loading_gen = None
            _load_spinner_timer.stop()
            load_overlay.hide()

        def _vtk_placeholder_message(text: str, *, error: bool = False) -> None:
            clr = "#f85149" if error else "#8b949e"
            fn = getattr(analysis, "_clear_qt_layout_items", None)
            if fn is not None:
                fn(vtk_host)
            lay = vtk_host.layout()
            if lay is None:
                lay = QVBoxLayout(vtk_host)
                lay.setContentsMargins(0, 0, 0, 0)
                vtk_host.setLayout(lay)
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
                analysis.idle_slice_band_controls_qt(slice_qt_bindings)
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
                analysis.idle_slice_band_controls_qt(slice_qt_bindings)
                _vtk_placeholder_message(
                    "Select a plan (.dcm or Pyramid .csv) and/or acquisition .csv.",
                    error=False,
                )
                status_lbl.setText("Ready — pick plan and/or CSV.")
                return

            measured = list(measured_unaligned)
            detector_align_caption: str | None = None
            align_note = ""
            align_cache_key = (pipeline_key, bool(cb_align.isChecked()))
            if cb_align.isChecked() and planned and measured:
                cached_aligned = _plot_cache.get("measured_aligned")
                cached_align_key = _plot_cache.get("align_cache_key")
                if cached_aligned is not None and cached_align_key == align_cache_key:
                    measured = list(cached_aligned)
                    info0 = _plot_cache.get("align_info")
                    if info0 is not None:
                        detector_align_caption = p.format_detector_align_caption(info0)
                else:
                    try:
                        measured, align_info = p.align_measured_to_plan_detector_xy(
                            planned, measured, a_is_x=False
                        )
                        _plot_cache["measured_aligned"] = measured
                        _plot_cache["align_info"] = align_info
                        _plot_cache["align_cache_key"] = align_cache_key
                        detector_align_caption = p.format_detector_align_caption(align_info)
                    except ValueError as ex:
                        align_note = f" Detector alignment skipped: {ex}"
            else:
                _plot_cache["align_cache_key"] = align_cache_key
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

            layer_mode_req, _, _ = resolve_layer_assign_mode(ctx.layer_assign_mode)
            layer_mode_plot = str(_plot_cache.get("layer_mode_run") or layer_mode_req)
            aggregate_plot = ctx.aggregate_spots and layer_mode_plot in ("auto", "gate_counter")

            auto_gap: float | None = None
            auto_xy: float | None = None
            auto_vp: float | None = None
            assign_plot = str(_plot_cache.get("auto_assign_method") or "episodes")
            if layer_mode_plot == "auto" and assign_plot == "episodes":
                auto_p = analysis.last_auto_layer_params()
                if auto_p is not None:
                    auto_gap = auto_p.episode_gap_s
                    auto_xy = auto_p.spot_xy_jump_mm
                    auto_vp = auto_p.viterbi_advance_penalty_mm2

            reuse_pl = _plot_cache.get("plotter")
            wet_use = float(sc_const.UPSTREAM_WET_SHIFTER_MM_DEFAULT)
            metric_use = _current_z_depth_metric()
            if cb_z_water.isChecked():
                wet_parsed = parse_upstream_wet_shifter_mm(e_wet.text())
                if wet_parsed is None:
                    _bump_load_generation_invalidate_async()
                    analysis.idle_slice_band_controls_qt(slice_qt_bindings)
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
                and _plot_cache.get("aligned") == cb_align.isChecked()
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
                aggregate_even_rows_after_odd=int(ctx.agg_even_n),
                spot_weight_mode=ctx.spot_weight_mode_run,
                detector_align_caption=detector_align_caption,
                bounds_xy_tick_mm=ctx.xy_tick_use,
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
                embed_qt=vtk_host,
                slice_qt=slice_qt_bindings,
                slice_band_init=slice_band_init,
            )
            _plot_cache["plotter"] = pl
            _plot_cache["aligned"] = cb_align.isChecked()
            _plot_cache["z_water_depth"] = cb_z_water.isChecked()
            _plot_cache["upstream_wet_mm"] = wet_use
            _plot_cache["z_depth_metric"] = metric_use
            try:
                _plot_cache["slice_on"] = bool(slice_chk.isChecked())
                if slice_sli.isEnabled():
                    _plot_cache["slice_center_i"] = int(slice_sli.value())
            except (AttributeError, TypeError, ValueError):
                _plot_cache["slice_on"] = bool(slice_band_init["slice_on"])
                _plot_cache["slice_center_i"] = int(slice_band_init["center_i"])
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
                auto_p = analysis.last_auto_layer_params()
                if auto_p is not None:
                    if assign_plot == "plan_sequential":
                        meas_line += " — plan-seq: +1 plan slot per deadtime break"
                        lbl_auto_tuning.setText(
                            "Deadtime = no A/B position fit; advance one plan slot per break "
                            f"(≥{auto_p.min_episode_rows} row(s) on current spot first)."
                        )
                    else:
                        meas_line += (
                            f" — auto Δt≥{auto_p.episode_gap_s:g} s, "
                            f"XY>{auto_p.spot_xy_jump_mm:g} mm, "
                            f"Viterbi {auto_p.viterbi_advance_penalty_mm2:g} mm²"
                        )
                        lbl_auto_tuning.setText(
                            "Inferred: "
                            f"Δt {auto_p.episode_gap_s:g} s · XY {auto_p.spot_xy_jump_mm:g} mm · "
                            f"weight ≥{auto_p.min_on_spot_weight_na:.3g} nA · "
                            f"≥{auto_p.min_episode_rows} row(s)/episode · "
                            f"Viterbi {auto_p.viterbi_advance_penalty_mm2:g} mm²"
                        )
                        diag_auto = analysis.last_auto_episode_diagnostics()
                        if diag_auto is not None:
                            meas_line += (
                                f" — episodes raw={diag_auto.n_raw_episodes}, "
                                f"aligned={diag_auto.n_after_align}/{diag_auto.n_plan}"
                            )
                            if not diag_auto.count_align_ok:
                                meas_line += " (spot-count align incomplete)"
            if measured and aggregate_plot:
                _cap = p.measured_spot_weight_caption(ctx.spot_weight_mode_run)
                if layer_mode_plot == "auto":
                    if assign_plot == "plan_sequential":
                        meas_line += f" (one {_cap}-weighted mean per plan spot)"
                    else:
                        meas_line += f" (one {_cap}-weighted mean per signal episode)"
                else:
                    meas_line += f" (one {_cap}-weighted mean per odd gate spot)"
                if ctx.agg_even_n > 0 and layer_mode_plot == "gate_counter":
                    meas_line += (
                        f"; up to {ctx.agg_even_n} good even-phase row(s) merged after odd→even"
                    )
            if cb_align.isChecked() and detector_align_caption:
                meas_line += f". {detector_align_caption}"
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
            status_lbl.setText(f"Updated. {'. '.join(parts)}.{align_note}{cache_note}")
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
            _plot_cache["measured_aligned"] = ld.measured_aligned
            _plot_cache["align_info"] = ld.align_info
            _plot_cache["align_cache_key"] = (
                (ld.pipeline_key, True) if ld.measured_aligned is not None else None
            )
            _plot_cache["label"] = ld.label
            _plot_cache["csv_display_name"] = ld.csv_display_name
            _plot_cache["layer_mode_run"] = ld.layer_mode_run
            _plot_cache["auto_assign_method"] = ld.auto_assign_method
            build_target = ld.csv_display_name or ld.label or "plan"
            build_note = f"Building 3D view — {build_target}…"
            load_overlay_msg.setText(build_note)
            status_lbl.setText(build_note)
            try:
                _finalize_refresh(ctx0, need_data=True)
            except RuntimeError as e:
                _plot_cache["plotter"] = None
                analysis.idle_slice_band_controls_qt(slice_qt_bindings)
                _vtk_placeholder_message(f"PyVista: {e}", error=True)
                status_lbl.setText(f"PyVista: {e}")
                logger.warning("PyVista runtime error during 3D refresh: %s", e)
            except Exception as e:
                _plot_cache["plotter"] = None
                analysis.idle_slice_band_controls_qt(slice_qt_bindings)
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
            _plot_cache["measured_aligned"] = None
            _plot_cache["align_info"] = None
            _plot_cache["align_cache_key"] = None
            _plot_cache["planned"] = None
            _plot_cache["plan_fwhm_xy"] = None
            _plot_cache["plan_mu"] = None
            _plot_cache["layer_mode_run"] = ""
            _plot_cache["auto_assign_method"] = "episodes"
            _plot_cache["plotter"] = None
            analysis.idle_slice_band_controls_qt(slice_qt_bindings)
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
            layer_mode_req, auto_assign_method, auto_infer = resolve_layer_assign_mode(
                layer_assign_mode
            )

            agg_even_n = parse_aggregate_even_tail_n(e_agg_even.text())
            if agg_even_n is None:
                status_lbl.setText(
                    f"Export: even-row merge must be 0–{sc_const.AGGREGATE_EVEN_TAIL_MAX}."
                )
                return

            sw_lbl_run = combo_sw.currentText().strip()
            sw_internal_run = _SW_MODE_BY_LABEL.get(sw_lbl_run, sc_const.SPOT_WEIGHT_MODE_DEFAULT)
            try:
                spot_weight_mode_run = analysis.normalize_measured_spot_weight_mode(sw_internal_run)
            except ValueError as ex:
                status_lbl.setText(f"Export: {ex}")
                return

            aggregate_spots = bool(cb_agg.isChecked())
            try:
                planned = _plot_cache.get("planned")
                plan_mu = _plot_cache.get("plan_mu")
                if planned is None or plan_mu is None:
                    planned, _, plan_mu, _, _ = analysis.planned_spot_xyz_and_counts_from_plan(dcm)
                label = str(_plot_cache.get("label") or "")
                layer_mode_run, aggregate_run = resolve_csv_load_layer_mode(
                    layer_mode=layer_mode_req,
                    plan_path=dcm,
                    csv_path=csv_path,
                    aggregate_spots=aggregate_spots,
                )
                measured = analysis.measured_spot_abc_from_csv(
                    csv_path,
                    max_points=None,
                    planned_xyz=planned,
                    a_is_x=False,
                    layer_mode=layer_mode_run,
                    aggregate_spots=aggregate_run,
                    aggregate_even_rows_after_odd=int(agg_even_n),
                    spot_weight_mode=spot_weight_mode_run,
                    auto_infer_params=auto_infer and layer_mode_run == "auto",
                    auto_assign_method=auto_assign_method,
                )
                if not measured:
                    status_lbl.setText("Export: no measured rows.")
                    return
                aligned = bool(cb_align.isChecked())
                if aligned:
                    measured, _align_info = analysis.align_measured_to_plan_detector_xy(
                        planned,
                        measured,
                        a_is_x=False,
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
                    positions_aligned_to_plan=aligned,
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
                        "aligned_measured_xy": "yes" if aligned else "no",
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
                _plot_cache["measured_aligned"] = None
                analysis.idle_slice_band_controls_qt(slice_qt_bindings)
                _vtk_placeholder_message(
                    "Select a plan (.dcm or Pyramid .csv) and/or acquisition .csv.",
                    error=False,
                )
                status_lbl.setText("Ready — pick plan and/or CSV.")
                return
            xy_tick_use = parse_bounds_xy_tick_mm(e_bxy.text())
            if xy_tick_use is None:
                _bump_load_generation_invalidate_async()
                analysis.idle_slice_band_controls_qt(slice_qt_bindings)
                _vtk_placeholder_message(
                    "XY ticks: 0 or spacing 0.05–500 mm.",
                    error=True,
                )
                status_lbl.setText("Fix XY ticks (mm).")
                return
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
                        analysis.idle_slice_band_controls_qt(slice_qt_bindings)
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
            agg_even_n = parse_aggregate_even_tail_n(e_agg_even.text())
            if agg_even_n is None:
                _bump_load_generation_invalidate_async()
                analysis.idle_slice_band_controls_qt(slice_qt_bindings)
                _vtk_placeholder_message(
                    f"Even merge: integer 0–{sc_const.AGGREGATE_EVEN_TAIL_MAX}.",
                    error=True,
                )
                status_lbl.setText("Fix even-row merge count.")
                return
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
                    int(agg_even_n),
                    spot_weight_mode_run,
                )
                ctx = GuiRefreshContext(
                    plan_path=plan_path if has_plan else None,
                    csv_path=csv_path if has_csv else None,
                    xy_tick_use=float(xy_tick_use),
                    qa_mode=qa_mode_run,
                    qa_pass_f=qa_pass_f,
                    qa_warn_f=qa_warn_f,
                    layer_assign_mode=layer_assign_mode,
                    aggregate_spots=bool(cb_agg.isChecked()),
                    agg_even_n=int(agg_even_n),
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
                        return pipeline_load_job(
                            ctx.plan_path,
                            ctx.csv_path,
                            layer_assign_mode=ctx.layer_assign_mode,
                            aggregate_spots=ctx.aggregate_spots,
                            aggregate_even_rows_after_odd=ctx.agg_even_n,
                            spot_weight_mode=ctx.spot_weight_mode_run,
                            auto_align=bool(cb_align.isChecked()),
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
                    analysis.idle_slice_band_controls_qt(slice_qt_bindings)
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
                    analysis.idle_slice_band_controls_qt(slice_qt_bindings)
                    _vtk_placeholder_message(f"PyVista: {e}", error=True)
                    status_lbl.setText(f"PyVista: {e}")
                    logger.warning("PyVista runtime error during 3D refresh: %s", e)
                except Exception as e:
                    _plot_cache["plotter"] = None
                    analysis.idle_slice_band_controls_qt(slice_qt_bindings)
                    _vtk_placeholder_message(f"Error: {e}", error=True)
                    status_lbl.setText(f"Error: {e}")
                    logger.exception("Unexpected error during 3D refresh")
            except Exception as e:
                _plot_cache["plotter"] = None
                analysis.idle_slice_band_controls_qt(slice_qt_bindings)
                _vtk_placeholder_message(f"Error: {e}", error=True)
                status_lbl.setText(f"Error: {e}")
                logger.exception("Unexpected error during 3D refresh")

        load_signals.finished.connect(_on_pipeline_load_finished)
        load_signals.failed.connect(_on_pipeline_load_failed)
        _debounce.timeout.connect(_do_refresh)

        def _commit_field_refresh() -> None:
            _debounce.stop()
            _do_refresh()

        for w in (
            e_dcm,
            e_csv,
            e_bxy,
            e_wet,
            e_agg_even,
        ):
            w.textChanged.connect(_schedule_refresh)
            w.editingFinished.connect(_commit_field_refresh)
        # QA thresholds: refresh on commit only so values like 0.5 can be typed without
        # debounced validation firing on intermediate "0" or "0.".
        for w in (e_qa_pass, e_qa_warn):
            w.editingFinished.connect(_commit_field_refresh)
        combo_sw.currentIndexChanged.connect(_do_refresh)
        combo_z_depth.currentIndexChanged.connect(_do_refresh)
        cb_weight_ch.toggled.connect(_do_refresh)
        cb_plan_fwhm.toggled.connect(_do_refresh)
        cb_meas_sigma.toggled.connect(_do_refresh)
        cb_z_water.toggled.connect(lambda _c: (_sync_wet_shifter_ui(), _do_refresh()))
        cb_align.toggled.connect(_do_refresh)
        cb_agg.toggled.connect(lambda _c: (_sync_layer_mode_ui(), _do_refresh()))
        cb_pqa.toggled.connect(lambda _c: (_sync_qa_lines(), _do_refresh()))

        def _on_qa_mode_changed() -> None:
            _stash_qa_thresholds()
            _apply_qa_mode_ui(refresh_fields=True)
            _do_refresh()

        rb_qa_pos.toggled.connect(lambda _c: _on_qa_mode_changed() if _c else None)
        rb_qa_dose.toggled.connect(lambda _c: _on_qa_mode_changed() if _c else None)
        cb_qa_lines.toggled.connect(_do_refresh)
        cb_qa_hide.toggled.connect(_do_refresh)
        cb_view_proj.toggled.connect(_do_refresh)
        btn_view_top.clicked.connect(lambda: _apply_quick_view("top"))
        btn_view_left.clicked.connect(lambda: _apply_quick_view("left"))
        btn_view_right.clicked.connect(lambda: _apply_quick_view("right"))
        rb_auto_episodes.toggled.connect(lambda _c: (_update_help(), _do_refresh()))
        rb_auto_seq.toggled.connect(lambda _c: (_update_help(), _do_refresh()))
        rb_gate.toggled.connect(lambda _c: (_update_help(), _do_refresh()))
        slice_chk.toggled.connect(lambda _c: persist())
        slice_sli.sliderReleased.connect(persist)

        app.aboutToQuit.connect(persist)

        win.show()
        finish_saved_window_layout(win, maximized=restore_maximized)
        QTimer.singleShot(0, _do_refresh)
        sys.exit(app.exec())
