"""
SpotCheck — **plan (DICOM) vs acquisition (CSV)** review GUI with embedded PyVista 3D.

**Regulatory / quality posture**

This application is **engineering and quality-assurance tooling**, not a medical device.
It supports visual and numerical review of raster‑scan ion plan data against vendor or
in‑house measurements. **Any clinical or safety‑critical use** is the responsibility of
the deploying organization, including validation, IQ/OQ/PQ, and applicable regulatory
submission. Tunable heuristics and defaults live in :mod:`spot_check.constants`; change them
only under controlled configuration management.

**Intended use**

- Load one **RT Ion Plan** (``.dcm``) and one **tabular acquisition** export (``.csv``).
- Assign CSV rows to nominal energy layers per configurable rules, then compare
  **measured vs planned** spot handling in 3D and optional **plan QA** coloring.
- The 3D view **refreshes automatically** when inputs validate (numeric fields use a
  short debounce while typing).

**Persistence**

Window geometry and control values are stored in ``.spot_check_gui_state.json``
under the project root (or current working directory). The file contains no patient
identifiers when paths are anonymized.

**Requirements:** ``pip install pydicom pyvista PySide6`` (VTK is bundled with PyVista).
For **large acquisitions** (hundreds of thousands+ rows), also install **scipy** so plan QA
and Viterbi layer costs use **cKDTree** acceleration (see :mod:`spot_check.analysis`).

**Diagnostics:** Set environment variable ``SPOT_CHECK_LOG`` to ``DEBUG``, ``INFO``,
``WARNING``, or ``ERROR`` (see :func:`spot_check.logging_utils.configure_logging`).
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import pydicom
except ImportError as e:  # pragma: no cover
    raise ImportError("RT Ion GUI requires pydicom. Install with: pip install pydicom") from e

try:
    from PySide6.QtCore import QEvent, QObject, QRunnable, Qt, QThreadPool, QTimer, Signal
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
from spot_check.constants import project_root
from spot_check.logging_utils import configure_logging

FOLDER = project_root()
_GUI_STATE_FILE = project_root() / ".spot_check_gui_state.json"
_LEGACY_GUI_STATE_FILE = project_root() / ".dicom_run_gui_state.json"
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


class _PipelineLoaderSignals(QObject):
    """Cross-thread load notifications; receiver lives on the GUI thread."""

    finished = Signal(object, int)
    failed = Signal(str, int)


class _PipelineLoadRunnable(QRunnable):
    """Runs DICOM + CSV ingestion off the GUI thread."""

    def __init__(
        self,
        fn: Any,
        signals: _PipelineLoaderSignals,
        generation: int,
    ) -> None:
        super().__init__()
        self._fn = fn
        self._signals = signals
        self._generation = generation

    def run(self) -> None:  # noqa: PLR6301 — QRunnable API
        try:
            out = self._fn()
            self._signals.finished.emit(out, self._generation)
        except Exception as exc:
            logger.exception("Background plan/CSV load failed")
            self._signals.failed.emit(str(exc), self._generation)


_MUTED_HINT = "#5c5c5c"
_MUTED_BODY = "#4a4a4a"
_MUTED_HELP = "#555555"

_DEFAULT_GUI_STATE: dict[str, object] = {
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
    # Increment when the JSON schema changes so migrations can be written if needed.
    "gui_state_schema_version": 1,
}


def _parse_layer_gap_s(raw: str) -> float | None:
    try:
        v = float(str(raw).strip())
        if v > 0.0 and v < 3600.0:
            return v
    except (ValueError, TypeError):
        pass
    return None


def _parse_refill_xy_tol_mm(raw: str) -> float | None:
    try:
        v = float(str(raw).strip())
        if v > 0.0 and v < 1_000.0:
            return v
    except (ValueError, TypeError):
        pass
    return None


def _parse_viterbi_penalty_mm2(raw: str) -> float | None:
    try:
        v = float(str(raw).strip())
        if v >= 0.0 and v <= 1.0e8:
            return v
    except (ValueError, TypeError):
        pass
    return None


def _parse_bounds_xy_tick_mm(raw: str) -> float | None:
    """Positive mm → granular ticks; 0 → coarse PyVista default (5 labels)."""
    try:
        v = float(str(raw).strip())
        if v == 0.0:
            return 0.0
        if 0.05 <= v <= 500.0:
            return v
    except (ValueError, TypeError):
        pass
    return None


def _parse_aggregate_even_tail_n(raw: str) -> int | None:
    """Gate-counter aggregate: 0 = close spot on odd→even only; max capped in
    :data:`spot_check.constants.AGGREGATE_EVEN_TAIL_MAX`."""
    try:
        v = int(float(str(raw).strip()))
        mx = int(sc_const.AGGREGATE_EVEN_TAIL_MAX)
        if 0 <= v <= mx:
            return v
    except (ValueError, TypeError):
        pass
    return None


def _parse_plan_qa_thresholds(pass_raw: str, warn_raw: str) -> tuple[float, float] | None:
    """Strict pass / warn limits in mm (XY distance to NN plan on layer): 0 < pass < warn ≤ 500."""
    try:
        a = float(str(pass_raw).strip())
        b = float(str(warn_raw).strip())
        if 0.0 < a < b <= 500.0:
            return a, b
    except (ValueError, TypeError):
        pass
    return None


@dataclass(frozen=True)
class _GuiRefreshContext:
    """Validated GUI inputs for one refresh (file paths + layering + QA)."""

    dcm: Path
    csv_path: Path
    xy_tick_use: float
    qa_pass_f: float
    qa_warn_f: float
    layer_mode: str
    gap: float
    xy_tol: float
    trust_stay: float
    vp_f: float
    aggregate_spots: bool
    agg_even_n: int
    spot_weight_mode_run: str
    pipeline_key: tuple[Any, ...]


def _gui_file_mtime(path: Path) -> float:
    try:
        return float(path.stat().st_mtime)
    except OSError:
        return -1.0


@dataclass(frozen=True)
class _PipelineLoadOK:
    """Result of background DICOM + CSV load (passed to the GUI thread)."""

    pipeline_key: tuple[Any, ...]
    label: str
    planned: list[tuple[float, float, float]]
    plan_fwhm_xy: Any
    n_plan_kept: int
    n_plan_raw: int
    measured_unaligned: list[tuple[float, ...]]
    csv_display_name: str
    measured_aligned: list[tuple[float, ...]] | None = None
    align_info: Any | None = None


def _pipeline_load_job(
    dcm: Path,
    csv_path: Path,
    *,
    layer_mode: str,
    gap: float,
    xy_tol: float,
    trust_stay: float,
    vp_f: float,
    aggregate_spots: bool,
    aggregate_even_rows_after_odd: int,
    spot_weight_mode: str,
    auto_align: bool = False,
) -> _PipelineLoadOK:
    """Parse plan + acquisition off the Qt GUI thread (optional detector XY align)."""
    p = analysis
    label = str(pydicom.dcmread(dcm, stop_before_pixels=True, force=True).get("RTPlanLabel", ""))
    planned, plan_fwhm_xy, n_plan_kept, n_plan_raw = p.planned_spot_xyz_and_counts_from_dicom(dcm)
    measured_unaligned = p.measured_spot_abc_from_csv(
        csv_path,
        max_points=None,
        planned_xyz=planned,
        a_is_x=False,
        layer_mode=layer_mode,
        layer_gap_s=gap,
        refill_same_spot_xy_tol_mm=xy_tol,
        refill_trust_time_gap_stay_dist_mm=trust_stay,
        viterbi_advance_penalty_mm2=vp_f,
        aggregate_spots=aggregate_spots,
        aggregate_even_rows_after_odd=int(aggregate_even_rows_after_odd),
        spot_weight_mode=spot_weight_mode,
    )
    if not measured_unaligned:
        raise ValueError("No measured rows to plot.")
    measured_aligned: list[tuple[float, ...]] | None = None
    align_info: Any | None = None
    if auto_align:
        measured_aligned, align_info = p.align_measured_to_plan_detector_xy(
            planned,
            measured_unaligned,
            a_is_x=False,
        )
    pipeline_key = (
        str(dcm.resolve()),
        _gui_file_mtime(dcm),
        str(csv_path.resolve()),
        _gui_file_mtime(csv_path),
        layer_mode,
        float(gap),
        float(xy_tol),
        float(vp_f),
        bool(aggregate_spots),
        int(aggregate_even_rows_after_odd),
        spot_weight_mode,
    )
    return _PipelineLoadOK(
        pipeline_key=pipeline_key,
        label=label,
        planned=planned,
        plan_fwhm_xy=plan_fwhm_xy,
        n_plan_kept=n_plan_kept,
        n_plan_raw=n_plan_raw,
        measured_unaligned=list(measured_unaligned),
        csv_display_name=csv_path.name,
        measured_aligned=list(measured_aligned) if measured_aligned is not None else None,
        align_info=align_info,
    )


def _spot_weight_mode_from_saved(raw: object) -> str:
    try:
        return analysis.normalize_measured_spot_weight_mode(str(raw))
    except (ValueError, TypeError):
        return sc_const.SPOT_WEIGHT_MODE_DEFAULT


def _load_gui_state() -> dict[str, object]:
    data: dict[str, object] = dict(_DEFAULT_GUI_STATE)
    state_file = _GUI_STATE_FILE
    if not state_file.is_file() and _LEGACY_GUI_STATE_FILE.is_file():
        state_file = _LEGACY_GUI_STATE_FILE
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


def _save_gui_state(
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
    plan_qa_pass_mm: float,
    plan_qa_warn_mm: float,
    plan_qa_draw_error_lines: bool,
    plan_qa_hide_pass_spots: bool,
    scale_plan_spots_by_dicom_fwhm: bool,
    measured_spots_sigma_world_mm: bool,
    z_axis_proton_water_depth_mm: bool,
    view_projection_perspective: bool,
    slice_band_on: bool,
    slice_band_center_i: int,
) -> None:
    try:
        _GUI_STATE_FILE.write_text(
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
                    "plan_qa_pass_mm": plan_qa_pass_mm,
                    "plan_qa_warn_mm": plan_qa_warn_mm,
                    "plan_qa_draw_error_lines": plan_qa_draw_error_lines,
                    "plan_qa_hide_pass_spots": plan_qa_hide_pass_spots,
                    "scale_plan_spots_by_dicom_fwhm": scale_plan_spots_by_dicom_fwhm,
                    "measured_spots_sigma_world_mm": measured_spots_sigma_world_mm,
                    "z_axis_proton_water_depth_mm": z_axis_proton_water_depth_mm,
                    "view_projection_perspective": view_projection_perspective,
                    "slice_band_on": slice_band_on,
                    "slice_band_center_i": slice_band_center_i,
                    "gui_state_schema_version": int(_DEFAULT_GUI_STATE["gui_state_schema_version"]),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("Could not write GUI state file %s: %s", _GUI_STATE_FILE, exc)


def _sanitize_tk_geometry(raw: object, *, default: str = "1400x900") -> str:
    """Accept saved Tk geometry strings; fall back if obviously invalid."""
    s = str(raw or "").strip()
    if not s:
        return default
    # "WxH" or "WxH+X+Y"
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


def _apply_saved_geometry(win: QMainWindow, raw: object) -> None:
    s = _sanitize_tk_geometry(raw, default="1400x900")
    try:
        tokens = s.replace("+", " ").split()
        wh = tokens[0].lower().split("x")
        w, h = int(float(wh[0])), int(float(wh[1]))
        win.resize(max(w, 400), max(h, 400))
        if len(tokens) >= 3:
            win.move(int(float(tokens[1])), int(float(tokens[2])))
    except (ValueError, IndexError, TypeError):
        win.resize(1400, 900)


def _geom_from_win(win: QMainWindow) -> str:
    g = win.geometry()
    return f"{g.width()}x{g.height()}+{g.x()}+{g.y()}"


def run_gui() -> None:
    configure_logging()
    saved = _load_gui_state()
    app = QApplication.instance() or QApplication(sys.argv)

    win = QMainWindow()
    win.setWindowTitle(f"SpotCheck v{__version__} — Plan vs acquisition")
    win.setMinimumSize(1040, 640)
    _apply_saved_geometry(win, saved.get("window_geometry"))

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
    vtk_placeholder = QLabel("3D view — pick .dcm + .csv; updates when inputs validate.")
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
        hint.setStyleSheet(f"color: {_MUTED_HINT};")
        hint.setMinimumWidth(0)
        hint.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        hint.setMaximumWidth(800)
        return hint

    raw_gap = saved.get("layer_gap_s", sc_const.TIME_LAYER_GAP_S_DEFAULT)
    try:
        gap0 = float(raw_gap)
        if gap0 <= 0:
            gap0 = sc_const.TIME_LAYER_GAP_S_DEFAULT
    except (TypeError, ValueError):
        gap0 = sc_const.TIME_LAYER_GAP_S_DEFAULT
    raw_xy = saved.get("refill_same_spot_xy_tol_mm", sc_const.REFILL_SAME_SPOT_XY_TOLERANCE_MM)
    try:
        xy0 = float(raw_xy)
        if xy0 <= 0:
            xy0 = sc_const.REFILL_SAME_SPOT_XY_TOLERANCE_MM
    except (TypeError, ValueError):
        xy0 = sc_const.REFILL_SAME_SPOT_XY_TOLERANCE_MM
    mode0 = str(saved.get("layer_assign_mode") or "gate_counter").strip().lower().replace("-", "_")
    if mode0 in ("time_gap", "plan_viterbi"):
        mode0 = "gate_counter"
    if mode0 not in ("unified", "gate_counter"):
        mode0 = "gate_counter"
    raw_vp = saved.get(
        "viterbi_advance_penalty_mm2", sc_const.VITERBI_LAYER_ADVANCE_PENALTY_MM2_DEFAULT
    )
    try:
        vp0 = float(raw_vp)
        if vp0 < 0:
            vp0 = sc_const.VITERBI_LAYER_ADVANCE_PENALTY_MM2_DEFAULT
    except (TypeError, ValueError):
        vp0 = sc_const.VITERBI_LAYER_ADVANCE_PENALTY_MM2_DEFAULT

    e_dcm = QLineEdit(str(saved.get("dcm_path") or ""))
    e_csv = QLineEdit(str(saved.get("csv_path") or ""))
    e_gap = QLineEdit(f"{gap0:g}")
    e_refill = QLineEdit(f"{xy0:g}")
    e_vit = QLineEdit(f"{vp0:g}")
    e_agg_even = QLineEdit(
        str(
            int(
                saved.get(
                    "aggregate_even_rows_after_odd", sc_const.AGGREGATE_EVEN_ROWS_AFTER_ODD_DEFAULT
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
    raw_qp = saved.get("plan_qa_pass_mm", sc_const.PLAN_QA_PASS_MM_DEFAULT)
    raw_qw = saved.get("plan_qa_warn_mm", sc_const.PLAN_QA_WARN_MM_DEFAULT)
    try:
        qp0 = float(raw_qp)
        qw0 = float(raw_qw)
        if not (0.0 < qp0 < qw0 <= 500.0):
            qp0, qw0 = (
                float(sc_const.PLAN_QA_PASS_MM_DEFAULT),
                float(sc_const.PLAN_QA_WARN_MM_DEFAULT),
            )
    except (TypeError, ValueError):
        qp0, qw0 = float(sc_const.PLAN_QA_PASS_MM_DEFAULT), float(sc_const.PLAN_QA_WARN_MM_DEFAULT)
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
    swm0 = _spot_weight_mode_from_saved(saved.get("spot_weight_mode"))
    combo_sw = QComboBox()
    for _m, lbl in _SPOT_WEIGHT_COMBO_LABELS:
        combo_sw.addItem(lbl, _m)
    combo_sw.setCurrentText(_SW_LABEL_BY_MODE[swm0])
    cb_plan_fwhm = QCheckBox("Plan: FWHM ellipses (DICOM 300A,0398)")
    cb_plan_fwhm.setChecked(_bool_saved("scale_plan_spots_by_dicom_fwhm", False))
    cb_meas_sigma = QCheckBox("Measured: σ ellipsoids (fit σ → XY mm; B→X, A→Y)")
    cb_meas_sigma.setChecked(_bool_saved("measured_spots_sigma_world_mm", False))
    cb_z_water = QCheckBox("Z: water depth (mm, proton CSDA approx.)")
    cb_z_water.setChecked(_bool_saved("z_axis_proton_water_depth_mm", True))
    cb_align = QCheckBox("Rigid XY align measured → plan (any rotation / A↔B)")
    cb_align.setChecked(_bool_saved("auto_align_detector_xy", True))
    cb_agg = QCheckBox("One measured point per odd gate phase (weighted mean)")
    cb_agg.setChecked(_bool_saved("aggregate_spots_by_gate", True))
    cb_pqa = QCheckBox("Color measured by plan XY distance (pass / warn / fail)")
    cb_pqa.setChecked(_bool_saved("plan_qa_coloring", True))
    cb_qa_lines = QCheckBox("QA lines: warn/fail → plan")
    cb_qa_lines.setChecked(_bool_saved("plan_qa_draw_error_lines", False))
    cb_qa_hide = QCheckBox("Hide pass-tier in 3D (warn+fail only)")
    cb_qa_hide.setChecked(_bool_saved("plan_qa_hide_pass_spots", False))
    cb_view_proj = QCheckBox("Projection view")
    cb_view_proj.setChecked(_bool_saved("view_projection_perspective", True))
    cb_view_proj.setToolTip("On: perspective (default). Off: orthogonal / parallel projection.")

    rb_unified = QRadioButton("Unified (Viterbi + timing / refill)")
    rb_gate = QRadioButton("Gate counter (odd=spot, even=deadtime)")
    layer_grp = QButtonGroup(win)
    layer_grp.addButton(rb_unified)
    layer_grp.addButton(rb_gate)
    if mode0 == "unified":
        rb_unified.setChecked(True)
    else:
        rb_gate.setChecked(True)

    unified_entries = (e_vit, e_gap, e_refill)

    help_lbl = QLabel()
    help_lbl.setWordWrap(True)
    help_lbl.setStyleSheet(f"color: {_MUTED_HELP}; font-size: 9pt;")
    help_lbl.setMinimumWidth(0)
    help_lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

    agg_intro_lbl = QLabel()
    agg_intro_lbl.setWordWrap(True)
    agg_intro_lbl.setStyleSheet(f"color: {_MUTED_HELP}; font-size: 9pt;")
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
    slice_status.setStyleSheet(f"color: {_MUTED_HINT}; font-size: 9pt;")
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
    status_lbl.setStyleSheet(f"color: {_MUTED_BODY};")
    status_lbl.setMinimumWidth(0)
    status_lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

    def persist() -> None:
        gap = _parse_layer_gap_s(e_gap.text())
        if gap is None:
            gap = sc_const.TIME_LAYER_GAP_S_DEFAULT
            e_gap.setText(f"{gap:g}")
        xy_tol = _parse_refill_xy_tol_mm(e_refill.text())
        if xy_tol is None:
            xy_tol = sc_const.REFILL_SAME_SPOT_XY_TOLERANCE_MM
            e_refill.setText(f"{xy_tol:g}")
        mode = "unified" if rb_unified.isChecked() else "gate_counter"
        vp = _parse_viterbi_penalty_mm2(e_vit.text())
        if vp is None:
            vp = sc_const.VITERBI_LAYER_ADVANCE_PENALTY_MM2_DEFAULT
            e_vit.setText(f"{vp:g}")
        xy_tick_save = _parse_bounds_xy_tick_mm(e_bxy.text())
        if xy_tick_save is None:
            xy_tick_save = float(sc_const.BOUNDS_XY_TICK_MM_DEFAULT)
            e_bxy.setText(f"{xy_tick_save:g}")
        qa_thr = _parse_plan_qa_thresholds(e_qa_pass.text(), e_qa_warn.text())
        if qa_thr is None:
            qa_thr = (
                float(sc_const.PLAN_QA_PASS_MM_DEFAULT),
                float(sc_const.PLAN_QA_WARN_MM_DEFAULT),
            )
            e_qa_pass.setText(f"{qa_thr[0]:g}")
            e_qa_warn.setText(f"{qa_thr[1]:g}")
        qa_pass_sv, qa_warn_sv = qa_thr
        tail_n = _parse_aggregate_even_tail_n(e_agg_even.text())
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
        _save_gui_state(
            dcm_path=e_dcm.text().strip(),
            csv_path=e_csv.text().strip(),
            window_geometry=_geom_from_win(win),
            layer_assign_mode=mode,
            layer_gap_s=gap,
            refill_same_spot_xy_tol_mm=xy_tol,
            viterbi_advance_penalty_mm2=vp,
            weight_measured_by_channel_sum=cb_weight_ch.isChecked(),
            spot_weight_mode=sw_mode_norm,
            aggregate_spots_by_gate=cb_agg.isChecked(),
            aggregate_even_rows_after_odd=int(tail_n),
            auto_align_detector_xy=cb_align.isChecked(),
            bounds_xy_tick_mm=float(xy_tick_save),
            plan_qa_coloring=cb_pqa.isChecked(),
            plan_qa_pass_mm=float(qa_pass_sv),
            plan_qa_warn_mm=float(qa_warn_sv),
            plan_qa_draw_error_lines=cb_qa_lines.isChecked(),
            plan_qa_hide_pass_spots=cb_qa_hide.isChecked(),
            scale_plan_spots_by_dicom_fwhm=cb_plan_fwhm.isChecked(),
            measured_spots_sigma_world_mm=cb_meas_sigma.isChecked(),
            z_axis_proton_water_depth_mm=cb_z_water.isChecked(),
            view_projection_perspective=cb_view_proj.isChecked(),
            slice_band_on=bool(slice_chk.isChecked()),
            slice_band_center_i=_sb_ci_persist,
        )

    def browse_dcm() -> None:
        init = str(Path(e_dcm.text().strip()).parent) if e_dcm.text().strip() else str(FOLDER)
        p, _ = QFileDialog.getOpenFileName(
            win,
            "Plan (.dcm)",
            init if Path(init).is_dir() else str(FOLDER),
            "DICOM (*.dcm);;All files (*.*)",
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
            "CSV (*.csv);;All files (*.*)",
        )
        if p:
            e_csv.setText(p)
            _do_refresh()

    def _sync_qa_lines() -> None:
        en = cb_pqa.isChecked()
        if not en:
            cb_qa_lines.setChecked(False)
            cb_qa_hide.setChecked(False)
        cb_qa_lines.setEnabled(en)
        cb_qa_hide.setEnabled(en)

    def _sync_agg_even() -> None:
        e_agg_even.setEnabled(cb_agg.isChecked())

    def _sync_unified_entries() -> None:
        u = rb_unified.isChecked()
        for w in unified_entries:
            w.setEnabled(u)

    def _update_help() -> None:
        lm = "unified" if rb_unified.isChecked() else "gate_counter"
        _sync_unified_entries()
        if lm == "unified":
            help_lbl.setText(
                "Unified: Viterbi stay or +1 layer vs plan; short Δt adds advance cost; "
                "long-gap same-spot XY blocks advance. B→X, A→Y."
            )
            agg_intro_lbl.setText(
                "If CSV has Gate Counter: optional merge per odd phase (weight = Display → Weight)."
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
    row_csv = QWidget()
    h_csv = QHBoxLayout(row_csv)
    h_csv.setContentsMargins(0, 0, 0, 0)
    h_csv.addWidget(e_csv, 1)
    b_csv = QPushButton("Browse…")
    b_csv.clicked.connect(browse_csv)
    h_csv.addWidget(b_csv)
    fl.addRow("Plan (.dcm)", row_dcm)
    fl.addRow("CSV", row_csv)

    gb_layer = QGroupBox("Layer assignment")
    vl_layer = QVBoxLayout(gb_layer)
    vl_layer.addWidget(rb_unified)
    vl_layer.addWidget(rb_gate)
    uni = QVBoxLayout()
    r1 = QHBoxLayout()
    r1.addWidget(QLabel("Advance (mm²)"))
    r1.addWidget(e_vit)
    r1.addWidget(_add_hint_lbl("↑ → fewer layer steps"), 1)
    uni.addLayout(r1)
    r2 = QHBoxLayout()
    r2.addWidget(QLabel("Min Δt (s)"))
    r2.addWidget(e_gap)
    r2.addWidget(_add_hint_lbl("~0.2 typical; short Δt adds cost"), 1)
    uni.addLayout(r2)
    r3 = QHBoxLayout()
    r3.addWidget(QLabel("Refill XY (mm)"))
    r3.addWidget(e_refill)
    r3.addWidget(
        _add_hint_lbl("Long gap + dXY ≤ this → refill block"),
        1,
    )
    uni.addLayout(r3)
    w_uni = QWidget()
    w_uni.setLayout(uni)
    vl_layer.addWidget(w_uni)
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
    vdisp.addWidget(cb_z_water)
    vdisp.addWidget(cb_align)
    rt = QHBoxLayout()
    rt.addWidget(QLabel("XY ticks (mm)"))
    rt.addWidget(e_bxy)
    rt.addWidget(_add_hint_lbl("0 = coarse; ~5 common"), 1)
    wtick = QWidget()
    wtick.setLayout(rt)
    vdisp.addWidget(wtick)

    gb_qa = QGroupBox("Plan QA (XY vs plan)")
    vqa = QVBoxLayout(gb_qa)
    vqa.addWidget(cb_pqa)
    vqa.addWidget(cb_qa_lines)
    vqa.addWidget(cb_qa_hide)
    qa_th = QHBoxLayout()
    qa_th.addWidget(QLabel("Pass ≤ (mm)"))
    qa_th.addWidget(e_qa_pass)
    qa_th.addWidget(QLabel("Warn ≤ (mm)"))
    qa_th.addWidget(e_qa_warn)
    wqa = QWidget()
    wqa.setLayout(qa_th)
    vqa.addWidget(wqa)
    vqa.addWidget(_add_hint_lbl("Need 0 < pass < warn."))

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
            f"Even rows after odd→even, good fits; 0=off; max {sc_const.AGGREGATE_EVEN_TAIL_MAX}"
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

    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    drawer_layout.addWidget(line)

    auto_hint = QLabel("3D refreshes when inputs validate. Numbers debounce briefly after typing.")
    auto_hint.setWordWrap(True)
    auto_hint.setStyleSheet(f"color: {_MUTED_HINT}; font-size: 9pt;")
    auto_hint.setMinimumWidth(0)
    auto_hint.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
    drawer_layout.addWidget(auto_hint)

    load_panel = QFrame()
    load_panel.setObjectName("loadPanel")
    load_panel.setStyleSheet(
        "#loadPanel { background-color: #161b22; border: 1px solid #30363d; border-radius: 6px; }"
    )
    load_panel_lay = QVBoxLayout(load_panel)
    load_panel_lay.setContentsMargins(10, 8, 10, 8)
    load_panel_lay.setSpacing(6)
    load_head = QHBoxLayout()
    load_spinner_lbl = QLabel(_LOAD_SPINNER_FRAMES[0])
    load_spinner_lbl.setStyleSheet("color: #58a6ff; font-size: 14pt;")
    load_spinner_lbl.setFixedWidth(22)
    load_msg_lbl = QLabel("Loading plan/CSV…")
    load_msg_lbl.setWordWrap(True)
    load_msg_lbl.setStyleSheet("color: #c9d1d9; font-weight: 600;")
    load_msg_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
    load_head.addWidget(load_spinner_lbl)
    load_head.addWidget(load_msg_lbl, 1)
    load_panel_lay.addLayout(load_head)
    load_bar = QProgressBar()
    load_bar.setRange(0, 0)
    load_bar.setFixedHeight(5)
    load_bar.setTextVisible(False)
    load_bar.setStyleSheet(
        "QProgressBar { background-color: #21262d; border: none; border-radius: 2px; }"
        "QProgressBar::chunk { background-color: #58a6ff; border-radius: 2px; }"
    )
    load_panel_lay.addWidget(load_bar)
    load_panel.hide()
    drawer_layout.addWidget(load_panel)

    drawer_layout.addWidget(status_lbl)
    foot = QLabel("pydicom · PyVista · PySide6. Engineering use only.")
    foot.setStyleSheet(f"color: {_MUTED_HINT}; font-size: 8pt;")
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

    _sync_qa_lines()
    _sync_agg_even()
    _update_help()

    _debounce = QTimer(win)
    _debounce.setSingleShot(True)
    _debounce.setInterval(_REFRESH_DEBOUNCE_MS)

    _plot_cache: dict[str, object] = {
        "pipeline_key": None,
        "planned": None,
        "plan_fwhm_xy": None,
        "n_plan_kept": 0,
        "n_plan_raw": 0,
        "measured_unaligned": None,
        "measured_aligned": None,
        "align_info": None,
        "align_cache_key": None,
        "label": "",
        "csv_display_name": "",
        "plotter": None,
        "aligned": None,
        "z_water_depth": False,
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
    load_signals = _PipelineLoaderSignals(win)
    pending_ctx: _GuiRefreshContext | None = None
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
        ch = _LOAD_SPINNER_FRAMES[_spinner_frame]
        load_spinner_lbl.setText(ch)
        load_overlay_spinner.setText(ch)

    _load_spinner_timer.timeout.connect(_tick_load_spinner)

    def _show_loading(generation: int, message: str) -> None:
        nonlocal _loading_gen, _spinner_frame
        _loading_gen = int(generation)
        _spinner_frame = 0
        ch = _LOAD_SPINNER_FRAMES[0]
        load_spinner_lbl.setText(ch)
        load_overlay_spinner.setText(ch)
        load_msg_lbl.setText(message)
        load_overlay_msg.setText(message)
        load_panel.show()
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
        load_panel.hide()
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

    def _finalize_refresh(ctx: _GuiRefreshContext, *, need_data: bool) -> None:
        nonlocal pending_ctx
        p = analysis
        pipeline_key = ctx.pipeline_key
        planned = _plot_cache.get("planned")
        plan_fwhm_xy = _plot_cache.get("plan_fwhm_xy")
        n_plan_kept = int(_plot_cache.get("n_plan_kept", 0))
        n_plan_raw = int(_plot_cache.get("n_plan_raw", 0))
        measured_unaligned = _plot_cache.get("measured_unaligned")
        label = str(_plot_cache.get("label", ""))
        csv_display_name = str(_plot_cache.get("csv_display_name", ""))
        if planned is None or measured_unaligned is None:
            logger.error(
                "Plot cache inconsistency: pipeline_key matched but "
                "planned or measured_unaligned is None"
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

        measured = list(measured_unaligned)
        detector_align_caption: str | None = None
        align_note = ""
        align_cache_key = (pipeline_key, bool(cb_align.isChecked()))
        if cb_align.isChecked():
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

        reuse_pl = _plot_cache.get("plotter")
        preserve_cam = (
            reuse_pl is not None
            and _plot_cache["pipeline_key"] == pipeline_key
            and _plot_cache.get("aligned") == cb_align.isChecked()
            and _plot_cache.get("z_water_depth") == cb_z_water.isChecked()
        )
        pl = p.show_comparison_3d_pyvista(
            planned,
            measured,
            title=f"{label} — plan vs {csv_display_name}",
            a_is_x=False,
            layer_mode=ctx.layer_mode,
            layer_gap_s=ctx.gap,
            refill_same_spot_xy_tol_mm=ctx.xy_tol,
            refill_trust_time_gap_stay_dist_mm=ctx.trust_stay,
            viterbi_advance_penalty_mm2=ctx.vp_f,
            weight_measured_by_channel=cb_weight_ch.isChecked(),
            aggregate_spots=ctx.aggregate_spots,
            aggregate_even_rows_after_odd=int(ctx.agg_even_n),
            spot_weight_mode=ctx.spot_weight_mode_run,
            detector_align_caption=detector_align_caption,
            bounds_xy_tick_mm=ctx.xy_tick_use,
            plan_qa_coloring=cb_pqa.isChecked(),
            plan_qa_pass_mm=ctx.qa_pass_f,
            plan_qa_warn_mm=ctx.qa_warn_f,
            plan_qa_draw_error_lines=cb_qa_lines.isChecked(),
            plan_qa_hide_pass_spots=cb_qa_hide.isChecked(),
            plan_fwhm_xy_mm=plan_fwhm_xy,
            scale_plan_spots_by_dicom_fwhm=cb_plan_fwhm.isChecked(),
            measured_spots_sigma_world_mm=cb_meas_sigma.isChecked(),
            z_axis_use_proton_water_depth_mm=cb_z_water.isChecked(),
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
        plan_line = f"Plan spots: {n_plan_kept} after meterset filter (weight>0"
        if n_plan_raw != n_plan_kept:
            plan_line += f"; {n_plan_raw} raw slots in CP maps before that filter"
        plan_line += ")"
        meas_line = f"Measured points: {n_meas} built and plotted"
        if ctx.aggregate_spots:
            _cap = p.measured_spot_weight_caption(ctx.spot_weight_mode_run)
            meas_line += f" (one {_cap}-weighted mean per odd gate spot)"
            if ctx.agg_even_n > 0 and ctx.layer_mode == "gate_counter":
                meas_line += (
                    f"; up to {ctx.agg_even_n} good even-phase row(s) merged after odd→even"
                )
        if cb_align.isChecked() and detector_align_caption:
            meas_line += f". {detector_align_caption}"
        if cb_pqa.isChecked():
            meas_line += (
                f". Plan QA colors: d≤{ctx.qa_pass_f:g} pass, "
                f"{ctx.qa_pass_f:g}<d≤{ctx.qa_warn_f:g} warn, d>{ctx.qa_warn_f:g} fail"
            )
            if cb_qa_hide.isChecked():
                meas_line += "; pass-tier spots hidden in 3D"
            if cb_qa_lines.isChecked():
                meas_line += "; error lines for warn+fail"
        if cb_meas_sigma.isChecked():
            meas_line += ". Measured: σ-sized ellipsoids in world XY (mm; see 3D caption)"
        status_lbl.setText(f"Updated. {plan_line}. {meas_line}.{align_note}{cache_note}")
        persist()
        pending_ctx = None

    def _on_pipeline_load_finished(res: object, generation: int) -> None:
        nonlocal pending_ctx, load_generation
        if generation != load_generation:
            return
        if not isinstance(res, _PipelineLoadOK):
            _hide_loading(generation)
            return
        ctx0 = pending_ctx
        if ctx0 is None:
            _hide_loading(generation)
            return
        ld = res
        _plot_cache["pipeline_key"] = ld.pipeline_key
        _plot_cache["planned"] = ld.planned
        _plot_cache["plan_fwhm_xy"] = ld.plan_fwhm_xy
        _plot_cache["n_plan_kept"] = ld.n_plan_kept
        _plot_cache["n_plan_raw"] = ld.n_plan_raw
        _plot_cache["measured_unaligned"] = ld.measured_unaligned
        _plot_cache["measured_aligned"] = ld.measured_aligned
        _plot_cache["align_info"] = ld.align_info
        _plot_cache["align_cache_key"] = (
            (ld.pipeline_key, True) if ld.measured_aligned is not None else None
        )
        _plot_cache["label"] = ld.label
        _plot_cache["csv_display_name"] = ld.csv_display_name
        build_note = f"Building 3D view — {ld.csv_display_name}…"
        load_msg_lbl.setText(build_note)
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
        _plot_cache["plotter"] = None
        analysis.idle_slice_band_controls_qt(slice_qt_bindings)
        short = msg if len(msg) < 160 else f"{msg[:157]}…"
        _vtk_placeholder_message(short, error=True)
        status_lbl.setText(short)
        _hide_loading(generation)

    def _bump_load_generation_invalidate_async() -> None:
        nonlocal load_generation, pending_ctx
        load_generation += 1
        pending_ctx = None

    def _do_refresh() -> None:
        nonlocal pending_ctx, load_generation
        persist()
        p = analysis
        dcm = Path(e_dcm.text().strip())
        csv_path = Path(e_csv.text().strip())
        if not dcm.is_file():
            _bump_load_generation_invalidate_async()
            _plot_cache["plotter"] = None
            _plot_cache["pipeline_key"] = None
            analysis.idle_slice_band_controls_qt(slice_qt_bindings)
            _vtk_placeholder_message("Select a valid plan (.dcm).", error=True)
            status_lbl.setText("Need valid .dcm.")
            return
        if not csv_path.is_file():
            _bump_load_generation_invalidate_async()
            _plot_cache["plotter"] = None
            _plot_cache["pipeline_key"] = None
            analysis.idle_slice_band_controls_qt(slice_qt_bindings)
            _vtk_placeholder_message("Select a valid CSV.", error=True)
            status_lbl.setText("Need valid CSV.")
            return
        xy_tick_use = _parse_bounds_xy_tick_mm(e_bxy.text())
        if xy_tick_use is None:
            _bump_load_generation_invalidate_async()
            analysis.idle_slice_band_controls_qt(slice_qt_bindings)
            _vtk_placeholder_message(
                "XY ticks: 0 or spacing 0.05–500 mm.",
                error=True,
            )
            status_lbl.setText("Fix XY ticks (mm).")
            return
        qa_pass_f = float(sc_const.PLAN_QA_PASS_MM_DEFAULT)
        qa_warn_f = float(sc_const.PLAN_QA_WARN_MM_DEFAULT)
        if cb_pqa.isChecked():
            qa_pair = _parse_plan_qa_thresholds(e_qa_pass.text(), e_qa_warn.text())
            if qa_pair is None:
                _bump_load_generation_invalidate_async()
                analysis.idle_slice_band_controls_qt(slice_qt_bindings)
                _vtk_placeholder_message(
                    "QA: 0 < pass < warn ≤ 500 (mm).",
                    error=True,
                )
                status_lbl.setText("Fix QA pass/warn (mm).")
                return
            qa_pass_f, qa_warn_f = qa_pair
        layer_mode = "unified" if rb_unified.isChecked() else "gate_counter"
        trust_stay = float(sc_const.REFILL_TRUST_TIME_GAP_STAY_DIST_MM)
        if layer_mode == "unified":
            gap = _parse_layer_gap_s(e_gap.text())
            if gap is None:
                _bump_load_generation_invalidate_async()
                analysis.idle_slice_band_controls_qt(slice_qt_bindings)
                _vtk_placeholder_message(
                    "Unified: enter min Δt (s), e.g. 0.2.",
                    error=True,
                )
                status_lbl.setText("Fix min Δt (s).")
                return
            xy_tol = _parse_refill_xy_tol_mm(e_refill.text())
            if xy_tol is None:
                _bump_load_generation_invalidate_async()
                analysis.idle_slice_band_controls_qt(slice_qt_bindings)
                _vtk_placeholder_message(
                    "Refill XY: positive mm (≤999).",
                    error=True,
                )
                status_lbl.setText("Fix refill XY (mm).")
                return
            vp_en = _parse_viterbi_penalty_mm2(e_vit.text())
            if vp_en is None:
                _bump_load_generation_invalidate_async()
                analysis.idle_slice_band_controls_qt(slice_qt_bindings)
                _vtk_placeholder_message(
                    "Advance penalty: mm², ≥0 (e.g. 400).",
                    error=True,
                )
                status_lbl.setText("Fix advance penalty (mm²).")
                return
        else:
            gap = float(sc_const.TIME_LAYER_GAP_S_DEFAULT)
            xy_tol = float(sc_const.REFILL_SAME_SPOT_XY_TOLERANCE_MM)
        vp_f = _parse_viterbi_penalty_mm2(e_vit.text())
        if vp_f is None:
            vp_f = float(sc_const.VITERBI_LAYER_ADVANCE_PENALTY_MM2_DEFAULT)
        agg_even_n = _parse_aggregate_even_tail_n(e_agg_even.text())
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
            sw_internal_run = _SW_MODE_BY_LABEL.get(sw_lbl_run, sc_const.SPOT_WEIGHT_MODE_DEFAULT)
            spot_weight_mode_run = p.normalize_measured_spot_weight_mode(sw_internal_run)

            pipeline_key = (
                str(dcm.resolve()),
                _gui_file_mtime(dcm),
                str(csv_path.resolve()),
                _gui_file_mtime(csv_path),
                layer_mode,
                float(gap),
                float(xy_tol),
                float(vp_f),
                bool(cb_agg.isChecked()),
                int(agg_even_n),
                spot_weight_mode_run,
            )
            ctx = _GuiRefreshContext(
                dcm=dcm,
                csv_path=csv_path,
                xy_tick_use=float(xy_tick_use),
                qa_pass_f=qa_pass_f,
                qa_warn_f=qa_warn_f,
                layer_mode=layer_mode,
                gap=float(gap),
                xy_tol=float(xy_tol),
                trust_stay=trust_stay,
                vp_f=float(vp_f),
                aggregate_spots=bool(cb_agg.isChecked()),
                agg_even_n=int(agg_even_n),
                spot_weight_mode_run=spot_weight_mode_run,
                pipeline_key=pipeline_key,
            )
            need_data = (
                pipeline_key != _plot_cache["pipeline_key"]
                or _plot_cache["measured_unaligned"] is None
            )

            if need_data:
                load_generation += 1
                gen = load_generation
                pending_ctx = ctx
                csv_name = csv_path.name
                load_note = f"Loading plan/CSV — {dcm.name}, {csv_name}…"
                if cb_align.isChecked():
                    load_note = f"Loading + aligning — {dcm.name}, {csv_name}…"
                status_lbl.setText(load_note)
                _show_loading(gen, load_note)

                def _job() -> _PipelineLoadOK:
                    return _pipeline_load_job(
                        ctx.dcm,
                        ctx.csv_path,
                        layer_mode=ctx.layer_mode,
                        gap=ctx.gap,
                        xy_tol=ctx.xy_tol,
                        trust_stay=ctx.trust_stay,
                        vp_f=ctx.vp_f,
                        aggregate_spots=ctx.aggregate_spots,
                        aggregate_even_rows_after_odd=ctx.agg_even_n,
                        spot_weight_mode=ctx.spot_weight_mode_run,
                        auto_align=bool(cb_align.isChecked()),
                    )

                load_pool.start(_PipelineLoadRunnable(_job, load_signals, gen))
                return

            planned = _plot_cache["planned"]
            measured_unaligned = _plot_cache["measured_unaligned"]
            if planned is None or measured_unaligned is None:
                logger.error(
                    "Plot cache inconsistency: pipeline_key matched but "
                    "planned or measured_unaligned is None"
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

    for w in (
        e_dcm,
        e_csv,
        e_gap,
        e_refill,
        e_vit,
        e_bxy,
        e_qa_pass,
        e_qa_warn,
        e_agg_even,
    ):
        w.textChanged.connect(_schedule_refresh)
        w.editingFinished.connect(
            lambda: (_debounce.stop(), _do_refresh())  # immediate + cancel pending debounce
        )
    combo_sw.currentIndexChanged.connect(_do_refresh)
    cb_weight_ch.toggled.connect(_do_refresh)
    cb_plan_fwhm.toggled.connect(_do_refresh)
    cb_meas_sigma.toggled.connect(_do_refresh)
    cb_z_water.toggled.connect(_do_refresh)
    cb_align.toggled.connect(_do_refresh)
    cb_agg.toggled.connect(lambda _c: (_sync_agg_even(), _do_refresh()))
    cb_pqa.toggled.connect(lambda _c: (_sync_qa_lines(), _do_refresh()))
    cb_qa_lines.toggled.connect(_do_refresh)
    cb_qa_hide.toggled.connect(_do_refresh)
    cb_view_proj.toggled.connect(_do_refresh)
    btn_view_top.clicked.connect(lambda: _apply_quick_view("top"))
    btn_view_left.clicked.connect(lambda: _apply_quick_view("left"))
    btn_view_right.clicked.connect(lambda: _apply_quick_view("right"))
    rb_unified.toggled.connect(lambda _c: (_update_help(), _do_refresh()))
    rb_gate.toggled.connect(lambda _c: (_update_help(), _do_refresh()))
    slice_chk.toggled.connect(lambda _c: persist())
    slice_sli.sliderReleased.connect(persist)

    app.aboutToQuit.connect(persist)

    win.show()
    QTimer.singleShot(0, _do_refresh)
    sys.exit(app.exec())


def main() -> None:
    """Console entry point for ``spot-check``."""
    run_gui()


if __name__ == "__main__":
    try:
        main()
    except ImportError:
        raise
    except Exception as exc:
        # stderr for operators without log config; full traceback when SPOT_CHECK_LOG allows
        configure_logging()
        logging.getLogger(__name__).exception("SpotCheck GUI startup failed")
        print("SpotCheck GUI startup failed:", exc, file=sys.stderr)
        raise
