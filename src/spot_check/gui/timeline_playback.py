"""Media-style timeline bar for acquisition window playback under the 3D view."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
from PySide6.QtCore import QElapsedTimer, Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QWidget,
)

from spot_check.analysis.viz.data import _time_slice_range_ms
from spot_check.constants import TIME_SLICE_WINDOW_FULL, TIME_SLICE_WINDOW_S_DEFAULT

_SYM_PLAY = "\u25B6"
_SYM_PAUSE = "\u2016"

_PLAYBACK_SPEEDS: tuple[tuple[str, float], ...] = (
    ("0.25×", 0.25),
    ("0.5×", 0.5),
    ("1×", 1.0),
    ("2×", 2.0),
    ("4×", 4.0),
    ("8×", 8.0),
)

_WINDOW_PRESETS: tuple[tuple[str, float], ...] = (
    ("0.1 s", 0.1),
    ("1 s", 1.0),
    ("10 s", 10.0),
    ("All", TIME_SLICE_WINDOW_FULL),
)

_TICK_MS = 33


def format_timeline_seconds(t: float) -> str:
    """Format seconds as ``m:ss.s`` for playback labels."""
    if not math.isfinite(t):
        return "—:—"
    t = max(0.0, float(t))
    minutes = int(t // 60)
    seconds = t - minutes * 60
    if minutes > 0:
        return f"{minutes}:{seconds:04.1f}"
    return f"0:{seconds:04.1f}"


def advance_playback_start_ms(
    start_ms: int,
    *,
    elapsed_ms: float,
    speed: float,
    start_min_ms: int,
    start_max_ms: int,
) -> tuple[int, bool]:
    """Advance window start; returns ``(new_start_ms, reached_end)``."""
    delta = int(round(float(elapsed_ms) * float(speed)))
    if delta <= 0:
        cur = int(np.clip(int(start_ms), start_min_ms, start_max_ms))
        return cur, cur >= start_max_ms
    nxt = int(start_ms) + delta
    if nxt >= start_max_ms:
        return int(start_max_ms), True
    return int(max(start_min_ms, nxt)), False


class TimelinePlaybackBar(QWidget):
    """Time-slice toggle, window width, play/pause, speed, scrub, and labels."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._playing = False
        self._start_min_ms = 0
        self._start_max_ms = 0
        self._t_max_s = 0.0
        self._speed = 1.0
        self._window_s = float(TIME_SLICE_WINDOW_S_DEFAULT)
        self._timeline_time_s = np.zeros(0, dtype=np.float64)
        self._time_slice_cfg: dict[str, bool | int | float] | None = None
        self._apply_slice: Callable[[], None] | None = None

        row = QHBoxLayout(self)
        row.setContentsMargins(8, 4, 8, 4)
        row.setSpacing(6)

        self._chk_slice = QCheckBox("Time")
        self._chk_slice.setToolTip("Enable acquisition time window on plan and measured spots")
        row.addWidget(self._chk_slice)

        self._combo_window = QComboBox()
        for label, _ in _WINDOW_PRESETS:
            self._combo_window.addItem(label)
        self._combo_window.setToolTip(
            "Window width; All = every spot from acquisition start through the playhead"
        )
        row.addWidget(self._combo_window)

        self._btn_play = QPushButton(_SYM_PLAY)
        self._btn_play.setToolTip("Play / pause acquisition window")
        row.addWidget(self._btn_play)

        self._combo_speed = QComboBox()
        for label, _ in _PLAYBACK_SPEEDS:
            self._combo_speed.addItem(label)
        self._combo_speed.setToolTip("Playback speed (real-time multiplier)")
        row.addWidget(self._combo_speed)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setTracking(True)
        row.addWidget(self._slider, 1)

        self._lbl_current = QLabel("—:—")
        self._lbl_current.setMinimumWidth(52)
        self._lbl_current.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(self._lbl_current)

        row.addWidget(QLabel("/"))

        self._lbl_total = QLabel("—:—")
        self._lbl_total.setMinimumWidth(52)
        self._lbl_total.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(self._lbl_total)

        self._timer = QTimer(self)
        self._timer.setInterval(_TICK_MS)
        self._timer.timeout.connect(self._on_timer_tick)
        self._elapsed = QElapsedTimer()

        self._chk_slice.toggled.connect(self._on_slice_toggled)
        self._btn_play.clicked.connect(self._toggle_play)
        self._slider.sliderPressed.connect(self._pause_playback)
        self._slider.valueChanged.connect(self._on_slider_changed)
        self._combo_speed.currentIndexChanged.connect(self._on_speed_changed)
        self._combo_window.currentIndexChanged.connect(self._on_window_changed)

        self.idle()

    def bindings_dict(self) -> dict[str, Any]:
        """Bindings passed to :func:`show_comparison_3d_pyvista` as ``time_slice_qt``."""
        return {
            "bar": self,
            "check": self._chk_slice,
            "window": self._combo_window,
            "play": self._btn_play,
            "speed": self._combo_speed,
            "slider": self._slider,
            "time_current": self._lbl_current,
            "time_total": self._lbl_total,
        }

    def slice_enabled(self) -> bool:
        return bool(self._chk_slice.isChecked())

    def start_ms(self) -> int:
        return int(self._slider.value())

    def speed_multiplier(self) -> float:
        return float(self._speed)

    def window_seconds(self) -> float:
        return float(self._window_s)

    def stop_playback(self) -> None:
        self._pause_playback()

    def idle(self) -> None:
        self.stop_playback()
        self._time_slice_cfg = None
        self._apply_slice = None
        self._timeline_time_s = np.zeros(0, dtype=np.float64)
        self._start_min_ms = 0
        self._start_max_ms = 0
        self._t_max_s = 0.0
        self._chk_slice.setEnabled(False)
        self._chk_slice.setChecked(False)
        self._combo_window.setEnabled(False)
        self._btn_play.setEnabled(False)
        self._combo_speed.setEnabled(False)
        self._slider.setEnabled(False)
        self._slider.setMinimum(0)
        self._slider.setMaximum(1)
        self._btn_play.setText(_SYM_PLAY)
        self._lbl_current.setText("—:—")
        self._lbl_total.setText("—:—")

    def wire(
        self,
        time_slice_cfg: dict[str, bool | int | float],
        meas_time_s: np.ndarray,
        *,
        apply_slice: Callable[[], None],
        saved_speed: float = 1.0,
    ) -> None:
        """Connect to plotter slice state after a successful 3D plot."""
        self.stop_playback()
        self._time_slice_cfg = time_slice_cfg
        self._apply_slice = apply_slice
        self._timeline_time_s = np.asarray(meas_time_s, dtype=np.float64).reshape(-1)

        if not bool(np.any(np.isfinite(self._timeline_time_s))):
            self.idle()
            return

        self._window_s = float(
            time_slice_cfg.get("window_s", TIME_SLICE_WINDOW_S_DEFAULT)
        )
        self._set_window_combo(self._window_s)
        self._sync_slider_range()

        cur_ms = int(time_slice_cfg.get("start_ms", self._start_min_ms))
        cur_ms = int(np.clip(cur_ms, self._start_min_ms, self._start_max_ms))
        time_slice_cfg["start_ms"] = cur_ms
        time_slice_cfg["window_s"] = self._window_s

        self._set_speed_combo(saved_speed)

        self._slider.blockSignals(True)
        self._slider.setValue(cur_ms)
        self._slider.blockSignals(False)

        self._chk_slice.blockSignals(True)
        self._chk_slice.setChecked(bool(time_slice_cfg.get("slice_on", False)))
        self._chk_slice.blockSignals(False)

        self._chk_slice.setEnabled(True)
        self._combo_window.setEnabled(True)
        self._slider.setEnabled(True)
        self._sync_playback_controls()
        self._refresh_labels()
        apply_slice()

    def _sync_slider_range(self) -> None:
        rng = _time_slice_range_ms(self._timeline_time_s, window_s=self._window_s)
        if rng is None:
            return
        start_min_ms, start_max_ms, _t_min, t_max = rng
        self._start_min_ms = int(start_min_ms)
        self._start_max_ms = int(start_max_ms)
        self._t_max_s = float(t_max)
        self._slider.blockSignals(True)
        self._slider.setMinimum(start_min_ms)
        self._slider.setMaximum(start_max_ms)
        self._slider.setSingleStep(1)
        cur = int(np.clip(int(self._slider.value()), start_min_ms, start_max_ms))
        self._slider.setValue(cur)
        self._slider.blockSignals(False)
        if self._time_slice_cfg is not None:
            self._time_slice_cfg["start_ms"] = cur

    def _sync_playback_controls(self) -> None:
        on = self.slice_enabled()
        self._btn_play.setEnabled(on)
        self._combo_speed.setEnabled(on)

    def _set_window_combo(self, window_s: float) -> None:
        best_i = 1
        best_d = abs(float(window_s) - 1.0)
        for i, (_, val) in enumerate(_WINDOW_PRESETS):
            d = abs(float(window_s) - float(val))
            if d < best_d:
                best_d = d
                best_i = i
        self._combo_window.blockSignals(True)
        self._combo_window.setCurrentIndex(best_i)
        self._combo_window.blockSignals(False)
        self._window_s = float(_WINDOW_PRESETS[best_i][1])

    def _set_speed_combo(self, speed: float) -> None:
        best_i = 2
        best_d = abs(float(speed) - 1.0)
        for i, (_, mult) in enumerate(_PLAYBACK_SPEEDS):
            d = abs(float(speed) - mult)
            if d < best_d:
                best_d = d
                best_i = i
        self._combo_speed.blockSignals(True)
        self._combo_speed.setCurrentIndex(best_i)
        self._combo_speed.blockSignals(False)
        self._speed = float(_PLAYBACK_SPEEDS[best_i][1])

    def _refresh_labels(self) -> None:
        if self._time_slice_cfg is None:
            self._lbl_current.setText("—:—")
            self._lbl_total.setText("—:—")
            return
        start_s = float(int(self._time_slice_cfg["start_ms"])) / 1000.0
        self._lbl_current.setText(format_timeline_seconds(start_s))
        self._lbl_total.setText(format_timeline_seconds(self._t_max_s))

    def _on_slice_toggled(self, checked: bool) -> None:
        if self._time_slice_cfg is None or self._apply_slice is None:
            return
        if not checked:
            self._pause_playback()
        self._time_slice_cfg["slice_on"] = bool(checked)
        self._sync_playback_controls()
        self._apply_slice()

    def _on_window_changed(self, index: int) -> None:
        if not (0 <= index < len(_WINDOW_PRESETS)):
            return
        self._window_s = float(_WINDOW_PRESETS[index][1])
        if self._time_slice_cfg is not None:
            self._time_slice_cfg["window_s"] = self._window_s
        self._sync_slider_range()
        if self._apply_slice is not None:
            self._apply_slice()

    def _toggle_play(self) -> None:
        if self._playing:
            self._pause_playback()
        else:
            self._start_playback()

    def _start_playback(self) -> None:
        if self._time_slice_cfg is None or self._apply_slice is None:
            return
        if not self.slice_enabled():
            return
        if int(self._time_slice_cfg["start_ms"]) >= self._start_max_ms:
            self._time_slice_cfg["start_ms"] = self._start_min_ms
            self._slider.blockSignals(True)
            self._slider.setValue(self._start_min_ms)
            self._slider.blockSignals(False)
            self._apply_slice()
        self._playing = True
        self._btn_play.setText(_SYM_PAUSE * 2)
        self._elapsed.start()
        self._timer.start()

    def _pause_playback(self) -> None:
        self._playing = False
        self._timer.stop()
        self._btn_play.setText(_SYM_PLAY)

    def _on_speed_changed(self, index: int) -> None:
        if 0 <= index < len(_PLAYBACK_SPEEDS):
            self._speed = float(_PLAYBACK_SPEEDS[index][1])

    def _on_slider_changed(self, val: int) -> None:
        if self._time_slice_cfg is None or self._apply_slice is None:
            return
        self._pause_playback()
        sm = int(np.clip(int(val), self._start_min_ms, self._start_max_ms))
        self._time_slice_cfg["start_ms"] = sm
        self._refresh_labels()
        if bool(self._time_slice_cfg.get("slice_on", False)):
            self._apply_slice()

    def _on_timer_tick(self) -> None:
        if not self._playing or self._time_slice_cfg is None or self._apply_slice is None:
            return
        if not self.slice_enabled():
            self._pause_playback()
            return
        elapsed = self._elapsed.restart()
        if elapsed <= 0:
            return
        cur = int(self._time_slice_cfg["start_ms"])
        nxt, at_end = advance_playback_start_ms(
            cur,
            elapsed_ms=float(elapsed),
            speed=self._speed,
            start_min_ms=self._start_min_ms,
            start_max_ms=self._start_max_ms,
        )
        self._time_slice_cfg["start_ms"] = nxt
        self._slider.blockSignals(True)
        self._slider.setValue(nxt)
        self._slider.blockSignals(False)
        self._refresh_labels()
        if bool(self._time_slice_cfg.get("slice_on", False)):
            self._apply_slice()
        if at_end:
            self._pause_playback()
