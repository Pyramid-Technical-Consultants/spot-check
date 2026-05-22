"""Media-style timeline bar for 1 s acquisition window playback under the 3D view."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
from PySide6.QtCore import QElapsedTimer, Qt, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QWidget,
)

from spot_check.analysis.viz.data import _time_slice_range_ms

_BAR_BG = "#161b22"
_BAR_BORDER = "#30363d"
_ACCENT = "#58a6ff"
_MUTED = "#8b949e"
_BTN_HOVER = "#21262d"

_PLAYBACK_SPEEDS: tuple[tuple[str, float], ...] = (
    ("0.25×", 0.25),
    ("0.5×", 0.5),
    ("1×", 1.0),
    ("2×", 2.0),
    ("4×", 4.0),
    ("8×", 8.0),
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


class TimelinePlaybackBar(QFrame):
    """Play/pause, speed, scrub slider, and current/total labels for the 1 s window."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("timelinePlaybackBar")
        self.setFixedHeight(52)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            f"#timelinePlaybackBar {{ background-color: {_BAR_BG}; "
            f"border-top: 1px solid {_BAR_BORDER}; }}"
        )

        self._playing = False
        self._start_min_ms = 0
        self._start_max_ms = 0
        self._t_max_s = 0.0
        self._speed = 1.0
        self._time_slice_cfg: dict[str, bool | int | float] | None = None
        self._apply_slice: Callable[[], None] | None = None

        row = QHBoxLayout(self)
        row.setContentsMargins(10, 8, 12, 8)
        row.setSpacing(8)

        self._btn_play = QPushButton("▶")
        self._btn_play.setFixedSize(36, 36)
        self._btn_play.setToolTip("Play / pause 1 s acquisition window")
        self._btn_play.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_play.setStyleSheet(
            f"QPushButton {{ background: transparent; color: {_ACCENT}; "
            f"border: 1px solid {_BAR_BORDER}; border-radius: 6px; font-size: 14pt; }}"
            f"QPushButton:hover {{ background: {_BTN_HOVER}; }}"
            f"QPushButton:disabled {{ color: #484f58; border-color: #21262d; }}"
        )
        row.addWidget(self._btn_play)

        self._combo_speed = QComboBox()
        self._combo_speed.setFixedHeight(36)
        for label, _ in _PLAYBACK_SPEEDS:
            self._combo_speed.addItem(label)
        self._combo_speed.setToolTip("Playback speed (real-time multiplier)")
        self._combo_speed.setStyleSheet(
            f"QComboBox {{ background: {_BTN_HOVER}; color: #e6edf3; "
            f"border: 1px solid {_BAR_BORDER}; border-radius: 6px; padding: 2px 8px; "
            f"min-width: 4.5em; }}"
            f"QComboBox:disabled {{ color: #484f58; }}"
            f"QComboBox::drop-down {{ border: none; }}"
        )
        row.addWidget(self._combo_speed)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setTracking(True)
        self._slider.setStyleSheet(
            f"QSlider::groove:horizontal {{ height: 6px; background: #21262d; "
            f"border-radius: 3px; }}"
            f"QSlider::handle:horizontal {{ width: 14px; margin: -5px 0; "
            f"background: {_ACCENT}; border-radius: 7px; }}"
            f"QSlider::sub-page:horizontal {{ background: #388bfd66; border-radius: 3px; }}"
        )
        row.addWidget(self._slider, 1)

        _lbl_style = (
            f"color: {_MUTED}; font-family: Consolas, 'Courier New', monospace; "
            f"font-size: 10pt; background: transparent;"
        )
        self._lbl_current = QLabel("—:—")
        self._lbl_current.setMinimumWidth(52)
        self._lbl_current.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._lbl_current.setStyleSheet(_lbl_style)
        row.addWidget(self._lbl_current)

        sep = QLabel("/")
        sep.setStyleSheet("color: #484f58; font-size: 10pt; background: transparent;")
        row.addWidget(sep)

        self._lbl_total = QLabel("—:—")
        self._lbl_total.setMinimumWidth(52)
        self._lbl_total.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._lbl_total.setStyleSheet(_lbl_style)
        row.addWidget(self._lbl_total)

        self._timer = QTimer(self)
        self._timer.setInterval(_TICK_MS)
        self._timer.timeout.connect(self._on_timer_tick)
        self._elapsed = QElapsedTimer()

        self._btn_play.clicked.connect(self._toggle_play)
        self._slider.sliderPressed.connect(self._pause_playback)
        self._slider.valueChanged.connect(self._on_slider_changed)
        self._combo_speed.currentIndexChanged.connect(self._on_speed_changed)

        self.idle()

    def bindings_dict(self) -> dict[str, Any]:
        """Bindings passed to :func:`show_comparison_3d_pyvista` as ``time_slice_qt``."""
        return {
            "bar": self,
            "play": self._btn_play,
            "speed": self._combo_speed,
            "slider": self._slider,
            "time_current": self._lbl_current,
            "time_total": self._lbl_total,
        }

    def start_ms(self) -> int:
        return int(self._slider.value())

    def speed_multiplier(self) -> float:
        return float(self._speed)

    def stop_playback(self) -> None:
        self._pause_playback()

    def idle(self) -> None:
        self.stop_playback()
        self._time_slice_cfg = None
        self._apply_slice = None
        self._start_min_ms = 0
        self._start_max_ms = 0
        self._t_max_s = 0.0
        self._btn_play.setEnabled(False)
        self._combo_speed.setEnabled(False)
        self._slider.setEnabled(False)
        self._slider.setMinimum(0)
        self._slider.setMaximum(1)
        self._btn_play.setText("▶")
        self._lbl_current.setText("—:—")
        self._lbl_total.setText("—:—")

    def wire(
        self,
        time_slice_cfg: dict[str, bool | int | float],
        meas_time_s: np.ndarray,
        *,
        window_s: float,
        apply_slice: Callable[[], None],
        saved_speed: float = 1.0,
    ) -> None:
        """Connect to plotter slice state after a successful 3D plot."""
        self.stop_playback()
        self._time_slice_cfg = time_slice_cfg
        self._apply_slice = apply_slice

        rng = _time_slice_range_ms(meas_time_s, window_s=window_s)
        if rng is None:
            self.idle()
            return

        start_min_ms, start_max_ms, _t_min, t_max = rng
        self._start_min_ms = int(start_min_ms)
        self._start_max_ms = int(start_max_ms)
        self._t_max_s = float(t_max)

        time_slice_cfg["slice_on"] = True
        cur_ms = int(time_slice_cfg.get("start_ms", start_min_ms))
        cur_ms = int(np.clip(cur_ms, start_min_ms, start_max_ms))
        time_slice_cfg["start_ms"] = cur_ms

        self._set_speed_combo(saved_speed)

        self._slider.blockSignals(True)
        self._slider.setMinimum(start_min_ms)
        self._slider.setMaximum(start_max_ms)
        self._slider.setSingleStep(1)
        self._slider.setValue(cur_ms)
        self._slider.blockSignals(False)

        self._btn_play.setEnabled(True)
        self._combo_speed.setEnabled(True)
        self._slider.setEnabled(True)
        self._refresh_labels()
        apply_slice()

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

    def _toggle_play(self) -> None:
        if self._playing:
            self._pause_playback()
        else:
            self._start_playback()

    def _start_playback(self) -> None:
        if self._time_slice_cfg is None or self._apply_slice is None:
            return
        if int(self._time_slice_cfg["start_ms"]) >= self._start_max_ms:
            self._time_slice_cfg["start_ms"] = self._start_min_ms
            self._slider.blockSignals(True)
            self._slider.setValue(self._start_min_ms)
            self._slider.blockSignals(False)
            self._apply_slice()
        self._playing = True
        self._btn_play.setText("⏸")
        self._elapsed.start()
        self._timer.start()

    def _pause_playback(self) -> None:
        self._playing = False
        self._timer.stop()
        self._btn_play.setText("▶")

    def _on_speed_changed(self, index: int) -> None:
        if 0 <= index < len(_PLAYBACK_SPEEDS):
            self._speed = float(_PLAYBACK_SPEEDS[index][1])

    def _on_slider_changed(self, val: int) -> None:
        if self._time_slice_cfg is None or self._apply_slice is None:
            return
        sm = int(np.clip(int(val), self._start_min_ms, self._start_max_ms))
        self._time_slice_cfg["start_ms"] = sm
        self._refresh_labels()
        self._apply_slice()

    def _on_timer_tick(self) -> None:
        if not self._playing or self._time_slice_cfg is None or self._apply_slice is None:
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
        self._apply_slice()
        if at_end:
            self._pause_playback()
