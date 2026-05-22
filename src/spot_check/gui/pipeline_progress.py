"""Pipeline progress overlay widget — phase step list and weighted progress bar."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QFontMetrics
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QProgressBar, QVBoxLayout, QWidget

from spot_check.pipeline.progress import ProgressEvent
from spot_check.pipeline.types import ALL_PHASE_IDS, PHASE_LABELS, PHASE_WEIGHTS

_STATUS_PENDING = "○"
_STATUS_ACTIVE = "●"
_STATUS_DONE = "✓"

_MUTED = "#8b949e"
_ACTIVE = "#58a6ff"
_DONE = "#3fb950"
_MSG = "#c9d1d9"
_SECTION = "#6e7681"
_BAR_TRACK = "#21262d"
_MSG_FONT_PT = 10


def _status_message_min_height() -> int:
    """Reserve space for two wrapped status lines (10pt) to avoid layout jumps."""
    font = QFont()
    font.setPointSize(_MSG_FONT_PT)
    font.setWeight(QFont.Weight.Medium)
    return QFontMetrics(font).lineSpacing() * 2


class PipelineProgressWidget(QFrame):
    """Shows pipeline phase status and a weighted overall progress bar."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("pipelineProgress")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setAutoFillBackground(False)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet("#pipelineProgress { background: transparent; }")
        self._phase_labels: dict[str, QLabel] = {}
        self._phase_status: dict[str, str] = dict.fromkeys(ALL_PHASE_IDS, _STATUS_PENDING)
        self._active_phase: str | None = None
        self._phase_fraction: dict[str, float] = dict.fromkeys(ALL_PHASE_IDS, 0.0)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)

        self._msg = QLabel("Starting pipeline…")
        self._msg.setWordWrap(True)
        self._msg.setMinimumHeight(_status_message_min_height())
        self._msg.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        self._msg.setStyleSheet(
            f"color: {_MSG}; font-size: {_MSG_FONT_PT}pt; font-weight: 500; "
            "background: transparent;"
        )
        lay.addWidget(self._msg)

        bar_row = QHBoxLayout()
        bar_row.setSpacing(10)
        self._bar = QProgressBar()
        self._bar.setRange(0, 1000)
        self._bar.setValue(0)
        self._bar.setFixedHeight(8)
        self._bar.setTextVisible(False)
        self._bar.setStyleSheet(
            f"QProgressBar {{ background-color: {_BAR_TRACK}; border: none; "
            "border-radius: 4px; }"
            f"QProgressBar::chunk {{ background-color: {_ACTIVE}; border-radius: 4px; }}"
        )
        bar_row.addWidget(self._bar, 1)
        self._pct = QLabel("0%")
        self._pct.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._pct.setFixedWidth(40)
        self._pct.setStyleSheet(
            f"color: {_MUTED}; font-size: 9pt; font-weight: 600; background: transparent;"
        )
        bar_row.addWidget(self._pct)
        lay.addLayout(bar_row)

        section = QLabel("PIPELINE STEPS")
        section.setStyleSheet(
            f"color: {_SECTION}; font-size: 8pt; font-weight: 600; "
            "letter-spacing: 0.6px; background: transparent;"
        )
        lay.addWidget(section)

        phase_box = QVBoxLayout()
        phase_box.setSpacing(2)
        for phase_id in ALL_PHASE_IDS:
            row = QLabel(f"{_STATUS_PENDING}  {PHASE_LABELS[phase_id]}")
            row.setStyleSheet(self._phase_style(_STATUS_PENDING))
            self._phase_labels[phase_id] = row
            phase_box.addWidget(row)
        lay.addLayout(phase_box)

    def reset(self, *, message: str = "Starting pipeline…") -> None:
        self._active_phase = None
        self._phase_fraction = dict.fromkeys(ALL_PHASE_IDS, 0.0)
        self._phase_status = dict.fromkeys(ALL_PHASE_IDS, _STATUS_PENDING)
        self._refresh_phase_rows()
        self._msg.setText(message)
        self._set_bar_value(0)

    def apply_event(self, event: ProgressEvent) -> None:
        phase_id = event.phase_id
        if phase_id not in self._phase_labels:
            return

        if self._active_phase is not None and phase_id != self._active_phase:
            idx_new = ALL_PHASE_IDS.index(phase_id)
            idx_old = ALL_PHASE_IDS.index(self._active_phase)
            if idx_new > idx_old:
                for pid in ALL_PHASE_IDS[:idx_new]:
                    self._phase_status[pid] = _STATUS_DONE
                    self._phase_fraction[pid] = 1.0

        self._active_phase = phase_id
        self._phase_status[phase_id] = _STATUS_ACTIVE

        if event.current is not None and event.total is not None and event.total > 0:
            frac = min(1.0, max(0.0, float(event.current) / float(event.total)))
            self._phase_fraction[phase_id] = frac
        elif event.step.endswith("_done") or event.step.endswith("_complete"):
            self._phase_status[phase_id] = _STATUS_DONE
            self._phase_fraction[phase_id] = 1.0

        if event.message:
            self._msg.setText(event.message)
        self._refresh_phase_rows()
        self._set_bar_value(int(round(self._overall_fraction() * 1000)))

    def set_visualize_phase(self, *, message: str) -> None:
        """Mark data phases done and activate visualization (main thread)."""
        for pid in ALL_PHASE_IDS[:-1]:
            self._phase_status[pid] = _STATUS_DONE
            self._phase_fraction[pid] = 1.0
        viz_id = ALL_PHASE_IDS[-1]
        self._active_phase = viz_id
        self._phase_status[viz_id] = _STATUS_ACTIVE
        self._msg.setText(message)
        self._refresh_phase_rows()
        self._set_bar_value(int(round(self._overall_fraction() * 1000)))

    def mark_complete(self) -> None:
        for pid in ALL_PHASE_IDS:
            self._phase_status[pid] = _STATUS_DONE
            self._phase_fraction[pid] = 1.0
        self._active_phase = None
        self._refresh_phase_rows()
        self._set_bar_value(1000)

    def _overall_fraction(self) -> float:
        total = 0.0
        for phase_id in ALL_PHASE_IDS:
            w = PHASE_WEIGHTS.get(phase_id, 0.0)
            total += w * self._phase_fraction.get(phase_id, 0.0)
        return min(1.0, max(0.0, total))

    def _set_bar_value(self, value: int) -> None:
        clamped = min(1000, max(0, int(value)))
        self._bar.setValue(clamped)
        pct = clamped / 10.0
        if pct >= 100.0:
            self._pct.setText("100%")
        elif pct <= 0.0:
            self._pct.setText("0%")
        else:
            self._pct.setText(f"{pct:.0f}%")

    @staticmethod
    def _phase_style(status: str) -> str:
        if status == _STATUS_ACTIVE:
            color = _ACTIVE
            weight = "600"
        elif status == _STATUS_DONE:
            color = _DONE
            weight = "500"
        else:
            color = _MUTED
            weight = "400"
        return (
            f"color: {color}; font-size: 9pt; font-weight: {weight}; "
            "padding: 2px 0; background: transparent;"
        )

    def _refresh_phase_rows(self) -> None:
        for phase_id, lbl in self._phase_labels.items():
            status = self._phase_status.get(phase_id, _STATUS_PENDING)
            lbl.setText(f"{status}  {PHASE_LABELS[phase_id]}")
            lbl.setStyleSheet(self._phase_style(status))
