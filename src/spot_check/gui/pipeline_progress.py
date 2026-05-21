"""Pipeline progress overlay widget — phase step list and weighted progress bar."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QLabel, QProgressBar, QVBoxLayout, QWidget

from spot_check.pipeline.progress import ProgressEvent
from spot_check.pipeline.types import ALL_PHASE_IDS, PHASE_LABELS, PHASE_WEIGHTS

_STATUS_PENDING = "○"
_STATUS_ACTIVE = "◉"
_STATUS_DONE = "✓"
_STATUS_SKIP = "—"

_MUTED = "#8b949e"
_ACTIVE = "#58a6ff"
_DONE = "#3fb950"
_MSG = "#c9d1d9"


class PipelineProgressWidget(QFrame):
    """Shows pipeline phase status and a weighted overall progress bar."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._phase_labels: dict[str, QLabel] = {}
        self._phase_status: dict[str, str] = dict.fromkeys(ALL_PHASE_IDS, _STATUS_PENDING)
        self._active_phase: str | None = None
        self._phase_fraction: dict[str, float] = dict.fromkeys(ALL_PHASE_IDS, 0.0)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        for phase_id in ALL_PHASE_IDS:
            row = QLabel(f"{_STATUS_PENDING}  {PHASE_LABELS[phase_id]}")
            row.setStyleSheet(f"color: {_MUTED}; font-size: 9pt;")
            self._phase_labels[phase_id] = row
            lay.addWidget(row)

        self._msg = QLabel("Loading…")
        self._msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._msg.setWordWrap(True)
        self._msg.setStyleSheet(f"color: {_MSG}; font-size: 11pt; font-weight: 600;")
        lay.addWidget(self._msg)

        self._bar = QProgressBar()
        self._bar.setRange(0, 1000)
        self._bar.setValue(0)
        self._bar.setFixedHeight(6)
        self._bar.setTextVisible(False)
        self._bar.setStyleSheet(
            "QProgressBar { background-color: #21262d; border: none; border-radius: 3px; }"
            "QProgressBar::chunk { background-color: #58a6ff; border-radius: 3px; }"
        )
        lay.addWidget(self._bar)

    def reset(self, *, message: str = "Starting pipeline…") -> None:
        self._active_phase = None
        self._phase_fraction = dict.fromkeys(ALL_PHASE_IDS, 0.0)
        self._phase_status = dict.fromkeys(ALL_PHASE_IDS, _STATUS_PENDING)
        self._refresh_phase_rows()
        self._msg.setText(message)
        self._bar.setValue(0)

    def apply_event(self, event: ProgressEvent) -> None:
        phase_id = event.phase_id
        if phase_id not in self._phase_labels:
            return

        # Mark earlier phases done when a later phase starts.
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

        self._msg.setText(event.message)
        self._refresh_phase_rows()
        self._bar.setValue(int(round(self._overall_fraction() * 1000)))

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
        self._bar.setValue(int(round(self._overall_fraction() * 1000)))

    def mark_complete(self) -> None:
        for pid in ALL_PHASE_IDS:
            self._phase_status[pid] = _STATUS_DONE
            self._phase_fraction[pid] = 1.0
        self._active_phase = None
        self._refresh_phase_rows()
        self._bar.setValue(1000)

    def _overall_fraction(self) -> float:
        total = 0.0
        for phase_id in ALL_PHASE_IDS:
            w = PHASE_WEIGHTS.get(phase_id, 0.0)
            total += w * self._phase_fraction.get(phase_id, 0.0)
        return min(1.0, max(0.0, total))

    def _refresh_phase_rows(self) -> None:
        for phase_id, lbl in self._phase_labels.items():
            status = self._phase_status.get(phase_id, _STATUS_PENDING)
            if status == _STATUS_ACTIVE:
                color = _ACTIVE
            elif status == _STATUS_DONE:
                color = _DONE
            else:
                color = _MUTED
            lbl.setText(f"{status}  {PHASE_LABELS[phase_id]}")
            lbl.setStyleSheet(f"color: {color}; font-size: 9pt;")
