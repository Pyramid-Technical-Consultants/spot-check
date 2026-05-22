"""Fullscreen loading overlay for the embedded 3D view."""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from spot_check.gui.pipeline_progress import PipelineProgressWidget
from spot_check.pipeline.progress import ProgressEvent

_SPINNER_FRAMES: tuple[str, ...] = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
_SPINNER_MS = 90

_OVERLAY_BG = "rgba(13, 17, 23, 0.78)"
_CARD_BG = "#161b22"
_CARD_BORDER = "#30363d"
_ACCENT = "#58a6ff"
_TITLE = "#e6edf3"
_SUBTITLE = "#8b949e"
_FOOTER = "#6e7681"


class LoadOverlayPanel(QFrame):
    """Dimmed fullscreen overlay with a centered progress card."""

    def __init__(self, host: QWidget, *, app_name: str = "SpotCheck") -> None:
        super().__init__(host)
        self._host = host
        self._app_name = str(app_name)
        self._spinner_frame = 0
        self._visible_for_load = False

        self.setObjectName("loadOverlay")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(f"#loadOverlay {{ background-color: {_OVERLAY_BG}; }}")
        self.setVisible(False)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addStretch(1)

        row = QHBoxLayout()
        row.addStretch(1)

        self._card = QFrame()
        self._card.setObjectName("loadOverlayCard")
        self._card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._card.setAutoFillBackground(True)
        self._card.setStyleSheet(
            f"#loadOverlayCard {{ background-color: {_CARD_BG}; "
            f"border: 1px solid {_CARD_BORDER}; border-radius: 10px; }}"
            "#loadOverlayCard QLabel { background: transparent; }"
        )

        card_lay = QVBoxLayout(self._card)
        card_lay.setContentsMargins(24, 22, 24, 20)
        card_lay.setSpacing(14)

        header = QHBoxLayout()
        header.setSpacing(14)
        self._spinner = QLabel(_SPINNER_FRAMES[0])
        self._spinner.setFixedWidth(36)
        self._spinner.setAlignment(
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop
        )
        self._spinner.setStyleSheet(
            f"color: {_ACCENT}; font-size: 26pt; padding-top: 2px; background: transparent;"
        )
        header.addWidget(self._spinner)

        title_col = QVBoxLayout()
        title_col.setSpacing(4)
        self._title = QLabel("Loading data")
        self._title.setStyleSheet(
            f"color: {_TITLE}; font-size: 13pt; font-weight: 600; background: transparent;"
        )
        self._subtitle = QLabel("Preparing files…")
        self._subtitle.setWordWrap(True)
        self._subtitle.setStyleSheet(
            f"color: {_SUBTITLE}; font-size: 10pt; background: transparent;"
        )
        title_col.addWidget(self._title)
        title_col.addWidget(self._subtitle)
        header.addLayout(title_col, 1)
        card_lay.addLayout(header)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.NoFrame)
        divider.setFixedHeight(1)
        divider.setStyleSheet(f"background-color: {_CARD_BORDER}; border: none;")
        card_lay.addWidget(divider)

        self._progress = PipelineProgressWidget()
        card_lay.addWidget(self._progress)

        self._footer = QLabel(f"{self._app_name} · pipeline running on background thread")
        self._footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._footer.setStyleSheet(f"color: {_FOOTER}; font-size: 8pt; background: transparent;")
        card_lay.addWidget(self._footer)

        row.addWidget(self._card)
        row.addStretch(1)
        root.addLayout(row)
        root.addStretch(1)

        self._timer = QTimer(self)
        self._timer.setInterval(_SPINNER_MS)
        self._timer.timeout.connect(self._tick_spinner)

    def reset(self, *, message: str) -> None:
        self._spinner_frame = 0
        self._spinner.setText(_SPINNER_FRAMES[0])
        self._subtitle.setText(message)
        self._progress.reset(message=message)

    def apply_event(self, event: ProgressEvent) -> None:
        self._progress.apply_event(event)
        if event.message:
            self._subtitle.setText(event.message)

    def set_visualize_phase(self, *, message: str) -> None:
        self._title.setText("Building 3D view")
        self._subtitle.setText(message)
        self._progress.set_visualize_phase(message=message)

    def mark_complete(self) -> None:
        self._progress.mark_complete()

    def sync_geometry(self) -> None:
        host_w = max(1, int(self._host.width()))
        host_h = max(1, int(self._host.height()))
        self.setGeometry(0, 0, host_w, host_h)
        card_w = min(420, max(300, int(host_w * 0.34)))
        self._card.setFixedWidth(card_w)
        self.raise_()

    def show_loading(self) -> None:
        self._visible_for_load = True
        self._title.setText("Loading data")
        self.sync_geometry()
        self.show()
        self.raise_()
        if not self._timer.isActive():
            self._timer.start()

    def hide_loading(self) -> None:
        self._visible_for_load = False
        self._timer.stop()
        self.hide()

    @property
    def is_loading(self) -> bool:
        return self._visible_for_load

    def _tick_spinner(self) -> None:
        self._spinner_frame = (self._spinner_frame + 1) % len(_SPINNER_FRAMES)
        self._spinner.setText(_SPINNER_FRAMES[self._spinner_frame])
