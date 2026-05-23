"""Frameless popup card for double-clicked spot metadata."""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, QPoint, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from spot_check.analysis.viz.spot_info import SpotInfoRow

_CARD_BG = "#161b22"
_CARD_BORDER = "#30363d"
_TITLE = "#e6edf3"
_LABEL = "#8b949e"
_VALUE = "#e6edf3"
_MARGIN = 12


class SpotInfoPopup(QFrame):
    """Non-modal spot inspect card; dismiss on Escape or click outside."""

    def __init__(self, host: QWidget) -> None:
        super().__init__(host, Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint)
        self._host = host
        self.setObjectName("spotInfoPopup")
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, False)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            f"#spotInfoPopup {{ background-color: {_CARD_BG}; "
            f"border: 1px solid {_CARD_BORDER}; border-radius: 8px; }}"
            "#spotInfoPopup QLabel { background: transparent; }"
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(_MARGIN, _MARGIN, _MARGIN, _MARGIN)
        root.setSpacing(8)

        self._heading = QLabel("Spot")
        self._heading.setStyleSheet(
            f"color: {_TITLE}; font-size: 11pt; font-weight: 600; background: transparent;"
        )
        root.addWidget(self._heading)

        self._grid_host = QWidget()
        self._grid = QGridLayout(self._grid_host)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(12)
        self._grid.setVerticalSpacing(4)
        root.addWidget(self._grid_host)

        self._hint = QLabel("Double-click another spot to inspect · Esc to close")
        self._hint.setWordWrap(True)
        self._hint.setStyleSheet(f"color: {_LABEL}; font-size: 8pt; background: transparent;")
        root.addWidget(self._hint)

        self.hide()

    def show_spot(
        self,
        *,
        title: str,
        rows: list[SpotInfoRow],
        local_x: int,
        local_y: int,
    ) -> None:
        self._heading.setText(title)
        while self._grid.count():
            item = self._grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        for row_i, row in enumerate(rows):
            lbl = QLabel(row.label)
            lbl.setStyleSheet(f"color: {_LABEL}; font-size: 9pt; background: transparent;")
            val = QLabel(row.value)
            val.setWordWrap(True)
            val.setStyleSheet(f"color: {_VALUE}; font-size: 9pt; background: transparent;")
            self._grid.addWidget(lbl, row_i, 0, Qt.AlignmentFlag.AlignTop)
            self._grid.addWidget(val, row_i, 1, Qt.AlignmentFlag.AlignTop)

        self.adjustSize()
        self._position_near(local_x, local_y)
        self.show()
        self.raise_()

    def _position_near(self, local_x: int, local_y: int) -> None:
        pad = 16
        offset_x, offset_y = 12, 12
        x = int(local_x) + offset_x
        y = int(local_y) + offset_y
        host_w = max(1, int(self._host.width()))
        host_h = max(1, int(self._host.height()))
        w = self.sizeHint().width()
        h = self.sizeHint().height()
        if x + w + pad > host_w:
            x = max(pad, int(local_x) - w - offset_x)
        if y + h + pad > host_h:
            y = max(pad, int(local_y) - h - offset_y)
        self.move(max(0, x), max(0, y))

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.hide()
            event.accept()
            return
        super().keyPressEvent(event)

    @staticmethod
    def dismiss_if_outside_click(global_pos: QPoint, popup: SpotInfoPopup | None) -> None:
        if popup is None or not popup.isVisible():
            return
        if not popup.geometry().contains(popup.mapFromGlobal(global_pos)):
            popup.hide()


def install_spot_popup_dismiss_filter(host: QWidget, popup: SpotInfoPopup) -> None:
    """Hide popup when user clicks outside it on the host widget."""

    class _DismissFilter(QObject):
        def eventFilter(self, obj: QWidget, event: QEvent) -> bool:
            if event.type() == QEvent.Type.MouseButtonPress and popup.isVisible():
                gp = event.globalPosition().toPoint()
                local = popup.mapFromGlobal(gp)
                if not popup.rect().contains(local):
                    popup.hide()
            return False

    filt = _DismissFilter(host)
    host.installEventFilter(filt)
    host._spot_info_dismiss_filter = filt  # type: ignore[attr-defined]
