"""Background loading on a worker thread."""

from __future__ import annotations

import logging
from typing import Any, Callable

from PySide6.QtCore import QObject, QRunnable, Signal

from spot_check.pipeline.progress import ProgressEvent

logger = logging.getLogger(__name__)


class PipelineLoaderSignals(QObject):
    finished = Signal(object, int)
    failed = Signal(str, int)
    progress = Signal(object, int)


class PipelineLoadRunnable(QRunnable):
    def __init__(
        self,
        fn: Callable[[], Any],
        signals: PipelineLoaderSignals,
        generation: int,
    ) -> None:
        super().__init__()
        self._fn = fn
        self._signals = signals
        self._generation = generation

    def run(self) -> None:  # noqa: PLR6301
        try:
            out = self._fn()
            self._signals.finished.emit(out, self._generation)
        except Exception as exc:
            logger.exception("Background plan/CSV load failed")
            self._signals.failed.emit(str(exc), self._generation)

    @staticmethod
    def emit_progress(
        signals: PipelineLoaderSignals,
        generation: int,
        event: ProgressEvent,
    ) -> None:
        signals.progress.emit(event, generation)
