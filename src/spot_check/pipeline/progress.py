"""Progress reporting for the data processing pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol


@dataclass(frozen=True)
class ProgressEvent:
    """One progress update from a pipeline phase."""

    phase_id: str
    step: str
    message: str
    current: int | None = None
    total: int | None = None


class ProgressSink(Protocol):
    """Receive progress events from pipeline phases (worker or main thread)."""

    def report(self, event: ProgressEvent) -> None: ...


class NullProgressSink:
    """No-op sink for tests and headless callers."""

    def report(self, event: ProgressEvent) -> None:
        del event


class CallbackProgressSink:
    """Forward events to a callable (e.g. Qt signal emit)."""

    def __init__(self, callback: Callable[[ProgressEvent], None]) -> None:
        self._callback = callback

    def report(self, event: ProgressEvent) -> None:
        self._callback(event)
