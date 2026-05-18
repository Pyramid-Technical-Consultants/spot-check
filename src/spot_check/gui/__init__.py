"""SpotCheck desktop application (PySide6 + PyVista)."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["main", "run_gui"]

if TYPE_CHECKING:
    from collections.abc import Callable


def __getattr__(name: str) -> "Callable[[], None]":
    if name in __all__:
        from spot_check.gui.app import main, run_gui

        return main if name == "main" else run_gui
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
