"""Logging setup for SpotCheck scripts and the desktop GUI.

Configures the root logger once (``basicConfig``) so operators can set ``SPOT_CHECK_LOG``
without touching application code.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Final, TextIO

_DEFAULT_FORMAT: Final[str] = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


def configure_logging(
    level: int | None = None,
    *,
    stream: TextIO | None = None,
    format_string: str = _DEFAULT_FORMAT,
) -> None:
    """
    Apply ``basicConfig`` once if the root logger has no handlers (typical in scripts).

    Level from ``level``, or env ``SPOT_CHECK_LOG`` (DEBUG/INFO/WARNING/ERROR), default WARNING.
    """
    root = logging.getLogger()
    if root.handlers:
        return
    if level is None:
        raw = (os.environ.get("SPOT_CHECK_LOG") or "WARNING").strip().upper()
        candidate = getattr(logging, raw, None)
        level = candidate if isinstance(candidate, int) else logging.WARNING
    logging.basicConfig(
        level=int(level),
        format=format_string,
        stream=stream if stream is not None else sys.stderr,
        force=False,
    )
