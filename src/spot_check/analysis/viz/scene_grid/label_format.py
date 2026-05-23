"""Numeric tick label formatting for scene grid."""

from __future__ import annotations


def format_grid_label(value: float, fmt: str) -> str:
    if fmt.startswith("%"):
        return fmt % float(value)
    if fmt:
        return fmt.format(float(value))
    return str(float(value))
