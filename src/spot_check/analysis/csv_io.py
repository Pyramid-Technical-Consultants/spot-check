"""Acquisition CSV text I/O (plain or gzip)."""

from __future__ import annotations

import csv
import gzip
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO

from spot_check.constants import GATE_COUNTER_KEY


def acquisition_csv_has_gate_counter(csv_path: Path) -> bool:
    """True when the acquisition CSV includes a Gate Counter column."""
    with open_acquisition_csv(csv_path) as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return False
    return GATE_COUNTER_KEY in [h.strip() for h in header]


def acquisition_csv_stem(csv_path: Path) -> str:
    """Stem for plan-tag inference and export naming (``.csv`` / ``.csv.gz``)."""
    name = csv_path.name
    lower = name.lower()
    if lower.endswith(".csv.gz"):
        return name[:-7]
    return csv_path.stem


@contextmanager
def open_acquisition_csv(csv_path: Path) -> Iterator[TextIO]:
    """Open acquisition export as UTF-8 text (BOM-tolerant), including ``.csv.gz``."""
    lower = csv_path.name.lower()
    if lower.endswith(".csv.gz"):
        with gzip.open(csv_path, mode="rt", encoding="utf-8-sig", newline="") as f:
            yield f
    elif lower.endswith(".csv"):
        with csv_path.open(newline="", encoding="utf-8-sig") as f:
            yield f
    else:
        raise ValueError(f"Expected .csv or .csv.gz, got {csv_path.name!r}")
