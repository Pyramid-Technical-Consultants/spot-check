"""Load planned spot sequences from Pyramid plan export CSV."""

from __future__ import annotations

import csv
import logging
import math
import re
from pathlib import Path
from typing import TextIO

import numpy as np

from spot_check.exceptions import PlanDataError

logger = logging.getLogger(__name__)

_REQUIRED_COLS = frozenset({"ENERGY", "X_POSITION", "Y_POSITION"})
_OPTIONAL_MU_COL = "CHARGE_REQ"
_OPTIONAL_FWHM_COL = "BEAM_SIZE"


def _normalize_plan_col(name: str) -> str:
    s = str(name or "").strip().lstrip("#").strip().upper()
    if "(" in s:
        s = s.split("(", 1)[0].strip()
    return s


def _parse_pyramid_plan_header_line(line: str) -> list[str]:
    raw = line.strip().lstrip("\ufeff")
    if raw.startswith("#"):
        raw = raw[1:]
    return [_normalize_plan_col(c) for c in next(csv.reader([raw]))]


def pyramid_plan_csv_column_names(csv_path: Path) -> frozenset[str] | None:
    """Return normalized column names when the file looks like a Pyramid plan CSV."""
    try:
        with csv_path.open(newline="", encoding="utf-8-sig") as f:
            line = f.readline()
    except OSError:
        return None
    if not line.strip():
        return None
    cols = _parse_pyramid_plan_header_line(line)
    if not _REQUIRED_COLS.issubset(cols):
        return None
    return frozenset(cols)


def is_pyramid_plan_csv(csv_path: Path) -> bool:
    return pyramid_plan_csv_column_names(csv_path) is not None


def plan_label_from_pyramid_csv_stem(stem: str) -> str:
    m = re.match(r"^(.+?)_cube(?:_|$)", stem, re.IGNORECASE)
    if m:
        return m.group(1)
    return stem


def plan_label_from_pyramid_csv(csv_path: Path) -> str:
    return plan_label_from_pyramid_csv_stem(csv_path.stem)


def _float_cell(raw: str) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float("nan")


_PyramidRow = tuple[float, float, float, float | None, float]


def _iter_pyramid_plan_rows(
    f: TextIO,
) -> tuple[list[str], list[_PyramidRow]]:
    header_line = f.readline()
    if not header_line.strip():
        raise PlanDataError("Pyramid plan CSV is empty")
    cols = _parse_pyramid_plan_header_line(header_line)
    if not _REQUIRED_COLS.issubset(cols):
        missing = sorted(_REQUIRED_COLS - set(cols))
        raise PlanDataError(
            f"Pyramid plan CSV missing column(s): {', '.join(missing)} "
            f"(expected X/Y position, energy, and optional charge / beam size)"
        )
    idx = {name: i for i, name in enumerate(cols)}
    ix = idx["X_POSITION"]
    iy = idx["Y_POSITION"]
    ie = idx["ENERGY"]
    imu = idx.get(_OPTIONAL_MU_COL)
    ibs = idx.get(_OPTIONAL_FWHM_COL)
    rows: list[_PyramidRow] = []
    for line_no, raw in enumerate(f, start=2):
        if not raw.strip():
            continue
        cells = next(csv.reader([raw]))
        if len(cells) <= max(ix, iy, ie):
            raise PlanDataError(f"Pyramid plan CSV row {line_no}: not enough columns")
        x = _float_cell(cells[ix])
        y = _float_cell(cells[iy])
        energy = _float_cell(cells[ie])
        mu = _float_cell(cells[imu]) if imu is not None and imu < len(cells) else float("nan")
        beam = (
            _float_cell(cells[ibs])
            if ibs is not None and ibs < len(cells)
            else float("nan")
        )
        if not all(math.isfinite(v) for v in (x, y, energy)):
            continue
        rows.append((x, y, energy, beam if math.isfinite(beam) and beam > 0.0 else None, mu))
    return cols, rows


def planned_spot_xyz_and_counts_from_pyramid_csv(
    csv_path: Path,
) -> tuple[list[tuple[float, float, float]], np.ndarray | None, np.ndarray, int, int]:
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        _, parsed = _iter_pyramid_plan_rows(f)
    n_raw = len(parsed)
    out: list[tuple[float, float, float]] = []
    mu_kept: list[float] = []
    fwhm_kept: list[tuple[float, float] | None] = []
    any_fwhm = False
    for x, y, energy, beam, mu in parsed:
        drop = not math.isfinite(mu) or mu <= 0.0
        if not drop:
            out.append((x, y, energy))
            mu_kept.append(mu)
            fwhm = (beam, beam) if beam is not None else None
            fwhm_kept.append(fwhm)
            if fwhm is not None:
                any_fwhm = True
    n_kept = len(out)
    if n_kept == 0:
        raise PlanDataError("No planned spots extracted from Pyramid plan CSV")
    fwhm_arr: np.ndarray | None = None
    if any_fwhm and fwhm_kept:
        fx = np.empty(len(fwhm_kept), dtype=np.float64)
        fy = np.empty(len(fwhm_kept), dtype=np.float64)
        for i, pair in enumerate(fwhm_kept):
            if pair is not None:
                fx[i] = float(pair[0])
                fy[i] = float(pair[1])
            else:
                fx[i] = np.nan
                fy[i] = np.nan
        fwhm_arr = np.column_stack([fx, fy])
    plan_mu = np.asarray(mu_kept, dtype=np.float64)
    logger.info(
        "Pyramid plan CSV loaded: %s — %s spots kept, %s raw rows",
        csv_path,
        n_kept,
        n_raw,
    )
    return out, fwhm_arr, plan_mu, n_kept, n_raw
