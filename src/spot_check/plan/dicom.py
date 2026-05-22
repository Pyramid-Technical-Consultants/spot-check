"""Load planned spot sequences from RT Ion DICOM."""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pydicom

from spot_check.analysis.csv_io import acquisition_csv_stem
from spot_check.constants import CSV_LABEL_RE, project_root
from spot_check.exceptions import PlanDataError

logger = logging.getLogger(__name__)

DEFAULT_PLAN_SEARCH_DIR = project_root()


def _flatten_ds_list(seq: Any) -> list:
    if seq is None:
        return []
    try:
        return list(item for item in seq if item is not None)
    except TypeError:
        return []


def infer_csv_plan_tag(stem: str) -> str | None:
    m = CSV_LABEL_RE.match(stem)
    return m.group(1) if m else None


def rt_plan_label_from_csv_stem(stem: str) -> str | None:
    tag = infer_csv_plan_tag(stem)
    if not tag:
        return None
    m = re.match(r"(T0G\d+)", tag, re.IGNORECASE)
    return m.group(1).upper() if m else None


def find_dicom_for_csv(
    csv_path: Path,
    folder: Path = DEFAULT_PLAN_SEARCH_DIR,
) -> Path:
    label = rt_plan_label_from_csv_stem(acquisition_csv_stem(csv_path))
    if not label:
        raise ValueError(f"Cannot infer RT plan label from CSV name: {csv_path.name}")
    for p in sorted(folder.glob("*.dcm")):
        ds = pydicom.dcmread(p, stop_before_pixels=True, force=True)
        plan = str(ds.get("RTPlanLabel", "") or "").strip().upper()
        if plan == label.upper():
            return p
    raise FileNotFoundError(f"No DICOM in {folder} with RTPlanLabel matching {label!r}")


def _scanning_spot_size_xy_mm_from_cp(cp: Any) -> tuple[float, float] | None:
    raw = cp.get("ScanningSpotSize", None)
    if raw is None:
        return None
    try:
        seq = list(raw)
        if len(seq) >= 2:
            fx, fy = float(seq[0]), float(seq[1])
            if (
                math.isfinite(fx)
                and math.isfinite(fy)
                and fx > 0.0
                and fy > 0.0
                and fx < 500.0
                and fy < 500.0
            ):
                return fx, fy
    except (TypeError, ValueError):
        pass
    return None


def _iter_planned_spot_slots_from_dataset(ds: Any):
    for beam in _flatten_ds_list(ds.get("IonBeamSequence")):
        for cp in _flatten_ds_list(beam.get("IonControlPointSequence")):
            n = int(cp.get("NumberOfScanSpotPositions", 0) or 0)
            sm = cp.get("ScanSpotPositionMap", None)
            if not n or sm is None:
                continue
            energy = float(cp.get("NominalBeamEnergy", 0.0) or 0.0)
            coords = list(sm)
            limit = min(len(coords), 2 * n)
            weights: list[float] | None
            raw_w = cp.get("ScanSpotMetersetWeights", None)
            if raw_w is None:
                weights = None
            else:
                try:
                    weights = [float(x) for x in raw_w]
                except (TypeError, ValueError):
                    weights = None
            fwhm_cp = _scanning_spot_size_xy_mm_from_cp(cp)
            for i in range(0, limit, 2):
                si = i // 2
                drop = False
                mu = float("nan")
                if weights is not None and si < len(weights):
                    mu = float(weights[si])
                    if not math.isfinite(mu) or mu <= 0.0:
                        drop = True
                tpl = (float(coords[i]), float(coords[i + 1]), energy)
                yield drop, tpl, fwhm_cp, mu


def planned_spot_xyz_and_counts_from_dicom_dataset(
    ds: Any,
) -> tuple[list[tuple[float, float, float]], np.ndarray | None, np.ndarray, int, int]:
    out: list[tuple[float, float, float]] = []
    mu_kept: list[float] = []
    fwhm_kept: list[tuple[float, float] | None] = []
    n_raw = 0
    any_fwhm = False
    for drop, t, fwhm, mu in _iter_planned_spot_slots_from_dataset(ds):
        n_raw += 1
        if not drop:
            out.append(t)
            mu_kept.append(mu)
            fwhm_kept.append(fwhm)
            if fwhm is not None:
                any_fwhm = True
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
    n_kept = len(out)
    if n_kept == 0:
        raise PlanDataError("No planned spots extracted from DICOM")
    plan_mu = np.asarray(mu_kept, dtype=np.float64)
    return out, fwhm_arr, plan_mu, n_kept, n_raw


def planned_spot_xyz_and_counts_from_dicom(
    dcm_path: Path,
) -> tuple[list[tuple[float, float, float]], np.ndarray | None, np.ndarray, int, int]:
    ds = pydicom.dcmread(dcm_path, stop_before_pixels=True, force=True)
    out, fwhm_arr, plan_mu, n_kept, n_raw = planned_spot_xyz_and_counts_from_dicom_dataset(ds)
    logger.info(
        "DICOM plan loaded: %s — %s spots kept, %s raw map slots",
        dcm_path,
        n_kept,
        n_raw,
    )
    return out, fwhm_arr, plan_mu, n_kept, n_raw


def planned_spot_xyz_from_dicom(dcm_path: Path) -> list[tuple[float, float, float]]:
    spots, _, _, _, _ = planned_spot_xyz_and_counts_from_dicom(dcm_path)
    return spots


def planned_spot_position_counts_from_dicom(dcm_path: Path) -> tuple[int, int]:
    ds = pydicom.dcmread(dcm_path, stop_before_pixels=True, force=True)
    n_raw = 0
    n_kept = 0
    for dropped, _t, _fw, _mu in _iter_planned_spot_slots_from_dataset(ds):
        n_raw += 1
        if not dropped:
            n_kept += 1
    return n_kept, n_raw
