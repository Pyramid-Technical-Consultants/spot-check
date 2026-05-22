"""Dispatch plan loading by file type (DICOM or Pyramid CSV)."""

from __future__ import annotations

from pathlib import Path

import pydicom

from spot_check.exceptions import PlanDataError

from .dicom import (
    planned_spot_xyz_and_counts_from_dicom,
    planned_spot_xyz_and_counts_from_dicom_dataset,
)
from .pyramid_csv import (
    is_pyramid_plan_csv,
    plan_label_from_pyramid_csv,
    planned_spot_xyz_and_counts_from_pyramid_csv,
)


def is_plan_dicom(path: Path) -> bool:
    return path.suffix.lower() == ".dcm"


def is_supported_plan_file(path: Path) -> bool:
    if not path.is_file():
        return False
    if is_plan_dicom(path):
        return True
    return path.suffix.lower() == ".csv" and is_pyramid_plan_csv(path)


def plan_label_from_path(plan_path: Path) -> str:
    if is_plan_dicom(plan_path):
        ds = pydicom.dcmread(plan_path, stop_before_pixels=True, force=True)
        return str(ds.get("RTPlanLabel", "") or "").strip()
    if is_pyramid_plan_csv(plan_path):
        return plan_label_from_pyramid_csv(plan_path)
    raise PlanDataError(f"Unsupported plan file: {plan_path.name}")


def planned_spot_xyz_and_counts_from_plan(
    plan_path: Path,
) -> tuple[list[tuple[float, float, float]], object, object, int, int]:
    if is_plan_dicom(plan_path):
        return planned_spot_xyz_and_counts_from_dicom(plan_path)
    if is_pyramid_plan_csv(plan_path):
        return planned_spot_xyz_and_counts_from_pyramid_csv(plan_path)
    raise PlanDataError(
        f"Unsupported plan file {plan_path.name!r}: use RT Ion DICOM (.dcm) "
        "or Pyramid plan CSV (X/Y position, energy, charge columns)"
    )


def load_plan_from_path(
    plan_path: Path,
) -> tuple[str, list[tuple[float, float, float]], object, object, int, int]:
    """Read plan label and spot table in one file pass (DICOM or Pyramid CSV)."""
    if is_plan_dicom(plan_path):
        ds = pydicom.dcmread(plan_path, stop_before_pixels=True, force=True)
        label = str(ds.get("RTPlanLabel", "") or "").strip()
        planned, fwhm, mu, n_kept, n_raw = planned_spot_xyz_and_counts_from_dicom_dataset(
            ds
        )
        return label, planned, fwhm, mu, n_kept, n_raw
    if is_pyramid_plan_csv(plan_path):
        label = plan_label_from_pyramid_csv(plan_path)
        planned, fwhm, mu, n_kept, n_raw = planned_spot_xyz_and_counts_from_pyramid_csv(
            plan_path
        )
        return label, planned, fwhm, mu, n_kept, n_raw
    raise PlanDataError(
        f"Unsupported plan file {plan_path.name!r}: use RT Ion DICOM (.dcm) "
        "or Pyramid plan CSV (X/Y position, energy, charge columns)"
    )
