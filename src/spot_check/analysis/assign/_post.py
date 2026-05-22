"""Post-assignment row transforms shared by layer assigners."""

from __future__ import annotations

from spot_check.models import DetectorRigidAlign2D


def apply_coarse_flat_to_rows(
    rows: list[tuple[float, ...]],
    *,
    transform: DetectorRigidAlign2D | None,
    a_is_x: bool,
) -> list[tuple[float, ...]]:
    if transform is None or not rows:
        return rows
    from spot_check.analysis.alignment import apply_detector_rigid2d_xy_to_measured_rows

    return apply_detector_rigid2d_xy_to_measured_rows(rows, transform, a_is_x=a_is_x)
