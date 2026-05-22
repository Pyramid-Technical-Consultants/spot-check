"""Total alignment caption for GUI status line."""

from __future__ import annotations

import math

from spot_check.analysis.alignment import (
    fine_align_measured_to_plan,
    format_total_detector_align_caption,
)
from spot_check.models import DetectorRigidAlign2D


def _measured_row(*, a: float, b: float, layer: float = 0.0) -> tuple[float, ...]:
    return (float(a), float(b), float(layer), 1.0, 0, float("nan"), float("nan"), 1.0)


def test_total_caption_none_when_no_align() -> None:
    assert format_total_detector_align_caption() is None


def test_total_caption_coarse_only() -> None:
    coarse = DetectorRigidAlign2D(
        theta_deg=1.25,
        tx_mm=-0.5,
        ty_mm=2.0,
        rms_nn_mm=3.0,
        rms_residual_mm=0.8,
        n_pairs=12,
        from_coarse_phase=True,
    )
    rows = [_measured_row(a=1.0, b=2.0)]
    cap = format_total_detector_align_caption(
        coarse=coarse,
        measured_base=rows,
        measured_final=rows,
        a_is_x=False,
    )
    assert cap is not None
    assert cap.startswith("Alignment:")
    assert "θ=1.25° CCW" in cap
    assert "t=(-0.5, 2) mm" in cap
    assert "QA RMS=0.8 mm" in cap
    assert "coarse flat" not in cap
    assert "Fine detector" not in cap


def test_total_caption_fine_only() -> None:
    planned = [(0.0, 0.0, 150.0), (5.0, 0.0, 150.0)]
    base_rows = [
        _measured_row(a=py + 0.4, b=px - 0.25, layer=0.0) for px, py, _ in planned
    ]
    out_rows, info = fine_align_measured_to_plan(
        planned,
        base_rows,
        allow_xy=True,
        allow_rotation=False,
        allow_scale=False,
    )
    assert info is not None
    cap = format_total_detector_align_caption(
        fine=info,
        measured_base=base_rows,
        measured_final=out_rows,
        a_is_x=False,
    )
    assert cap is not None
    assert "Alignment:" in cap
    assert "QA RMS=" in cap
    assert math.isfinite(info.rms_after_mm)
    assert f"QA RMS={info.rms_after_mm:.3g} mm" in cap
    assert "Fine detector" not in cap


def test_total_caption_coarse_plus_fine_from_rows() -> None:
    planned = [(0.0, 0.0, 150.0), (4.0, 1.0, 150.0), (-2.0, 3.0, 150.0)]
    coarse = DetectorRigidAlign2D(
        theta_deg=0.5,
        tx_mm=0.1,
        ty_mm=-0.2,
        rms_nn_mm=2.0,
        rms_residual_mm=1.1,
        n_pairs=3,
        from_coarse_phase=True,
    )
    # Post-coarse rows (pipeline measured_unaligned) with small residual error.
    post_coarse = [
        _measured_row(a=py + 0.05, b=px + 0.03, layer=0.0) for px, py, _ in planned
    ]
    out_rows, fine_info = fine_align_measured_to_plan(
        planned,
        post_coarse,
        allow_xy=True,
        allow_rotation=True,
        allow_scale=False,
    )
    assert fine_info is not None
    cap = format_total_detector_align_caption(
        coarse=coarse,
        fine=fine_info,
        measured_base=post_coarse,
        measured_final=out_rows,
        a_is_x=False,
    )
    assert cap is not None
    assert cap.count("θ=") == 1
    assert "Detector align" not in cap
    assert "Fine detector" not in cap
    assert f"QA RMS={fine_info.rms_after_mm:.3g} mm" in cap
