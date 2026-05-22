"""Fine detector alignment (post-aggregate weighted Gauss–Newton)."""

from __future__ import annotations

import math

import numpy as np

from spot_check.analysis.alignment import fine_align_measured_to_plan, measured_plan_xy_from_row
from spot_check.analysis.spatial import nominal_layer_energies_mev
from spot_check.pipeline.phases.fine_align import run_fine_align_phase
from spot_check.pipeline.progress import CallbackProgressSink, ProgressEvent
from spot_check.pipeline.types import PHASE_FINE_ALIGN, PipelineConfig, PipelineState


def _layer_idx_for_plan_spot(planned: list[tuple[float, float, float]], *, i: int) -> int:
    pe = float(planned[i][2])
    for ell, enn in enumerate(nominal_layer_energies_mev(planned)):
        if abs(pe - enn) <= 1e-4:
            return ell
    return 0


def _invert_fine_xy(
    px: float,
    py: float,
    *,
    theta_deg: float,
    sx: float,
    sy: float,
    tx: float,
    ty: float,
) -> tuple[float, float]:
    """Given ``p`` matched by ``p ≈ R(θ) @ diag(sx,sy) @ m + t``, return ``m`` in plan XY."""
    rad = math.radians(float(theta_deg))
    cos_t, sin_t = math.cos(rad), math.sin(rad)
    r_mat = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)
    pvec = np.array([float(px), float(py)], dtype=np.float64)
    tvec = np.array([float(tx), float(ty)], dtype=np.float64)
    v = r_mat.T @ (pvec - tvec)
    return float(v[0] / float(sx)), float(v[1] / float(sy))


def _measured_tuple_from_syn_plan_xy(mx: float, my: float, layer: float) -> tuple[float, ...]:
    """a_is_x=False: Fit A=my, Fit B=mx."""
    return (
        float(my),
        float(mx),
        float(layer),
        1.0,
        0,
        float("nan"),
        float("nan"),
        1.0,
    )


def test_fine_align_all_off_returns_none_info() -> None:
    planned = [
        (0.0, 0.0, 150.0),
        (3.5, -1.0, 150.0),
        (2.0, 2.75, 150.0),
    ]
    rows = [
        _measured_tuple_from_syn_plan_xy(px, py, 0.0) for px, py, _e in planned
    ]
    out, info = fine_align_measured_to_plan(
        planned,
        rows,
        allow_xy=False,
        allow_rotation=False,
        allow_scale=False,
    )
    assert info is None
    assert out == rows


def test_fine_align_xy_only_zeros_rotation_scale() -> None:
    planned = [(0.0, 0.0, 150.0), (5.0, 0.0, 150.0)]
    shift = (-1.15, 0.72)
    rows = [
        _measured_tuple_from_syn_plan_xy(px - shift[0], py - shift[1], 0.0)
        for px, py, _ in planned
    ]
    _, info = fine_align_measured_to_plan(
        planned,
        rows,
        allow_xy=True,
        allow_rotation=False,
        allow_scale=False,
    )
    assert info is not None
    assert abs(info.theta_deg) < 1e-3
    assert abs(info.sx - 1.0) < 5e-3
    assert abs(info.sy - 1.0) < 5e-3
    assert abs(info.tx_mm - shift[0]) < 5e-2
    assert abs(info.ty_mm - shift[1]) < 5e-2


def test_fine_align_recovers_delivery_similarity() -> None:
    planned = [
        (10.0, 0.0, 140.0),
        (14.5, -2.25, 140.0),
        (11.0, 6.75, 140.0),
        (17.75, 4.125, 140.0),
    ]
    truth = {"theta_deg": 4.75, "sx": 1.035, "sy": 0.985, "tx": 1.625, "ty": -0.4}
    rows: list[tuple[float, ...]] = []
    for i, (px, py, _) in enumerate(planned):
        mx, my = _invert_fine_xy(
            px,
            py,
            theta_deg=truth["theta_deg"],
            sx=truth["sx"],
            sy=truth["sy"],
            tx=truth["tx"],
            ty=truth["ty"],
        )
        li = float(_layer_idx_for_plan_spot(planned, i=i))
        rows.append(_measured_tuple_from_syn_plan_xy(mx, my, li))

    for i, row in enumerate(rows):
        mx, my = measured_plan_xy_from_row(row, a_is_x=False)
        px, py, _pe = planned[i]
        ux, uy = mx * truth["sx"], my * truth["sy"]
        rad = math.radians(truth["theta_deg"])
        c, s = math.cos(rad), math.sin(rad)
        xf = ux * c - uy * s + truth["tx"]
        yf = ux * s + uy * c + truth["ty"]
        assert abs(xf - px) < 1e-5 and abs(yf - py) < 1e-5

    out, info = fine_align_measured_to_plan(planned, rows, a_is_x=False)
    assert info is not None and len(out) == len(rows)
    rtol = 0.15
    assert abs(info.theta_deg - truth["theta_deg"]) < 0.22
    assert abs(info.sx - truth["sx"]) < rtol
    assert abs(info.sy - truth["sy"]) < rtol
    assert abs(info.tx_mm - truth["tx"]) < 0.2
    assert abs(info.ty_mm - truth["ty"]) < 0.2


def test_fine_align_improves_when_delivery_order_differs_from_plan() -> None:
    """Layer-NN pairing must work when row count matches plan but order does not."""
    planned = [
        (10.0, 0.0, 140.0),
        (14.5, -2.25, 140.0),
        (11.0, 6.75, 140.0),
        (17.75, 4.125, 140.0),
    ]
    truth = {"theta_deg": 4.75, "sx": 1.035, "sy": 0.985, "tx": 1.625, "ty": -0.4}
    rows: list[tuple[float, ...]] = []
    for i, (px, py, _) in enumerate(planned):
        mx, my = _invert_fine_xy(
            px,
            py,
            theta_deg=truth["theta_deg"],
            sx=truth["sx"],
            sy=truth["sy"],
            tx=truth["tx"],
            ty=truth["ty"],
        )
        li = float(_layer_idx_for_plan_spot(planned, i=i))
        rows.append(_measured_tuple_from_syn_plan_xy(mx, my, li))
    permuted = [rows[2], rows[0], rows[3], rows[1]]

    out, info = fine_align_measured_to_plan(planned, permuted, a_is_x=False)
    assert info is not None and len(out) == len(permuted)
    assert info.rms_after_mm <= info.rms_before_mm + 1e-3
    rtol = 0.15
    assert abs(info.theta_deg - truth["theta_deg"]) < 0.22
    assert abs(info.sx - truth["sx"]) < rtol
    assert abs(info.sy - truth["sy"]) < rtol


def test_run_fine_align_phase_reports_progress_phase_id() -> None:
    planned = [
        (0.0, 0.0, 150.0),
        (3.5, -1.0, 150.0),
        (2.0, 2.75, 150.0),
    ]
    truth = dict(theta_deg=-2.0, sx=1.02, sy=0.99, tx=-0.4, ty=0.66)
    rows = []
    for i, (px, py, _) in enumerate(planned):
        mx, my = _invert_fine_xy(
            px,
            py,
            theta_deg=truth["theta_deg"],
            sx=truth["sx"],
            sy=truth["sy"],
            tx=truth["tx"],
            ty=truth["ty"],
        )
        rows.append(_measured_tuple_from_syn_plan_xy(mx, my, 0.0))

    st = PipelineState()
    st.planned = planned
    st.measured_unaligned = rows
    cfg = PipelineConfig(
        plan_path=None,
        csv_path=None,
        layer_assign_mode="time_gap",
        aggregate_spots=False,
        spot_weight_mode="channel_sum",
    )
    events: list[ProgressEvent] = []
    run_fine_align_phase(st, cfg, CallbackProgressSink(events.append))
    assert PHASE_FINE_ALIGN in {ev.phase_id for ev in events}
    assert st.measured_fine_aligned is not None
