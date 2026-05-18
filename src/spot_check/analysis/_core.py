"""
RT Ion plan vs acquisition — DICOM + CSV → PyVista 3D comparison.

**Regulatory:** This is engineering analysis software, not a medical device. Operational
qualification and clinical use are the responsibility of the deploying site. Tunable defaults
and heuristics live in :mod:`spot_check.constants`; do not change them without documented review.

Planned beams: spot (x, y, nominal energy) from the Ion Control Point maps; optional
**Scanning Spot Size** (300A,0398) FWHM in mm (gantry X, Y) can scale plan markers as thin
ellipsoids in 3D (see ``scale_plan_spots_by_dicom_fwhm`` in :func:`show_comparison_3d_pyvista`).

Layer assignment (nominal energy index per row):

- **time_gap** — ``TIME_LAYER_GAP_S_DEFAULT`` plus refill heuristics (constants below).
  ``Gate Counter`` is ignored unless ``aggregate_spots=True``.

- **plan_viterbi** — global decode: each row keeps measured A/B; layer index comes from
  a monotone path (stay or +1 layer) minimizing distance-to-plan, plus a penalty for
  advancing. No invented coordinates; works when the machine changes energy with no timing gap.

- **unified** — same Viterbi objective, but the advance penalty is **per row**: base geometry
  penalty plus extras when ``Δt < layer_gap_s`` (discourage stepping without a timing “slot”)
  and when a long gap still has **same-spot XY** (refill; strongly block stepping). Tunes
  together with ``viterbi_advance_penalty_mm2``, ``layer_gap_s``, and refill mm settings.

- **gate_counter** — uses CSV ``Gate Counter``: **odd** values mark a **spot** phase (many
  consecutive rows may share the same count — one spot); **even** marks **deadtime** (also
  many rows per value). Nominal layer follows DICOM spot order; **spot index advances only
  when the gate count changes to a new odd value**. Requires ``planned_xyz``.

**Refill (time_gap mode):** a large Δt only *suppresses* a nominal energy step if the fit returns
near the **same plan XY** as the last row before the gap (``REFILL_SAME_SPOT_XY_TOLERANCE_MM``),
otherwise a step may occur when the next slice explains XY better
(``_layer_advance_plausible_vs_refill``).

**Spot aggregation (optional):** when ``aggregate_spots=True``, each run of consecutive rows with
the same **odd** ``Gate Counter`` value is one delivered spot (**even** gate ends the spot), or
with ``aggregate_even_rows_after_odd`` > 0 in **gate_counter** mode, up to that many **even-phase**
rows with valid fits after each odd→even transition are merged into the same weighted mean.
Rows in that run collapse to one point: **weighted means** of fit mean positions
(after imputation), nominal layer index, and fit σ_A / σ_B when present. Weights come from
``spot_weight_mode``: IX512 channel sum and/or Fit Amplitude A/B columns (see
:func:`measured_spot_weight_from_row`). Each measured row is a 7-tuple
``(..., σ_A, σ_B)`` (σ may be NaN); aggregation replaces runs with one weighted-mean row.

**Detector XY alignment (optional):** :func:`align_measured_to_plan_detector_xy` matches each
measured row to the nearest plan spot on its assigned nominal layer, then fits a 2D rigid
transform (rotation + translation) minimizing RMSE to those plan targets and applies it to
all measured A/B positions in plan coordinates (see :class:`DetectorRigidAlign2D`). Large
detector rotations (90° and beyond) and A↔B axis swaps are handled via multi-start ICP with
coarse rotation seeds.

**Plan QA coloring (optional):** colors each measured point from Euclidean XY distance to the
nearest plan spot on its assigned layer: green ≤ pass mm (default 1), amber between pass and
warn mm (default 3), red > warn mm (see :func:`measured_rgba_by_plan_qa`). Measured markers
are drawn as flat circular discs; optional **spot-weight opacity** (see
``weight_measured_by_channel``) or
Optionally draw **error lines** from warn/fail points to that nearest plan spot (see
``plan_qa_draw_error_lines`` in :func:`show_comparison_3d_pyvista`).
When ``plan_qa_hide_pass_spots`` is true, pass-tier measured points are dropped from the cloud
so only warn and fail markers are drawn (counts in the caption still include all tiers).

In 3D, **nominal-layer band** filtering can be driven from the GUI when ``embed_qt`` / ``slice_qt``
are passed (PySide6 host + checkbox / slider bindings from :mod:`spot_check.gui`). Standalone
runs use PyVista widgets: **5-layer slice** vs **full stack**, with a **center layer** control in
plan order.
Plan QA error lines follow the same filter.
"""

from __future__ import annotations

import bisect
import csv
import importlib.util
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

if importlib.util.find_spec("pydicom") is None:  # pragma: no cover
    raise ImportError("RT Ion analysis requires pydicom. Install with: pip install pydicom")

try:
    import pyvista as pv
except ImportError:  # pragma: no cover
    pv = None  # type: ignore[assignment]

import logging

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:  # pragma: no cover
    tk = None  # type: ignore[assignment, misc]
    ttk = None  # type: ignore[assignment, misc]

from spot_check.constants import (
    _DEFAULT_PLAN_FWHM_MM_WHEN_MISSING,
    _MEASURED_COLOR_3D,
    _PARTIAL_AXIS_MEAS_COLOR_3D,
    _PLAN_COLOR_3D,
    _PLAN_FWHM_GLYPH_Z_SPAN_FRAC,
    _PLAN_QA_DOSE_UNDER_FAIL_HEX,
    _PLAN_QA_DOSE_UNDER_WARN_HEX,
    _PLAN_QA_FAIL_HEX,
    _PLAN_QA_PASS_HEX,
    _PLAN_QA_WARN_HEX,
    _SPOT_WEIGHT_MODES,
    AGGREGATE_EVEN_ROWS_AFTER_ODD_DEFAULT,
    AGGREGATE_EVEN_TAIL_MAX,
    BOUNDS_XY_TICK_MM_DEFAULT,
    CHANNEL_SUM_KEY,
    DETECTOR_ALIGN_MAX_FIT_SAMPLES,
    DISPLAY_GLYPH_INSTANCE_CAP,
    DISPLAY_POINT_MESH_TARGET,
    FIT_AMPLITUDE_A_KEY,
    FIT_AMPLITUDE_B_KEY,
    GATE_COUNTER_KEY,
    MEASURED_SIGMA_GLYPH_FALLBACK_MM,
    MEASURED_SIGMA_GLYPH_MAX_MM,
    MEASURED_SIGMA_GLYPH_MIN_MM,
    MEASURED_SIGMA_GLYPH_SCALE_DEFAULT,
    PLAN_QA_DOSE_PASS_PP_DEFAULT,
    PLAN_QA_DOSE_WARN_PP_DEFAULT,
    PLAN_QA_PASS_MM_DEFAULT,
    PLAN_QA_WARN_MM_DEFAULT,
    REFILL_REJECT_EXTRA_MM,
    REFILL_REJECT_RATIO,
    REFILL_SAME_SPOT_XY_TOLERANCE_MM,
    REFILL_TRUST_TIME_GAP_STAY_DIST_MM,
    SIGMA_A_KEY,
    SIGMA_B_KEY,
    SPOT_WEIGHT_MODE_DEFAULT,
    TIME_LAYER_GAP_S_DEFAULT,
    UNIFIED_SAME_SPOT_REFILL_BLOCK_MM2,
    UNIFIED_SHORT_DT_EXTRA_MM2,
    VITERBI_LAYER_ADVANCE_PENALTY_MM2_DEFAULT,
    project_root,
)
from spot_check.exceptions import (
    AcquisitionDataError,
    GeometryConfigError,
    PlanDataError,
)
from spot_check.geometry import (
    PYVISTA_CUBE_AXES_GRID,
    PYVISTA_CUBE_AXES_LOCATION,
    PYVISTA_CUBE_AXES_TICKS,
    apply_pyvista_cube_z_axis,
    n_cube_axis_labels_for_mm_step,
    nominal_mev_to_plot_z,
)
from spot_check.geometry import (
    cube_z_axis_spec as _cube_z_axis_spec,
)
from spot_check.models import Comparison3DData, CubeZAxisSpec, DetectorRigidAlign2D

FOLDER = project_root()
logger = logging.getLogger(__name__)

_CubeZAxisSpec = CubeZAxisSpec


def _opt_float_cell(row: dict[str, str], key: str) -> float | None:
    raw = (row.get(key) or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


@dataclass(frozen=True)
class _PlanImputeLookup:
    """Nearest plan spot along X or Y only; sorted-axis queries O(log N)."""

    pts: np.ndarray  # (N, 2) float64
    ord_x: np.ndarray  # indices, argsort pts[:, 0]
    ord_y: np.ndarray  # indices, argsort pts[:, 1]

    @classmethod
    def from_xy(cls, xy: np.ndarray) -> _PlanImputeLookup | None:
        arr = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        if arr.shape[0] == 0:
            return None
        return cls(
            pts=arr,
            ord_x=np.argsort(arr[:, 0]),
            ord_y=np.argsort(arr[:, 1]),
        )


def _impute_plan_axis_fast(
    lk: _PlanImputeLookup,
    mx: float | None,
    my: float | None,
) -> tuple[float, float]:
    """Copy measured axis; fill the other from the closest plan spot on that axis (1D).

    Matches the original linear scan: all spots achieving min |Δaxis| are considered;
    tie-break is the smallest row index (plan / bucket order).
    """
    px, py = lk.pts[:, 0], lk.pts[:, 1]
    ox, oy = lk.ord_x, lk.ord_y
    if mx is not None and my is not None:
        return float(mx), float(my)
    if mx is None and my is None:
        i0 = int(ox[0])
        return float(px[i0]), float(py[i0])
    tol_coord = 1e-9

    def _pick_after_expanding(
        spv: np.ndarray, ordv: np.ndarray, mcoord: float
    ) -> tuple[float, float]:
        """spv = sorted coordinate; ordv maps sorted pos -> row index."""
        n = int(spv.shape[0])
        if n == 0:
            return float("nan"), float("nan")
        idx = int(np.searchsorted(spv, mcoord, side="left"))
        pos_cand: list[int] = []
        if idx > 0:
            pos_cand.append(idx - 1)
        if idx < n:
            pos_cand.append(idx)
        best_d = min(abs(float(spv[i]) - mcoord) for i in pos_cand)
        tol = max(tol_coord, 1e-12 * max(1.0, abs(mcoord)))
        lo = min(pos_cand)
        hi = max(pos_cand)
        while lo > 0 and abs(float(spv[lo - 1]) - mcoord) <= best_d + tol:
            lo -= 1
        while hi + 1 < n and abs(float(spv[hi + 1]) - mcoord) <= best_d + tol:
            hi += 1
        cand_rows = ordv[lo : hi + 1].astype(np.int64, copy=False)
        best_j = int(np.min(cand_rows))
        return float(px[best_j]), float(py[best_j])

    if mx is not None:
        spx = px[ox]
        bx, by = _pick_after_expanding(spx, ox, float(mx))
        if math.isnan(bx):
            return float(mx), 0.0
        return bx, by
    spv_y = py[oy]
    bx, by = _pick_after_expanding(spv_y, oy, float(my))  # type: ignore[arg-type]
    if math.isnan(bx):
        return 0.0, float(my)  # type: ignore[arg-type]
    return bx, by


def _plan_impute_lookups_per_layer(layer_xy: list[np.ndarray]) -> list[_PlanImputeLookup | None]:
    out: list[_PlanImputeLookup | None] = []
    for arr in layer_xy:
        out.append(_PlanImputeLookup.from_xy(np.asarray(arr, dtype=np.float64)))
    return out


def _plan_xy_from_optional_ab(
    a_opt: float | None,
    b_opt: float | None,
    *,
    a_is_x: bool,
) -> tuple[float | None, float | None, int]:
    """Map optional raw fit A/B to plan (X,Y). Returns partial: 0=both, 1=raw A missing, 2=raw B
    missing, -1=both missing."""
    if a_opt is not None and b_opt is not None:
        mx, my = fit_ab_to_plan_xy(a_opt, b_opt, a_is_x=a_is_x)
        return mx, my, 0
    if a_opt is None and b_opt is None:
        return None, None, -1
    if a_is_x:
        if a_opt is None:
            return None, float(b_opt), 1
        return float(a_opt), None, 2
    if a_opt is None:
        return float(b_opt), None, 1
    return None, float(a_opt), 2


def _ab_from_plan_xy(mx: float, my: float, *, a_is_x: bool) -> tuple[float, float]:
    return (mx, my) if a_is_x else (my, mx)


def _channel_sum_na_from_row(row: dict[str, str]) -> float:
    """Display weight proxy (nA); use 1.0 if column missing or invalid."""
    raw = (row.get(CHANNEL_SUM_KEY) or "").strip()
    if not raw:
        return 1.0
    try:
        return max(float(raw), 1e-9)
    except ValueError:
        return 1.0


def normalize_measured_spot_weight_mode(mode: str) -> str:
    """Return valid spot weight key: ``channel_sum`` | ``fit_amplitude_a`` | ``fit_amplitude_b``."""
    m = str(mode).strip().lower().replace("-", "_")
    aliases = {
        "ix512": "channel_sum",
        "channel_sum": "channel_sum",
        "channel": "channel_sum",
        "fit_amplitude_a": "fit_amplitude_a",
        "amplitude_a": "fit_amplitude_a",
        "amp_a": "fit_amplitude_a",
        "fa": "fit_amplitude_a",
        "fit_amplitude_b": "fit_amplitude_b",
        "amplitude_b": "fit_amplitude_b",
        "amp_b": "fit_amplitude_b",
        "fb": "fit_amplitude_b",
    }
    out = aliases.get(m, m)
    if out not in _SPOT_WEIGHT_MODES:
        raise ValueError(
            f"spot_weight_mode must be one of {sorted(_SPOT_WEIGHT_MODES)}, got {mode!r}"
        )
    return out


def measured_spot_weight_caption(mode: str) -> str:
    """Short label for plot caption (aggregate / tint source)."""
    m = normalize_measured_spot_weight_mode(mode)
    if m == "channel_sum":
        return "IX512 channel sum"
    if m == "fit_amplitude_a":
        return "Fit Amplitude A"
    return "Fit Amplitude B"


def _sigma_cell_to_float(v: float | None) -> float:
    if v is None:
        return float("nan")
    try:
        f = float(v)
        return f if math.isfinite(f) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _measured_row_with_sigma(
    a: float,
    b: float,
    layer: float,
    weight: float,
    partial: int,
    sa: float | None,
    sb: float | None,
    *,
    channel_sum_na: float | None = None,
) -> tuple[float, ...]:
    """One measured row as 8-tuple ``(A, B, layer, weight, partial, σ_A, σ_B, channel_sum_nA)``."""
    ch = float(channel_sum_na) if channel_sum_na is not None else float(weight)
    return (
        float(a),
        float(b),
        float(layer),
        float(weight),
        int(partial),
        _sigma_cell_to_float(sa),
        _sigma_cell_to_float(sb),
        ch,
    )


def measured_charge_na_from_tuple(tup: tuple[float, ...]) -> float:
    """Measured charge (nA) for dose QA from row weight (``spot_weight_mode`` at load).

    Tuple index 3 is the same weight used for aggregation and opacity tint. Index 7 is the
    IX512 channel sum when present; it is used only if weight is missing or non-positive.
    """
    if len(tup) > 3:
        w = float(tup[3])
        if math.isfinite(w) and w > 0.0:
            return w
    if len(tup) > 7:
        ch = float(tup[7])
        if math.isfinite(ch) and ch > 0.0:
            return ch
    return 1e-18


def measured_spot_weight_from_row(row: dict[str, str], mode: str) -> float:
    """Positive weight for aggregation tuple[3] and 3D tint input; falls back to channel sum if cell
    empty."""
    m = normalize_measured_spot_weight_mode(mode)
    if m == "channel_sum":
        return _channel_sum_na_from_row(row)
    key = FIT_AMPLITUDE_A_KEY if m == "fit_amplitude_a" else FIT_AMPLITUDE_B_KEY
    raw = (row.get(key) or "").strip()
    if not raw:
        return _channel_sum_na_from_row(row)
    try:
        return max(float(raw), 1e-9)
    except ValueError:
        return _channel_sum_na_from_row(row)


def _probe_csv_columns_for_measured_weights(
    csv_path: Path,
    *,
    aggregate_spots: bool,
    spot_weight_mode: str,
) -> None:
    swm = normalize_measured_spot_weight_mode(spot_weight_mode)
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        pr = csv.DictReader(f)
        fn = pr.fieldnames
        if not fn:
            return
        if aggregate_spots and GATE_COUNTER_KEY not in fn:
            raise ValueError(f"aggregate_spots needs a “{GATE_COUNTER_KEY}” column (found {fn!r})")
        has_ch = CHANNEL_SUM_KEY in fn
        if swm == "channel_sum" and not has_ch:
            raise ValueError(
                f"spot_weight_mode={swm!r} requires CSV column {CHANNEL_SUM_KEY!r} "
                f"(found columns: {list(fn)!r})"
            )
        # Fit amp A/B optional when IX512 sum exists; empty/missing cells use channel sum.
        if swm == "fit_amplitude_a" and FIT_AMPLITUDE_A_KEY not in fn and not has_ch:
            raise ValueError(
                f"spot_weight_mode={swm!r} needs {FIT_AMPLITUDE_A_KEY!r} or {CHANNEL_SUM_KEY!r} "
                f"(found columns: {list(fn)!r})"
            )
        if swm == "fit_amplitude_b" and FIT_AMPLITUDE_B_KEY not in fn and not has_ch:
            raise ValueError(
                f"spot_weight_mode={swm!r} needs {FIT_AMPLITUDE_B_KEY!r} or {CHANNEL_SUM_KEY!r} "
                f"(found columns: {list(fn)!r})"
            )


def _gate_int_from_row(row: dict[str, str], gc_key: str) -> int | None:
    """Parse gate counter cell as int; None if missing or invalid."""
    g_raw = (row.get(gc_key) or "").strip()
    if not g_raw:
        return None
    try:
        return int(float(g_raw))
    except ValueError:
        return None


def _weighted_mean_masked(
    values: list[float],
    weights: list[float],
    mask: list[bool],
) -> float:
    num = 0.0
    den = 0.0
    for v, w, ok in zip(values, weights, mask):
        if not ok or not math.isfinite(v):
            continue
        ww = max(float(w), 1e-18)
        num += ww * v
        den += ww
    if den <= 0.0:
        return float("nan")
    return num / den


def _finalize_spot_channel_weighted(
    buf: list[tuple[float, float, float, float, int, float | None, float | None]],
) -> tuple[float, float, float, float, int, float, float]:
    """Collapse one spot: weighted mean of A/B, layer, partial code, σ (weights in tuple buf column
    3)."""
    if not buf:
        raise ValueError("empty spot aggregation buffer")
    ws = [max(float(r[3]), 1e-18) for r in buf]
    sw = float(sum(ws))
    a_mean = float(sum(w * r[0] for w, r in zip(ws, buf)) / sw)
    b_mean = float(sum(w * r[1] for w, r in zip(ws, buf)) / sw)
    lay_mean = float(sum(w * float(r[2]) for w, r in zip(ws, buf)) / sw)
    pcds = [r[4] for r in buf]
    pcd_out = int(max(pcds)) if any(p > 0 for p in pcds) else 0
    sig_a_mean = _weighted_mean_masked(
        [r[5] if r[5] is not None else 0.0 for r in buf],
        ws,
        [r[5] is not None and math.isfinite(float(r[5])) for r in buf],
    )
    sig_b_mean = _weighted_mean_masked(
        [r[6] if r[6] is not None else 0.0 for r in buf],
        ws,
        [r[6] is not None and math.isfinite(float(r[6])) for r in buf],
    )
    ch_vals = [float(r[7]) if len(r) > 7 else float(r[3]) for r in buf]
    ch_ok = [math.isfinite(float(v)) and float(v) > 0 for v in ch_vals]
    ch_mean = _weighted_mean_masked(ch_vals, ws, ch_ok)
    return (a_mean, b_mean, lay_mean, sw, pcd_out, sig_a_mean, sig_b_mean, ch_mean)


def _apply_gate_spot_aggregation(
    rows: list[tuple[float, float, float, float, int]],
    gates: list[int],
    sigmas: list[tuple[float | None, float | None]],
) -> list[tuple[float, float, float, float, int, float, float]]:
    """Group consecutive odd Gate Counter phases; even gates flush. Returns 7-tuples (last two =
    σ)."""
    if not (len(rows) == len(gates) == len(sigmas)):
        raise ValueError("rows, gates, sigmas length mismatch for spot aggregation")
    out: list[tuple[float, float, float, float, int, float, float]] = []
    buf: list[tuple[float, float, float, float, int, float | None, float | None]] = []
    prev_g: int | None = None

    def flush() -> None:
        if not buf:
            return
        out.append(_finalize_spot_channel_weighted(buf))
        buf.clear()

    for r, g, (sa, sb) in zip(rows, gates, sigmas):
        if g % 2 == 0:
            flush()
            prev_g = g
            continue
        if prev_g is not None and prev_g % 2 == 1 and g != prev_g:
            flush()
        prev_g = g
        ch_n = float(r[7]) if len(r) > 7 else float(r[3])
        buf.append((r[0], r[1], r[2], r[3], r[4], sa, sb, ch_n))
    flush()
    return out


def _nearest_layer_index_from_plan_energy(z: float, layer_e: list[float]) -> int:
    """Index of closest nominal layer energy to plan spot z (MeV)."""
    if not layer_e:
        return 0
    zf = float(z)
    best_i = 0
    best_d = float("inf")
    for i, e in enumerate(layer_e):
        d = abs(zf - float(e))
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def _measured_tuple_for_spot_weighted_mean(
    t: tuple[float, ...],
) -> tuple[float, float, float, float, int, float | None, float | None]:
    """Unpack measured row for :func:`_finalize_spot_channel_weighted`."""
    a, b = float(t[0]), float(t[1])
    lay = float(t[2])
    w = float(t[3]) if len(t) >= 4 else 1.0
    pcd = int(t[4]) if len(t) >= 5 else 0
    sa: float | None = None
    sb: float | None = None
    if len(t) >= 6:
        raw = t[5]
        if raw is not None and math.isfinite(float(raw)):
            sa = float(raw)
    if len(t) >= 7:
        raw = t[6]
        if raw is not None and math.isfinite(float(raw)):
            sb = float(raw)
    ch = float(t[7]) if len(t) >= 8 else float(w)
    return (a, b, lay, w, pcd, sa, sb, ch)


def _hex_to_rgb_u8(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.strip().lstrip("#")
    if len(h) == 6:
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return 214, 39, 40


def _measured_alpha_u8_from_channel_weights(
    weights: np.ndarray,
    *,
    gamma: float = 0.5,
    alpha_floor: float = 0.1,
    alpha_ceil: float = 0.95,
) -> np.ndarray:
    """Per-point alpha uint8 from IX512 channel-sum style weights (same mapping as 3D measured
    tint)."""
    w = np.maximum(np.asarray(weights, dtype=np.float64), 1e-18)
    lo, hi = float(np.percentile(w, 4)), float(np.percentile(w, 96))
    if hi <= lo * 1.001:
        wn = np.ones_like(w, dtype=np.float64)
    else:
        wn = np.clip((w - lo) / (hi - lo), 0.0, 1.0)
    alpha = alpha_floor + (alpha_ceil - alpha_floor) * np.power(wn, gamma)
    return (255.0 * alpha).astype(np.uint8)


def measured_rgba_by_channel_weight(
    weights: np.ndarray,
    *,
    color_hex: str = _MEASURED_COLOR_3D,
    gold_mask: np.ndarray | None = None,
    gold_hex: str = _PARTIAL_AXIS_MEAS_COLOR_3D,
    gamma: float = 0.5,
    alpha_floor: float = 0.1,
    alpha_ceil: float = 0.95,
) -> np.ndarray:
    """Per-point RGBA uint8; low channel sum -> lower alpha (de-emphasized)."""
    w = np.maximum(np.asarray(weights, dtype=np.float64), 1e-18)
    n = w.shape[0]
    gm = np.asarray(gold_mask, dtype=bool) if gold_mask is not None else np.zeros(n, dtype=bool)
    if gm.shape[0] != n:
        raise ValueError("gold_mask length must match weights length")
    r0, g0, b0 = _hex_to_rgb_u8(color_hex)
    r1, g1, b1 = _hex_to_rgb_u8(gold_hex)
    rgba = np.zeros((n, 4), dtype=np.uint8)
    rgba[:, 0] = np.where(gm, np.uint8(r1), np.uint8(r0))
    rgba[:, 1] = np.where(gm, np.uint8(g1), np.uint8(g0))
    rgba[:, 2] = np.where(gm, np.uint8(b1), np.uint8(b0))
    rgba[:, 3] = _measured_alpha_u8_from_channel_weights(
        w, gamma=gamma, alpha_floor=alpha_floor, alpha_ceil=alpha_ceil
    )
    return rgba


def _layer_xy_kdtrees_for_qa(
    layer_xyz: list[np.ndarray],
) -> list[Any | None]:
    """2D cKDTree per nominal layer (plan X/Y mm) for NN queries; None when scipy missing or
    empty."""
    trees: list[Any | None] = []
    for arr in layer_xyz:
        a2 = np.asarray(arr, dtype=np.float64).reshape(-1, 3)
        if a2.shape[0] == 0 or _cKDTree is None:
            trees.append(None)
            continue
        xy = a2[:, 0:2]
        trees.append(_cKDTree(xy))
    return trees


def layer_nn_plan_xy_distances_and_expected_xyz(
    planned_xyz: list[tuple[float, float, float]],
    measured_rows: list[tuple[float, ...]],
    *,
    a_is_x: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """XY distance (mm) to nearest plan spot on each row's nominal layer and that spot's plan
    (x,y,energy)."""
    if not planned_xyz:
        raise ValueError("plan is empty")
    layer_e = nominal_layer_energies_mev(planned_xyz)
    if not layer_e:
        raise ValueError("plan has no nominal energy layers")
    layer_xyz = _plan_xyz_by_energy_layer(planned_xyz, layer_e)
    hi = len(layer_e) - 1
    n = len(measured_rows)
    if n == 0:
        return np.zeros(0, dtype=np.float64), np.zeros((0, 3), dtype=np.float64)

    li_raw = np.rint(np.asarray([float(t[2]) for t in measured_rows], dtype=np.float64)).astype(
        np.intp, copy=False
    )
    np.clip(li_raw, 0, hi, out=li_raw)

    if a_is_x:
        mx = np.fromiter((float(t[0]) for t in measured_rows), dtype=np.float64, count=n)
        my = np.fromiter((float(t[1]) for t in measured_rows), dtype=np.float64, count=n)
    else:
        mx = np.fromiter((float(t[1]) for t in measured_rows), dtype=np.float64, count=n)
        my = np.fromiter((float(t[0]) for t in measured_rows), dtype=np.float64, count=n)
    meas_xy = np.column_stack([mx, my])

    out_d = np.full(n, np.inf, dtype=np.float64)
    out_xyz = np.full((n, 3), np.nan, dtype=np.float64)
    trees = _layer_xy_kdtrees_for_qa(layer_xyz)

    for ell in range(len(layer_e)):
        mask = li_raw == ell
        if not np.any(mask):
            continue
        arr = np.asarray(layer_xyz[ell], dtype=np.float64).reshape(-1, 3)
        if arr.shape[0] == 0:
            continue
        q = meas_xy[mask]
        tree = trees[ell] if ell < len(trees) else None
        if tree is not None:
            dist, idx = _kdtree_query_k1(tree, q)
            dist = np.asarray(dist, dtype=np.float64).reshape(-1)
            idx = np.asarray(idx, dtype=np.intp).reshape(-1)
            out_d[mask] = dist
            out_xyz[mask] = arr[idx]
        else:
            xy_layer = arr[:, 0:2]
            d2 = np.sum((xy_layer[None, :, :] - q[:, None, :]) ** 2, axis=2)
            j = np.argmin(d2, axis=1)
            out_d[mask] = np.sqrt(d2[np.arange(q.shape[0], dtype=np.intp), j])
            out_xyz[mask] = arr[j]
    return out_d, out_xyz


def layer_nn_plan_match_for_measured(
    planned_xyz: list[tuple[float, float, float]],
    plan_mu: np.ndarray | None,
    measured_rows: list[tuple[float, ...]],
    *,
    a_is_x: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest plan spot on each row's layer: distance (mm), expected XYZ, meterset weight (MU)."""
    dist, exp_xyz = layer_nn_plan_xy_distances_and_expected_xyz(
        planned_xyz, measured_rows, a_is_x=a_is_x
    )
    n = int(dist.shape[0])
    exp_mu = np.full(n, np.nan, dtype=np.float64)
    if plan_mu is None or len(plan_mu) != len(planned_xyz):
        return dist, exp_xyz, exp_mu

    layer_e = nominal_layer_energies_mev(planned_xyz)
    if not layer_e:
        return dist, exp_xyz, exp_mu

    mu_buckets: list[list[float]] = [[] for _ in layer_e]
    for i, (px, py, pe) in enumerate(planned_xyz):
        pf = float(pe)
        mu_v = float(plan_mu[i])
        for k, e in enumerate(layer_e):
            if abs(pf - float(e)) <= 1e-4:
                mu_buckets[k].append(mu_v)
                break

    li_raw = np.rint(np.asarray([float(t[2]) for t in measured_rows], dtype=np.float64)).astype(
        np.intp, copy=False
    )
    hi = len(layer_e) - 1
    np.clip(li_raw, 0, hi, out=li_raw)

    if a_is_x:
        mx = np.fromiter((float(t[0]) for t in measured_rows), dtype=np.float64, count=n)
        my = np.fromiter((float(t[1]) for t in measured_rows), dtype=np.float64, count=n)
    else:
        mx = np.fromiter((float(t[1]) for t in measured_rows), dtype=np.float64, count=n)
        my = np.fromiter((float(t[0]) for t in measured_rows), dtype=np.float64, count=n)
    meas_xy = np.column_stack([mx, my])

    layer_xyz = _plan_xyz_by_energy_layer(planned_xyz, layer_e)
    trees = _layer_xy_kdtrees_for_qa(layer_xyz)

    for ell in range(len(layer_e)):
        mask = li_raw == ell
        if not np.any(mask):
            continue
        arr = np.asarray(layer_xyz[ell], dtype=np.float64).reshape(-1, 3)
        mu_arr = np.asarray(mu_buckets[ell], dtype=np.float64).reshape(-1)
        if arr.shape[0] == 0 or mu_arr.shape[0] != arr.shape[0]:
            continue
        q = meas_xy[mask]
        tree = trees[ell] if ell < len(trees) else None
        if tree is not None:
            _dist, idx = _kdtree_query_k1(tree, q)
            idx = np.asarray(idx, dtype=np.intp).reshape(-1)
            exp_mu[mask] = mu_arr[idx]
        else:
            xy_layer = arr[:, 0:2]
            d2 = np.sum((xy_layer[None, :, :] - q[:, None, :]) ** 2, axis=2)
            j = np.argmin(d2, axis=1)
            exp_mu[mask] = mu_arr[j]
    return dist, exp_xyz, exp_mu


def distances_measured_xy_to_layer_nn_plan_mm(
    planned_xyz: list[tuple[float, float, float]],
    measured_rows: list[tuple[float, ...]],
    *,
    a_is_x: bool,
) -> np.ndarray:
    """Plan XY distance (mm) from each measured row to the nearest plan spot on its nominal
    layer."""
    d, _ = layer_nn_plan_xy_distances_and_expected_xyz(planned_xyz, measured_rows, a_is_x=a_is_x)
    return d


def measured_rgba_by_plan_qa(
    dist_mm: np.ndarray,
    *,
    pass_mm: float,
    warn_mm: float,
    alpha_u8: np.ndarray | None = None,
) -> np.ndarray:
    """RGBA per point: pass (≤pass_mm), warn (pass..warn], fail (>warn_mm). Alpha from ``alpha_u8``
    or opaque."""
    if warn_mm <= pass_mm or pass_mm < 0:
        raise ValueError("plan QA: require 0 ≤ pass_mm < warn_mm")
    d = np.asarray(dist_mm, dtype=np.float64).reshape(-1)
    n = int(d.shape[0])
    if alpha_u8 is not None:
        au = np.asarray(alpha_u8, dtype=np.uint8).reshape(-1)
        if au.shape[0] != n:
            raise ValueError("alpha_u8 length must match dist_mm")
    rp, gp, bp = _hex_to_rgb_u8(_PLAN_QA_PASS_HEX)
    rw, gw, bw = _hex_to_rgb_u8(_PLAN_QA_WARN_HEX)
    rf, gf, bf = _hex_to_rgb_u8(_PLAN_QA_FAIL_HEX)
    rgba = np.zeros((n, 4), dtype=np.uint8)
    pass_m = d <= pass_mm
    fail_m = d > warn_mm
    warn_m = ~pass_m & ~fail_m
    rgba[pass_m, 0] = np.uint8(rp)
    rgba[pass_m, 1] = np.uint8(gp)
    rgba[pass_m, 2] = np.uint8(bp)
    rgba[warn_m, 0] = np.uint8(rw)
    rgba[warn_m, 1] = np.uint8(gw)
    rgba[warn_m, 2] = np.uint8(bw)
    rgba[fail_m, 0] = np.uint8(rf)
    rgba[fail_m, 1] = np.uint8(gf)
    rgba[fail_m, 2] = np.uint8(bf)
    if alpha_u8 is None:
        rgba[:, 3] = np.uint8(255)
    else:
        rgba[:, 3] = au
    return rgba


def measured_rgba_by_plan_dose_qa(
    signed_delta_pp: np.ndarray,
    *,
    pass_pp: float,
    warn_pp: float,
    alpha_u8: np.ndarray | None = None,
) -> np.ndarray:
    """RGBA per point from signed layer dose error (pp): + = over-dose, − = under-dose.

    Over: yellow warn / red fail (same as position QA). Under: sky warn / violet fail.
    """
    if warn_pp <= pass_pp or pass_pp < 0:
        raise ValueError("dose QA: require 0 ≤ pass_pp < warn_pp")
    s = np.asarray(signed_delta_pp, dtype=np.float64).reshape(-1)
    n = int(s.shape[0])
    if alpha_u8 is not None:
        au = np.asarray(alpha_u8, dtype=np.uint8).reshape(-1)
        if au.shape[0] != n:
            raise ValueError("alpha_u8 length must match signed_delta_pp")
    rp, gp, bp = _hex_to_rgb_u8(_PLAN_QA_PASS_HEX)
    rw, gw, bw = _hex_to_rgb_u8(_PLAN_QA_WARN_HEX)
    rf, gf, bf = _hex_to_rgb_u8(_PLAN_QA_FAIL_HEX)
    ruw, guw, buw = _hex_to_rgb_u8(_PLAN_QA_DOSE_UNDER_WARN_HEX)
    ruf, guf, buf = _hex_to_rgb_u8(_PLAN_QA_DOSE_UNDER_FAIL_HEX)
    rgba = np.zeros((n, 4), dtype=np.uint8)
    finite = np.isfinite(s)
    a = np.abs(s)
    pass_m = finite & (a <= pass_pp)
    over_warn_m = finite & (s > pass_pp) & (s <= warn_pp)
    over_fail_m = finite & (s > warn_pp)
    under_warn_m = finite & (s < -pass_pp) & (s >= -warn_pp)
    under_fail_m = finite & (s < -warn_pp)
    rgba[pass_m, 0] = np.uint8(rp)
    rgba[pass_m, 1] = np.uint8(gp)
    rgba[pass_m, 2] = np.uint8(bp)
    rgba[over_warn_m, 0] = np.uint8(rw)
    rgba[over_warn_m, 1] = np.uint8(gw)
    rgba[over_warn_m, 2] = np.uint8(bw)
    rgba[over_fail_m, 0] = np.uint8(rf)
    rgba[over_fail_m, 1] = np.uint8(gf)
    rgba[over_fail_m, 2] = np.uint8(bf)
    rgba[under_warn_m, 0] = np.uint8(ruw)
    rgba[under_warn_m, 1] = np.uint8(guw)
    rgba[under_warn_m, 2] = np.uint8(buw)
    rgba[under_fail_m, 0] = np.uint8(ruf)
    rgba[under_fail_m, 1] = np.uint8(guf)
    rgba[under_fail_m, 2] = np.uint8(buf)
    if alpha_u8 is None:
        rgba[:, 3] = np.uint8(255)
    else:
        rgba[:, 3] = au
    return rgba


def plan_dose_qa_tier_counts(
    signed_delta_pp: np.ndarray,
    *,
    pass_pp: float,
    warn_pp: float,
) -> tuple[int, int, int, int, int, int]:
    """Counts: pass, over_warn, over_fail, under_warn, under_fail (non-finite excluded)."""
    s = np.asarray(signed_delta_pp, dtype=np.float64).reshape(-1)
    finite = np.isfinite(s)
    a = np.abs(s)
    n_pass = int(np.count_nonzero(finite & (a <= pass_pp)))
    n_over_warn = int(np.count_nonzero(finite & (s > pass_pp) & (s <= warn_pp)))
    n_over_fail = int(np.count_nonzero(finite & (s > warn_pp)))
    n_under_warn = int(np.count_nonzero(finite & (s < -pass_pp) & (s >= -warn_pp)))
    n_under_fail = int(np.count_nonzero(finite & (s < -warn_pp)))
    return n_pass, n_over_warn, n_over_fail, n_under_warn, n_under_fail


def _plan_qa_error_line_polylines(
    meas_pts_view: np.ndarray,
    expected_plan_xyz: np.ndarray,
    dist_mm: np.ndarray,
    *,
    pass_mm: float,
    warn_mm: float,
    use_proton_water_depth_mm: bool = False,
) -> tuple[Any, Any]:
    """Separate line sets for warn-tier and fail-tier points (measured → NN plan spot), view Z."""
    if pv is None:
        return None, None
    d = np.asarray(dist_mm, dtype=np.float64).reshape(-1)
    n = int(d.shape[0])
    if meas_pts_view.shape[0] != n or expected_plan_xyz.shape != (n, 3):
        raise ValueError("shape mismatch building plan QA error lines")
    pass_m = d <= pass_mm
    fail_m = d > warn_mm
    warn_m = ~pass_m & ~fail_m

    def _build(idxs: np.ndarray) -> Any:
        pts_list: list[np.ndarray] = []
        lines_list: list[int] = []
        v = 0
        for i in idxs:
            exp = expected_plan_xyz[i]
            if not np.all(np.isfinite(exp)):
                continue
            p0 = meas_pts_view[i]
            zm = nominal_mev_to_plot_z(
                np.array([float(exp[2])], dtype=np.float64),
                use_proton_water_depth_mm=use_proton_water_depth_mm,
            )
            p1 = np.array(
                [
                    float(exp[0]),
                    float(exp[1]),
                    float(zm[0]),
                ],
                dtype=np.float64,
            )
            pts_list.append(p0)
            pts_list.append(p1)
            lines_list.extend((2, v, v + 1))
            v += 2
        if not pts_list:
            return None
        points = np.vstack(pts_list)
        lines = np.asarray(lines_list, dtype=np.int64)
        return pv.PolyData(points, lines=lines)

    warn_idx = np.flatnonzero(warn_m)
    fail_idx = np.flatnonzero(fail_m)
    return _build(warn_idx), _build(fail_idx)


def plan_qa_pass_warn_fail_counts(
    dist_mm: np.ndarray,
    *,
    pass_mm: float,
    warn_mm: float,
) -> tuple[int, int, int]:
    d = np.asarray(dist_mm, dtype=np.float64).reshape(-1)
    n_pass = int(np.count_nonzero(d <= pass_mm))
    n_fail = int(np.count_nonzero(d > warn_mm))
    n_warn = int(d.size) - n_pass - n_fail
    return n_pass, n_warn, n_fail


def format_plan_qa_caption(
    *,
    pass_mm: float,
    warn_mm: float,
    n_pass: int,
    n_warn: int,
    n_fail: int,
) -> str:
    return (
        f"Plan QA: pass d≤{pass_mm:g} mm; warn {pass_mm:g}<d≤{warn_mm:g} mm; fail d>{warn_mm:g} mm "
        f"({n_pass} pass / {n_warn} warn / {n_fail} fail)."
    )


def _layer_plan_mu_by_energy_layer(
    planned_xyz: list[tuple[float, float, float]],
    plan_mu: np.ndarray,
    layer_energies: list[float],
) -> list[np.ndarray]:
    buckets: list[list[float]] = [[] for _ in layer_energies]
    for i, (_px, _py, pe) in enumerate(planned_xyz):
        pf = float(pe)
        mu_v = float(plan_mu[i])
        for k, e in enumerate(layer_energies):
            if abs(pf - float(e)) <= 1e-4:
                buckets[k].append(mu_v)
                break
    return [np.asarray(b, dtype=np.float64) for b in buckets]


def layer_nn_local_spot_index_on_layer(
    planned_xyz: list[tuple[float, float, float]],
    measured_rows: list[tuple[float, ...]],
    *,
    a_is_x: bool = False,
) -> np.ndarray:
    """Nearest plan spot index within each row's nominal energy layer."""
    n = len(measured_rows)
    out = np.full(n, -1, dtype=np.intp)
    if not planned_xyz or n == 0:
        return out
    layer_e = nominal_layer_energies_mev(planned_xyz)
    if not layer_e:
        return out
    li_raw = np.rint(np.asarray([float(t[2]) for t in measured_rows], dtype=np.float64)).astype(
        np.intp, copy=False
    )
    hi = len(layer_e) - 1
    np.clip(li_raw, 0, hi, out=li_raw)
    if a_is_x:
        mx = np.fromiter((float(t[0]) for t in measured_rows), dtype=np.float64, count=n)
        my = np.fromiter((float(t[1]) for t in measured_rows), dtype=np.float64, count=n)
    else:
        mx = np.fromiter((float(t[1]) for t in measured_rows), dtype=np.float64, count=n)
        my = np.fromiter((float(t[0]) for t in measured_rows), dtype=np.float64, count=n)
    meas_xy = np.column_stack([mx, my])
    layer_xyz = _plan_xyz_by_energy_layer(planned_xyz, layer_e)
    trees = _layer_xy_kdtrees_for_qa(layer_xyz)
    for ell in range(len(layer_e)):
        mask = li_raw == ell
        if not np.any(mask):
            continue
        arr = np.asarray(layer_xyz[ell], dtype=np.float64).reshape(-1, 3)
        if arr.shape[0] == 0:
            continue
        q = meas_xy[mask]
        tree = trees[ell] if ell < len(trees) else None
        if tree is not None:
            _dist, idx = _kdtree_query_k1(tree, q)
            out[mask] = np.asarray(idx, dtype=np.intp).reshape(-1)
        else:
            xy_layer = arr[:, 0:2]
            d2 = np.sum((xy_layer[None, :, :] - q[:, None, :]) ** 2, axis=2)
            out[mask] = np.argmin(d2, axis=1).astype(np.intp)
    return out


def plan_dose_fraction_deviation_pp(
    planned_xyz: list[tuple[float, float, float]],
    plan_mu: np.ndarray | None,
    measured_rows: list[tuple[float, ...]],
    *,
    a_is_x: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Layer-relative dose QA: |meas_frac − plan_frac| in percentage points per row."""
    n = len(measured_rows)
    dev_pp = np.full(n, np.nan, dtype=np.float64)
    plan_frac_out = np.full(n, np.nan, dtype=np.float64)
    meas_frac_out = np.full(n, np.nan, dtype=np.float64)
    dist_mm, _exp_xyz = layer_nn_plan_xy_distances_and_expected_xyz(
        planned_xyz, measured_rows, a_is_x=a_is_x
    )
    if plan_mu is None or len(plan_mu) != len(planned_xyz) or n == 0 or not planned_xyz:
        return dev_pp, plan_frac_out, meas_frac_out, dist_mm

    layer_e = nominal_layer_energies_mev(planned_xyz)
    if not layer_e:
        return dev_pp, plan_frac_out, meas_frac_out, dist_mm

    plan_mu_arr = np.asarray(plan_mu, dtype=np.float64)
    layer_mu = _layer_plan_mu_by_energy_layer(planned_xyz, plan_mu_arr, layer_e)
    plan_frac_by_layer: list[np.ndarray] = []
    for mu_arr in layer_mu:
        if mu_arr.size == 0:
            plan_frac_by_layer.append(np.zeros(0, dtype=np.float64))
            continue
        total = float(np.nansum(mu_arr))
        if not math.isfinite(total) or total <= 0.0:
            plan_frac_by_layer.append(np.full(mu_arr.shape[0], np.nan, dtype=np.float64))
        else:
            plan_frac_by_layer.append(mu_arr / total)

    li_raw = np.rint(np.asarray([float(t[2]) for t in measured_rows], dtype=np.float64)).astype(
        np.intp, copy=False
    )
    hi = len(layer_e) - 1
    np.clip(li_raw, 0, hi, out=li_raw)
    local_idx = layer_nn_local_spot_index_on_layer(
        planned_xyz, measured_rows, a_is_x=a_is_x
    )
    charges_by_layer = [
        np.zeros(layer_mu[ell].shape[0], dtype=np.float64) for ell in range(len(layer_e))
    ]
    for i, tup in enumerate(measured_rows):
        ell = int(li_raw[i])
        j = int(local_idx[i])
        if j < 0 or ell >= len(charges_by_layer) or j >= charges_by_layer[ell].shape[0]:
            continue
        ch = measured_charge_na_from_tuple(tup)
        if math.isfinite(ch) and ch > 0.0:
            charges_by_layer[ell][j] += ch

    for i, _tup in enumerate(measured_rows):
        ell = int(li_raw[i])
        j = int(local_idx[i])
        if ell >= len(plan_frac_by_layer) or j < 0 or j >= plan_frac_by_layer[ell].shape[0]:
            continue
        pf = float(plan_frac_by_layer[ell][j])
        layer_total = float(np.sum(charges_by_layer[ell]))
        if not math.isfinite(pf) or not math.isfinite(layer_total) or layer_total <= 0.0:
            continue
        mf = float(charges_by_layer[ell][j]) / layer_total
        plan_frac_out[i] = pf
        meas_frac_out[i] = mf
        dev_pp[i] = abs(mf - pf) * 100.0

    return dev_pp, plan_frac_out, meas_frac_out, dist_mm


def format_plan_dose_qa_caption(
    *,
    pass_pp: float,
    warn_pp: float,
    n_pass: int,
    n_over_warn: int,
    n_over_fail: int,
    n_under_warn: int,
    n_under_fail: int,
    spot_weight_mode: str = SPOT_WEIGHT_MODE_DEFAULT,
) -> str:
    w_lbl = measured_spot_weight_caption(spot_weight_mode)
    return (
        f"Dose QA (layer %): pass |Δ|≤{pass_pp:g} pp ({n_pass}). "
        f"Over: yellow {pass_pp:g}<Δ≤{warn_pp:g} pp ({n_over_warn}), "
        f"red Δ>{warn_pp:g} pp ({n_over_fail}). "
        f"Under: cyan −{warn_pp:g}≤Δ<−{pass_pp:g} pp ({n_under_warn}), "
        f"violet Δ<−{warn_pp:g} pp ({n_under_fail}). "
        f"Plan MU vs measured {w_lbl}."
    )


def nominal_layer_energies_mev(planned_xyz: list[tuple[float, float, float]]) -> list[float]:
    out: list[float] = []
    last: float | None = None
    for *_, e in planned_xyz:
        ef = float(e)
        if last is None or ef != last:
            out.append(ef)
            last = ef
    return out


def fit_ab_to_plan_xy(a: float, b: float, *, a_is_x: bool) -> tuple[float, float]:
    return (a, b) if a_is_x else (b, a)


def _min_xy_dist_to_nominal_energy(
    planned_xyz: list[tuple[float, float, float]],
    e_nom: float,
    mx: float,
    my: float,
) -> float:
    best = float("inf")
    for px, py, pe in planned_xyz:
        if abs(float(pe) - float(e_nom)) > 1e-4:
            continue
        d = float(np.hypot(mx - px, my - py))
        if d < best:
            best = d
    return best


def _layer_advance_plausible_vs_refill(
    planned_xyz: list[tuple[float, float, float]],
    layer_energies: list[float],
    layer: int,
    mx: float,
    my: float,
    *,
    trust_time_gap_stay_dist_mm: float = REFILL_TRUST_TIME_GAP_STAY_DIST_MM,
    layer_trees: list[Any] | None = None,
) -> bool:
    """After a gap that is not a same-spot XY return: True → advance nominal energy layer."""
    if layer >= len(layer_energies) - 1:
        return False
    if layer_trees is not None and len(layer_trees) == len(layer_energies):

        def _min_dist_mm(li: int) -> float:
            if li < 0 or li >= len(layer_trees):
                return float("inf")
            tr = layer_trees[li]
            if tr is None:
                return float("inf")
            q = np.array([[mx, my]], dtype=np.float64)
            d, _ = _kdtree_query_k1(tr, q)
            return float(np.asarray(d, dtype=np.float64).reshape(-1)[0])

        d_stay = _min_dist_mm(layer)
        d_next = _min_dist_mm(layer + 1)
        if not math.isfinite(d_stay):
            d_stay = float("inf")
        if not math.isfinite(d_next):
            d_next = float("inf")
        if d_stay > trust_time_gap_stay_dist_mm:
            return True
        ratio_base = max(d_stay, 1.0)
        worse = (d_next > d_stay + REFILL_REJECT_EXTRA_MM) or (
            d_next > REFILL_REJECT_RATIO * ratio_base
        )
        return not worse
    e0 = layer_energies[layer]
    e1 = layer_energies[layer + 1]
    d_stay = _min_xy_dist_to_nominal_energy(planned_xyz, e0, mx, my)
    d_next = _min_xy_dist_to_nominal_energy(planned_xyz, e1, mx, my)
    if not math.isfinite(d_stay):
        d_stay = float("inf")
    if not math.isfinite(d_next):
        d_next = float("inf")
    if d_stay > trust_time_gap_stay_dist_mm:
        return True
    ratio_base = max(d_stay, 1.0)
    worse = (d_next > d_stay + REFILL_REJECT_EXTRA_MM) or (
        d_next > REFILL_REJECT_RATIO * ratio_base
    )
    return not worse


_DETECTOR_ALIGN_ICP_MAX_ITER = 25
_DETECTOR_ALIGN_ICP_TOL_MM = 0.05
_DETECTOR_ALIGN_COARSE_ANGLES_DEG: tuple[int, ...] = tuple(range(0, 360, 15))


def _detector_align_coarse_angles_deg(n_samples: int) -> tuple[int, ...]:
    """Fewer rotation seeds for large acquisitions (ICP cost scales with sample count)."""
    n = int(n_samples)
    if n > 100_000:
        return (0, 90, 180, 270)
    if n > 50_000:
        return tuple(range(0, 360, 45))
    if n > 10_000:
        return tuple(range(0, 360, 30))
    return _DETECTOR_ALIGN_COARSE_ANGLES_DEG


def _rotation_matrix_2d(theta_deg: float) -> np.ndarray:
    th = math.radians(float(theta_deg))
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def _measured_xy_for_align(
    row: tuple[float, ...],
    *,
    a_is_x: bool,
    swap_ab_axes: bool,
) -> tuple[float, float]:
    """Plan-frame XY used for alignment; ``swap_ab_axes`` maps Fit A→X and Fit B→Y."""
    a, b = float(row[0]), float(row[1])
    if swap_ab_axes:
        return a, b
    return measured_plan_xy_from_row(row, a_is_x=a_is_x)


def _build_align_samples(
    measured_rows: list[tuple[float, ...]],
    layer_xy: list[np.ndarray],
    hi: int,
    *,
    a_is_x: bool,
    swap_ab_axes: bool,
) -> tuple[np.ndarray, np.ndarray]:
    m_acc: list[list[float]] = []
    li_acc: list[int] = []
    for tup in measured_rows:
        li = int(round(float(tup[2])))
        if li < 0:
            li = 0
        elif li > hi:
            li = hi
        arr = np.asarray(layer_xy[li], dtype=np.float64).reshape(-1, 2)
        if arr.shape[0] == 0:
            continue
        mx, my = _measured_xy_for_align(tup, a_is_x=a_is_x, swap_ab_axes=swap_ab_axes)
        m_acc.append([mx, my])
        li_acc.append(li)
    if not m_acc:
        return np.zeros((0, 2), dtype=np.float64), np.zeros(0, dtype=np.intp)
    return np.asarray(m_acc, dtype=np.float64), np.asarray(li_acc, dtype=np.intp)


def _subsample_align_indices(n: int, max_n: int) -> np.ndarray:
    """Deterministic stride subsample for alignment fit (full transform applied to all rows)."""
    n = int(n)
    cap = int(max_n)
    if n <= 0:
        return np.zeros(0, dtype=np.intp)
    if n <= cap:
        return np.arange(n, dtype=np.intp)
    step = int(math.ceil(n / cap))
    return np.arange(0, n, step, dtype=np.intp)


def _kdtree_query_k1(tree: Any, q: np.ndarray) -> tuple[Any, Any]:
    """Nearest-neighbor query; parallel workers only for large batches (avoids Windows overhead)."""
    pts = np.asarray(q, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    nq = int(pts.shape[0])
    if nq == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.intp)
    if nq >= 256:
        try:
            return tree.query(pts, k=1, workers=-1)
        except TypeError:
            pass
    return tree.query(pts, k=1)


def _layer_xy_kdtrees_2d(layer_xy: list[np.ndarray]) -> list[Any | None]:
    """2D cKDTree per nominal layer for fast NN during detector alignment."""
    trees: list[Any | None] = []
    for arr in layer_xy:
        a2 = np.asarray(arr, dtype=np.float64).reshape(-1, 2)
        if a2.shape[0] == 0 or _cKDTree is None:
            trees.append(None)
            continue
        trees.append(_cKDTree(a2))
    return trees


def _layer_nn_plan_targets(
    meas_xy: np.ndarray,
    layer_idx: np.ndarray,
    layer_xy: list[np.ndarray],
    *,
    layer_trees: list[Any | None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest plan spot on each row's assigned layer for the given measured XY positions."""
    n = int(meas_xy.shape[0])
    if n == 0:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0, 2), dtype=np.float64)
    trees = _layer_xy_kdtrees_2d(layer_xy) if layer_trees is None else layer_trees
    m_out = np.empty((n, 2), dtype=np.float64)
    p_out = np.empty((n, 2), dtype=np.float64)
    keep = np.zeros(n, dtype=bool)
    li_arr = np.asarray(layer_idx, dtype=np.intp).reshape(-1)
    for ell in np.unique(li_arr):
        mask = li_arr == int(ell)
        if not np.any(mask):
            continue
        arr = np.asarray(layer_xy[int(ell)], dtype=np.float64).reshape(-1, 2)
        if arr.shape[0] == 0:
            continue
        q = np.asarray(meas_xy[mask], dtype=np.float64).reshape(-1, 2)
        tree = trees[int(ell)] if int(ell) < len(trees) else None
        if tree is not None:
            _, idx = _kdtree_query_k1(tree, q)
            idx = np.asarray(idx, dtype=np.intp).reshape(-1)
            m_out[mask] = q
            p_out[mask] = arr[idx]
            keep[mask] = True
        else:
            d2 = np.sum((arr[None, :, :] - q[:, None, :]) ** 2, axis=2)
            j = np.argmin(d2, axis=1)
            m_out[mask] = q
            p_out[mask] = arr[j]
            keep[mask] = True
    if not np.any(keep):
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0, 2), dtype=np.float64)
    return m_out[keep], p_out[keep]


def _layer_nn_rms_mm(
    meas_xy: np.ndarray,
    layer_idx: np.ndarray,
    layer_xy: list[np.ndarray],
    r_mat: np.ndarray,
    tvec: np.ndarray,
    *,
    layer_trees: list[Any | None] | None = None,
) -> float:
    """RMS NN distance (mm) after applying ``r_mat @ m + tvec`` with per-layer matching."""
    trans = (np.asarray(r_mat, dtype=np.float64) @ meas_xy.T).T + np.asarray(
        tvec, dtype=np.float64
    ).reshape(1, 2)
    m_pairs, p_pairs = _layer_nn_plan_targets(trans, layer_idx, layer_xy, layer_trees=layer_trees)
    if int(m_pairs.shape[0]) == 0:
        return float("inf")
    diff = m_pairs - p_pairs
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def _icp_rigid_layer_nn(
    meas_xy: np.ndarray,
    layer_idx: np.ndarray,
    layer_xy: list[np.ndarray],
    *,
    layer_trees: list[Any | None] | None = None,
    r_init: np.ndarray | None = None,
    t_init: np.ndarray | None = None,
    max_iter: int = _DETECTOR_ALIGN_ICP_MAX_ITER,
    tol_mm: float = _DETECTOR_ALIGN_ICP_TOL_MM,
) -> tuple[np.ndarray, np.ndarray, float, float, int]:
    """Iterative per-layer NN + Kabsch; returns cumulative ``R``, ``t``, RMS before/after."""
    trees = _layer_xy_kdtrees_2d(layer_xy) if layer_trees is None else layer_trees
    r_acc = (
        np.eye(2, dtype=np.float64)
        if r_init is None
        else np.asarray(r_init, dtype=np.float64).reshape(2, 2).copy()
    )
    t_acc = (
        np.zeros(2, dtype=np.float64)
        if t_init is None
        else np.asarray(t_init, dtype=np.float64).reshape(2).copy()
    )
    rms_nn = float("nan")
    rms_res = float("inf")
    prev = float("inf")
    n_iter = 0
    for n_iter in range(1, int(max_iter) + 1):
        trans = (r_acc @ meas_xy.T).T + t_acc
        m_pairs, p_pairs = _layer_nn_plan_targets(trans, layer_idx, layer_xy, layer_trees=trees)
        if int(m_pairs.shape[0]) < 2:
            break
        diff0 = m_pairs - p_pairs
        rms_nn = float(np.sqrt(np.mean(np.sum(diff0 * diff0, axis=1))))
        r_step, t_step, _, rms_res = _kabsch_rigid_2d(m_pairs, p_pairs)
        t_acc = r_step @ t_acc + t_step
        r_acc = r_step @ r_acc
        if math.isfinite(prev) and abs(prev - rms_res) < float(tol_mm):
            break
        prev = rms_res
    return r_acc, t_acc, rms_nn, rms_res, n_iter


def _detector_align_multistart_icp(
    meas_xy: np.ndarray,
    layer_idx: np.ndarray,
    layer_xy: list[np.ndarray],
    *,
    layer_trees: list[Any | None] | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float, int, int]:
    """Try coarse rotation seeds; return best cumulative ``R``, ``t``, RMS, ICP iters."""
    if int(meas_xy.shape[0]) < 2:
        raise ValueError("detector alignment needs at least 2 measured rows with plan spots")
    trees = _layer_xy_kdtrees_2d(layer_xy) if layer_trees is None else layer_trees
    best_rms = float("inf")
    best: tuple[np.ndarray, np.ndarray, float, float, int] | None = None
    centroid = np.mean(meas_xy, axis=0)
    angle_seeds = _detector_align_coarse_angles_deg(int(meas_xy.shape[0]))
    for init_deg in angle_seeds:
        r_seed = _rotation_matrix_2d(init_deg)
        t_seed = centroid - r_seed @ centroid
        r_acc, t_acc, rms_nn, rms_res, n_iter = _icp_rigid_layer_nn(
            meas_xy,
            layer_idx,
            layer_xy,
            layer_trees=trees,
            r_init=r_seed,
            t_init=t_seed,
        )
        if not math.isfinite(rms_res) or int(meas_xy.shape[0]) < 2:
            continue
        holdout = _layer_nn_rms_mm(meas_xy, layer_idx, layer_xy, r_acc, t_acc, layer_trees=trees)
        if holdout < best_rms:
            best_rms = holdout
            best = (r_acc, t_acc, rms_nn, holdout, n_iter)
    if best is None:
        raise ValueError("detector ICP alignment failed for all rotation seeds")
    r_acc, t_acc, rms_nn, rms_res, n_iter = best
    return r_acc, t_acc, rms_nn, rms_res, n_iter, int(meas_xy.shape[0])


def measured_plan_xy_from_row(row: tuple[float, ...], *, a_is_x: bool) -> tuple[float, float]:
    """Plan-frame (X, Y) mm from stored fit A/B row (same convention as the 3D plot)."""
    a, b = float(row[0]), float(row[1])
    return (a, b) if a_is_x else (b, a)


def measured_row_with_plan_xy(
    row: tuple[float, ...],
    x: float,
    y: float,
    *,
    a_is_x: bool,
) -> tuple[float, ...]:
    tail = row[2:]
    if a_is_x:
        return (float(x), float(y), *tail)
    return (float(y), float(x), *tail)


def _kabsch_rigid_2d(
    meas: np.ndarray,
    plan: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return (R, t, rms_nn, rms_residual). Shapes (n,2), (n,2)."""
    n = int(meas.shape[0])
    if n == 0:
        raise ValueError("no point pairs for detector alignment")
    if n == 1:
        r_mat = np.eye(2, dtype=np.float64)
        tvec = (plan[0] - meas[0]).astype(np.float64)
        rms_nn = float(np.linalg.norm(meas[0] - plan[0]))
        rms_res = 0.0
        return r_mat, tvec, rms_nn, rms_res
    c_m = meas.mean(axis=0)
    c_p = plan.mean(axis=0)
    diff0 = meas - plan
    rms_nn = float(np.sqrt(np.mean(np.sum(diff0 * diff0, axis=1))))
    m_c = meas - c_m
    p_c = plan - c_p
    h = m_c.T @ p_c
    u, _, vt = np.linalg.svd(h)
    r_mat = vt.T @ u.T
    if float(np.linalg.det(r_mat)) < 0.0:
        vt2 = vt.copy()
        vt2[1, :] *= -1.0
        r_mat = vt2.T @ u.T
    tvec = c_p - r_mat @ c_m
    aligned = (r_mat @ meas.T).T + tvec
    diff1 = aligned - plan
    rms_res = float(np.sqrt(np.mean(np.sum(diff1 * diff1, axis=1))))
    return r_mat, tvec, rms_nn, rms_res


def _apply_rigid_xy_to_measured_rows(
    measured_rows: list[tuple[float, ...]],
    r_mat: np.ndarray,
    tvec: np.ndarray,
    *,
    a_is_x: bool,
    swap_ab_axes: bool,
) -> list[tuple[float, ...]]:
    """Apply ``r_mat @ m + tvec`` to every row (vectorized XY; tuple tails preserved)."""
    n = len(measured_rows)
    if n == 0:
        return []
    a_col = np.fromiter((float(r[0]) for r in measured_rows), dtype=np.float64, count=n)
    b_col = np.fromiter((float(r[1]) for r in measured_rows), dtype=np.float64, count=n)
    if swap_ab_axes or a_is_x:
        m = np.column_stack([a_col, b_col])
    else:
        m = np.column_stack([b_col, a_col])
    w = (np.asarray(r_mat, dtype=np.float64) @ m.T).T + np.asarray(tvec, dtype=np.float64).reshape(
        1, 2
    )
    out: list[tuple[float, ...]] = []
    if a_is_x:
        for i, row in enumerate(measured_rows):
            out.append((float(w[i, 0]), float(w[i, 1]), *row[2:]))
    else:
        for i, row in enumerate(measured_rows):
            out.append((float(w[i, 1]), float(w[i, 0]), *row[2:]))
    return out


def align_measured_to_plan_detector_xy(
    planned_xyz: list[tuple[float, float, float]],
    measured_rows: list[tuple[float, ...]],
    *,
    a_is_x: bool,
    max_fit_samples: int = DETECTOR_ALIGN_MAX_FIT_SAMPLES,
) -> tuple[list[tuple[float, ...]], DetectorRigidAlign2D]:
    """Per-layer NN + multi-start ICP rigid XY, apply to every measured row.

    Handles arbitrary in-plane rotation (including 90°+ detector placement) and optional
    Fit A↔B axis swap by trying both axis conventions and coarse rotation seeds (every
    15°), then refining with iterative closest-point + Kabsch on each row's assigned
    nominal layer. For large acquisitions, the rigid fit uses a deterministic subsample
    (``max_fit_samples``); the returned transform is applied to **all** rows.
    """
    if not planned_xyz or not measured_rows:
        raise ValueError("plan and measured rows are required for detector alignment")
    layer_e = nominal_layer_energies_mev(planned_xyz)
    if not layer_e:
        raise ValueError("plan has no nominal energy layers")
    layer_xy = _plan_xy_by_energy_layer(planned_xyz, layer_e)
    hi = len(layer_e) - 1
    layer_trees = _layer_xy_kdtrees_2d(layer_xy)

    best: (
        tuple[
            np.ndarray,
            np.ndarray,
            float,
            float,
            bool,
            int,
            int,
            int,
        ]
        | None
    ) = None
    best_rms = float("inf")
    for swap_ab in (False, True):
        meas_xy, layer_idx = _build_align_samples(
            measured_rows,
            layer_xy,
            hi,
            a_is_x=a_is_x,
            swap_ab_axes=swap_ab,
        )
        n_all = int(meas_xy.shape[0])
        if n_all < 2:
            continue
        fit_idx = _subsample_align_indices(n_all, int(max_fit_samples))
        meas_fit = meas_xy[fit_idx]
        layer_fit = layer_idx[fit_idx]
        try:
            r_mat, tvec, rms_nn, rms_res, n_iter, _n_fit = _detector_align_multistart_icp(
                meas_fit,
                layer_fit,
                layer_xy,
                layer_trees=layer_trees,
            )
        except ValueError:
            continue
        if rms_res < best_rms:
            best_rms = rms_res
            best = (r_mat, tvec, rms_nn, rms_res, swap_ab, n_iter, n_all, int(fit_idx.size))

    if best is None:
        raise ValueError(
            "detector alignment needs at least 2 measured rows with plan spots on their layer "
            "(check layer assignment, plan, and detector orientation)"
        )

    r_mat, tvec, rms_nn, rms_res, swap_ab, n_iter, n_all, n_fit = best
    theta = float(math.degrees(math.atan2(float(r_mat[1, 0]), float(r_mat[0, 0]))))
    info = DetectorRigidAlign2D(
        theta_deg=theta,
        tx_mm=float(tvec[0]),
        ty_mm=float(tvec[1]),
        rms_nn_mm=rms_nn,
        rms_residual_mm=rms_res,
        n_pairs=n_all,
        ab_axes_swapped=bool(swap_ab),
        icp_iterations=int(n_iter),
        n_pairs_fit=int(n_fit),
    )
    out_rows = _apply_rigid_xy_to_measured_rows(
        measured_rows,
        r_mat,
        tvec,
        a_is_x=a_is_x,
        swap_ab_axes=swap_ab,
    )
    return out_rows, info


def format_detector_align_caption(info: DetectorRigidAlign2D) -> str:
    swap_note = "; Fit A↔B swapped for search" if info.ab_axes_swapped else ""
    fit_note = ""
    if info.n_pairs_fit > 0 and info.n_pairs_fit < info.n_pairs:
        fit_note = f", fit n={info.n_pairs_fit}/{info.n_pairs}"
    return (
        f"Detector align: θ={info.theta_deg:.5g}° CCW, t=({info.tx_mm:.5g}, {info.ty_mm:.5g}) mm, "
        f"RMS after={info.rms_residual_mm:.5g} mm (NN RMS before={info.rms_nn_mm:.5g} mm, "
        f"n={info.n_pairs}{fit_note}, ICP={info.icp_iterations}{swap_note})."
    )


def energies_for_measured_time_layers(
    layer_energies: list[float],
    measured_abc: list[tuple[float, ...]],
) -> list[float]:
    if not layer_energies or not measured_abc:
        return []
    hi = len(layer_energies) - 1
    out: list[float] = []
    for tup in measured_abc:
        idx = int(round(float(tup[2])))
        if idx < 0:
            idx = 0
        elif idx > hi:
            idx = hi
        out.append(layer_energies[idx])
    return out


def _plan_xy_by_energy_layer(
    planned_xyz: list[tuple[float, float, float]],
    layer_energies: list[float],
) -> list[np.ndarray]:
    buckets: list[list[list[float]]] = [[] for _ in layer_energies]
    for px, py, pe in planned_xyz:
        pf = float(pe)
        for k, e in enumerate(layer_energies):
            if abs(pf - float(e)) <= 1e-4:
                buckets[k].append([float(px), float(py)])
                break
    return [
        np.asarray(b, dtype=np.float64) if b else np.zeros((0, 2), dtype=np.float64)
        for b in buckets
    ]


def _plan_xyz_by_energy_layer(
    planned_xyz: list[tuple[float, float, float]],
    layer_energies: list[float],
) -> list[np.ndarray]:
    """Same layer bucketing as :func:`_plan_xy_by_energy_layer`; each bucket is (n, 3) X, Y, energy
    (MeV)."""
    buckets: list[list[list[float]]] = [[] for _ in layer_energies]
    for px, py, pe in planned_xyz:
        pf = float(pe)
        for k, e in enumerate(layer_energies):
            if abs(pf - float(e)) <= 1e-4:
                buckets[k].append([float(px), float(py), float(pe)])
                break
    return [
        np.asarray(b, dtype=np.float64) if b else np.zeros((0, 3), dtype=np.float64)
        for b in buckets
    ]


try:
    from scipy.spatial import cKDTree as _cKDTree
except ImportError:  # pragma: no cover
    _cKDTree = None


def _build_layer_kdtrees(layer_xy: list[np.ndarray]) -> list[Any] | None:
    """Per-layer 2D nearest point; speeds time_gap refill vs scanning the full plan."""
    if _cKDTree is None:
        return None
    trees: list[Any] = []
    for arr in layer_xy:
        a = np.asarray(arr, dtype=np.float64).reshape(-1, 2)
        if a.shape[0] == 0:
            trees.append(None)
        else:
            trees.append(_cKDTree(a))
    return trees


def _nearest_sqdist_sq_mm2_chunked(
    meas_xy: np.ndarray, plan_xy: np.ndarray, chunk: int = 2048
) -> np.ndarray:
    """min_j ||meas_i - plan_j||^2 for each row i (pure NumPy)."""
    if plan_xy.shape[0] == 0:
        return np.full(meas_xy.shape[0], np.inf, dtype=np.float64)
    out = np.empty(meas_xy.shape[0], dtype=np.float64)
    n = meas_xy.shape[0]
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        diff = meas_xy[s:e, None, :] - plan_xy[None, :, :]
        out[s:e] = (diff * diff).sum(axis=2).min(axis=1)
    return out


def _nearest_sqdist_sq_mm2_to_points(meas_xy: np.ndarray, plan_xy: np.ndarray) -> np.ndarray:
    """Squared distance (mm^2) from each meas row to nearest plan row."""
    if plan_xy.shape[0] == 0:
        return np.full(meas_xy.shape[0], np.inf, dtype=np.float64)
    n, m = meas_xy.shape[0], plan_xy.shape[0]
    if _cKDTree is not None and m > 0 and (n * m > 40_000 or n > 25_000):
        tree = _cKDTree(plan_xy)
        d, _ = _kdtree_query_k1(tree, meas_xy)
        return np.asarray(d, dtype=np.float64) ** 2
    return _nearest_sqdist_sq_mm2_chunked(meas_xy, plan_xy)


def _emit_sqdist_to_layers_mm2(meas_xy: np.ndarray, layer_plan_xy: list[np.ndarray]) -> np.ndarray:
    """Squared XY distance (mm²) from each measured row to nearest plan spot on each layer."""
    n, L = meas_xy.shape[0], len(layer_plan_xy)
    cost = np.zeros((n, L), dtype=np.float64)
    trees: list[Any | None] = []
    for pts in layer_plan_xy:
        a = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        if _cKDTree is not None and a.shape[0] > 0:
            trees.append(_cKDTree(a))
        else:
            trees.append(None)
    for k in range(L):
        pts = layer_plan_xy[k]
        a = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        if a.shape[0] == 0:
            cost[:, k] = np.inf
            continue
        t = trees[k]
        if t is not None:
            d, _ = _kdtree_query_k1(t, meas_xy)
            cost[:, k] = np.asarray(d, dtype=np.float64) ** 2
        else:
            cost[:, k] = _nearest_sqdist_sq_mm2_chunked(meas_xy, a)
    return cost


def viterbi_monotone_layer_assign(
    emit_sq_mm2: np.ndarray,
    advance_penalty_mm2: float | np.ndarray,
) -> np.ndarray:
    """
    Minimize sum_i emit[i, ell_i] + sum of advance penalties at steps where ell increases,
    subject to ell non-decreasing and ell[i] - ell[i-1] in {0, 1}.

    ``advance_penalty_mm2`` may be a scalar (same cost every step) or length-``n`` vector:
    penalty applied when transitioning into row ``i`` (index ``i``), for ``i >= 1``.
    """
    n, L = emit_sq_mm2.shape
    if L == 0:
        raise ValueError("no nominal layers")
    ap = np.asarray(advance_penalty_mm2, dtype=np.float64)
    if ap.size == 1:
        pen_row = np.full(n, float(ap.flat[0]), dtype=np.float64)
    else:
        ap = ap.reshape(-1)
        if ap.shape[0] != n:
            raise ValueError(
                f"advance_penalty_mm2 must be scalar or length n={n}, got length {ap.shape[0]}"
            )
        pen_row = ap
    if (pen_row < 0).any():
        raise ValueError("advance_penalty_mm2 must be >= 0")
    inf = np.inf
    C = np.full((n, L), inf, dtype=np.float64)
    back = np.zeros((n, L), dtype=np.int32)
    C[0, 0] = float(emit_sq_mm2[0, 0])
    C[0, 1:] = inf
    idx_hi = np.arange(1, L, dtype=np.int32)
    idx_lo = np.arange(0, L - 1, dtype=np.int32)
    for i in range(1, n):
        p_add = float(pen_row[i])
        C[i, 0] = float(emit_sq_mm2[i, 0]) + C[i - 1, 0]
        back[i, 0] = 0
        stay = C[i - 1, 1:L]
        adv = C[i - 1, : L - 1] + p_add
        stay_better = stay <= adv
        C[i, 1:L] = emit_sq_mm2[i, 1:L] + np.where(stay_better, stay, adv)
        back[i, 1:L] = np.where(stay_better, idx_hi, idx_lo)
    k_end = int(np.argmin(C[n - 1]))
    layers = np.zeros(n, dtype=np.int32)
    k = k_end
    layers[n - 1] = k
    for i in range(n - 1, 0, -1):
        k = int(back[i, k])
        layers[i - 1] = k
    return layers


def build_unified_advance_penalty_mm2(
    times_s: np.ndarray,
    meas_xy: np.ndarray,
    *,
    base_penalty_mm2: float,
    layer_gap_s: float,
    refill_same_spot_xy_tol_mm: float,
    short_dt_extra_mm2: float = UNIFIED_SHORT_DT_EXTRA_MM2,
    same_spot_refill_block_mm2: float = UNIFIED_SAME_SPOT_REFILL_BLOCK_MM2,
) -> np.ndarray:
    """
    Per-row Viterbi advance penalty: ``base`` + time/refill modifiers on steps ``i>=1``.
    Row index ``i`` uses ``Δt`` and ``ΔXY`` from valid row ``i-1`` to ``i``.
    """
    n = meas_xy.shape[0]
    pen = np.full(n, float(base_penalty_mm2), dtype=np.float64)
    if n <= 1:
        return pen
    dt = np.diff(times_s.astype(np.float64))
    dxy = np.linalg.norm(np.diff(meas_xy.astype(np.float64), axis=0), axis=1)
    long_gap = dt >= float(layer_gap_s)
    same_small = dxy <= float(refill_same_spot_xy_tol_mm)
    pen[1:] += np.where(~long_gap, float(short_dt_extra_mm2), 0.0)
    pen[1:] += np.where(long_gap & same_small, float(same_spot_refill_block_mm2), 0.0)
    return pen


def measured_spot_abc_from_csv(
    csv_path: Path,
    *,
    max_points: int | None = None,
    layer_mode: str = "time_gap",
    layer_gap_s: float = TIME_LAYER_GAP_S_DEFAULT,
    refill_same_spot_xy_tol_mm: float = REFILL_SAME_SPOT_XY_TOLERANCE_MM,
    refill_trust_time_gap_stay_dist_mm: float = REFILL_TRUST_TIME_GAP_STAY_DIST_MM,
    viterbi_advance_penalty_mm2: float = VITERBI_LAYER_ADVANCE_PENALTY_MM2_DEFAULT,
    planned_xyz: list[tuple[float, float, float]] | None = None,
    a_is_x: bool = False,
    aggregate_spots: bool = False,
    aggregate_even_rows_after_odd: int = AGGREGATE_EVEN_ROWS_AFTER_ODD_DEFAULT,
    spot_weight_mode: str = SPOT_WEIGHT_MODE_DEFAULT,
) -> list[tuple[float, ...]]:
    """Measured rows as (A mm, B mm, layer index, spot weight, partial code).

    Index **3** is a positive **weight** from ``spot_weight_mode`` (IX512 sum and/or Fit Amplitude
    A/B; see :func:`measured_spot_weight_from_row`) for aggregation and optional 3D opacity.

    Returns 8-tuples ``(A, B, layer, weight, partial, σ_A, σ_B, channel_sum_nA)`` (σ may be NaN).

    If ``aggregate_spots`` is True, requires ``Gate Counter`` on the CSV; each contiguous run of
    rows with the same **odd** gate value is one spot (even gate closes a spot), optionally extended
    in **gate_counter** mode by up to ``aggregate_even_rows_after_odd`` **even-phase** rows with
    valid amplitude + position after each odd→even transition (0 = legacy close on transition).
    Aggregated rows use weighted means of position, layer, and σ.
    """
    mode = layer_mode.strip().lower().replace("-", "_")
    if mode not in ("time_gap", "plan_viterbi", "unified", "gate_counter"):
        raise ValueError(
            "layer_mode must be 'time_gap', 'plan_viterbi', 'unified', or 'gate_counter'"
        )

    swm = normalize_measured_spot_weight_mode(spot_weight_mode)

    if aggregate_spots:
        ne = int(aggregate_even_rows_after_odd)
        if ne < 0 or ne > AGGREGATE_EVEN_TAIL_MAX:
            raise ValueError(
                f"aggregate_even_rows_after_odd must be in [0, {AGGREGATE_EVEN_TAIL_MAX}], "
                f"got {ne!r}"
            )

    _probe_csv_columns_for_measured_weights(
        csv_path, aggregate_spots=aggregate_spots, spot_weight_mode=swm
    )
    if mode == "time_gap":
        if layer_gap_s <= 0:
            raise ValueError("layer_gap_s must be > 0")
        if refill_same_spot_xy_tol_mm <= 0:
            raise ValueError("refill_same_spot_xy_tol_mm must be > 0")
        if refill_trust_time_gap_stay_dist_mm <= 0:
            raise ValueError("refill_trust_time_gap_stay_dist_mm must be > 0")
    elif mode == "plan_viterbi":
        if viterbi_advance_penalty_mm2 < 0:
            raise ValueError("viterbi_advance_penalty_mm2 must be >= 0")
        if not planned_xyz:
            raise ValueError("plan_viterbi requires planned_xyz from the RT plan")
    elif mode == "unified":
        if layer_gap_s <= 0:
            raise ValueError("layer_gap_s must be > 0")
        if refill_same_spot_xy_tol_mm <= 0:
            raise ValueError("refill_same_spot_xy_tol_mm must be > 0")
        if viterbi_advance_penalty_mm2 < 0:
            raise ValueError("viterbi_advance_penalty_mm2 must be >= 0")
        if not planned_xyz:
            raise ValueError("unified requires planned_xyz from the RT plan")
    else:  # gate_counter
        if not planned_xyz:
            raise ValueError("gate_counter requires planned_xyz from the RT plan")

    layer_energies: list[float] | None = None
    max_layer: int | None = None
    if planned_xyz:
        layer_energies = nominal_layer_energies_mev(planned_xyz)
        if layer_energies:
            max_layer = len(layer_energies) - 1

    if mode in ("plan_viterbi", "unified"):
        if not layer_energies or max_layer is None:
            raise ValueError(
                "plan-based layer modes require a plan with at least one nominal energy layer"
            )
        layer_xy = _plan_xy_by_energy_layer(planned_xyz, layer_energies)  # type: ignore[arg-type]
        plan_xy2 = np.asarray(
            [(float(px), float(py)) for px, py, _ in planned_xyz],  # type: ignore[misc]
            dtype=np.float64,
        )
        global_lk = _PlanImputeLookup.from_xy(plan_xy2)
        if global_lk is None:
            raise ValueError("plan has no scan spots for imputation / Viterbi")
        layer_lks = _plan_impute_lookups_per_layer(layer_xy)
        ab_buf: list[tuple[float, float]] = []
        xy_buf: list[list[float]] = []
        t_buf: list[float] = []
        w_buf: list[float] = []
        ch_buf: list[float] = []
        partial_plan_xy: list[tuple[float | None, float | None]] = []
        partial_codes: list[int] = []
        gates_acc: list[int] = []
        sig_acc: list[tuple[float | None, float | None]] = []
        fa_key = "Fit Amplitude A (nA)"
        a_key = "Fit Mean Position A (mm)"
        b_key = "Fit Mean Position B (mm)"
        with csv_path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return []
            time_key = reader.fieldnames[0]
            for row in reader:
                if not (row.get(fa_key) or "").strip():
                    continue
                try:
                    t = float(row[time_key])
                except ValueError:
                    continue
                a_opt = _opt_float_cell(row, a_key)
                b_opt = _opt_float_cell(row, b_key)
                mx_p, my_p, pcd = _plan_xy_from_optional_ab(a_opt, b_opt, a_is_x=a_is_x)
                if pcd < 0:
                    continue
                mx_i, my_i = _impute_plan_axis_fast(global_lk, mx_p, my_p)
                a_fin, b_fin = _ab_from_plan_xy(mx_i, my_i, a_is_x=a_is_x)
                ab_buf.append((a_fin, b_fin))
                xy_buf.append([mx_i, my_i])
                t_buf.append(t)
                w_buf.append(measured_spot_weight_from_row(row, swm))
                ch_buf.append(_channel_sum_na_from_row(row))
                partial_plan_xy.append((mx_p, my_p))
                partial_codes.append(pcd)
                sig_acc.append(
                    (_opt_float_cell(row, SIGMA_A_KEY), _opt_float_cell(row, SIGMA_B_KEY))
                )
                if aggregate_spots:
                    g_cell = _gate_int_from_row(row, GATE_COUNTER_KEY)
                    if g_cell is None:
                        raise ValueError(
                            f"aggregate_spots: missing/invalid {GATE_COUNTER_KEY!r} on a fit row"
                        )
                    gates_acc.append(g_cell)
                if max_points is not None and len(ab_buf) >= max_points:
                    break
        if not ab_buf:
            return []
        meas_xy = np.asarray(xy_buf, dtype=np.float64)
        emit = _emit_sqdist_to_layers_mm2(meas_xy, layer_xy)
        if mode == "plan_viterbi":
            pen = viterbi_advance_penalty_mm2
        else:
            pen = build_unified_advance_penalty_mm2(
                np.asarray(t_buf, dtype=np.float64),
                meas_xy,
                base_penalty_mm2=viterbi_advance_penalty_mm2,
                layer_gap_s=layer_gap_s,
                refill_same_spot_xy_tol_mm=refill_same_spot_xy_tol_mm,
            )
        layers_idx = viterbi_monotone_layer_assign(emit, pen)
        hi = max_layer
        for i, (mx_p, my_p) in enumerate(partial_plan_xy):
            if partial_codes[i] == 0:
                continue
            efi = int(layers_idx[i])
            if efi < 0:
                efi = 0
            elif efi > hi:
                efi = hi
            lk_ref = layer_lks[efi]
            if lk_ref is None:
                lk_ref = global_lk
            mx_f, my_f = _impute_plan_axis_fast(lk_ref, mx_p, my_p)
            a_fin, b_fin = _ab_from_plan_xy(mx_f, my_f, a_is_x=a_is_x)
            ab_buf[i] = (a_fin, b_fin)

        out: list[tuple[float, ...]] = []
        for i, ((a, b), ell, wch, ch_n, pcd) in enumerate(
            zip(ab_buf, layers_idx.tolist(), w_buf, ch_buf, partial_codes)
        ):
            efi = int(ell)
            if efi < 0:
                efi = 0
            elif efi > hi:
                efi = hi
            sa, sb = sig_acc[i]
            out.append(
                _measured_row_with_sigma(
                    a, b, float(efi), wch, pcd, sa, sb, channel_sum_na=ch_n
                )
            )
        if aggregate_spots:
            return _apply_gate_spot_aggregation(out, gates_acc, sig_acc)
        return out

    if mode == "gate_counter":
        if not layer_energies or max_layer is None:
            raise ValueError("gate_counter requires a plan with at least one nominal energy layer")
        layer_xy_gc = _plan_xy_by_energy_layer(planned_xyz, layer_energies)  # type: ignore[arg-type]
        spots_per = [
            int(np.asarray(arr, dtype=np.float64).reshape(-1, 2).shape[0]) for arr in layer_xy_gc
        ]
        if sum(spots_per) == 0:
            raise ValueError("gate_counter: plan has no spots")
        cumul: list[int] = [0]
        for c in spots_per:
            cumul.append(cumul[-1] + c)
        plan_xy2_gc = np.asarray(
            [(float(px), float(py)) for px, py, _ in planned_xyz],  # type: ignore[misc]
            dtype=np.float64,
        )
        global_lk_gc = _PlanImputeLookup.from_xy(plan_xy2_gc)
        if global_lk_gc is None:
            raise ValueError("gate_counter: plan has no XY spots")
        layer_lks_gc = _plan_impute_lookups_per_layer(layer_xy_gc)
        hi_gc = max_layer
        out_gc: list[tuple[float, ...]] = []
        spot_buf_gc: list[tuple[float, float, float, float, int, float | None, float | None]] = []
        prev_gate: int | None = None
        pending_even_tail = 0
        even_tail_max = int(aggregate_even_rows_after_odd) if aggregate_spots else 0
        i_spot = 0
        eff_li = 0
        n_gc_raw = 0
        fa_key = "Fit Amplitude A (nA)"
        a_key = "Fit Mean Position A (mm)"
        b_key = "Fit Mean Position B (mm)"
        with csv_path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return []
            gc_key = GATE_COUNTER_KEY
            if gc_key not in reader.fieldnames:
                raise ValueError(
                    f"CSV has no “{GATE_COUNTER_KEY}” column (columns: {list(reader.fieldnames)!r})"
                )
            for row in reader:
                g_raw = (row.get(gc_key) or "").strip()
                if not g_raw:
                    continue
                try:
                    g = int(float(g_raw))
                except ValueError:
                    continue
                if g != prev_gate:
                    if aggregate_spots:
                        if even_tail_max > 0:
                            if g % 2 == 1:
                                if spot_buf_gc:
                                    out_gc.append(_finalize_spot_channel_weighted(spot_buf_gc))
                                    spot_buf_gc.clear()
                                pending_even_tail = 0
                                eff_li = max(0, min(bisect.bisect_right(cumul, i_spot) - 1, hi_gc))
                                i_spot += 1
                            elif prev_gate is not None and prev_gate % 2 == 1 and g % 2 == 0:
                                pending_even_tail = even_tail_max
                            else:
                                if spot_buf_gc:
                                    out_gc.append(_finalize_spot_channel_weighted(spot_buf_gc))
                                    spot_buf_gc.clear()
                                pending_even_tail = 0
                                if g % 2 == 1:
                                    eff_li = max(
                                        0, min(bisect.bisect_right(cumul, i_spot) - 1, hi_gc)
                                    )
                                    i_spot += 1
                        else:
                            if prev_gate is not None and prev_gate % 2 == 1 and spot_buf_gc:
                                out_gc.append(_finalize_spot_channel_weighted(spot_buf_gc))
                                spot_buf_gc.clear()
                            pending_even_tail = 0
                            if g % 2 == 1:
                                eff_li = max(0, min(bisect.bisect_right(cumul, i_spot) - 1, hi_gc))
                                i_spot += 1
                    else:
                        if g % 2 == 1:
                            eff_li = max(0, min(bisect.bisect_right(cumul, i_spot) - 1, hi_gc))
                            i_spot += 1
                    prev_gate = g
                allow_even_slurp = (
                    aggregate_spots and even_tail_max > 0 and g % 2 == 0 and pending_even_tail > 0
                )
                if g % 2 == 0 and not allow_even_slurp:
                    continue
                if not (row.get(fa_key) or "").strip():
                    continue
                a_opt = _opt_float_cell(row, a_key)
                b_opt = _opt_float_cell(row, b_key)
                mx_p, my_p, pcd = _plan_xy_from_optional_ab(a_opt, b_opt, a_is_x=a_is_x)
                if pcd < 0:
                    continue
                lk_row = layer_lks_gc[eff_li] if eff_li < len(layer_lks_gc) else None
                lk_use = lk_row or global_lk_gc
                mx, my = _impute_plan_axis_fast(lk_use, mx_p, my_p)
                a_fin, b_fin = _ab_from_plan_xy(mx, my, a_is_x=a_is_x)
                w_ch = measured_spot_weight_from_row(row, swm)
                ch_n = _channel_sum_na_from_row(row)
                sa = _opt_float_cell(row, SIGMA_A_KEY)
                sb = _opt_float_cell(row, SIGMA_B_KEY)
                if aggregate_spots:
                    spot_buf_gc.append((a_fin, b_fin, float(eff_li), w_ch, int(pcd), sa, sb, ch_n))
                    n_gc_raw += 1
                    if allow_even_slurp:
                        pending_even_tail -= 1
                        if pending_even_tail == 0 and spot_buf_gc:
                            out_gc.append(_finalize_spot_channel_weighted(spot_buf_gc))
                            spot_buf_gc.clear()
                    if max_points is not None and n_gc_raw >= max_points:
                        if spot_buf_gc:
                            out_gc.append(_finalize_spot_channel_weighted(spot_buf_gc))
                            spot_buf_gc.clear()
                        break
                else:
                    out_gc.append(
                        _measured_row_with_sigma(
                            a_fin,
                            b_fin,
                            float(eff_li),
                            w_ch,
                            int(pcd),
                            sa,
                            sb,
                            channel_sum_na=ch_n,
                        )
                    )
                    if max_points is not None and len(out_gc) >= max_points:
                        break
        if aggregate_spots and spot_buf_gc:
            out_gc.append(_finalize_spot_channel_weighted(spot_buf_gc))
        return out_gc

    out: list[tuple[float, ...]] = []
    gates_tg: list[int] = []
    sig_tg: list[tuple[float | None, float | None]] = []
    layer = 0
    prev_t: float | None = None
    prev_mx: float | None = None
    prev_my: float | None = None
    plan_xy2_tg: np.ndarray | None = None
    global_lk_tg: _PlanImputeLookup | None = None
    layer_lks_tg: list[_PlanImputeLookup | None] | None = None
    layer_trees_tg: list[Any] | None = None
    if planned_xyz:
        plan_xy2_tg = np.asarray(
            [(float(px), float(py)) for px, py, _ in planned_xyz],
            dtype=np.float64,
        )
        global_lk_tg = _PlanImputeLookup.from_xy(plan_xy2_tg)
        if layer_energies:
            layer_xy_tg = _plan_xy_by_energy_layer(planned_xyz, layer_energies)
            layer_lks_tg = _plan_impute_lookups_per_layer(layer_xy_tg)
            layer_trees_tg = _build_layer_kdtrees(layer_xy_tg)

    fa_key = "Fit Amplitude A (nA)"
    a_key = "Fit Mean Position A (mm)"
    b_key = "Fit Mean Position B (mm)"

    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return []
        time_key = reader.fieldnames[0]

        for row in reader:
            if not (row.get(fa_key) or "").strip():
                continue
            try:
                t = float(row[time_key])
            except ValueError:
                continue
            a_opt = _opt_float_cell(row, a_key)
            b_opt = _opt_float_cell(row, b_key)
            mx_p, my_p, pcd = _plan_xy_from_optional_ab(a_opt, b_opt, a_is_x=a_is_x)
            if pcd < 0:
                continue

            li_use = layer if max_layer is None else min(layer, max_layer)
            if global_lk_tg is None:
                mx = float(mx_p or 0.0)
                my = float(my_p or 0.0)
            else:
                lk_row: _PlanImputeLookup | None = None
                if layer_lks_tg and 0 <= li_use < len(layer_lks_tg):
                    lk_row = layer_lks_tg[li_use]
                lk_row = lk_row or global_lk_tg
                mx, my = _impute_plan_axis_fast(lk_row, mx_p, my_p)
            a_fin, b_fin = _ab_from_plan_xy(mx, my, a_is_x=a_is_x)

            same_spot_refill = False
            if prev_t is not None and (t - prev_t) >= layer_gap_s:
                same_spot_refill = (
                    prev_mx is not None
                    and prev_my is not None
                    and float(np.hypot(mx - prev_mx, my - prev_my)) <= refill_same_spot_xy_tol_mm
                )
                if not same_spot_refill:
                    if max_layer is None:
                        layer += 1
                    elif layer < max_layer and layer_energies and planned_xyz is not None:
                        if _layer_advance_plausible_vs_refill(
                            planned_xyz,
                            layer_energies,
                            layer,
                            mx,
                            my,
                            trust_time_gap_stay_dist_mm=refill_trust_time_gap_stay_dist_mm,
                            layer_trees=layer_trees_tg,
                        ):
                            layer += 1

            eff_layer = layer if max_layer is None else min(layer, max_layer)
            sa = _opt_float_cell(row, SIGMA_A_KEY)
            sb = _opt_float_cell(row, SIGMA_B_KEY)
            out.append(
                _measured_row_with_sigma(
                    a_fin,
                    b_fin,
                    float(eff_layer),
                    measured_spot_weight_from_row(row, swm),
                    int(pcd),
                    sa,
                    sb,
                    channel_sum_na=_channel_sum_na_from_row(row),
                )
            )
            if aggregate_spots:
                g_cell = _gate_int_from_row(row, GATE_COUNTER_KEY)
                if g_cell is None:
                    raise ValueError(
                        f"aggregate_spots: missing/invalid {GATE_COUNTER_KEY!r} on a fit row"
                    )
                gates_tg.append(g_cell)
                sig_tg.append((sa, sb))
            prev_t = t
            prev_mx, prev_my = mx, my
            if max_points is not None and len(out) >= max_points:
                break

    if aggregate_spots:
        return _apply_gate_spot_aggregation(out, gates_tg, sig_tg)
    return out


def _plan_energy_bounds_mev(planned_xyz: list[tuple[float, float, float]]) -> tuple[float, float]:
    zs = [z for _, _, z in planned_xyz]
    if not zs:
        return 0.0, 0.0
    return max(zs), min(zs)


# Cached templates for instanced FWHM / σ ellipsoids (see _instanced_axis_aligned_ellipsoids).
_GLYPH_UNIT_SPHERE: dict[tuple[int, int], Any] = {}


def _unit_sphere_glyph_template(phi_resolution: int, theta_resolution: int) -> Any:
    if pv is None:
        raise RuntimeError("Install PyVista for GPU 3D: pip install pyvista")
    key = (int(phi_resolution), int(theta_resolution))
    tpl = _GLYPH_UNIT_SPHERE.get(key)
    if tpl is None:
        tpl = pv.Sphere(
            radius=1.0,
            phi_resolution=int(phi_resolution),
            theta_resolution=int(theta_resolution),
        )
        _GLYPH_UNIT_SPHERE[key] = tpl
    return tpl


def _disc_point_add_mesh_kwargs(*, point_size: float) -> dict[str, Any]:
    """Sharp screen-space circular discs (VTK sphere impostors), flat and unlit."""
    return {
        "render_points_as_spheres": True,
        "lighting": False,
        "ambient": 1.0,
        "diffuse": 0.0,
        "specular": 0.0,
        "smooth_shading": False,
        "point_size": float(point_size),
    }


def _instanced_axis_aligned_ellipsoids(
    centers: np.ndarray,
    semiaxes_xyz: np.ndarray,
    *,
    phi_resolution: int = 14,
    theta_resolution: int = 14,
) -> Any:
    """Axis-aligned ellipsoids in scene units (mm): per-row X/Y/Z semiaxes.

    ``pyvista.PolyData.glyph(..., scale=<3-component array>, orient=False)`` does not
    apply per-axis scaling reliably on VTK 9.6 (glyphs stay near unit size), so we
    instance a unit sphere template with explicit point transforms.
    """
    if pv is None:
        raise RuntimeError("Install PyVista for GPU 3D: pip install pyvista")
    centers_u = np.asarray(centers, dtype=np.float64, order="C")
    semi_u = np.asarray(semiaxes_xyz, dtype=np.float64, order="C")
    n = int(centers_u.shape[0])
    if n == 0:
        return pv.PolyData(np.zeros((0, 3), dtype=np.float64))
    if centers_u.shape != (n, 3):
        raise ValueError("centers must have shape (n, 3)")
    if semi_u.shape != (n, 3):
        raise ValueError("semiaxes_xyz must have shape (n, 3)")

    tpl = _unit_sphere_glyph_template(phi_resolution, theta_resolution)
    tpl_pts = np.asarray(tpl.points, dtype=np.float64)
    m = int(tpl_pts.shape[0])
    tpl_faces = np.asarray(tpl.faces, dtype=np.int64).reshape(tpl.n_cells, 4)
    if not bool(np.all(tpl_faces[:, 0] == 3)):
        raise RuntimeError("internal: unit-sphere template must be all triangles")

    pts = (tpl_pts[np.newaxis, :, :] * semi_u[:, np.newaxis, :]).reshape(-1, 3)
    pts += np.repeat(centers_u, m, axis=0)

    tri = tpl_faces[:, 1:4]
    off = (np.arange(n, dtype=np.int64) * m)[:, np.newaxis, np.newaxis]
    inst_tris = (tri[np.newaxis, :, :] + off).reshape(-1, 3)
    face_arr = np.empty((inst_tris.shape[0], 4), dtype=np.int64)
    face_arr[:, 0] = 3
    face_arr[:, 1:4] = inst_tris
    return pv.PolyData(pts, face_arr.ravel())


def _plan_spot_fwhm_glyph_mesh(plan_pts: np.ndarray, fwhm_xy_mm: np.ndarray) -> Any:
    """At each plan point, an axis-aligned ellipsoid with X/Y semiaxis = FWHM/2 (mm) and thin Z."""
    if pv is None:
        raise RuntimeError("Install PyVista for GPU 3D: pip install pyvista")
    n = int(plan_pts.shape[0])
    if fwhm_xy_mm.shape != (n, 2):
        raise ValueError("plan_fwhm_xy_mm must have shape (n_plan, 2)")
    fx = fwhm_xy_mm[:, 0].astype(np.float64, copy=False)
    fy = fwhm_xy_mm[:, 1].astype(np.float64, copy=False)
    good = np.isfinite(fx) & np.isfinite(fy) & (fx > 0.0) & (fy > 0.0)
    med_x = float(np.nanmedian(fx[good])) if np.any(good) else _DEFAULT_PLAN_FWHM_MM_WHEN_MISSING
    med_y = float(np.nanmedian(fy[good])) if np.any(good) else _DEFAULT_PLAN_FWHM_MM_WHEN_MISSING
    if not math.isfinite(med_x) or med_x <= 0.0:
        med_x = _DEFAULT_PLAN_FWHM_MM_WHEN_MISSING
    if not math.isfinite(med_y) or med_y <= 0.0:
        med_y = _DEFAULT_PLAN_FWHM_MM_WHEN_MISSING
    sx = np.where(good, fx, med_x) * 0.5
    sy = np.where(good, fy, med_y) * 0.5
    zptp = float(np.ptp(plan_pts[:, 2])) if n else 1.0
    if not math.isfinite(zptp) or zptp <= 0.0:
        zptp = 1.0
    sz = max(zptp * _PLAN_FWHM_GLYPH_Z_SPAN_FRAC, 1e-9)
    scal = np.column_stack([sx, sy, np.full(n, sz, dtype=np.float64)])
    return _instanced_axis_aligned_ellipsoids(plan_pts, scal)


def _measured_spot_sigma_glyph_mesh(
    meas_pts: np.ndarray,
    sigma_xy_mm: np.ndarray,
    *,
    sigma_scale: float = MEASURED_SIGMA_GLYPH_SCALE_DEFAULT,
    rgba: np.ndarray | None = None,
) -> Any:
    """Per measured point, an axis-aligned ellipsoid: X/Y semiaxis = σ×scale (mm), diameter =
    2×scale×σ; thin Z."""
    if pv is None:
        raise RuntimeError("Install PyVista for GPU 3D: pip install pyvista")
    n = int(meas_pts.shape[0])
    if n == 0:
        return pv.PolyData(np.zeros((0, 3), dtype=np.float64))
    sig = np.asarray(sigma_xy_mm, dtype=np.float64).reshape(n, 2)
    if sig.shape[0] != n:
        raise ValueError("sigma_xy_mm row count must match meas_pts")
    sx_raw = sig[:, 0]
    sy_raw = sig[:, 1]
    good = np.isfinite(sx_raw) & np.isfinite(sy_raw) & (sx_raw > 0.0) & (sy_raw > 0.0)
    fb = float(MEASURED_SIGMA_GLYPH_FALLBACK_MM)
    med_x = float(np.nanmedian(sx_raw[good])) if np.any(good) else fb
    med_y = float(np.nanmedian(sy_raw[good])) if np.any(good) else fb
    if not math.isfinite(med_x) or med_x <= 0.0:
        med_x = fb
    if not math.isfinite(med_y) or med_y <= 0.0:
        med_y = fb
    scale = float(sigma_scale)
    sx = np.where(good, sx_raw, med_x).astype(np.float64, copy=False) * scale
    sy = np.where(good, sy_raw, med_y).astype(np.float64, copy=False) * scale
    sx = np.clip(sx, float(MEASURED_SIGMA_GLYPH_MIN_MM), float(MEASURED_SIGMA_GLYPH_MAX_MM))
    sy = np.clip(sy, float(MEASURED_SIGMA_GLYPH_MIN_MM), float(MEASURED_SIGMA_GLYPH_MAX_MM))
    zptp = float(np.ptp(meas_pts[:, 2]))
    if not math.isfinite(zptp) or zptp <= 0.0:
        zptp = 1.0
    sz = max(zptp * float(_PLAN_FWHM_GLYPH_Z_SPAN_FRAC), 1e-9)
    scal = np.column_stack([sx, sy, np.full(n, sz, dtype=np.float64)])
    tpl = _unit_sphere_glyph_template(14, 14)
    n_g = int(tpl.n_points)
    centers = np.asarray(meas_pts, dtype=np.float64, order="C")
    g = _instanced_axis_aligned_ellipsoids(centers, scal)
    if rgba is not None:
        rgba_u8 = np.asarray(rgba, dtype=np.uint8).reshape(-1, 4)
        if int(rgba_u8.shape[0]) == n:
            if int(g.n_points) == n_g * n:
                g["rgba"] = np.repeat(rgba_u8, n_g, axis=0)
            else:
                logger.warning(
                    "Measured σ glyph RGBA expansion skipped (point count mismatch: %s vs %s×%s)",
                    g.n_points,
                    n,
                    n_g,
                )
    return g


def prepare_comparison_3d_data(
    planned_xyz: list[tuple[float, float, float]],
    measured_abc: list[tuple[float, ...]],
    *,
    a_is_x: bool,
    max_measured_draw: int | None = None,
    plan_fwhm_xy_mm: np.ndarray | None = None,
) -> Comparison3DData:
    if not planned_xyz:
        raise PlanDataError("No planned spots extracted from DICOM")
    if not measured_abc:
        raise AcquisitionDataError("No measured points found in CSV")
    n_plan = len(planned_xyz)
    fwhm_arr: np.ndarray | None = None
    if plan_fwhm_xy_mm is not None:
        fa = np.asarray(plan_fwhm_xy_mm, dtype=np.float64).reshape(-1)
        if fa.size != 2 * n_plan:
            raise ValueError("plan_fwhm_xy_mm must have length 2 * n_plan or shape (n_plan, 2)")
        fwhm_arr = fa.reshape(n_plan, 2)
    e_hi, e_lo = _plan_energy_bounds_mev(planned_xyz)
    plan_xyz = np.asarray(planned_xyz, dtype=np.float64).reshape(-1, 3)
    layer_e = nominal_layer_energies_mev(planned_xyz)

    rows = list(measured_abc)
    if max_measured_draw is not None and len(rows) > max_measured_draw:
        rows = rows[:max_measured_draw]

    z_mapped = energies_for_measured_time_layers(layer_e, rows)

    if a_is_x:
        xlab, ylab = "Fit A (mm)", "Fit B (mm)"
        mx = [t[0] for t in rows]
        my = [t[1] for t in rows]
    else:
        xlab, ylab = "Fit B (mm)", "Fit A (mm)"
        mx = [t[1] for t in rows]
        my = [t[0] for t in rows]

    wts: list[float] = []
    parts: list[int] = []
    for t in rows:
        wts.append(float(t[3]) if len(t) >= 4 else 1.0)
        parts.append(int(t[4]) if len(t) >= 5 else 0)
    meas_weight = np.asarray(wts, dtype=np.float64)
    meas_partial_raw = np.asarray(parts, dtype=np.int8)

    meas_xyz = np.column_stack([mx, my, z_mapped]).astype(np.float64)
    sig_plot_x: list[float] = []
    sig_plot_y: list[float] = []
    for t in rows:
        if len(t) >= 7:
            try:
                sa = float(t[5])
                if not math.isfinite(sa):
                    sa = float("nan")
            except (TypeError, ValueError):
                sa = float("nan")
            try:
                sb = float(t[6])
                if not math.isfinite(sb):
                    sb = float("nan")
            except (TypeError, ValueError):
                sb = float("nan")
        else:
            sa, sb = float("nan"), float("nan")
        if a_is_x:
            sig_plot_x.append(sa)
            sig_plot_y.append(sb)
        else:
            sig_plot_x.append(sb)
            sig_plot_y.append(sa)
    meas_sigma_xy = np.column_stack(
        [np.asarray(sig_plot_x, dtype=np.float64), np.asarray(sig_plot_y, dtype=np.float64)]
    )
    return Comparison3DData(
        plan_xyz=plan_xyz,
        meas_xyz=meas_xyz,
        xlab=xlab,
        ylab=ylab,
        e_hi=e_hi,
        e_lo=e_lo,
        meas_weight=meas_weight,
        meas_partial_raw=meas_partial_raw,
        plan_fwhm_xy_mm=fwhm_arr,
        meas_sigma_xy_mm=meas_sigma_xy,
    )


def _energy_slice_mask(energy_mev: np.ndarray, lo_mev: float, hi_mev: float) -> np.ndarray:
    """Inclusive nominal-energy band; ``lo_mev``, ``hi_mev`` may be in either order."""
    a, b = sorted((float(lo_mev), float(hi_mev)))
    e = np.asarray(energy_mev, dtype=np.float64).reshape(-1)
    return (e >= a) & (e <= b)


def _nominal_layer_index_band_mev(
    layer_energies_mev: Sequence[float],
    center_index: int,
    *,
    half_width: int = 2,
) -> tuple[float, float]:
    """Inclusive MeV range for up to ``2 * half_width + 1`` consecutive plan layers around
    ``center_index``."""
    n = len(layer_energies_mev)
    if n == 0:
        return 0.0, 0.0
    c = int(np.clip(int(center_index), 0, n - 1))
    hw = int(max(0, half_width))
    i0 = max(0, c - hw)
    i1 = min(n - 1, c + hw)
    band = [float(layer_energies_mev[j]) for j in range(i0, i1 + 1)]
    return float(min(band)), float(max(band))


_VTK_TK_PUMP: dict[str, Any] = {"after_id": None, "plotter": None}


def _vtk_rendering_tk_dll_present() -> bool:
    """Return True if the VTK–Tk bridge library is present (often absent in pip wheels on
    Windows)."""
    try:
        from pathlib import Path

        import vtkmodules

        libs = Path(vtkmodules.__file__).resolve().parent.parent / "vtk.libs"
        if not libs.is_dir():
            return False
        if any(libs.glob("vtkRenderingTk*.dll")):
            return True
        if any(libs.glob("libvtkRenderingTk*.so*")):
            return True
    except Exception:
        return False
    return False


def _stop_tk_vtk_event_pump(tk_master: Any) -> None:
    """Cancel VTK event pumping for a separate PyVista window coordinated with Tk."""
    aid = _VTK_TK_PUMP.get("after_id")
    if tk is not None and tk_master is not None and aid is not None:
        try:
            tk_master.after_cancel(aid)
        except (tk.TclError, ValueError, TypeError):
            pass
    _VTK_TK_PUMP["after_id"] = None
    prev = _VTK_TK_PUMP.get("plotter")
    _VTK_TK_PUMP["plotter"] = None
    if prev is not None:
        try:
            prev.close()
        except Exception:
            pass


def _ensure_pyvista_iren_initialized(plotter: Any) -> None:
    """``Plotter.show(interactive_update=True)`` skips ``iren.initialize()`` on VTK 9.2.3+, but
    :meth:`RenderWindowInteractor.process_events` requires an initialized interactor."""
    try:
        iren_wrap = getattr(plotter, "iren", None)
        if iren_wrap is None:
            return
        if not bool(getattr(iren_wrap, "initialized", False)):
            iren_wrap.initialize()
    except Exception:
        pass


def _start_tk_vtk_event_pump(tk_master: Any, plotter: Any) -> None:
    """Drive a non-embedded PyVista window while Tk's mainloop runs (``interactive_update``
    mode)."""
    if tk is None:
        return
    _stop_tk_vtk_event_pump(tk_master)
    _VTK_TK_PUMP["plotter"] = plotter

    def pump() -> None:
        plr = _VTK_TK_PUMP.get("plotter")
        if plr is None:
            _VTK_TK_PUMP["after_id"] = None
            return
        rw = getattr(plr, "render_window", None)
        if rw is None:
            _VTK_TK_PUMP["after_id"] = None
            _VTK_TK_PUMP["plotter"] = None
            return
        try:
            if plr.iren is not None:
                if not bool(getattr(plr.iren, "initialized", False)):
                    _ensure_pyvista_iren_initialized(plr)
                plr.iren.process_events()
        except Exception:
            pass
        try:
            if rw is not None:
                rw.Render()
        except Exception:
            pass
        try:
            _VTK_TK_PUMP["after_id"] = tk_master.after(33, pump)
        except (tk.TclError, RuntimeError):
            _VTK_TK_PUMP["after_id"] = None

    _VTK_TK_PUMP["after_id"] = tk_master.after(33, pump)


def _show_tk_vtk_fallback_panel(parent: Any) -> None:
    """Explain separate-window 3D when ``vtkRenderingTk`` is not in the VTK wheel."""
    if tk is None:
        return
    inner = tk.Frame(parent, bg="#0d1117")
    inner.pack(fill=tk.BOTH, expand=True)
    msg = (
        "3D view is open in a separate window.\n\n"
        "This Python environment’s VTK build does not ship vtkRenderingTk (typical for "
        "`pip install vtk` on Windows), so the renderer cannot be embedded in this pane.\n\n"
        "Options: use conda-forge VTK built with Tk, or keep using the separate window — "
        "slice controls in the drawer still apply after Show 3D."
    )
    tk.Label(
        inner,
        text=msg,
        bg="#0d1117",
        fg="#8b949e",
        font=("", 10),
        justify=tk.LEFT,
        wraplength=420,
    ).pack(anchor="n", padx=16, pady=20)


def idle_slice_band_controls(slice_tk: dict[str, Any] | None) -> None:
    """Disable 3D layer-band widgets until a plot exists (Tk GUI)."""
    if slice_tk is None or tk is None:
        return
    try:
        slice_tk["scale"].configure(state=tk.DISABLED)
        slice_tk["checkbtn"].state(["disabled"])
        slice_tk["status_var"].set("Run Show 3D to enable the layer band.")
        slice_tk["var_slice"].set(False)
    except (tk.TclError, KeyError):
        pass


def _wire_slice_band_controls(
    slice_tk: dict[str, Any],
    slice_cfg: dict[str, bool | int],
    layer_energies_plan: list[float],
    n_plan_layers: int,
    apply_slice: Any,
) -> None:
    if tk is None or n_plan_layers <= 0:
        return
    var_slice = slice_tk["var_slice"]
    scale = slice_tk["scale"]
    chk = slice_tk["checkbtn"]
    status_var = slice_tk["status_var"]

    def band_line() -> str:
        ci = int(slice_cfg["center_i"])
        emid = float(layer_energies_plan[ci])
        if not bool(slice_cfg["slice_on"]):
            return (
                f"Full stack: {n_plan_layers} nominal layer(s). "
                f"Slider ref — index {ci}, {emid:.2f} MeV (plan order)."
            )
        lo_m, hi_m = _nominal_layer_index_band_mev(layer_energies_plan, ci, half_width=2)
        return f"5-layer band: [{lo_m:.2f}, {hi_m:.2f}] MeV (center idx {ci}, {emid:.2f} MeV)."

    def refresh() -> None:
        status_var.set(band_line())

    def on_scale(val: str) -> None:
        slice_cfg["center_i"] = int(np.clip(int(round(float(val))), 0, n_plan_layers - 1))
        if bool(slice_cfg["slice_on"]):
            apply_slice()
        refresh()

    def on_chk() -> None:
        slice_cfg["slice_on"] = bool(var_slice.get())
        apply_slice()
        refresh()

    var_slice.set(bool(slice_cfg["slice_on"]))
    scale.configure(
        from_=0,
        to=max(0, n_plan_layers - 1),
        resolution=1,
        state=tk.NORMAL,
    )
    scale.set(int(slice_cfg["center_i"]))
    chk.state(["!disabled"])
    scale.configure(command=on_scale)
    chk.configure(command=on_chk)
    refresh()


def _embed_pyvista_plotter_in_tk(parent: Any, plotter: Any) -> Any:
    """Place an existing PyVista plotter's render window inside a Tk container (~left pane)."""
    from vtkmodules.tk.vtkTkRenderWindowInteractor import vtkTkRenderWindowInteractor

    if tk is None:
        raise RuntimeError("tkinter is required to embed the PyVista plotter")
    parent.update_idletasks()
    w = max(parent.winfo_width(), 320)
    h = max(parent.winfo_height(), 240)
    inner = tk.Frame(parent, bg="#0d1117")
    inner.pack(fill=tk.BOTH, expand=True)
    rw = plotter.render_window
    iren = vtkTkRenderWindowInteractor(inner, rw=rw, width=w, height=h)
    iren.pack(fill=tk.BOTH, expand=True)
    iren.Initialize()
    rw.SetInteractor(iren)
    if getattr(plotter, "iren", None) is not None:
        plotter.iren.interactor = iren
    plotter.render()

    def _on_resize(event: tk.Event) -> None:
        if event.widget is not parent:
            return
        nw = max(int(event.width), 2)
        nh = max(int(event.height), 2)
        try:
            rw.SetSize(nw, nh)
            plotter.render()
        except Exception:
            pass

    parent.bind("<Configure>", _on_resize)
    return iren


def _clear_qt_layout_items(parent: Any) -> None:
    """Remove all widgets from ``parent``'s ``QVBoxLayout`` (Qt embed pane)."""
    lay = parent.layout()
    if lay is None:
        return
    while lay.count():
        item = lay.takeAt(0)
        w = item.widget()
        if w is not None:
            w.deleteLater()


def _embed_pyvista_plotter_in_qt(parent: Any, plotter: Any) -> Any:
    """Place PyVista's render window in a Qt widget (works with pip VTK on Windows)."""
    from PySide6.QtWidgets import QVBoxLayout
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

    _clear_qt_layout_items(parent)
    lay = parent.layout()
    if lay is None:
        lay = QVBoxLayout(parent)
        lay.setContentsMargins(0, 0, 0, 0)
        parent.setLayout(lay)
    rw = plotter.render_window
    vtk_widget = QVTKRenderWindowInteractor(parent, rw=rw)
    lay.addWidget(vtk_widget)
    if getattr(plotter, "iren", None) is not None:
        plotter.iren.interactor = vtk_widget._Iren
    vtk_widget.Initialize()
    # QVTK replaces the vtkRenderWindowInteractor. PyVista's theme style was applied to the
    # old interactor; without re-applying, VTK's default (often joystick camera) feels broken.
    try:
        plotter.enable_interactor_style()
    except Exception:
        try:
            plotter.enable_trackball_style()
        except Exception:
            pass
    # Without show(), Plotter.render() skips vtk Render() until this runs (see plotter._first_time).
    try:
        plotter._on_first_render_request()
    except Exception:
        pass
    plotter.render()
    return vtk_widget


def apply_comparison_3d_camera_view(
    plotter: Any,
    view: str,
    *,
    zoom: float = 1.05,
    render: bool = True,
) -> None:
    """Snap camera to a standard view: ``top`` (XY), ``left`` (−X), or ``right`` (+X)."""
    v = str(view).strip().lower()
    if v == "top":
        plotter.view_xy()
    elif v == "left":
        plotter.view_yz(negative=True)
    elif v == "right":
        plotter.view_yz(negative=False)
    else:
        raise ValueError(f"unknown 3D view {view!r} (expected top, left, or right)")
    try:
        plotter.camera.zoom(float(zoom))
    except Exception:
        pass
    if render:
        plotter.render()


def idle_slice_band_controls_qt(slice_qt: dict[str, Any] | None) -> None:
    if slice_qt is None:
        return
    try:
        slice_qt["slider"].setEnabled(False)
        slice_qt["check"].setEnabled(False)
        slice_qt["status"].setText("Layer band enables after a successful 3D plot.")
    except Exception:
        pass


def _wire_slice_band_controls_qt(
    slice_qt: dict[str, Any],
    slice_cfg: dict[str, bool | int],
    layer_energies_plan: list[float],
    n_plan_layers: int,
    apply_slice: Any,
) -> None:
    chk: Any = slice_qt["check"]
    sli: Any = slice_qt["slider"]
    status: Any = slice_qt["status"]
    if n_plan_layers <= 0:
        return

    def band_line() -> str:
        ci = int(slice_cfg["center_i"])
        emid = float(layer_energies_plan[ci])
        if not bool(slice_cfg["slice_on"]):
            return (
                f"Full stack: {n_plan_layers} nominal layer(s). "
                f"Slider ref — index {ci}, {emid:.2f} MeV (plan order)."
            )
        lo_m, hi_m = _nominal_layer_index_band_mev(layer_energies_plan, ci, half_width=2)
        return f"5-layer band: [{lo_m:.2f}, {hi_m:.2f}] MeV (center idx {ci}, {emid:.2f} MeV)."

    def refresh() -> None:
        status.setText(band_line())

    def on_sli(val: int) -> None:
        slice_cfg["center_i"] = int(np.clip(int(val), 0, n_plan_layers - 1))
        if bool(slice_cfg["slice_on"]):
            apply_slice()
        refresh()

    def on_chk(checked: bool) -> None:
        slice_cfg["slice_on"] = bool(checked)
        apply_slice()
        refresh()

    prev_chk = slice_qt.get("_slice_chk_handler")
    prev_sli = slice_qt.get("_slice_sli_handler")
    if prev_chk is not None:
        try:
            chk.toggled.disconnect(prev_chk)
        except (TypeError, RuntimeError):
            pass
    if prev_sli is not None:
        try:
            sli.valueChanged.disconnect(prev_sli)
        except (TypeError, RuntimeError):
            pass

    sli.setMinimum(0)
    sli.setMaximum(max(0, n_plan_layers - 1))
    sli.setSingleStep(1)
    try:
        sli.setTracking(True)
    except Exception:
        pass
    sli.setValue(int(slice_cfg["center_i"]))
    chk.setChecked(bool(slice_cfg["slice_on"]))
    chk.setEnabled(True)
    sli.setEnabled(True)
    slice_qt["_slice_sli_handler"] = on_sli
    slice_qt["_slice_chk_handler"] = on_chk
    sli.valueChanged.connect(on_sli)
    chk.toggled.connect(on_chk)
    refresh()


def show_comparison_3d_pyvista(
    planned_xyz: list[tuple[float, float, float]],
    measured_abc: list[tuple[float, ...]],
    *,
    title: str,
    a_is_x: bool,
    max_measured_draw: int | None = None,
    layer_mode: str | None = None,
    layer_gap_s: float | None = None,
    refill_same_spot_xy_tol_mm: float | None = None,
    refill_trust_time_gap_stay_dist_mm: float | None = None,
    viterbi_advance_penalty_mm2: float | None = None,
    weight_measured_by_channel: bool = True,
    aggregate_spots: bool = False,
    aggregate_even_rows_after_odd: int = 0,
    spot_weight_mode: str = SPOT_WEIGHT_MODE_DEFAULT,
    detector_align_caption: str | None = None,
    bounds_xy_tick_mm: float | None = None,
    plan_qa_coloring: bool = False,
    plan_qa_mode: str = "position",
    plan_qa_pass_mm: float = PLAN_QA_PASS_MM_DEFAULT,
    plan_qa_warn_mm: float = PLAN_QA_WARN_MM_DEFAULT,
    plan_qa_pass_pp: float = PLAN_QA_DOSE_PASS_PP_DEFAULT,
    plan_qa_warn_pp: float = PLAN_QA_DOSE_WARN_PP_DEFAULT,
    plan_mu: np.ndarray | None = None,
    plan_qa_draw_error_lines: bool = False,
    plan_qa_hide_pass_spots: bool = False,
    plan_fwhm_xy_mm: np.ndarray | None = None,
    scale_plan_spots_by_dicom_fwhm: bool = True,
    measured_spots_sigma_world_mm: bool = False,
    measured_sigma_glyph_scale: float | None = None,
    reuse_plotter: Any | None = None,
    reuse_camera: bool = False,
    reembed_qt: bool = True,
    embed_parent: Any | None = None,
    slice_tk: dict[str, Any] | None = None,
    embed_qt: Any | None = None,
    slice_qt: dict[str, Any] | None = None,
    slice_band_init: dict[str, bool | int] | None = None,
    z_axis_use_proton_water_depth_mm: bool = True,
    view_projection_perspective: bool = True,
) -> Any:
    if pv is None:
        raise RuntimeError("Install PyVista for GPU 3D: pip install pyvista")

    qa_mode = str(plan_qa_mode).strip().lower().replace("-", "_")
    if qa_mode not in ("position", "dose"):
        raise GeometryConfigError("plan_qa_mode must be 'position' or 'dose'")
    if plan_qa_coloring:
        if qa_mode == "dose":
            if float(plan_qa_warn_pp) <= float(plan_qa_pass_pp):
                raise GeometryConfigError("plan_qa_warn_pp must be greater than plan_qa_pass_pp")
        elif float(plan_qa_warn_mm) <= float(plan_qa_pass_mm):
            raise GeometryConfigError("plan_qa_warn_mm must be greater than plan_qa_pass_mm")
    if plan_qa_draw_error_lines and not plan_qa_coloring:
        raise GeometryConfigError("plan_qa_draw_error_lines requires plan_qa_coloring")
    if plan_qa_draw_error_lines and qa_mode == "dose":
        raise GeometryConfigError("plan_qa_draw_error_lines applies to position QA only")
    qa_hide_pass_spots = bool(plan_qa_hide_pass_spots) and bool(plan_qa_coloring)
    qa_draw_lines = bool(plan_qa_draw_error_lines) and qa_mode == "position"

    prep = prepare_comparison_3d_data(
        planned_xyz,
        measured_abc,
        a_is_x=a_is_x,
        max_measured_draw=max_measured_draw,
        plan_fwhm_xy_mm=plan_fwhm_xy_mm,
    )

    plan_pts = prep.plan_xyz.copy()
    meas_pts = prep.meas_xyz.copy()
    use_depth_z = bool(z_axis_use_proton_water_depth_mm)
    plan_pts[:, 2] = nominal_mev_to_plot_z(plan_pts[:, 2], use_proton_water_depth_mm=use_depth_z)
    meas_pts[:, 2] = nominal_mev_to_plot_z(meas_pts[:, 2], use_proton_water_depth_mm=use_depth_z)

    n_m = meas_pts.shape[0]
    _POINT_SIZE_3D = 9
    meas_cloud = pv.PolyData(meas_pts)
    gold_mask: np.ndarray | None = None
    if prep.meas_partial_raw is not None and prep.meas_partial_raw.shape[0] == n_m:
        gold_mask = prep.meas_partial_raw > 0

    rows_for_qa = list(measured_abc)[:n_m]
    plan_qa_caption_extra = ""
    dist_qa: np.ndarray | None = None
    exp_xyz_qa: np.ndarray | None = None
    weight_alpha_u8: np.ndarray | None = None
    if (
        weight_measured_by_channel
        and prep.meas_weight is not None
        and int(prep.meas_weight.shape[0]) == n_m
    ):
        weight_alpha_u8 = _measured_alpha_u8_from_channel_weights(prep.meas_weight)
    qa_metric: np.ndarray | None = None
    if plan_qa_coloring:
        if qa_mode == "dose":
            dev_pp, plan_frac_qa, meas_frac_qa, dist_qa = plan_dose_fraction_deviation_pp(
                planned_xyz, plan_mu, rows_for_qa, a_is_x=a_is_x
            )
            qa_metric = dev_pp
            signed_pp = np.where(
                np.isfinite(plan_frac_qa) & np.isfinite(meas_frac_qa),
                (meas_frac_qa - plan_frac_qa) * 100.0,
                np.nan,
            )
            _dist_nn, exp_xyz_qa = layer_nn_plan_xy_distances_and_expected_xyz(
                planned_xyz, rows_for_qa, a_is_x=a_is_x
            )
            meas_cloud["rgba"] = measured_rgba_by_plan_dose_qa(
                signed_pp,
                pass_pp=float(plan_qa_pass_pp),
                warn_pp=float(plan_qa_warn_pp),
                alpha_u8=weight_alpha_u8,
            )
            npass, now, nof, nuw, nuf = plan_dose_qa_tier_counts(
                signed_pp, pass_pp=float(plan_qa_pass_pp), warn_pp=float(plan_qa_warn_pp)
            )
            plan_qa_caption_extra = format_plan_dose_qa_caption(
                pass_pp=float(plan_qa_pass_pp),
                warn_pp=float(plan_qa_warn_pp),
                n_pass=npass,
                n_over_warn=now,
                n_over_fail=nof,
                n_under_warn=nuw,
                n_under_fail=nuf,
                spot_weight_mode=spot_weight_mode,
            )
        else:
            dist_qa, exp_xyz_qa = layer_nn_plan_xy_distances_and_expected_xyz(
                planned_xyz, rows_for_qa, a_is_x=a_is_x
            )
            qa_metric = dist_qa
            meas_cloud["rgba"] = measured_rgba_by_plan_qa(
                dist_qa,
                pass_mm=plan_qa_pass_mm,
                warn_mm=plan_qa_warn_mm,
                alpha_u8=weight_alpha_u8,
            )
            npass, nwarn, nfail = plan_qa_pass_warn_fail_counts(
                dist_qa, pass_mm=plan_qa_pass_mm, warn_mm=plan_qa_warn_mm
            )
            plan_qa_caption_extra = format_plan_qa_caption(
                pass_mm=plan_qa_pass_mm,
                warn_mm=plan_qa_warn_mm,
                n_pass=npass,
                n_warn=nwarn,
                n_fail=nfail,
            )
            if qa_draw_lines:
                plan_qa_caption_extra += " Lines: warn+fail → NN plan spot."
    elif (
        weight_measured_by_channel
        and prep.meas_weight is not None
        and int(prep.meas_weight.shape[0]) == n_m
    ):
        meas_cloud["rgba"] = measured_rgba_by_channel_weight(
            prep.meas_weight,
            gold_mask=gold_mask,
        )

    use_rgba = plan_qa_coloring or (gold_mask is not None and bool(np.any(gold_mask)))
    if use_rgba and "rgba" not in meas_cloud.point_data and gold_mask is not None:
        r0, g0, b0 = _hex_to_rgb_u8(_MEASURED_COLOR_3D)
        r1, g1, b1 = _hex_to_rgb_u8(_PARTIAL_AXIS_MEAS_COLOR_3D)
        gm = gold_mask
        rgba = np.zeros((n_m, 4), dtype=np.uint8)
        rgba[:, 0] = np.where(gm, np.uint8(r1), np.uint8(r0))
        rgba[:, 1] = np.where(gm, np.uint8(g1), np.uint8(g0))
        rgba[:, 2] = np.where(gm, np.uint8(b1), np.uint8(b0))
        rgba[:, 3] = 255
        meas_cloud["rgba"] = rgba

    meas_idx = np.arange(int(n_m), dtype=np.int64)
    if qa_hide_pass_spots and qa_metric is not None:
        d_q = np.asarray(qa_metric, dtype=np.float64).reshape(-1)
        if int(d_q.shape[0]) != int(n_m):
            raise ValueError(
                "plan_qa_hide_pass_spots: QA metric length does not match measured count"
            )
        pass_thr = float(plan_qa_pass_pp) if qa_mode == "dose" else float(plan_qa_pass_mm)
        keep = ~(np.isfinite(d_q) & (d_q <= pass_thr))
        n_pass_pts = int(np.count_nonzero(np.isfinite(d_q) & (d_q <= pass_thr)))
        idx = np.flatnonzero(keep)
        meas_idx = idx.astype(np.int64)
        if idx.size == 0:
            meas_cloud = pv.PolyData(np.empty((0, 3), dtype=np.float64))
            meas_pts = np.empty((0, 3), dtype=np.float64)
            n_m = 0
        else:
            meas_cloud = meas_cloud.extract_points(idx)
            meas_pts = np.asarray(meas_cloud.points, dtype=np.float64)
            n_m = int(meas_pts.shape[0])
        if n_pass_pts > 0:
            plan_qa_caption_extra += (
                f" Omitting {n_pass_pts} pass-tier measured spot(s); {n_m} warn/fail drawn."
            )

    meas_pts_final = np.asarray(meas_cloud.points, dtype=np.float64).copy()
    meas_e_final = np.asarray(prep.meas_xyz[meas_idx, 2], dtype=np.float64).reshape(-1)
    dist_qa_draw: np.ndarray | None = None
    exp_xyz_qa_draw: np.ndarray | None = None
    if dist_qa is not None:
        dist_qa_draw = np.asarray(dist_qa, dtype=np.float64).reshape(-1)[meas_idx]
    if exp_xyz_qa is not None:
        exp_xyz_qa_draw = np.asarray(exp_xyz_qa, dtype=np.float64).reshape(-1, 3)[meas_idx]
    meas_rgba_final: np.ndarray | None = None
    if "rgba" in meas_cloud.point_data:
        meas_rgba_final = np.asarray(meas_cloud.point_data["rgba"]).copy()

    sig_scale_eff = float(
        MEASURED_SIGMA_GLYPH_SCALE_DEFAULT
        if measured_sigma_glyph_scale is None
        else measured_sigma_glyph_scale
    )
    n_sig_src = int(prep.meas_xyz.shape[0])
    if prep.meas_sigma_xy_mm is None or int(prep.meas_sigma_xy_mm.shape[0]) != n_sig_src:
        meas_sigma_all = np.full((n_sig_src, 2), np.nan, dtype=np.float64)
    else:
        meas_sigma_all = np.asarray(prep.meas_sigma_xy_mm, dtype=np.float64).reshape(n_sig_src, 2)
    meas_sigma_final = (
        meas_sigma_all[meas_idx] if n_sig_src > 0 else np.zeros((0, 2), dtype=np.float64)
    )
    display_perf_note = ""
    n_m_pre_display = int(n_m)
    if plan_qa_coloring and n_m_pre_display > 80_000 and _cKDTree is None:
        logger.warning(
            "Install scipy for much faster plan QA on large acquisitions (%s measured rows).",
            n_m_pre_display,
        )

    want_sigma_glyphs = bool(measured_spots_sigma_world_mm) and n_m_pre_display > 0

    if want_sigma_glyphs and n_m_pre_display > DISPLAY_GLYPH_INSTANCE_CAP:
        step = int(math.ceil(n_m_pre_display / DISPLAY_GLYPH_INSTANCE_CAP))
        sub = np.arange(0, n_m_pre_display, step, dtype=np.intp)
        display_perf_note += (
            f" Measured σ ellipsoid stride {step} "
            f"(~{sub.size} of {n_m_pre_display} spots for display)."
        )
        meas_pts_final = meas_pts_final[sub]
        meas_e_final = meas_e_final[sub]
        if dist_qa_draw is not None:
            dist_qa_draw = dist_qa_draw[sub]
        if exp_xyz_qa_draw is not None:
            exp_xyz_qa_draw = exp_xyz_qa_draw[sub]
        if meas_rgba_final is not None:
            meas_rgba_final = meas_rgba_final[sub]
        meas_sigma_final = meas_sigma_final[sub]
        n_m = int(sub.size)
    elif not want_sigma_glyphs and n_m_pre_display > DISPLAY_POINT_MESH_TARGET:
        step = int(math.ceil(n_m_pre_display / DISPLAY_POINT_MESH_TARGET))
        sub = np.arange(0, n_m_pre_display, step, dtype=np.intp)
        display_perf_note += (
            f" Measured mesh stride {step} (~{sub.size} of {n_m_pre_display} points for display)."
        )
        meas_pts_final = meas_pts_final[sub]
        meas_e_final = meas_e_final[sub]
        if dist_qa_draw is not None:
            dist_qa_draw = dist_qa_draw[sub]
        if exp_xyz_qa_draw is not None:
            exp_xyz_qa_draw = exp_xyz_qa_draw[sub]
        if meas_rgba_final is not None:
            meas_rgba_final = meas_rgba_final[sub]
        meas_sigma_final = meas_sigma_final[sub]
        n_m = int(sub.size)

    meas_sigma_glyphs = want_sigma_glyphs and n_m > 0

    def _make_measured_view_mesh(
        pts: np.ndarray,
        sig_xy: np.ndarray,
        rgba: np.ndarray | None,
    ) -> Any:
        if pts.shape[0] == 0:
            return pv.PolyData(np.zeros((0, 3), dtype=np.float64))
        if meas_sigma_glyphs:
            return _measured_spot_sigma_glyph_mesh(
                pts,
                sig_xy,
                sigma_scale=sig_scale_eff,
                rgba=rgba,
            )
        m = pv.PolyData(pts)
        if rgba is not None:
            m["rgba"] = rgba
        return m

    plan_e_mev = np.asarray(prep.plan_xyz[:, 2], dtype=np.float64).reshape(-1)
    x_all = np.r_[plan_pts[:, 0], meas_pts[:, 0]]
    y_all = np.r_[plan_pts[:, 1], meas_pts[:, 1]]
    x_min, x_max = float(np.min(x_all)), float(np.max(x_all))
    y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
    eff_tick = (
        float(BOUNDS_XY_TICK_MM_DEFAULT) if bounds_xy_tick_mm is None else float(bounds_xy_tick_mm)
    )
    if eff_tick > 0.0 and math.isfinite(eff_tick):
        n_xlabels = n_cube_axis_labels_for_mm_step(x_min, x_max, eff_tick)
        n_ylabels = n_cube_axis_labels_for_mm_step(y_min, y_max, eff_tick)
    else:
        n_xlabels = 5
        n_ylabels = 5

    plan_cloud = pv.PolyData(plan_pts)

    plan_allow_fwhm_glyphs = bool(scale_plan_spots_by_dicom_fwhm) and int(plan_pts.shape[0]) <= int(
        DISPLAY_GLYPH_INSTANCE_CAP
    )
    if bool(scale_plan_spots_by_dicom_fwhm) and int(plan_pts.shape[0]) > int(
        DISPLAY_GLYPH_INSTANCE_CAP
    ):
        display_perf_note += (
            f" Plan FWHM ellipsoids disabled ({plan_pts.shape[0]} spots > "
            f"{DISPLAY_GLYPH_INSTANCE_CAP} glyph budget — use points."
        )

    # Sharp circular disc sprites (VTK sphere impostors — not square GL points or gaussian blur).
    point_r = _POINT_SIZE_3D
    point_kw = _disc_point_add_mesh_kwargs(point_size=point_r)

    plan_rendered_fwhm_glyphs = False
    plan_glyphs: Any = None
    if (
        plan_allow_fwhm_glyphs
        and prep.plan_fwhm_xy_mm is not None
        and prep.plan_fwhm_xy_mm.shape[0] == plan_pts.shape[0]
        and bool(np.any(np.isfinite(prep.plan_fwhm_xy_mm)))
    ):
        try:
            plan_glyphs = _plan_spot_fwhm_glyph_mesh(plan_pts, prep.plan_fwhm_xy_mm)
            plan_rendered_fwhm_glyphs = True
        except Exception:
            plan_rendered_fwhm_glyphs = False
            plan_glyphs = None
    else:
        plan_glyphs = None

    pl = reuse_plotter
    saved_camera_position: Any = None
    if pl is not None:
        if reuse_camera:
            try:
                saved_camera_position = pl.camera_position
            except Exception:
                saved_camera_position = None
        # pl.clear() does not remove vtkCubeAxesActor; stale axes keep wrong Z ticks.
        try:
            pl.remove_bounds_axes()
        except Exception:
            pass
        pl.clear()
        try:
            if pl.renderer.cube_axes_actor is not None:
                pl.remove_bounds_axes()
        except Exception:
            pass
    if pl is None:
        pl = pv.Plotter(window_size=(1440, 960), title="Plan vs measured (PyVista)")
    pl.set_background("#0d1117")
    try:
        pl.enable_anti_aliasing("msaa")
    except (TypeError, ValueError):
        try:
            pl.enable_anti_aliasing()
        except Exception:
            pass

    if plan_rendered_fwhm_glyphs and plan_glyphs is not None:
        plan_actor = pl.add_mesh(
            plan_glyphs,
            color=_PLAN_COLOR_3D,
            opacity=0.45,
            pickable=True,
            smooth_shading=False,
            lighting=False,
        )
    else:
        plan_actor = pl.add_mesh(
            plan_cloud,
            color=_PLAN_COLOR_3D,
            opacity=0.45,
            pickable=True,
            **point_kw,
        )

    line_warn_actor: Any | None = None
    line_fail_actor: Any | None = None
    if (
        plan_qa_coloring
        and qa_draw_lines
        and dist_qa_draw is not None
        and exp_xyz_qa_draw is not None
    ):
        lines_warn, lines_fail = _plan_qa_error_line_polylines(
            meas_pts_final,
            exp_xyz_qa_draw,
            dist_qa_draw,
            pass_mm=plan_qa_pass_mm,
            warn_mm=plan_qa_warn_mm,
            use_proton_water_depth_mm=use_depth_z,
        )
        if lines_warn is not None:
            line_warn_actor = pl.add_mesh(
                lines_warn,
                color=_PLAN_QA_WARN_HEX,
                line_width=2,
                opacity=0.7,
                pickable=False,
            )
        if lines_fail is not None:
            line_fail_actor = pl.add_mesh(
                lines_fail,
                color=_PLAN_QA_FAIL_HEX,
                line_width=2,
                opacity=0.7,
                pickable=False,
            )

    meas_view0 = _make_measured_view_mesh(meas_pts_final, meas_sigma_final, meas_rgba_final)
    meas_actor: Any | None = None
    if n_m > 0:
        if meas_sigma_glyphs:
            has_rgba = meas_rgba_final is not None and "rgba" in meas_view0.point_data
            if has_rgba:
                meas_actor = pl.add_mesh(
                    meas_view0,
                    scalars="rgba",
                    rgba=True,
                    smooth_shading=False,
                    lighting=False,
                    opacity=1.0,
                    pickable=True,
                )
            else:
                meas_actor = pl.add_mesh(
                    meas_view0,
                    color=_MEASURED_COLOR_3D,
                    smooth_shading=False,
                    lighting=False,
                    opacity=1.0,
                    pickable=True,
                )
        elif meas_rgba_final is not None:
            meas_actor = pl.add_mesh(
                meas_view0,
                scalars="rgba",
                rgba=True,
                opacity=1.0,
                pickable=True,
                **point_kw,
            )
        else:
            meas_actor = pl.add_mesh(
                meas_view0,
                color=_MEASURED_COLOR_3D,
                opacity=1.0,
                pickable=True,
                **point_kw,
            )

    e_rng_lo = float(prep.e_lo)
    e_rng_hi = float(prep.e_hi)
    if e_rng_hi <= e_rng_lo:
        e_rng_lo -= 0.5
        e_rng_hi += 0.5

    layer_energies_plan = nominal_layer_energies_mev(planned_xyz)
    n_plan_layers = len(layer_energies_plan)
    _center_default = (
        int(np.clip(n_plan_layers // 2, 0, max(0, n_plan_layers - 1))) if n_plan_layers else 0
    )
    slice_cfg: dict[str, bool | int] = {
        "slice_on": False,
        "center_i": _center_default,
    }
    if slice_band_init:
        if "slice_on" in slice_band_init:
            slice_cfg["slice_on"] = bool(slice_band_init["slice_on"])
        if "center_i" in slice_band_init:
            ci0 = int(slice_band_init["center_i"])
            slice_cfg["center_i"] = int(np.clip(ci0, 0, max(0, n_plan_layers - 1)))
    # Filled when embedding in Qt so slice callback can repaint the QVTK widget after updates.
    _qt_vtk_embed: dict[str, Any] = {"widget": None}
    _cube_axes: dict[str, Any] = {"actor": None, "z_spec": None}

    def _empty_poly() -> Any:
        return pv.PolyData(np.zeros((0, 3), dtype=np.float64))

    def _slice_lo_hi_mev() -> tuple[float, float]:
        if not bool(slice_cfg["slice_on"]) or n_plan_layers == 0:
            return e_rng_lo, e_rng_hi
        return _nominal_layer_index_band_mev(
            layer_energies_plan, int(slice_cfg["center_i"]), half_width=2
        )

    def _scene_z_for_cube_axes(
        pm: np.ndarray,
        mm: np.ndarray,
    ) -> np.ndarray:
        if bool(slice_cfg["slice_on"]):
            z_parts: list[np.ndarray] = []
            if np.any(pm):
                z_parts.append(np.asarray(plan_pts[pm, 2], dtype=np.float64))
            if np.any(mm):
                z_parts.append(np.asarray(meas_pts_final[mm, 2], dtype=np.float64))
            if z_parts:
                return np.concatenate(z_parts)
        return np.r_[plan_pts[:, 2], meas_pts[:, 2]]

    def _apply_cube_z_axis(actor: Any, z_spec: _CubeZAxisSpec) -> None:
        apply_pyvista_cube_z_axis(
            actor,
            z_spec,
            x_min=float(x_min),
            x_max=float(x_max),
            y_min=float(y_min),
            y_max=float(y_max),
        )

    def _sync_cube_z_axis(
        pm: np.ndarray | None = None,
        mm: np.ndarray | None = None,
    ) -> None:
        actor = _cube_axes.get("actor")
        z_spec0 = _cube_axes.get("z_spec")
        if actor is None or z_spec0 is None:
            return
        if pm is None or mm is None:
            lo_m, hi_m = _slice_lo_hi_mev()
            pm = _energy_slice_mask(plan_e_mev, lo_m, hi_m)
            mm = (
                _energy_slice_mask(meas_e_final, lo_m, hi_m)
                if meas_e_final.size > 0
                else np.zeros(0, dtype=bool)
            )
        z_spec = _cube_z_axis_spec(
            _scene_z_for_cube_axes(pm, mm),
            use_proton_water_depth_mm=use_depth_z,
            tick_mm=eff_tick,
        )
        _cube_axes["z_spec"] = z_spec
        try:
            _apply_cube_z_axis(actor, z_spec)
        except Exception as exc:
            logger.warning("Cube Z-axis refresh failed: %s", exc)

    def _apply_nominal_energy_slice() -> None:
        nonlocal line_warn_actor, line_fail_actor
        lo_m, hi_m = _slice_lo_hi_mev()
        pm = _energy_slice_mask(plan_e_mev, lo_m, hi_m)
        mm = (
            _energy_slice_mask(meas_e_final, lo_m, hi_m)
            if meas_e_final.size > 0
            else np.zeros(0, dtype=bool)
        )

        if plan_rendered_fwhm_glyphs and prep.plan_fwhm_xy_mm is not None:
            if not np.any(pm):
                plan_actor.mapper.dataset = _empty_poly()
            else:
                plan_actor.mapper.dataset = _plan_spot_fwhm_glyph_mesh(
                    plan_pts[pm], prep.plan_fwhm_xy_mm[pm]
                )
        else:
            if not np.any(pm):
                plan_actor.mapper.dataset = _empty_poly()
            else:
                plan_actor.mapper.dataset = pv.PolyData(plan_pts[pm])

        if meas_actor is not None:
            if not np.any(mm):
                meas_actor.mapper.dataset = _empty_poly()
            else:
                sub_pts = meas_pts_final[mm]
                sub_sig = meas_sigma_final[mm]
                sub_rgba = meas_rgba_final[mm] if meas_rgba_final is not None else None
                meas_actor.mapper.dataset = _make_measured_view_mesh(sub_pts, sub_sig, sub_rgba)

        if line_warn_actor is not None:
            pl.remove_actor(line_warn_actor)
            line_warn_actor = None
        if line_fail_actor is not None:
            pl.remove_actor(line_fail_actor)
            line_fail_actor = None
        if (
            plan_qa_coloring
            and qa_draw_lines
            and dist_qa_draw is not None
            and exp_xyz_qa_draw is not None
            and np.any(mm)
        ):
            lw, lf = _plan_qa_error_line_polylines(
                meas_pts_final[mm],
                exp_xyz_qa_draw[mm],
                dist_qa_draw[mm],
                pass_mm=plan_qa_pass_mm,
                warn_mm=plan_qa_warn_mm,
                use_proton_water_depth_mm=use_depth_z,
            )
            if lw is not None:
                line_warn_actor = pl.add_mesh(
                    lw,
                    color=_PLAN_QA_WARN_HEX,
                    line_width=2,
                    opacity=0.7,
                    pickable=False,
                )
            if lf is not None:
                line_fail_actor = pl.add_mesh(
                    lf,
                    color=_PLAN_QA_FAIL_HEX,
                    line_width=2,
                    opacity=0.7,
                    pickable=False,
                )
        _sync_cube_z_axis(pm, mm)
        pl.render()
        qw = None
        if slice_qt is not None:
            qw = slice_qt.get("_qt_vtk_widget")
        if qw is None:
            qw = _qt_vtk_embed.get("widget")
        if qw is not None:
            try:
                qw.update()
            except Exception:
                pass

    if n_plan_layers > 0 and embed_parent is None and embed_qt is None:

        def _on_center_layer(value: float) -> None:
            ci = int(round(float(value)))
            slice_cfg["center_i"] = int(np.clip(ci, 0, n_plan_layers - 1))
            _apply_nominal_energy_slice()

        def _on_slice_mode_checkbox(checked: bool) -> None:
            slice_cfg["slice_on"] = bool(checked)
            _apply_nominal_energy_slice()

        _slider_rng = (0.0, float(max(0, n_plan_layers - 1)))
        _slider_val = float(slice_cfg["center_i"])
        try:
            pl.add_slider_widget(
                _on_center_layer,
                rng=_slider_rng,
                value=_slider_val,
                title="center plan layer (5 around)",
                pointa=(0.02, 0.92),
                pointb=(0.40, 0.92),
                fmt="%.0f",
                style="modern",
                interaction_event="always",
            )
        except (TypeError, ValueError):
            try:
                pl.add_slider_widget(
                    _on_center_layer,
                    rng=_slider_rng,
                    value=_slider_val,
                    title="center plan layer (5 around)",
                    pointa=(0.02, 0.92),
                    pointb=(0.40, 0.92),
                    fmt="%.0f",
                    interaction_event="always",
                )
            except TypeError:
                pl.add_slider_widget(
                    _on_center_layer,
                    rng=_slider_rng,
                    value=_slider_val,
                    title="center plan layer (5 around)",
                    pointa=(0.02, 0.92),
                    pointb=(0.40, 0.92),
                    fmt="%.0f",
                )
        pl.add_checkbox_button_widget(
            _on_slice_mode_checkbox,
            value=False,
            position=(14, 118),
            size=22,
            border_size=4,
        )
        pl.add_text(
            "5-layer slice",
            position=(42, 121),
            font_size=11,
            color="#c9d1d9",
        )

    pl.add_axes(
        line_width=4,
        x_color="#79c0ff",
        y_color="#56d364",
        z_color="#d2a8ff",
        cone_radius=0.4,
        shaft_length=0.7,
    )

    _lm = (layer_mode or "time_gap").strip().lower().replace("-", "_")
    if _lm == "plan_viterbi":
        _vp = (
            VITERBI_LAYER_ADVANCE_PENALTY_MM2_DEFAULT
            if viterbi_advance_penalty_mm2 is None
            else viterbi_advance_penalty_mm2
        )
        caption = (
            f"n_plan={plan_pts.shape[0]}, n_meas={n_m}, MeV [{prep.e_lo:.1f}, {prep.e_hi:.1f}]. "
            f"Layers: Viterbi vs plan, advance penalty {_vp:g} mm^2."
        )
    elif _lm == "unified":
        _vp = (
            VITERBI_LAYER_ADVANCE_PENALTY_MM2_DEFAULT
            if viterbi_advance_penalty_mm2 is None
            else viterbi_advance_penalty_mm2
        )
        _gap = TIME_LAYER_GAP_S_DEFAULT if layer_gap_s is None else layer_gap_s
        _xytol = (
            REFILL_SAME_SPOT_XY_TOLERANCE_MM
            if refill_same_spot_xy_tol_mm is None
            else refill_same_spot_xy_tol_mm
        )
        caption = (
            f"n_plan={plan_pts.shape[0]}, n_meas={n_m}, MeV [{prep.e_lo:.1f}, {prep.e_hi:.1f}]. "
            f"Layers: unified Viterbi + dt>={_gap:g} s / same-spot<={_xytol:g} mm gates, "
            f"base {_vp:g} mm^2."
        )
    elif _lm == "gate_counter":
        caption = (
            f"n_plan={plan_pts.shape[0]}, n_meas={n_m}, MeV [{prep.e_lo:.1f}, {prep.e_hi:.1f}]. "
            f"Layers: CSV Gate Counter — odd bucket=spot, even=deadtime; "
            f"spot index advances when count changes; layer from plan order ({GATE_COUNTER_KEY})."
        )
    else:
        _gap = TIME_LAYER_GAP_S_DEFAULT if layer_gap_s is None else layer_gap_s
        _xytol = (
            REFILL_SAME_SPOT_XY_TOLERANCE_MM
            if refill_same_spot_xy_tol_mm is None
            else refill_same_spot_xy_tol_mm
        )
        _trust = (
            REFILL_TRUST_TIME_GAP_STAY_DIST_MM
            if refill_trust_time_gap_stay_dist_mm is None
            else refill_trust_time_gap_stay_dist_mm
        )
        caption = (
            f"n_plan={plan_pts.shape[0]}, n_meas={n_m}, MeV [{prep.e_lo:.1f}, {prep.e_hi:.1f}]. "
            f"Layers: time gap | dt>={_gap:g} s, same-spot<={_xytol:g} mm, far-slice>{_trust:g} mm."
        )
    if prep.meas_partial_raw is not None:
        n_gold = int(np.count_nonzero(prep.meas_partial_raw))
        if n_gold:
            caption += (
                f" Gold: {n_gold} one-axis row(s); missing fit axis from plan at nominal layer."
            )
    if aggregate_spots:
        _sw = measured_spot_weight_caption(spot_weight_mode)
        caption += (
            f" Measured spots aggregated: {_sw}-weighted mean XY + σ per odd "
            f"{GATE_COUNTER_KEY} phase."
        )
        if aggregate_even_rows_after_odd > 0:
            caption += (
                f" Gate-counter: up to {aggregate_even_rows_after_odd} even-phase row(s) "
                "with good fits merged after each odd→even switch."
            )
    if plan_rendered_fwhm_glyphs:
        caption += (
            " Plan: FWHM ellipsoids from DICOM Scanning Spot Size (300A,0398); "
            "semiaxis = FWHM/2 in X/Y, thin along scene Z."
        )
    if meas_sigma_glyphs:
        caption += (
            " Measured: world-space σ ellipsoids — X/Y diameter (mm) = "
            f"{2.0 * sig_scale_eff:g}× fit σ ({SIGMA_A_KEY} / {SIGMA_B_KEY} axes mapped like A/B); "
            "thin disk along scene Z."
        )
    if weight_measured_by_channel and prep.meas_weight is not None:
        caption += f" Measured opacity ∝ {measured_spot_weight_caption(spot_weight_mode)}."
    if detector_align_caption:
        caption += " " + detector_align_caption.strip()
    if plan_qa_caption_extra:
        caption += " " + plan_qa_caption_extra
    if n_plan_layers > 0:
        if embed_qt is not None:
            caption += (
                " Right-hand panel: toggle 5-layer slice and drag the center layer slider "
                "(unchecked = full MeV range)."
            )
        elif embed_parent is None:
            caption += (
                " Upper left: enable 5-layer slice (checkbox) to show up to five consecutive "
                "nominal-energy layers (DICOM order) around the center layer index on the "
                "slider; leave the checkbox off to view the full MeV range."
            )
        elif _vtk_rendering_tk_dll_present():
            caption += (
                " Right-hand panel: toggle 5-layer slice and drag the center layer slider "
                "(unchecked = full MeV range)."
            )
        else:
            caption += (
                " Separate 3D window: 5-layer slice uses the right-hand panel "
                "(pip VTK omits Tk embedding)."
            )
    if display_perf_note:
        caption += display_perf_note
    if use_depth_z:
        caption += (
            " Z axis: proton CSDA water-equivalent depth (mm); tick step follows XY bounds (mm)."
        )
    pl.add_text(title, position="upper_left", font_size=11, color="#f0f6fc", shadow=True)
    pl.add_text(caption, position="lower_left", font_size=9, color="#8b949e")

    # Scene Z: negative depth/mm or −E×view_scale (shallow toward top); see nominal_mev_to_plot_z.
    # ``axes_ranges`` maps bounding-box corners to tick labels (mm or MeV).
    lo_m0, hi_m0 = _slice_lo_hi_mev()
    pm0 = _energy_slice_mask(plan_e_mev, lo_m0, hi_m0)
    mm0 = (
        _energy_slice_mask(meas_e_final, lo_m0, hi_m0)
        if meas_e_final.size > 0
        else np.zeros(0, dtype=bool)
    )
    z_all = _scene_z_for_cube_axes(pm0, mm0)
    z_spec_init = _cube_z_axis_spec(
        z_all,
        use_proton_water_depth_mm=use_depth_z,
        tick_mm=eff_tick,
    )
    _cube_axes["z_spec"] = z_spec_init
    bounds_axes = (
        float(x_min),
        float(x_max),
        float(y_min),
        float(y_max),
        z_spec_init.zmin_scene,
        z_spec_init.zmax_scene,
    )
    axes_ranges_scene = (
        float(x_min),
        float(x_max),
        float(y_min),
        float(y_max),
        z_spec_init.z_label_at_min,
        z_spec_init.z_label_at_max,
    )
    _cube_axes["actor"] = pl.show_bounds(
        bounds=bounds_axes,
        axes_ranges=axes_ranges_scene,
        grid=PYVISTA_CUBE_AXES_GRID,
        ticks=PYVISTA_CUBE_AXES_TICKS,
        location=PYVISTA_CUBE_AXES_LOCATION,
        all_edges=False,
        color="#8b949e",
        xtitle=prep.xlab,
        ytitle=prep.ylab,
        ztitle=z_spec_init.ztitle,
        n_xlabels=n_xlabels,
        n_ylabels=n_ylabels,
        n_zlabels=z_spec_init.n_zlabels,
        show_xaxis=True,
        show_yaxis=True,
        show_zaxis=True,
        show_xlabels=True,
        show_ylabels=True,
        show_zlabels=True,
        padding=0.06,
        fmt="%.4g",
    )
    try:
        _apply_cube_z_axis(_cube_axes["actor"], z_spec_init)
    except Exception as exc:
        logger.warning("Initial cube Z-axis apply failed: %s", exc)

    if reuse_plotter is None or not reuse_camera:
        pl.reset_camera()
        pl.camera.zoom(1.05)
    elif saved_camera_position is not None:
        try:
            pl.camera_position = saved_camera_position
        except Exception:
            pl.reset_camera()
            pl.camera.zoom(1.05)
    else:
        pl.reset_camera()
        pl.camera.zoom(1.05)

    try:
        pl.camera.parallel_projection = not bool(view_projection_perspective)
    except Exception:
        pass

    _sync_cube_z_axis()
    _apply_nominal_energy_slice()

    if embed_qt is not None:
        try:
            if reembed_qt:
                w = _embed_pyvista_plotter_in_qt(embed_qt, pl)
                _qt_vtk_embed["widget"] = w
                if slice_qt is not None:
                    slice_qt["_qt_vtk_widget"] = w
            else:
                pl.render()
                qw = slice_qt.get("_qt_vtk_widget") if slice_qt is not None else None
                if qw is None:
                    qw = _qt_vtk_embed.get("widget")
                if qw is not None:
                    try:
                        qw.update()
                    except Exception:
                        pass
            _sync_cube_z_axis()
            pl.render()
        except Exception:
            idle_slice_band_controls_qt(slice_qt)
            raise
        if slice_qt is not None:
            if n_plan_layers > 0:
                _wire_slice_band_controls_qt(
                    slice_qt,
                    slice_cfg,
                    layer_energies_plan,
                    n_plan_layers,
                    _apply_nominal_energy_slice,
                )
            else:
                idle_slice_band_controls_qt(slice_qt)
    elif embed_parent is not None:
        if tk is None:
            pl.show()
            idle_slice_band_controls(slice_tk)
        else:
            for _child in embed_parent.winfo_children():
                _child.destroy()
            tk_top = embed_parent.winfo_toplevel()
            _stop_tk_vtk_event_pump(tk_top)
            embedded = False
            if _vtk_rendering_tk_dll_present():
                try:
                    _embed_pyvista_plotter_in_tk(embed_parent, pl)
                    embedded = True
                except Exception:
                    embedded = False
            if not embedded:
                _show_tk_vtk_fallback_panel(embed_parent)
                pl.show(interactive_update=True, auto_close=False, interactive=True)
                _ensure_pyvista_iren_initialized(pl)
                _start_tk_vtk_event_pump(tk_top, pl)
            if slice_tk is not None:
                if n_plan_layers > 0:
                    _wire_slice_band_controls(
                        slice_tk,
                        slice_cfg,
                        layer_energies_plan,
                        n_plan_layers,
                        _apply_nominal_energy_slice,
                    )
                else:
                    idle_slice_band_controls(slice_tk)
    else:
        pl.show()
    return pl
