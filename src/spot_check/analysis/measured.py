"""Measured."""

from __future__ import annotations

from spot_check.analysis._imports import *  # noqa: F403
from spot_check.analysis.assign.base import (
    finalize_measured_assign_coverage,
    plan_spots_without_assignment_data,
)
from spot_check.analysis.assign.types import AssignCsvParams, MeasuredAssignResult
from spot_check.analysis.csv_io import open_acquisition_csv
from spot_check.analysis.layers import _opt_float_cell
from spot_check.analysis.spatial import (
    _plan_xy_from_optional_ab,
    xy_sigma_flier_keep_mask,
)


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

def measured_row_time_s(tup: tuple[float, ...]) -> float:
    """Acquisition ``time (s)`` when present as optional tuple index 8."""
    if len(tup) > 8:
        try:
            t = float(tup[8])
            return t if math.isfinite(t) else float("nan")
        except (TypeError, ValueError):
            return float("nan")
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
    time_s: float | None = None,
) -> tuple[float, ...]:
    """One measured row as 8- or 9-tuple (optional index 8 = acquisition time in seconds)."""
    ch = float(channel_sum_na) if channel_sum_na is not None else float(weight)
    row: tuple[float, ...] = (
        float(a),
        float(b),
        float(layer),
        float(weight),
        int(partial),
        _sigma_cell_to_float(sa),
        _sigma_cell_to_float(sb),
        ch,
    )
    if time_s is not None:
        try:
            t = float(time_s)
            if math.isfinite(t):
                return row + (t,)
        except (TypeError, ValueError):
            pass
    return row

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
    spot_weight_mode: str,
) -> None:
    swm = normalize_measured_spot_weight_mode(spot_weight_mode)
    with open_acquisition_csv(csv_path) as f:
        pr = csv.DictReader(f)
        fn = pr.fieldnames
        if not fn:
            return
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


_GateSpotBufRow = tuple[float, float, float, float, int, float | None, float | None, float, float]


def _measured_tuple_for_spot_weighted_mean(
    t: tuple[float, ...],
) -> _GateSpotBufRow:
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
    t_s = measured_row_time_s(t)
    return (a, b, lay, w, pcd, sa, sb, ch, t_s)


def _finalize_spot_channel_weighted(
    buf: list[_GateSpotBufRow],
) -> tuple[float, float, float, float, int, float, float, float, float]:
    """Collapse one spot: weighted mean of A/B, layer, partial, σ, and acquisition time."""
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
    t_vals = [float(r[8]) if len(r) > 8 else float("nan") for r in buf]
    t_ok = [math.isfinite(float(v)) for v in t_vals]
    t_mean = _weighted_mean_masked(t_vals, ws, t_ok)
    return (a_mean, b_mean, lay_mean, sw, pcd_out, sig_a_mean, sig_b_mean, ch_mean, t_mean)


def gate_counter_aggregated_layer_indices_from_csv(
    csv_path: Path,
    spots_per_layer: Sequence[int],
    *,
    max_layer: int,
    a_is_x: bool,
    spot_weight_mode: str,
    max_points: int | None = None,
) -> np.ndarray:
    """Nominal layer index per gate_counter spot (``aggregate_spots=True`` semantics).

    Matches :func:`measured_spot_abc_from_csv` gate_counter aggregation (weighted mean of
    per-row ``eff_li`` in each spot buffer) without plan imputation or A/B output.
    """
    cumul: list[int] = [0]
    for c in spots_per_layer:
        cumul.append(cumul[-1] + int(c))
    hi = int(max_layer)
    swm = normalize_measured_spot_weight_mode(spot_weight_mode)
    layers: list[int] = []
    spot_buf: list[_GateSpotBufRow] = []
    prev_gate: int | None = None
    i_spot = 0
    eff_li = 0
    n_raw = 0
    fa_key = FIT_AMPLITUDE_A_KEY
    a_key = "Fit Mean Position A (mm)"
    b_key = "Fit Mean Position B (mm)"
    gc_key = GATE_COUNTER_KEY

    def flush_layer() -> None:
        if spot_buf:
            layers.append(int(_finalize_spot_channel_weighted(spot_buf)[2]))
            spot_buf.clear()

    with open_acquisition_csv(csv_path) as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or gc_key not in reader.fieldnames:
            return np.zeros(0, dtype=np.int64)
        for row in reader:
            g_raw = (row.get(gc_key) or "").strip()
            if not g_raw:
                continue
            try:
                g = int(float(g_raw))
            except ValueError:
                continue
            if g != prev_gate:
                if prev_gate is not None and prev_gate % 2 == 1 and spot_buf:
                    flush_layer()
                if g % 2 == 1:
                    eff_li = max(0, min(bisect.bisect_right(cumul, i_spot) - 1, hi))
                    i_spot += 1
                prev_gate = g
            if g % 2 == 0:
                continue
            if not (row.get(fa_key) or "").strip():
                continue
            a_opt = _opt_float_cell(row, a_key)
            b_opt = _opt_float_cell(row, b_key)
            _, _, pcd = _plan_xy_from_optional_ab(a_opt, b_opt, a_is_x=a_is_x)
            # Gate-counter aggregation matches legacy CSV semantics: keep partial one-axis fits;
            # only rows with both position axes missing are skipped.
            if pcd < 0:
                continue
            w_ch = measured_spot_weight_from_row(row, swm)
            ch_n = _channel_sum_na_from_row(row)
            spot_buf.append((0.0, 0.0, float(eff_li), w_ch, 0, None, None, ch_n, float("nan")))
            n_raw += 1
            if max_points is not None and n_raw >= max_points:
                if spot_buf:
                    flush_layer()
                break
    if spot_buf:
        flush_layer()
    return np.asarray(layers, dtype=np.int64)


def _gate_phase_spot_ids_for_rows(gates: Sequence[int]) -> list[int]:
    """Monotonic spot id per row from Gate Counter (odd=on-spot; even → ``-1``)."""
    spot_ids: list[int] = []
    cur = -1
    prev_g: int | None = None
    for g in gates:
        if g < 0:
            spot_ids.append(-1)
            continue
        if g % 2 == 0:
            if prev_g is not None and prev_g % 2 == 1:
                cur += 1
            spot_ids.append(-1)
            prev_g = g
            continue
        if prev_g is not None and prev_g % 2 == 1 and g != prev_g:
            cur += 1
        elif cur < 0:
            cur = 0
        spot_ids.append(cur)
        prev_g = g
    return spot_ids


def _aggregate_rows_by_spot_id_map(
    rows: list[tuple[float, ...]],
    spot_ids: Sequence[int],
) -> dict[int, tuple[float, ...]]:
    """Weighted mean per ``spot_id``; keys are spot ids with at least one row."""
    if len(rows) != len(spot_ids):
        raise ValueError(
            f"rows and spot_ids length mismatch ({len(rows)} vs {len(spot_ids)})"
        )
    buckets: dict[int, list[_GateSpotBufRow]] = {}
    for row, sid in zip(rows, spot_ids):
        if sid < 0:
            continue
        buf_row = _measured_tuple_for_spot_weighted_mean(row)
        if sid not in buckets:
            buckets[sid] = []
        buckets[sid].append(buf_row)
    return {sid: _finalize_spot_channel_weighted(buf) for sid, buf in buckets.items()}


def aggregate_measured_rows_by_spot_id(
    rows: list[tuple[float, ...]],
    spot_ids: Sequence[int],
) -> list[tuple[float, ...]]:
    """Post-assignment weighted mean per ``spot_id``; first-seen group order."""
    if not rows:
        return []
    order: list[int] = []
    seen: set[int] = set()
    for sid in spot_ids:
        if sid < 0 or sid in seen:
            continue
        seen.add(sid)
        order.append(sid)
    by_id = _aggregate_rows_by_spot_id_map(rows, spot_ids)
    return [by_id[sid] for sid in order if sid in by_id]


def _plan_sequential_aggregated_per_plan_spot(
    rows: list[tuple[float, ...]],
    spot_ids: Sequence[int],
    *,
    n_plan_spots: int,
    planned_xyz: list[tuple[float, float, float]],
    spots_per_layer: Sequence[int],
    a_is_x: bool,
) -> list[tuple[float, ...]]:
    """One aggregated row per plan spot that has assigned acquisition data (no plan-only fill)."""
    from spot_check.analysis.layers import delivery_layer_indices

    plan_layer = delivery_layer_indices(n_plan_spots, spots_per_layer)
    by_pi = _aggregate_rows_by_spot_id_map(rows, spot_ids)
    out: list[tuple[float, ...]] = []
    for i in sorted(by_pi):
        if i < 0 or i >= n_plan_spots:
            continue
        t = by_pi[i]
        a, b = float(t[0]), float(t[1])
        if not (math.isfinite(a) and math.isfinite(b)):
            continue
        layer_i = float(int(plan_layer[i]))
        lay = float(t[2])
        if not math.isfinite(lay):
            lay = layer_i
        sa = float(t[5]) if len(t) > 5 and math.isfinite(float(t[5])) else None
        sb = float(t[6]) if len(t) > 6 and math.isfinite(float(t[6])) else None
        ch = float(t[7]) if len(t) > 7 and math.isfinite(float(t[7])) else float(t[3])
        out.append(
            _measured_row_with_sigma(
                a,
                b,
                lay,
                float(t[3]),
                int(t[4]),
                sa,
                sb,
                channel_sum_na=ch,
                time_s=measured_row_time_s(t),
            )
        )
    return out


def _aggregate_assigned_rows(
    rows: list[tuple[float, ...]],
    spot_ids: Sequence[int],
    *,
    aggregate_spots: bool,
) -> list[tuple[float, ...]]:
    if not aggregate_spots:
        return rows
    return aggregate_measured_rows_by_spot_id(rows, spot_ids)


def filter_assigned_xy_fliers(
    result: MeasuredAssignResult,
    planned_xyz: list[tuple[float, float, float]],
    *,
    n_sigma: float,
) -> MeasuredAssignResult:
    """Drop assigned rows whose plan-frame XY offset exceeds ``n_sigma`` fit σ vs layer-NN plan."""
    if not result.rows or not planned_xyz:
        return result
    keep = xy_sigma_flier_keep_mask(
        result.rows,
        planned_xyz,
        n_sigma=float(n_sigma),
        a_is_x=result.a_is_x,
    )
    if bool(np.all(keep)):
        return result
    idx = np.flatnonzero(keep).astype(np.intp, copy=False)
    rows = [result.rows[int(i)] for i in idx]
    spot_ids = [result.spot_ids[int(i)] for i in idx]
    gates = [result.gates[int(i)] for i in idx] if result.gates else []
    return finalize_measured_assign_coverage(
        MeasuredAssignResult(
            rows=rows,
            spot_ids=spot_ids,
            layer_mode=result.layer_mode,
            assign_method=result.assign_method,
            n_plan_spots=result.n_plan_spots,
            planned_xyz=result.planned_xyz,
            spots_per_layer=result.spots_per_layer,
            a_is_x=result.a_is_x,
            gates=gates,
        ),
        planned_xyz=planned_xyz,
    )


def aggregate_measured_assign_result(
    result: MeasuredAssignResult,
    *,
    aggregate_spots: bool,
) -> list[tuple[float, ...]]:
    """Collapse assigned rows by spot id when ``aggregate_spots`` is True."""
    if not aggregate_spots:
        return result.rows
    if not result.rows:
        return []
    spot_ids = result.spot_ids
    if result.gates and any(g >= 0 for g in result.gates):
        spot_ids = _gate_phase_spot_ids_for_rows(result.gates)
    if (
        result.layer_mode == "auto"
        and result.assign_method == "plan_sequential"
        and result.planned_xyz is not None
        and result.spots_per_layer is not None
    ):
        return _plan_sequential_aggregated_per_plan_spot(
            result.rows,
            spot_ids,
            n_plan_spots=result.n_plan_spots,
            planned_xyz=result.planned_xyz,
            spots_per_layer=result.spots_per_layer,
            a_is_x=result.a_is_x,
        )
    return aggregate_measured_rows_by_spot_id(result.rows, spot_ids)


def assign_measured_from_csv(
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
    spot_weight_mode: str = SPOT_WEIGHT_MODE_DEFAULT,
    auto_episode_gap_s: float = TIME_LAYER_GAP_S_DEFAULT,
    auto_min_on_spot_weight_na: float = AUTO_MIN_ON_SPOT_WEIGHT_NA_DEFAULT,
    auto_spot_xy_jump_mm: float = AUTO_SPOT_XY_JUMP_MM_DEFAULT,
    auto_min_episode_rows: int = AUTO_MIN_EPISODE_ROWS_DEFAULT,
    auto_infer_params: bool = True,
    auto_assign_method: str = "episodes",
    heal_partial_fit_axes: bool = False,
    coarse_flat_transform: DetectorRigidAlign2D | None = None,
    skip_column_probe: bool = False,
    preloaded_auto_columns: object | None = None,
) -> MeasuredAssignResult:
    """Assign layers/spots from CSV without spot aggregation.

    See :func:`measured_spot_abc_from_csv` for mode documentation.
    """
    from spot_check.analysis.assign import run_layer_assignment

    params = AssignCsvParams(
        csv_path=csv_path,
        max_points=max_points,
        planned_xyz=planned_xyz,
        a_is_x=a_is_x,
        spot_weight_mode=spot_weight_mode,
        skip_column_probe=skip_column_probe,
        heal_partial_fit_axes=heal_partial_fit_axes,
        coarse_flat_transform=coarse_flat_transform,
        preloaded_auto_columns=preloaded_auto_columns,
        layer_gap_s=layer_gap_s,
        refill_same_spot_xy_tol_mm=refill_same_spot_xy_tol_mm,
        refill_trust_time_gap_stay_dist_mm=refill_trust_time_gap_stay_dist_mm,
        viterbi_advance_penalty_mm2=viterbi_advance_penalty_mm2,
        auto_episode_gap_s=auto_episode_gap_s,
        auto_min_on_spot_weight_na=auto_min_on_spot_weight_na,
        auto_spot_xy_jump_mm=auto_spot_xy_jump_mm,
        auto_min_episode_rows=auto_min_episode_rows,
        auto_infer_params=auto_infer_params,
        auto_assign_method=auto_assign_method,
    )
    return run_layer_assignment(layer_mode, params)


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
    aggregate_spots: bool = True,
    spot_weight_mode: str = SPOT_WEIGHT_MODE_DEFAULT,
    auto_episode_gap_s: float = TIME_LAYER_GAP_S_DEFAULT,
    auto_min_on_spot_weight_na: float = AUTO_MIN_ON_SPOT_WEIGHT_NA_DEFAULT,
    auto_spot_xy_jump_mm: float = AUTO_SPOT_XY_JUMP_MM_DEFAULT,
    auto_min_episode_rows: int = AUTO_MIN_EPISODE_ROWS_DEFAULT,
    auto_infer_params: bool = True,
    auto_assign_method: str = "episodes",
    heal_partial_fit_axes: bool = False,
) -> list[tuple[float, ...]]:
    """Measured rows as (A mm, B mm, layer index, spot weight, partial code).

    Index **3** is a positive **weight** from ``spot_weight_mode`` (IX512 sum and/or Fit Amplitude
    A/B; see :func:`measured_spot_weight_from_row`) for aggregation and optional 3D opacity.

    Returns 8-tuples ``(A, B, layer, weight, partial, σ_A, σ_B, channel_sum_nA)`` (σ may be NaN).

    By default, rows missing Fit Mean Position A or B are dropped. Set
    ``heal_partial_fit_axes=True`` to keep rows with exactly one missing axis and fill the gap
    from the nearest plan spot at the assigned layer (partial code 1 or 2). Rows with both axes
    missing are always dropped.

    **auto** never reads ``Gate Counter``. With ``auto_assign_method='episodes'``, deadtime
    segmentation aligns spot count to the plan and layers follow **acquisition time** (earliest
    spot -> layer 0 = highest nominal energy). With ``auto_assign_method='plan_sequential'``,
    the first on-spot burst is plan spot **0** and delivery advances **+1** after deadtime gaps;
    after each deadtime break (no fit on either position axis), assignment advances by exactly one
    plan slot once the current spot has at least one row (no skipped plan indices).
    Layers then match gate_counter plan-slot indexing.

    When ``auto_infer_params`` is True (default), episode gap, XY jump, on-spot weight floor,
    min episode rows, and Viterbi advance penalty are derived from the CSV timing/weights and
    plan geometry (:func:`infer_auto_layer_params`); explicit ``auto_*`` / ``viterbi_*`` kwargs
    are ignored. Set ``auto_infer_params=False`` to use the keyword values (tests, scripts).

    **Spot aggregation** (``aggregate_spots=True``) runs after assignment: rows with the same
    assignment id are collapsed via :func:`aggregate_measured_rows_by_spot_id` (weighted means of
    A/B, layer, σ, and weight). Ids come from the assignment stage (episode/span index, gate-counter
    spot phase, time-gap delivery spot, or gate phases when that column is present).
    """
    assigned = assign_measured_from_csv(
        csv_path,
        max_points=max_points,
        layer_mode=layer_mode,
        layer_gap_s=layer_gap_s,
        refill_same_spot_xy_tol_mm=refill_same_spot_xy_tol_mm,
        refill_trust_time_gap_stay_dist_mm=refill_trust_time_gap_stay_dist_mm,
        viterbi_advance_penalty_mm2=viterbi_advance_penalty_mm2,
        planned_xyz=planned_xyz,
        a_is_x=a_is_x,
        spot_weight_mode=spot_weight_mode,
        auto_episode_gap_s=auto_episode_gap_s,
        auto_min_on_spot_weight_na=auto_min_on_spot_weight_na,
        auto_spot_xy_jump_mm=auto_spot_xy_jump_mm,
        auto_min_episode_rows=auto_min_episode_rows,
        auto_infer_params=auto_infer_params,
        auto_assign_method=auto_assign_method,
        heal_partial_fit_axes=heal_partial_fit_axes,
    )
    return aggregate_measured_assign_result(assigned, aggregate_spots=aggregate_spots)


__all__ = [
    "AssignCsvParams",
    "MeasuredAssignResult",
    "aggregate_measured_assign_result",
    "assign_measured_from_csv",
    "filter_assigned_xy_fliers",
    "finalize_measured_assign_coverage",
    "gate_counter_aggregated_layer_indices_from_csv",
    "measured_charge_na_from_tuple",
    "measured_row_time_s",
    "measured_spot_abc_from_csv",
    "measured_spot_weight_caption",
    "measured_spot_weight_from_row",
    "normalize_measured_spot_weight_mode",
    "plan_spots_without_assignment_data",
]
