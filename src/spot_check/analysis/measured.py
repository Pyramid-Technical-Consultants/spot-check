"""Measured."""

from __future__ import annotations

from spot_check.analysis._imports import *  # noqa: F403
from spot_check.analysis.csv_io import open_acquisition_csv
from spot_check.analysis.layers import (
    _impute_plan_axis_fast,
    _layer_advance_plausible_vs_refill,
    _opt_float_cell,
    _plan_impute_lookups_per_layer,
    _PlanImputeLookup,
    delivery_layer_indices,
    layer_indices_by_acquisition_time,
    viterbi_monotone_layer_assign,
)
from spot_check.analysis.spatial import (
    _ab_from_plan_xy,
    _build_layer_kdtrees,
    _emit_sqdist_to_layers_mm2,
    _plan_xy_by_energy_layer,
    _plan_xy_from_optional_ab,
    fit_position_row_ok,
    nominal_layer_energies_mev,
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


_GateSpotBufRow = tuple[float, float, float, float, int, float | None, float | None, float]


def _finalize_spot_channel_weighted(
    buf: list[_GateSpotBufRow],
) -> tuple[float, float, float, float, int, float, float, float]:
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
            spot_buf.append((0.0, 0.0, float(eff_li), w_ch, 0, None, None, ch_n))
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
    """One output row per plan delivery index (0 … n_plan-1); nominal layer from plan order."""
    from spot_check.analysis.layers import delivery_layer_indices

    plan_layer = delivery_layer_indices(n_plan_spots, spots_per_layer)
    by_pi = _aggregate_rows_by_spot_id_map(rows, spot_ids)
    out: list[tuple[float, ...]] = []
    for i in range(n_plan_spots):
        layer_i = float(int(plan_layer[i]))
        t = by_pi.get(i)
        if t is None:
            px, py, _ = planned_xyz[i]
            a, b = _ab_from_plan_xy(float(px), float(py), a_is_x=a_is_x)
            out.append(
                _measured_row_with_sigma(
                    a,
                    b,
                    layer_i,
                    1e-18,
                    0,
                    None,
                    None,
                    channel_sum_na=1e-18,
                )
            )
            continue
        a, b = float(t[0]), float(t[1])
        if not (math.isfinite(a) and math.isfinite(b)):
            px, py, _ = planned_xyz[i]
            a, b = _ab_from_plan_xy(float(px), float(py), a_is_x=a_is_x)
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
    return (a, b, lay, w, pcd, sa, sb, ch)

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
    align_detector_xy_before_assign: bool = False,
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
    rows are assigned in **plan delivery order** (spot 0 = first spot of the highest-energy layer);
    after each deadtime break (no fit on either position axis), assignment advances by exactly one
    plan slot once the current spot has at least one row (no skipped plan indices).
    Layers then match gate_counter plan-slot indexing.

    When ``auto_infer_params`` is True (default), episode gap, XY jump, on-spot weight floor,
    min episode rows, and Viterbi advance penalty are derived from the CSV timing/weights and
    plan geometry (:func:`infer_auto_layer_params`); explicit ``auto_*`` / ``viterbi_*`` kwargs
    are ignored. Set ``auto_infer_params=False`` to use the keyword values (tests, scripts).

    **Pre-assignment detector align** (``align_detector_xy_before_assign=True``): rigid 2D map on
    flattened plan + on-spot measured XY before episode or plan-sequential assignment (``auto``
    only). Use when the detector frame is misaligned; post-assignment align only moves A/B after
    indices are fixed.

    **Spot aggregation** (``aggregate_spots=True``) runs after assignment: rows with the same
    assignment id are collapsed via :func:`aggregate_measured_rows_by_spot_id` (weighted means of
    A/B, layer, σ, and weight). Ids come from the assignment stage (episode/span index, gate-counter
    spot phase, time-gap delivery spot, or gate phases when that column is present).
    """
    mode = layer_mode.strip().lower().replace("-", "_")
    if mode not in ("time_gap", "plan_viterbi", "auto", "gate_counter"):
        raise ValueError(
            "layer_mode must be 'time_gap', 'plan_viterbi', 'auto', or 'gate_counter'"
        )

    swm = normalize_measured_spot_weight_mode(spot_weight_mode)

    _probe_csv_columns_for_measured_weights(csv_path, spot_weight_mode=swm)
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
    elif mode == "auto":
        if auto_episode_gap_s <= 0:
            raise ValueError("auto_episode_gap_s must be > 0")
        if auto_spot_xy_jump_mm <= 0:
            raise ValueError("auto_spot_xy_jump_mm must be > 0")
        if auto_min_on_spot_weight_na < 0:
            raise ValueError("auto_min_on_spot_weight_na must be >= 0")
        if int(auto_min_episode_rows) < 1:
            raise ValueError("auto_min_episode_rows must be >= 1")
        if viterbi_advance_penalty_mm2 < 0:
            raise ValueError("viterbi_advance_penalty_mm2 must be >= 0")
        if not planned_xyz:
            raise ValueError("auto requires planned_xyz from the RT plan")
        assign_m = str(auto_assign_method).strip().lower().replace("-", "_")
        if assign_m == "sequential":
            assign_m = "plan_sequential"
        from spot_check.constants import AUTO_ASSIGN_METHODS

        if assign_m not in AUTO_ASSIGN_METHODS:
            raise ValueError(
                f"auto_assign_method must be one of {sorted(AUTO_ASSIGN_METHODS)}, got {assign_m!r}"
            )
    else:  # gate_counter
        if not planned_xyz:
            raise ValueError("gate_counter requires planned_xyz from the RT plan")

    layer_energies: list[float] | None = None
    max_layer: int | None = None
    if planned_xyz:
        layer_energies = nominal_layer_energies_mev(planned_xyz)
        if layer_energies:
            max_layer = len(layer_energies) - 1

    if mode == "plan_viterbi":
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
        w_buf: list[float] = []
        ch_buf: list[float] = []
        partial_plan_xy: list[tuple[float | None, float | None]] = []
        partial_codes: list[int] = []
        gates_acc: list[int] = []
        sig_acc: list[tuple[float | None, float | None]] = []
        fa_key = "Fit Amplitude A (nA)"
        a_key = "Fit Mean Position A (mm)"
        b_key = "Fit Mean Position B (mm)"
        with open_acquisition_csv(csv_path) as f:
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
                if not fit_position_row_ok(pcd, heal_partial_fit_axes=heal_partial_fit_axes):
                    continue
                mx_i, my_i = _impute_plan_axis_fast(global_lk, mx_p, my_p)
                a_fin, b_fin = _ab_from_plan_xy(mx_i, my_i, a_is_x=a_is_x)
                ab_buf.append((a_fin, b_fin))
                xy_buf.append([mx_i, my_i])
                w_buf.append(measured_spot_weight_from_row(row, swm))
                ch_buf.append(_channel_sum_na_from_row(row))
                partial_plan_xy.append((mx_p, my_p))
                partial_codes.append(pcd)
                sig_acc.append(
                    (_opt_float_cell(row, SIGMA_A_KEY), _opt_float_cell(row, SIGMA_B_KEY))
                )
                g_cell = _gate_int_from_row(row, GATE_COUNTER_KEY)
                gates_acc.append(int(g_cell) if g_cell is not None else -1)
                if max_points is not None and len(ab_buf) >= max_points:
                    break
        if not ab_buf:
            return []
        meas_xy = np.asarray(xy_buf, dtype=np.float64)
        emit = _emit_sqdist_to_layers_mm2(meas_xy, layer_xy)
        layers_idx = viterbi_monotone_layer_assign(emit, viterbi_advance_penalty_mm2)
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
            zip(ab_buf, layers_idx, w_buf, ch_buf, partial_codes)
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
        agg_ids = (
            _gate_phase_spot_ids_for_rows(gates_acc)
            if any(g >= 0 for g in gates_acc)
            else [int(layers_idx[i]) for i in range(len(out))]
        )
        return _aggregate_assigned_rows(out, agg_ids, aggregate_spots=aggregate_spots)

    if mode == "auto":
        from spot_check.analysis.auto_columns import load_auto_fit_columns_from_csv
        from spot_check.analysis.episodes import (
            cols_with_delivery_weights,
            segment_align_auto_columns,
        )
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

        cols = load_auto_fit_columns_from_csv(
            csv_path,
            global_lk=global_lk,
            a_is_x=a_is_x,
            spot_weight_mode=swm,
            max_points=max_points,
            include_deadtime_rows=assign_m == "plan_sequential",
            heal_partial_fit_axes=heal_partial_fit_axes,
        )
        if len(cols) == 0:
            return []

        if align_detector_xy_before_assign:
            from spot_check.analysis.alignment import align_auto_fit_columns_to_plan_xy

            cols, _pre_align = align_auto_fit_columns_to_plan_xy(
                cols,
                planned_xyz,  # type: ignore[arg-type]
                a_is_x=a_is_x,
            )

        n_plan_spots = len(planned_xyz)  # type: ignore[arg-type]
        spots_per_layer = [
            int(np.asarray(arr, dtype=np.float64).reshape(-1, 2).shape[0])
            for arr in layer_xy
        ]

        from spot_check.constants import (
            AUTO_EDGE_DEAD_RATIO_DEFAULT,
            AUTO_EDGE_TINY_MERGE_ROWS,
        )

        gap_s = float(auto_episode_gap_s)
        xy_jump = float(auto_spot_xy_jump_mm)
        min_w = float(auto_min_on_spot_weight_na)
        min_rows = int(auto_min_episode_rows)
        dead_ratio = float(AUTO_EDGE_DEAD_RATIO_DEFAULT)
        tiny_merge = int(AUTO_EDGE_TINY_MERGE_ROWS)
        plan_seq_radius_mm2: float | None = None
        if auto_infer_params:
            from spot_check.analysis.auto_params import infer_auto_layer_params

            auto_p = infer_auto_layer_params(cols, planned_xyz)  # type: ignore[arg-type]
            gap_s = auto_p.episode_gap_s
            xy_jump = auto_p.spot_xy_jump_mm
            min_w = auto_p.min_on_spot_weight_na
            min_rows = auto_p.min_episode_rows
            dead_ratio = auto_p.dead_ratio
            tiny_merge = auto_p.tiny_merge_rows
            plan_seq_radius_mm2 = float(auto_p.plan_seq_cluster_radius_mm2)

        assign_m_run = assign_m
        if assign_m_run == "episodes":
            aligned_groups, _diag = segment_align_auto_columns(
                cols,
                n_plan_spots=n_plan_spots,
                episode_gap_s=gap_s,
                min_on_spot_weight_na=min_w,
                spot_xy_jump_mm=xy_jump,
                min_episode_rows=min_rows,
                dead_ratio=dead_ratio,
                tiny_merge_rows=tiny_merge,
                plan_xy=plan_xy2,
            )
            layers_idx_auto = layer_indices_by_acquisition_time(
                aligned_groups, spots_per_layer
            )
        else:
            from spot_check.analysis.plan_sequential import (
                assign_plan_indices_sequential,
                plan_spot_index_per_span,
                sequential_spans_from_plan_indices,
            )

            plan_idx = assign_plan_indices_sequential(
                cols,
                plan_xy2,
                min_rows_on_spot=1,
                cluster_radius_mm2=plan_seq_radius_mm2,
            )
            aligned_groups = sequential_spans_from_plan_indices(plan_idx)
            plan_layers = delivery_layer_indices(n_plan_spots, spots_per_layer)
            spot_pi = plan_spot_index_per_span(plan_idx, aligned_groups)
            layers_idx_auto = plan_layers[spot_pi]

        cols_w = cols_with_delivery_weights(cols)

        hi_auto = max_layer

        def _clamp_layer(efi: int) -> int:
            if efi < 0:
                return 0
            if efi > hi_auto:
                return hi_auto
            return efi

        out_rows: list[tuple[float, ...]] = []
        spot_ids: list[int] = []
        for ei, (s, e) in enumerate(aligned_groups):
            efi = _clamp_layer(int(layers_idx_auto[ei]))
            lk_ref = layer_lks[efi] if efi < len(layer_lks) else None
            if lk_ref is None:
                lk_ref = global_lk
            for ri in range(s, e):
                pcd = int(cols.pcd[ri])
                mx_pp = float(cols.mx_p[ri])
                my_pp = float(cols.my_p[ri])
                if pcd == 0:
                    a_fin, b_fin = float(cols.a[ri]), float(cols.b[ri])
                else:
                    mx_f, my_f = _impute_plan_axis_fast(
                        lk_ref,
                        mx_pp if mx_pp == mx_pp else None,
                        my_pp if my_pp == my_pp else None,
                    )
                    a_fin, b_fin = _ab_from_plan_xy(mx_f, my_f, a_is_x=a_is_x)
                sa_v = float(cols.sa[ri])
                sb_v = float(cols.sb[ri])
                sa = sa_v if sa_v == sa_v else None
                sb = sb_v if sb_v == sb_v else None
                wch = float(cols_w.weight[ri])
                ch_n = float(cols_w.ch_n[ri])
                out_rows.append(
                    _measured_row_with_sigma(
                        a_fin,
                        b_fin,
                        float(efi),
                        wch,
                        pcd,
                        sa,
                        sb,
                        channel_sum_na=ch_n,
                    )
                )
                if assign_m_run == "plan_sequential":
                    spot_ids.append(int(plan_idx[ri]))
                else:
                    spot_ids.append(ei)
                if max_points is not None and len(out_rows) >= max_points:
                    if assign_m_run == "plan_sequential" and aggregate_spots:
                        return _plan_sequential_aggregated_per_plan_spot(
                            out_rows,
                            spot_ids,
                            n_plan_spots=n_plan_spots,
                            planned_xyz=planned_xyz,  # type: ignore[arg-type]
                            spots_per_layer=spots_per_layer,
                            a_is_x=a_is_x,
                        )
                    return out_rows

        if assign_m_run == "plan_sequential" and aggregate_spots:
            return _plan_sequential_aggregated_per_plan_spot(
                out_rows,
                spot_ids,
                n_plan_spots=n_plan_spots,
                planned_xyz=planned_xyz,  # type: ignore[arg-type]
                spots_per_layer=spots_per_layer,
                a_is_x=a_is_x,
            )
        return _aggregate_assigned_rows(
            out_rows, spot_ids, aggregate_spots=aggregate_spots
        )

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
        spot_ids_gc: list[int] = []
        prev_gate: int | None = None
        spot_id_gc = -1
        i_spot = 0
        eff_li = 0
        fa_key = "Fit Amplitude A (nA)"
        a_key = "Fit Mean Position A (mm)"
        b_key = "Fit Mean Position B (mm)"
        with open_acquisition_csv(csv_path) as f:
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
                    if g % 2 == 1:
                        spot_id_gc += 1
                        eff_li = max(0, min(bisect.bisect_right(cumul, i_spot) - 1, hi_gc))
                        i_spot += 1
                    prev_gate = g
                if g % 2 == 0:
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
                spot_ids_gc.append(spot_id_gc)
                if max_points is not None and len(out_gc) >= max_points:
                    break
        return _aggregate_assigned_rows(
            out_gc, spot_ids_gc, aggregate_spots=aggregate_spots
        )

    out: list[tuple[float, ...]] = []
    spot_ids_tg: list[int] = []
    gates_tg: list[int] = []
    layer = 0
    timing_spot_id = 0
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

    with open_acquisition_csv(csv_path) as f:
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
            if not fit_position_row_ok(pcd, heal_partial_fit_axes=heal_partial_fit_axes):
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
                    timing_spot_id += 1
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
            g_cell = _gate_int_from_row(row, GATE_COUNTER_KEY)
            gates_tg.append(int(g_cell) if g_cell is not None else -1)
            spot_ids_tg.append(timing_spot_id)
            prev_t = t
            prev_mx, prev_my = mx, my
            if max_points is not None and len(out) >= max_points:
                break

    return _aggregate_assigned_rows(
        out,
        _gate_phase_spot_ids_for_rows(gates_tg)
        if aggregate_spots and any(g >= 0 for g in gates_tg)
        else spot_ids_tg,
        aggregate_spots=aggregate_spots,
    )
