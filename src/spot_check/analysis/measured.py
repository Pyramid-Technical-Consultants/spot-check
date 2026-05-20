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
    aggregate_spots: bool,
    spot_weight_mode: str,
    gate_counter_aggregate: bool = False,
) -> None:
    swm = normalize_measured_spot_weight_mode(spot_weight_mode)
    with open_acquisition_csv(csv_path) as f:
        pr = csv.DictReader(f)
        fn = pr.fieldnames
        if not fn:
            return
        if gate_counter_aggregate and GATE_COUNTER_KEY not in fn:
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


def _apply_gate_spot_aggregation(
    rows: list[tuple[float, ...]],
    gates: list[int],
    sigmas: list[tuple[float | None, float | None]],
) -> list[tuple[float, ...]]:
    """Group consecutive odd Gate Counter phases; even gates flush. Returns 7-tuples (last two =
    σ)."""
    if not (len(rows) == len(gates) == len(sigmas)):
        raise ValueError("rows, gates, sigmas length mismatch for spot aggregation")
    out: list[tuple[float, ...]] = []
    buf: list[_GateSpotBufRow] = []
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
    Layers then match gate_counter plan-slot indexing. With ``aggregate_spots=True``, each spot
    span becomes one **weighted-mean** row; when False, every on-spot CSV row is kept.

    When ``auto_infer_params`` is True (default), episode gap, XY jump, on-spot weight floor,
    min episode rows, and Viterbi advance penalty are derived from the CSV timing/weights and
    plan geometry (:func:`infer_auto_layer_params`); explicit ``auto_*`` / ``viterbi_*`` kwargs
    are ignored. Set ``auto_infer_params=False`` to use the keyword values (tests, scripts).

    In **gate_counter** mode with ``aggregate_spots=True``, requires ``Gate Counter`` on the CSV;
    each contiguous run of rows with the same **odd** gate value is one spot (even gate closes a
    spot). Aggregated rows use weighted means of position, layer, and σ.
    """
    mode = layer_mode.strip().lower().replace("-", "_")
    if mode not in ("time_gap", "plan_viterbi", "auto", "gate_counter"):
        raise ValueError(
            "layer_mode must be 'time_gap', 'plan_viterbi', 'auto', or 'gate_counter'"
        )

    swm = normalize_measured_spot_weight_mode(spot_weight_mode)

    _probe_csv_columns_for_measured_weights(
        csv_path,
        aggregate_spots=aggregate_spots,
        spot_weight_mode=swm,
        gate_counter_aggregate=aggregate_spots and mode == "gate_counter",
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
        if aggregate_spots:
            return _apply_gate_spot_aggregation(out, gates_acc, sig_acc)
        return out

    if mode == "auto":
        from spot_check.analysis.auto_columns import load_auto_fit_columns_from_csv
        from spot_check.analysis.episodes import (
            aggregate_spans_batch,
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
                min_rows_on_spot=min_rows,
                cluster_radius_mm2=plan_seq_radius_mm2,
            )
            aligned_groups = sequential_spans_from_plan_indices(plan_idx)
            plan_layers = delivery_layer_indices(n_plan_spots, spots_per_layer)
            spot_pi = plan_spot_index_per_span(plan_idx, aligned_groups)
            layers_idx_auto = plan_layers[spot_pi]

        cols_w = cols_with_delivery_weights(cols)
        aggs = aggregate_spans_batch(cols_w, aligned_groups)

        hi_auto = max_layer

        def _clamp_layer(efi: int) -> int:
            if efi < 0:
                return 0
            if efi > hi_auto:
                return hi_auto
            return efi

        if not aggregate_spots:
            out_rows: list[tuple[float, ...]] = []
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
                    if max_points is not None and len(out_rows) >= max_points:
                        return out_rows
            return out_rows

        ab_buf_auto = [(a.a, a.b) for a in aggs]
        w_buf_auto = [a.weight for a in aggs]
        ch_buf_auto = [a.ch_n for a in aggs]
        partial_codes_auto = [a.pcd for a in aggs]
        partial_plan_xy_auto = [(a.mx_pp, a.my_pp) for a in aggs]
        sig_acc_auto = [(a.sa, a.sb) for a in aggs]

        for i, (mx_pp2, my_pp2) in enumerate(partial_plan_xy_auto):
            if partial_codes_auto[i] == 0:
                continue
            efi = _clamp_layer(int(layers_idx_auto[i]))
            lk_ref = layer_lks[efi]
            if lk_ref is None:
                lk_ref = global_lk
            mx_f, my_f = _impute_plan_axis_fast(lk_ref, mx_pp2, my_pp2)
            ab_buf_auto[i] = _ab_from_plan_xy(mx_f, my_f, a_is_x=a_is_x)

        out_auto: list[tuple[float, ...]] = []
        for i, ((a, b), ell, wch, ch_n, pcd) in enumerate(
            zip(ab_buf_auto, layers_idx_auto, w_buf_auto, ch_buf_auto, partial_codes_auto)
        ):
            efi = _clamp_layer(int(ell))
            sa, sb = sig_acc_auto[i]
            out_auto.append(
                _measured_row_with_sigma(
                    a, b, float(efi), wch, pcd, sa, sb, channel_sum_na=ch_n
                )
            )
        return out_auto

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
        i_spot = 0
        eff_li = 0
        n_gc_raw = 0
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
                    if aggregate_spots:
                        if prev_gate is not None and prev_gate % 2 == 1 and spot_buf_gc:
                            out_gc.append(_finalize_spot_channel_weighted(spot_buf_gc))
                            spot_buf_gc.clear()
                        if g % 2 == 1:
                            eff_li = max(0, min(bisect.bisect_right(cumul, i_spot) - 1, hi_gc))
                            i_spot += 1
                    else:
                        if g % 2 == 1:
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
                if aggregate_spots:
                    spot_buf_gc.append((a_fin, b_fin, float(eff_li), w_ch, int(pcd), sa, sb, ch_n))
                    n_gc_raw += 1
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
