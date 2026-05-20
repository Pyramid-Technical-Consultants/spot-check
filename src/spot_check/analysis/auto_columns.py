"""Columnar acquisition table for ``layer_mode='auto'`` (millions of CSV rows)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import numpy as np

from spot_check.analysis.csv_io import open_acquisition_csv
from spot_check.analysis.layers import _PlanImputeLookup
from spot_check.constants import (
    CHANNEL_SUM_KEY,
    FIT_AMPLITUDE_A_KEY,
    FIT_AMPLITUDE_B_KEY,
    SIGMA_A_KEY,
    SIGMA_B_KEY,
)

_FIT_POS_A = "Fit Mean Position A (mm)"
_FIT_POS_B = "Fit Mean Position B (mm)"


@dataclass(frozen=True)
class AutoFitColumns:
    """Filtered fit rows as contiguous float arrays (one row per index)."""

    t: np.ndarray
    mx: np.ndarray
    my: np.ndarray
    a: np.ndarray
    b: np.ndarray
    mx_p: np.ndarray
    my_p: np.ndarray
    weight: np.ndarray
    ch_n: np.ndarray
    fit_a: np.ndarray
    pcd: np.ndarray
    sa: np.ndarray
    sb: np.ndarray

    def __len__(self) -> int:
        return int(self.t.shape[0])


def _empty_columns() -> AutoFitColumns:
    z = np.zeros(0, dtype=np.float64)
    return AutoFitColumns(
        t=z,
        mx=z,
        my=z,
        a=z,
        b=z,
        mx_p=z,
        my_p=z,
        weight=z,
        ch_n=z,
        fit_a=z,
        pcd=np.zeros(0, dtype=np.int32),
        sa=z,
        sb=z,
    )


def _header_column_map(header_line: str) -> dict[str, int]:
    names = [h.strip() for h in header_line.split(",")]
    return {name: i for i, name in enumerate(names)}


def _cell_float(parts: list[str], col: int | None) -> float:
    if col is None or col >= len(parts):
        return float("nan")
    raw = parts[col]
    if not raw:
        return float("nan")
    return float(raw)


def _plan_xy_partial_arrays(
    a_mm: np.ndarray,
    b_mm: np.ndarray,
    *,
    a_is_x: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized partial-plan XY and partial codes (NaN = missing axis)."""
    n = a_mm.shape[0]
    mx_p = np.full(n, np.nan, dtype=np.float64)
    my_p = np.full(n, np.nan, dtype=np.float64)
    pcd = np.full(n, -1, dtype=np.int32)
    has_a = np.isfinite(a_mm)
    has_b = np.isfinite(b_mm)
    both = has_a & has_b
    only_b = has_b & ~has_a
    only_a = has_a & ~has_b
    if a_is_x:
        mx_p[both] = a_mm[both]
        my_p[both] = b_mm[both]
        mx_p[only_b] = b_mm[only_b]
        my_p[only_a] = a_mm[only_a]
    else:
        mx_p[both] = b_mm[both]
        my_p[both] = a_mm[both]
        mx_p[only_b] = b_mm[only_b]
        my_p[only_a] = a_mm[only_a]
    pcd[both] = 0
    pcd[only_b] = 1
    pcd[only_a] = 2
    return mx_p, my_p, pcd


def _position_row_keep_mask(
    pcd: np.ndarray,
    *,
    heal_partial_fit_axes: bool,
    include_deadtime_rows: bool,
) -> np.ndarray:
    """Row mask for auto column load: full fits; optional partial heal; plan-seq deadtime."""
    pc = np.asarray(pcd, dtype=np.int32)
    if heal_partial_fit_axes:
        pos_ok = pc >= 0
    else:
        pos_ok = pc == 0
    if include_deadtime_rows:
        return pos_ok | (pc < 0)
    return pos_ok


def position_fit_deadtime_mask(cols: AutoFitColumns) -> np.ndarray:
    """True when neither Fit Mean Position A nor B has a finite value (deadtime).

    Includes acquisition rows with no fit amplitude (timeline gaps in the CSV).
    """
    return ~(np.isfinite(cols.mx_p) | np.isfinite(cols.my_p))


def _grow_f64(buf: np.ndarray, new_cap: int) -> np.ndarray:
    out = np.empty(new_cap, dtype=np.float64)
    out[: buf.shape[0]] = buf
    return out


def _parse_acquisition_prealloc(
    line_source: TextIO | Iterator[str],
    col_map: dict[str, int],
    *,
    swm: str,
    max_points: int | None,
    include_deadtime_rows: bool = False,
) -> tuple[np.ndarray, ...]:
    col_time = col_map.get("time (s)", 0)
    col_fa = col_map[FIT_AMPLITUDE_A_KEY]
    col_a = col_map[_FIT_POS_A]
    col_b = col_map[_FIT_POS_B]
    col_ch = col_map.get(CHANNEL_SUM_KEY)
    col_sa = col_map.get(SIGMA_A_KEY)
    col_sb = col_map.get(SIGMA_B_KEY)
    col_wa = col_map.get(FIT_AMPLITUDE_A_KEY)
    col_wb = col_map.get(FIT_AMPLITUDE_B_KEY)

    n_cap = 4096
    if max_points is not None:
        n_cap = max_points

    t = np.empty(n_cap, dtype=np.float64)
    a_mm = np.empty(n_cap, dtype=np.float64)
    b_mm = np.empty(n_cap, dtype=np.float64)
    w_arr = np.empty(n_cap, dtype=np.float64)
    ch_arr = np.empty(n_cap, dtype=np.float64)
    fa_arr = np.empty(n_cap, dtype=np.float64)
    sa = np.empty(n_cap, dtype=np.float64)
    sb = np.empty(n_cap, dtype=np.float64)

    use_ch = swm == "channel_sum"
    use_wa = swm == "fit_amplitude_a"
    use_wb = swm == "fit_amplitude_b"

    i = 0
    for line in line_source:
        line = line.rstrip("\n\r")
        if not line:
            continue
        parts = line.split(",")
        has_fa = col_fa < len(parts) and bool(parts[col_fa].strip())
        if not has_fa:
            if not include_deadtime_rows:
                continue
            nan = float("nan")
            t[i] = _cell_float(parts, col_time)
            a_mm[i] = nan
            b_mm[i] = nan
            w_arr[i] = nan
            fa_arr[i] = nan
            ch_arr[i] = nan
            sa[i] = nan
            sb[i] = nan
            i += 1
            if i >= n_cap:
                if max_points is not None:
                    break
                n_cap = max(n_cap * 2, i + 4096)
                t = _grow_f64(t, n_cap)
                a_mm = _grow_f64(a_mm, n_cap)
                b_mm = _grow_f64(b_mm, n_cap)
                w_arr = _grow_f64(w_arr, n_cap)
                ch_arr = _grow_f64(ch_arr, n_cap)
                fa_arr = _grow_f64(fa_arr, n_cap)
                sa = _grow_f64(sa, n_cap)
                sb = _grow_f64(sb, n_cap)
            continue
        t[i] = _cell_float(parts, col_time)
        a_mm[i] = _cell_float(parts, col_a)
        b_mm[i] = _cell_float(parts, col_b)
        if use_ch and col_ch is not None:
            v = _cell_float(parts, col_ch)
            w_arr[i] = max(v, 1e-9) if v == v and v > 0 else 1.0
        elif use_wa and col_wa is not None:
            v = _cell_float(parts, col_wa)
            if v == v and v > 0:
                w_arr[i] = max(v, 1e-9)
            else:
                c = _cell_float(parts, col_ch)
                w_arr[i] = max(c, 1e-9) if c == c and c > 0 else 1.0
        elif use_wb and col_wb is not None:
            v = _cell_float(parts, col_wb)
            if v == v and v > 0:
                w_arr[i] = max(v, 1e-9)
            else:
                c = _cell_float(parts, col_ch)
                w_arr[i] = max(c, 1e-9) if c == c and c > 0 else 1.0
        else:
            w_arr[i] = 1.0
        v_fa = _cell_float(parts, col_fa)
        fa_arr[i] = max(v_fa, 1e-9) if v_fa == v_fa and v_fa > 0 else 1.0
        if col_ch is not None:
            c = _cell_float(parts, col_ch)
            ch_arr[i] = max(c, 1e-9) if c == c and c > 0 else 1.0
        else:
            ch_arr[i] = 1.0
        sa[i] = _cell_float(parts, col_sa)
        sb[i] = _cell_float(parts, col_sb)
        i += 1
        if i >= n_cap:
            if max_points is not None:
                break
            n_cap = max(n_cap * 2, i + 4096)
            t = _grow_f64(t, n_cap)
            a_mm = _grow_f64(a_mm, n_cap)
            b_mm = _grow_f64(b_mm, n_cap)
            w_arr = _grow_f64(w_arr, n_cap)
            ch_arr = _grow_f64(ch_arr, n_cap)
            fa_arr = _grow_f64(fa_arr, n_cap)
            sa = _grow_f64(sa, n_cap)
            sb = _grow_f64(sb, n_cap)

    if i == 0:
        z = np.zeros(0, dtype=np.float64)
        return z, z, z, z, z, z, z, z

    return (
        t[:i],
        a_mm[:i],
        b_mm[:i],
        w_arr[:i],
        ch_arr[:i],
        fa_arr[:i],
        sa[:i],
        sb[:i],
    )


def load_auto_fit_columns_from_csv(
    csv_path: Path,
    *,
    global_lk: _PlanImputeLookup,
    a_is_x: bool,
    spot_weight_mode: str,
    max_points: int | None = None,
    include_deadtime_rows: bool = False,
    heal_partial_fit_axes: bool = False,
) -> AutoFitColumns:
    """Read CSV into column arrays; skips rows without fit amplitude.

    Rows with fit amplitude but no position on either A/B axis are kept as deadtime
    (``position_fit_deadtime_mask``) only when ``include_deadtime_rows``; they are not imputed
    to a plan XY. Partial one-axis rows are dropped unless ``heal_partial_fit_axes``.
    """
    with open_acquisition_csv(csv_path) as f:
        header = f.readline()
        if not header:
            return _empty_columns()
        col_map = _header_column_map(header.rstrip("\n\r"))
        for key in (FIT_AMPLITUDE_A_KEY, _FIT_POS_A, _FIT_POS_B):
            if key not in col_map:
                raise ValueError(f"CSV missing required column {key!r} for auto mode")

        swm = spot_weight_mode.strip().lower().replace("-", "_")
        parsed = _parse_acquisition_prealloc(
            f,
            col_map,
            swm=swm,
            max_points=max_points,
            include_deadtime_rows=include_deadtime_rows,
        )
    t, a_mm, b_mm, w_arr, ch_arr, fa_arr, sa, sb = parsed
    if t.shape[0] == 0:
        return _empty_columns()

    mx_p, my_p, pcd = _plan_xy_partial_arrays(a_mm, b_mm, a_is_x=a_is_x)
    keep = _position_row_keep_mask(
        pcd,
        heal_partial_fit_axes=heal_partial_fit_axes,
        include_deadtime_rows=include_deadtime_rows,
    )
    if not np.all(keep):
        t = t[keep]
        a_mm = a_mm[keep]
        b_mm = b_mm[keep]
        w_arr = w_arr[keep]
        ch_arr = ch_arr[keep]
        fa_arr = fa_arr[keep]
        sa = sa[keep]
        sb = sb[keep]
        mx_p = mx_p[keep]
        my_p = my_p[keep]
        pcd = pcd[keep]
    if t.shape[0] == 0:
        return _empty_columns()
    has_pos = np.isfinite(mx_p) | np.isfinite(my_p)

    mx_i, my_i = global_lk.impute_xy_arrays(mx_p, my_p)
    nan = float("nan")
    mx_i = np.where(has_pos, mx_i, nan)
    my_i = np.where(has_pos, my_i, nan)
    if a_is_x:
        a_arr, b_arr = mx_i, my_i
    else:
        a_arr, b_arr = my_i, mx_i

    return AutoFitColumns(
        t=t,
        mx=mx_i,
        my=my_i,
        a=a_arr,
        b=b_arr,
        mx_p=mx_p,
        my_p=my_p,
        weight=w_arr,
        ch_n=ch_arr,
        fit_a=fa_arr,
        pcd=pcd,
        sa=sa,
        sb=sb,
    )
