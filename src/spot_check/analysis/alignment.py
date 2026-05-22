"""Alignment."""

from __future__ import annotations

from typing import NamedTuple

from spot_check.analysis._imports import *  # noqa: F403
from spot_check.analysis.auto_columns import AutoFitColumns, position_fit_deadtime_mask
from spot_check.analysis.spatial import (
    _cKDTree,
    _kdtree_query_k1,
    _layer_nn_plan_targets,
    _layer_nn_rms_mm,
    _layer_xy_kdtrees_2d,
    build_layer_nn_plan_context,
    layer_nn_dist_and_expected_from_xy,
    layer_nn_plan_xy_distances_and_expected_xyz,
    meas_layer_indices_and_xy_from_rows,
)

_last_detector_align: DetectorRigidAlign2D | None = None

_DETECTOR_ALIGN_ICP_MAX_ITER = 25
_DETECTOR_ALIGN_ICP_TOL_MM = 0.05
_DETECTOR_ALIGN_COARSE_ANGLES_DEG: tuple[int, ...] = tuple(range(0, 360, 15))
_COARSE_FLAT_PRESCREEN_ANGLES_DEG: tuple[int, ...] = (0, 90, 180, 270)
_COARSE_FLAT_FINALIST_KEEP_MM = 1.5
_COARSE_FLAT_MAX_FINALISTS = 4
_COARSE_FLAT_RMS_STOP_MM = 0.05

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


def _plan_xy_signs(*, flip_plan_x: bool, flip_plan_y: bool) -> np.ndarray:
    return np.array(
        [-1.0 if flip_plan_x else 1.0, -1.0 if flip_plan_y else 1.0],
        dtype=np.float64,
    )


def _detector_align_orientation_variants() -> tuple[tuple[bool, bool, bool], ...]:
    """(swap_ab_axes, flip_plan_x, flip_plan_y) combinations for coarse search."""
    return tuple(
        (swap_ab, flip_x, flip_y)
        for swap_ab in (False, True)
        for flip_x in (False, True)
        for flip_y in (False, True)
    )


def _measured_ab_to_plan_xy_array(
    a: np.ndarray,
    b: np.ndarray,
    *,
    a_is_x: bool,
    swap_ab_axes: bool,
) -> np.ndarray:
    if swap_ab_axes:
        return np.column_stack([a, b])
    if a_is_x:
        return np.column_stack([a, b])
    return np.column_stack([b, a])


def _plan_xy_array_to_measured_ab(
    plan_xy: np.ndarray,
    *,
    a_is_x: bool,
    swap_ab_axes: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if swap_ab_axes or a_is_x:
        return plan_xy[:, 0].copy(), plan_xy[:, 1].copy()
    return plan_xy[:, 1].copy(), plan_xy[:, 0].copy()


def _transform_plan_xy_rigid(
    plan_xy: np.ndarray,
    r_mat: np.ndarray,
    tvec: np.ndarray,
    *,
    flip_plan_x: bool,
    flip_plan_y: bool,
) -> np.ndarray:
    """Apply optional plan-axis mirrors then ``R @ m + t`` on finite rows."""
    m = np.asarray(plan_xy, dtype=np.float64).reshape(-1, 2)
    signs = _plan_xy_signs(flip_plan_x=flip_plan_x, flip_plan_y=flip_plan_y)
    oriented = m * signs.reshape(1, 2)
    out = m.copy()
    r_arr = np.asarray(r_mat, dtype=np.float64).reshape(2, 2)
    t_arr = np.asarray(tvec, dtype=np.float64).reshape(2)
    finite = np.isfinite(oriented).all(axis=1)
    if np.any(finite):
        out[finite] = (r_arr @ oriented[finite].T).T + t_arr
    return out


def _measured_xy_for_align(
    row: tuple[float, ...],
    *,
    a_is_x: bool,
    swap_ab_axes: bool,
    flip_plan_x: bool = False,
    flip_plan_y: bool = False,
) -> tuple[float, float]:
    """Plan-frame XY used for alignment; optional A/B swap and axis mirrors."""
    a, b = float(row[0]), float(row[1])
    if swap_ab_axes:
        mx, my = a, b
    else:
        mx, my = measured_plan_xy_from_row(row, a_is_x=a_is_x)
    sx, sy = _plan_xy_signs(flip_plan_x=flip_plan_x, flip_plan_y=flip_plan_y)
    return float(mx * sx), float(my * sy)

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
        if not (math.isfinite(mx) and math.isfinite(my)):
            continue
        m_acc.append([mx, my])
        li_acc.append(li)
    if not m_acc:
        return np.zeros((0, 2), dtype=np.float64), np.zeros(0, dtype=np.intp)
    return np.asarray(m_acc, dtype=np.float64), np.asarray(li_acc, dtype=np.intp)


def _build_align_samples_delivery_order(
    measured_rows: list[tuple[float, ...]],
    planned_xyz: list[tuple[float, float, float]],
    *,
    a_is_x: bool,
    swap_ab_axes: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Measured XY and matching plan XY when row *i* is plan delivery spot *i*."""
    n = min(len(measured_rows), len(planned_xyz))
    m_acc: list[list[float]] = []
    p_acc: list[list[float]] = []
    for i in range(n):
        mx, my = _measured_xy_for_align(
            measured_rows[i], a_is_x=a_is_x, swap_ab_axes=swap_ab_axes
        )
        if not (math.isfinite(mx) and math.isfinite(my)):
            continue
        px, py = float(planned_xyz[i][0]), float(planned_xyz[i][1])
        m_acc.append([mx, my])
        p_acc.append([px, py])
    if not m_acc:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0, 2), dtype=np.float64)
    return np.asarray(m_acc, dtype=np.float64), np.asarray(p_acc, dtype=np.float64)

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

def _icp_rigid_fixed_plan_targets(
    meas_xy: np.ndarray,
    plan_xy: np.ndarray,
    *,
    r_init: np.ndarray | None = None,
    t_init: np.ndarray | None = None,
    max_iter: int = _DETECTOR_ALIGN_ICP_MAX_ITER,
    tol_mm: float = _DETECTOR_ALIGN_ICP_TOL_MM,
) -> tuple[np.ndarray, np.ndarray, float, float, int]:
    """ICP with known plan XY per row (delivery-order 1:1 assignment)."""
    m = np.asarray(meas_xy, dtype=np.float64).reshape(-1, 2)
    p = np.asarray(plan_xy, dtype=np.float64).reshape(-1, 2)
    if m.shape[0] != p.shape[0] or m.shape[0] == 0:
        raise ValueError("delivery-order alignment needs matching finite row counts")
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
        trans = (r_acc @ m.T).T + t_acc
        if int(trans.shape[0]) < 1:
            break
        diff0 = trans - p
        rms_nn = float(np.sqrt(np.mean(np.sum(diff0 * diff0, axis=1))))
        r_step, t_step, _, rms_res = _kabsch_rigid_2d(trans, p)
        t_acc = r_step @ t_acc + t_step
        r_acc = r_step @ r_acc
        if math.isfinite(prev) and abs(prev - rms_res) < float(tol_mm):
            break
        prev = rms_res
    return r_acc, t_acc, rms_nn, rms_res, n_iter


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
        finite = np.isfinite(trans).all(axis=1)
        if int(np.count_nonzero(finite)) < 1:
            break
        m_pairs, p_pairs = _layer_nn_plan_targets(
            trans[finite], layer_idx[finite], layer_xy, layer_trees=trees
        )
        if int(m_pairs.shape[0]) < 1:
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


def last_detector_align_info() -> DetectorRigidAlign2D | None:
    """Last rigid detector align (pre- or post-assignment), if any."""
    return _last_detector_align


def _set_last_detector_align(info: DetectorRigidAlign2D | None) -> None:
    global _last_detector_align
    _last_detector_align = info


def _flat_nn_plan_targets(
    meas_xy: np.ndarray,
    plan_xy: np.ndarray,
    *,
    plan_tree: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest plan spot in flattened 2D (all layers on one Z plane)."""
    m = np.asarray(meas_xy, dtype=np.float64).reshape(-1, 2)
    p_all = np.asarray(plan_xy, dtype=np.float64).reshape(-1, 2)
    if m.shape[0] == 0 or p_all.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0, 2), dtype=np.float64)
    finite = np.isfinite(m).all(axis=1)
    if not np.any(finite):
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0, 2), dtype=np.float64)
    q = m[finite]
    tree = plan_tree
    if tree is None:
        if _cKDTree is None:
            raise ValueError("scipy is required for flat-plan detector alignment")
        tree = _cKDTree(p_all)
    _, idx = _kdtree_query_k1(tree, q)
    idx_flat = np.asarray(idx, dtype=np.intp).ravel()
    return q, p_all[idx_flat]


def _flat_nn_rms_mm(
    meas_xy: np.ndarray,
    plan_xy: np.ndarray,
    r_mat: np.ndarray,
    tvec: np.ndarray,
    *,
    plan_tree: Any | None = None,
) -> float:
    trans = (np.asarray(r_mat, dtype=np.float64) @ meas_xy.T).T + np.asarray(
        tvec, dtype=np.float64
    ).reshape(2)
    m_pairs, p_pairs = _flat_nn_plan_targets(trans, plan_xy, plan_tree=plan_tree)
    if int(m_pairs.shape[0]) < 1:
        return float("inf")
    diff = m_pairs - p_pairs
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def _icp_rigid_flat_plan_nn(
    meas_xy: np.ndarray,
    plan_xy: np.ndarray,
    *,
    plan_tree: Any | None = None,
    r_init: np.ndarray | None = None,
    t_init: np.ndarray | None = None,
    max_iter: int = _DETECTOR_ALIGN_ICP_MAX_ITER,
    tol_mm: float = _DETECTOR_ALIGN_ICP_TOL_MM,
) -> tuple[np.ndarray, np.ndarray, float, float, int]:
    """ICP: global NN to flattened plan XY, then Kabsch (ignores nominal layer / Z)."""
    m = np.asarray(meas_xy, dtype=np.float64).reshape(-1, 2)
    p_all = np.asarray(plan_xy, dtype=np.float64).reshape(-1, 2)
    if m.shape[0] < 1 or p_all.shape[0] < 1:
        raise ValueError("flat-plan alignment needs measured and plan XY")
    tree = plan_tree
    if tree is None:
        if _cKDTree is None:
            raise ValueError("scipy is required for flat-plan detector alignment")
        tree = _cKDTree(p_all)
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
        trans = (r_acc @ m.T).T + t_acc
        finite = np.isfinite(trans).all(axis=1)
        if int(np.count_nonzero(finite)) < 1:
            break
        m_pairs, p_pairs = _flat_nn_plan_targets(
            trans[finite], p_all, plan_tree=tree
        )
        if int(m_pairs.shape[0]) < 1:
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


def _detector_align_multistart_icp_flat(
    meas_xy: np.ndarray,
    plan_xy: np.ndarray,
    *,
    plan_tree: Any | None = None,
    angle_seeds: tuple[int, ...] | None = None,
    rms_stop_mm: float = _COARSE_FLAT_RMS_STOP_MM,
) -> tuple[np.ndarray, np.ndarray, float, float, int, int]:
    """Coarse rotation seeds + flat-plan ICP."""
    if int(meas_xy.shape[0]) < 1:
        raise ValueError("detector alignment needs at least one finite measured row with plan")
    best_rms = float("inf")
    best: tuple[np.ndarray, np.ndarray, float, float, int] | None = None
    centroid = np.nanmean(meas_xy, axis=0)
    if not (math.isfinite(float(centroid[0])) and math.isfinite(float(centroid[1]))):
        raise ValueError("detector alignment needs finite measured XY")
    seeds = (
        tuple(int(d) for d in angle_seeds)
        if angle_seeds is not None
        else _detector_align_coarse_angles_deg(int(meas_xy.shape[0]))
    )
    for init_deg in seeds:
        r_seed = _rotation_matrix_2d(init_deg)
        t_seed = centroid - r_seed @ centroid
        r_acc, t_acc, rms_nn, rms_res, n_iter = _icp_rigid_flat_plan_nn(
            meas_xy,
            plan_xy,
            plan_tree=plan_tree,
            r_init=r_seed,
            t_init=t_seed,
        )
        if not math.isfinite(rms_res):
            continue
        if rms_res < best_rms:
            best_rms = rms_res
            best = (r_acc, t_acc, rms_nn, rms_res, n_iter)
        if best_rms <= float(rms_stop_mm):
            break
    if best is None:
        raise ValueError("detector ICP alignment failed for all rotation seeds")
    r_acc, t_acc, rms_nn, rms_res, n_iter = best
    return r_acc, t_acc, rms_nn, rms_res, n_iter, int(meas_xy.shape[0])


class _CoarseFlatMeasBlocks(NamedTuple):
    normal_xy: np.ndarray
    swap_xy: np.ndarray
    n_live: int


class _CoarseFlatFitCandidate(NamedTuple):
    rms_res: float
    swap_ab: bool
    flip_plan_x: bool
    flip_plan_y: bool
    r_mat: np.ndarray
    tvec: np.ndarray
    rms_nn: float
    n_iter: int


def _prepare_coarse_flat_meas_blocks(
    cols: AutoFitColumns,
    *,
    use_on_spot_only: bool,
) -> _CoarseFlatMeasBlocks:
    """Live on-spot plan-frame XY blocks for normal and swapped A/B conventions."""
    n = len(cols)
    if n == 0:
        z = np.zeros((0, 2), dtype=np.float64)
        return _CoarseFlatMeasBlocks(z, z, 0)
    live = np.ones(n, dtype=bool)
    if use_on_spot_only:
        live = ~position_fit_deadtime_mask(cols)
    normal = _auto_cols_plan_xy_matrix(cols, swap_ab_axes=False)
    swap = _auto_cols_plan_xy_matrix(cols, swap_ab_axes=True)
    finite = live & np.isfinite(normal).all(axis=1) & np.isfinite(swap).all(axis=1)
    return _CoarseFlatMeasBlocks(normal[finite], swap[finite], int(np.count_nonzero(finite)))


def _oriented_meas_xy(
    blocks: _CoarseFlatMeasBlocks,
    *,
    swap_ab_axes: bool,
    flip_plan_x: bool,
    flip_plan_y: bool,
) -> np.ndarray:
    base = blocks.swap_xy if swap_ab_axes else blocks.normal_xy
    return base * _plan_xy_signs(flip_plan_x=flip_plan_x, flip_plan_y=flip_plan_y).reshape(1, 2)


def _coarse_flat_holdout_rms(
    blocks: _CoarseFlatMeasBlocks,
    cand: _CoarseFlatFitCandidate,
    plan_xy: np.ndarray,
    plan_tree: Any | None,
) -> float:
    """NN RMS on all live rows after applying a fitted candidate transform."""
    meas_full = _oriented_meas_xy(
        blocks,
        swap_ab_axes=cand.swap_ab,
        flip_plan_x=cand.flip_plan_x,
        flip_plan_y=cand.flip_plan_y,
    )
    return _flat_nn_rms_mm(meas_full, plan_xy, cand.r_mat, cand.tvec, plan_tree=plan_tree)


def _coarse_flat_finalist_keys(
    candidates: list[_CoarseFlatFitCandidate],
    *,
    blocks: _CoarseFlatMeasBlocks,
    plan_xy: np.ndarray,
    plan_tree: Any | None,
) -> set[tuple[bool, bool, bool]]:
    """Orientations to refine with the full rotation seed set."""
    if not candidates:
        return set()
    def _holdout(cand: _CoarseFlatFitCandidate) -> float:
        return _coarse_flat_holdout_rms(blocks, cand, plan_xy, plan_tree)

    ranked = sorted(candidates, key=_holdout)
    best_ps = _coarse_flat_holdout_rms(blocks, ranked[0], plan_xy, plan_tree)
    keys: set[tuple[bool, bool, bool]] = set()
    for cand in ranked:
        key = (cand.swap_ab, cand.flip_plan_x, cand.flip_plan_y)
        if key in keys:
            continue
        hold = _coarse_flat_holdout_rms(blocks, cand, plan_xy, plan_tree)
        if keys and hold > best_ps + float(_COARSE_FLAT_FINALIST_KEEP_MM):
            break
        keys.add(key)
        if len(keys) >= int(_COARSE_FLAT_MAX_FINALISTS):
            break
    for cand in ranked[:2]:
        keys.add((cand.swap_ab, cand.flip_plan_x, cand.flip_plan_y))
    return keys


def _coarse_flat_fit_candidate(
    meas_fit: np.ndarray,
    plan_xy: np.ndarray,
    plan_tree: Any | None,
    *,
    swap_ab: bool,
    flip_x: bool,
    flip_y: bool,
    angle_seeds: tuple[int, ...],
) -> _CoarseFlatFitCandidate | None:
    try:
        r_mat, tvec, rms_nn, rms_res, n_iter, _ = _detector_align_multistart_icp_flat(
            meas_fit,
            plan_xy,
            plan_tree=plan_tree,
            angle_seeds=angle_seeds,
        )
    except ValueError:
        return None
    return _CoarseFlatFitCandidate(
        rms_res=float(rms_res),
        swap_ab=bool(swap_ab),
        flip_plan_x=bool(flip_x),
        flip_plan_y=bool(flip_y),
        r_mat=r_mat,
        tvec=tvec,
        rms_nn=float(rms_nn),
        n_iter=int(n_iter),
    )


def _auto_cols_plan_xy_matrix(
    cols: AutoFitColumns,
    *,
    swap_ab_axes: bool,
) -> np.ndarray:
    """Plan-frame XY from auto columns (same convention for coarse fit and apply)."""
    if swap_ab_axes:
        return np.column_stack(
            [
                np.asarray(cols.a, dtype=np.float64),
                np.asarray(cols.b, dtype=np.float64),
            ]
        )
    return np.column_stack(
        [
            np.asarray(cols.mx, dtype=np.float64),
            np.asarray(cols.my, dtype=np.float64),
        ]
    )


def _write_auto_cols_plan_xy(
    plan_out: np.ndarray,
    *,
    a_is_x: bool,
    swap_ab_axes: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Write transformed plan XY back to mx/my and refreshed Fit A/B arrays."""
    if swap_ab_axes:
        a, b = _plan_xy_array_to_measured_ab(plan_out, a_is_x=a_is_x, swap_ab_axes=True)
        if a_is_x:
            mx, my = a.copy(), b.copy()
        else:
            mx, my = b.copy(), a.copy()
        return mx, my, a, b
    mx = plan_out[:, 0].copy()
    my = plan_out[:, 1].copy()
    if a_is_x:
        return mx, my, mx.copy(), my.copy()
    return mx, my, my.copy(), mx.copy()


def _build_align_samples_from_auto_cols(
    cols: AutoFitColumns,
    *,
    use_on_spot_only: bool,
    swap_ab_axes: bool,
    flip_plan_x: bool = False,
    flip_plan_y: bool = False,
) -> np.ndarray:
    """Plan-frame XY from columnar auto rows (position-fit on-spot by default)."""
    blocks = _prepare_coarse_flat_meas_blocks(cols, use_on_spot_only=use_on_spot_only)
    if blocks.n_live < 1:
        return np.zeros((0, 2), dtype=np.float64)
    return _oriented_meas_xy(
        blocks,
        swap_ab_axes=swap_ab_axes,
        flip_plan_x=flip_plan_x,
        flip_plan_y=flip_plan_y,
    )


def _apply_rigid_to_auto_fit_columns(
    cols: AutoFitColumns,
    r_mat: np.ndarray,
    tvec: np.ndarray,
    *,
    a_is_x: bool,
    swap_ab_axes: bool = False,
    flip_plan_x: bool = False,
    flip_plan_y: bool = False,
) -> AutoFitColumns:
    """Apply orientation + rigid map to auto-fit columns; refresh stored A/B and plan XY."""
    from dataclasses import replace

    n = len(cols)
    if n == 0:
        return cols
    plan_xy = _auto_cols_plan_xy_matrix(cols, swap_ab_axes=swap_ab_axes)
    plan_out = _transform_plan_xy_rigid(
        plan_xy,
        r_mat,
        tvec,
        flip_plan_x=flip_plan_x,
        flip_plan_y=flip_plan_y,
    )
    mx, my, a, b = _write_auto_cols_plan_xy(
        plan_out,
        a_is_x=a_is_x,
        swap_ab_axes=swap_ab_axes,
    )
    mp = np.column_stack(
        [
            np.array(cols.mx_p, dtype=np.float64),
            np.array(cols.my_p, dtype=np.float64),
        ]
    )
    mp_out = _transform_plan_xy_rigid(
        mp,
        r_mat,
        tvec,
        flip_plan_x=flip_plan_x,
        flip_plan_y=flip_plan_y,
    )
    return replace(cols, mx=mx, my=my, mx_p=mp_out[:, 0], my_p=mp_out[:, 1], a=a, b=b)


def fit_coarse_flat_align_from_auto_columns(
    cols: AutoFitColumns,
    planned_xyz: list[tuple[float, float, float]],
    *,
    max_fit_samples: int = DETECTOR_ALIGN_MAX_FIT_SAMPLES,
    use_on_spot_only: bool = True,
) -> DetectorRigidAlign2D:
    """Fit rigid flat 2D map (flattened plan + on-spot measured XY); does not mutate columns."""
    if not planned_xyz or len(cols) == 0:
        raise ValueError("plan and auto columns are required for coarse flat alignment")
    plan_xy = np.asarray(
        [(float(px), float(py)) for px, py, _ in planned_xyz],
        dtype=np.float64,
    )
    if plan_xy.shape[0] < 1:
        raise ValueError("plan has no scan spots for coarse flat alignment")
    plan_tree = _cKDTree(plan_xy) if _cKDTree is not None else None
    blocks = _prepare_coarse_flat_meas_blocks(cols, use_on_spot_only=use_on_spot_only)
    if blocks.n_live < 1:
        raise ValueError(
            "coarse flat alignment needs at least one on-spot row with finite "
            "fit XY and a matching plan spot in 2D"
        )
    fit_idx = _subsample_align_indices(blocks.n_live, int(max_fit_samples))
    n_all = blocks.n_live
    n_fit = int(fit_idx.size)
    full_seeds = _detector_align_coarse_angles_deg(n_fit)
    prescreen_only = set(full_seeds).issubset(set(_COARSE_FLAT_PRESCREEN_ANGLES_DEG))

    prescreened: list[_CoarseFlatFitCandidate] = []
    for swap_ab, flip_x, flip_y in _detector_align_orientation_variants():
        meas_fit = _oriented_meas_xy(
            blocks,
            swap_ab_axes=swap_ab,
            flip_plan_x=flip_x,
            flip_plan_y=flip_y,
        )[fit_idx]
        cand = _coarse_flat_fit_candidate(
            meas_fit,
            plan_xy,
            plan_tree,
            swap_ab=swap_ab,
            flip_x=flip_x,
            flip_y=flip_y,
            angle_seeds=full_seeds if prescreen_only else _COARSE_FLAT_PRESCREEN_ANGLES_DEG,
        )
        if cand is not None:
            prescreened.append(cand)

    if not prescreened:
        raise ValueError(
            "coarse flat alignment needs at least one on-spot row with finite "
            "fit XY and a matching plan spot in 2D"
        )

    def _holdout(cand: _CoarseFlatFitCandidate) -> float:
        return _coarse_flat_holdout_rms(blocks, cand, plan_xy, plan_tree)

    if prescreen_only:
        best = min(prescreened, key=_holdout)
    else:
        finalist_keys = _coarse_flat_finalist_keys(
            prescreened,
            blocks=blocks,
            plan_xy=plan_xy,
            plan_tree=plan_tree,
        )
        by_key = {
            (c.swap_ab, c.flip_plan_x, c.flip_plan_y): c for c in prescreened
        }
        for swap_ab, flip_x, flip_y in _detector_align_orientation_variants():
            if (swap_ab, flip_x, flip_y) not in finalist_keys:
                continue
            meas_fit = _oriented_meas_xy(
                blocks,
                swap_ab_axes=swap_ab,
                flip_plan_x=flip_x,
                flip_plan_y=flip_y,
            )[fit_idx]
            cand = _coarse_flat_fit_candidate(
                meas_fit,
                plan_xy,
                plan_tree,
                swap_ab=swap_ab,
                flip_x=flip_x,
                flip_y=flip_y,
                angle_seeds=full_seeds,
            )
            if cand is None:
                continue
            key = (swap_ab, flip_x, flip_y)
            prev = by_key.get(key)
            if prev is None or cand.rms_res < prev.rms_res:
                by_key[key] = cand
        best = min(by_key.values(), key=_holdout)

    holdout_rms = _coarse_flat_holdout_rms(blocks, best, plan_xy, plan_tree)

    r_mat = best.r_mat
    tvec = best.tvec
    rms_nn = best.rms_nn
    swap_ab = best.swap_ab
    flip_x = best.flip_plan_x
    flip_y = best.flip_plan_y
    n_iter = best.n_iter
    theta = float(math.degrees(math.atan2(float(r_mat[1, 0]), float(r_mat[0, 0]))))
    return DetectorRigidAlign2D(
        theta_deg=theta,
        tx_mm=float(tvec[0]),
        ty_mm=float(tvec[1]),
        rms_nn_mm=float(rms_nn),
        rms_residual_mm=float(holdout_rms),
        n_pairs=n_all,
        ab_axes_swapped=bool(swap_ab),
        flip_plan_x=bool(flip_x),
        flip_plan_y=bool(flip_y),
        icp_iterations=int(n_iter),
        n_pairs_fit=int(n_fit),
        from_coarse_phase=True,
    )


def apply_coarse_flat_transform_to_auto_fit_columns(
    cols: AutoFitColumns,
    info: DetectorRigidAlign2D,
    *,
    a_is_x: bool,
) -> AutoFitColumns:
    """Apply coarse flat rigid map (orientation mirrors + rotation) to auto-fit columns."""
    r_mat = _rotation_matrix_2d(info.theta_deg)
    tvec = np.asarray([info.tx_mm, info.ty_mm], dtype=np.float64)
    return _apply_rigid_to_auto_fit_columns(
        cols,
        r_mat,
        tvec,
        a_is_x=a_is_x,
        swap_ab_axes=bool(info.ab_axes_swapped),
        flip_plan_x=bool(info.flip_plan_x),
        flip_plan_y=bool(info.flip_plan_y),
    )


def align_auto_fit_columns_to_plan_xy(
    cols: AutoFitColumns,
    planned_xyz: list[tuple[float, float, float]],
    *,
    a_is_x: bool = False,
    max_fit_samples: int = DETECTOR_ALIGN_MAX_FIT_SAMPLES,
    use_on_spot_only: bool = True,
) -> tuple[AutoFitColumns, DetectorRigidAlign2D]:
    """Fit and apply coarse flat rigid 2D align on flattened plan + on-spot measured XY."""
    info = fit_coarse_flat_align_from_auto_columns(
        cols,
        planned_xyz,
        max_fit_samples=max_fit_samples,
        use_on_spot_only=use_on_spot_only,
    )
    out_cols = apply_coarse_flat_transform_to_auto_fit_columns(cols, info, a_is_x=a_is_x)
    _set_last_detector_align(info)
    return out_cols, info


def _detector_align_multistart_delivery(
    meas_xy: np.ndarray,
    plan_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float, int, int]:
    """Rotation search when measured row *i* maps to plan spot *i*."""
    if int(meas_xy.shape[0]) < 1:
        raise ValueError("detector alignment needs at least one finite measured/plan pair")
    best_rms = float("inf")
    best: tuple[np.ndarray, np.ndarray, float, float, int] | None = None
    centroid = np.nanmean(meas_xy, axis=0)
    if not (math.isfinite(float(centroid[0])) and math.isfinite(float(centroid[1]))):
        raise ValueError("detector alignment needs finite measured XY")
    angle_seeds = _detector_align_coarse_angles_deg(int(meas_xy.shape[0]))
    for init_deg in angle_seeds:
        r_seed = _rotation_matrix_2d(init_deg)
        t_seed = centroid - r_seed @ centroid
        r_acc, t_acc, rms_nn, rms_res, n_iter = _icp_rigid_fixed_plan_targets(
            meas_xy,
            plan_xy,
            r_init=r_seed,
            t_init=t_seed,
        )
        if not math.isfinite(rms_res):
            continue
        holdout = float(
            np.sqrt(
                np.mean(
                    np.sum(
                        ((r_acc @ meas_xy.T).T + t_acc - plan_xy) ** 2,
                        axis=1,
                    )
                )
            )
        )
        if holdout < best_rms:
            best_rms = holdout
            best = (r_acc, t_acc, rms_nn, holdout, n_iter)
    if best is None:
        raise ValueError("detector ICP alignment failed for all rotation seeds")
    r_acc, t_acc, rms_nn, rms_res, n_iter = best
    return r_acc, t_acc, rms_nn, rms_res, n_iter, int(meas_xy.shape[0])


def _detector_align_multistart_icp(
    meas_xy: np.ndarray,
    layer_idx: np.ndarray,
    layer_xy: list[np.ndarray],
    *,
    layer_trees: list[Any | None] | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float, int, int]:
    """Try coarse rotation seeds; return best cumulative ``R``, ``t``, RMS, ICP iters."""
    if int(meas_xy.shape[0]) < 1:
        raise ValueError("detector alignment needs at least one finite measured row with plan")
    trees = _layer_xy_kdtrees_2d(layer_xy) if layer_trees is None else layer_trees
    best_rms = float("inf")
    best: tuple[np.ndarray, np.ndarray, float, float, int] | None = None
    centroid = np.nanmean(meas_xy, axis=0)
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
        if not math.isfinite(rms_res):
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
    flip_plan_x: bool = False,
    flip_plan_y: bool = False,
) -> list[tuple[float, ...]]:
    """Apply orientation mirrors + ``r_mat @ m + tvec`` (vectorized XY; tuple tails preserved)."""
    n = len(measured_rows)
    if n == 0:
        return []
    a_col = np.fromiter((float(r[0]) for r in measured_rows), dtype=np.float64, count=n)
    b_col = np.fromiter((float(r[1]) for r in measured_rows), dtype=np.float64, count=n)
    plan_xy = _measured_ab_to_plan_xy_array(
        a_col,
        b_col,
        a_is_x=a_is_x,
        swap_ab_axes=swap_ab_axes,
    )
    plan_out = _transform_plan_xy_rigid(
        plan_xy,
        r_mat,
        tvec,
        flip_plan_x=flip_plan_x,
        flip_plan_y=flip_plan_y,
    )
    a_out, b_out = _plan_xy_array_to_measured_ab(
        plan_out,
        a_is_x=a_is_x,
        swap_ab_axes=swap_ab_axes,
    )
    out: list[tuple[float, ...]] = []
    for i, row in enumerate(measured_rows):
        out.append((float(a_out[i]), float(b_out[i]), *row[2:]))
    return out


def apply_detector_rigid2d_xy_to_measured_rows(
    measured_rows: list[tuple[float, ...]],
    info: DetectorRigidAlign2D,
    *,
    a_is_x: bool,
) -> list[tuple[float, ...]]:
    """Apply rigid map from :class:`~spot_check.models.DetectorRigidAlign2D` to row A/B XY."""
    r_mat = _rotation_matrix_2d(info.theta_deg)
    tvec = np.asarray([info.tx_mm, info.ty_mm], dtype=np.float64)
    return _apply_rigid_xy_to_measured_rows(
        measured_rows,
        r_mat,
        tvec,
        a_is_x=a_is_x,
        swap_ab_axes=bool(info.ab_axes_swapped),
        flip_plan_x=bool(info.flip_plan_x),
        flip_plan_y=bool(info.flip_plan_y),
    )


def _apply_similarity_xy_to_measured_rows(
    measured_rows: list[tuple[float, ...]],
    *,
    theta_deg: float,
    sx: float,
    sy: float,
    tx_mm: float,
    ty_mm: float,
    a_is_x: bool,
) -> list[tuple[float, ...]]:
    """Apply ``R(θ) @ diag(sx,sy) @ m + t`` to every row (vectorized XY; tuple tails preserved)."""
    r_mat = _rotation_matrix_2d(theta_deg)
    return _apply_scaled_rotation_xy_to_measured_rows(
        measured_rows,
        r_mat=r_mat,
        sx=sx,
        sy=sy,
        tx_mm=tx_mm,
        ty_mm=ty_mm,
        a_is_x=a_is_x,
    )


def _apply_scaled_rotation_xy_to_measured_rows(
    measured_rows: list[tuple[float, ...]],
    *,
    r_mat: np.ndarray,
    sx: float,
    sy: float,
    tx_mm: float,
    ty_mm: float,
    a_is_x: bool,
) -> list[tuple[float, ...]]:
    n = len(measured_rows)
    if n == 0:
        return []
    a_col = np.fromiter((float(r[0]) for r in measured_rows), dtype=np.float64, count=n)
    b_col = np.fromiter((float(r[1]) for r in measured_rows), dtype=np.float64, count=n)
    m = np.column_stack([a_col, b_col]) if a_is_x else np.column_stack([b_col, a_col])
    r_arr = np.asarray(r_mat, dtype=np.float64)
    t_arr = np.array([float(tx_mm), float(ty_mm)], dtype=np.float64)
    sx_f, sy_f = float(sx), float(sy)
    finite = np.isfinite(m).all(axis=1)
    scaled = m.copy()
    scaled[:, 0] *= sx_f
    scaled[:, 1] *= sy_f
    w = scaled.copy()
    if np.any(finite):
        w[finite] = (r_arr @ scaled[finite].T).T + t_arr
    out: list[tuple[float, ...]] = []
    if a_is_x:
        for i, row in enumerate(measured_rows):
            out.append((float(w[i, 0]), float(w[i, 1]), *row[2:]))
    else:
        for i, row in enumerate(measured_rows):
            out.append((float(w[i, 1]), float(w[i, 0]), *row[2:]))
    return out


def format_detector_align_caption(info: DetectorRigidAlign2D) -> str:
    stage = "coarse flat" if info.from_coarse_phase else "detector"
    orient_parts: list[str] = []
    if info.ab_axes_swapped:
        orient_parts.append("Fit A↔B swapped")
    if info.flip_plan_x:
        orient_parts.append("mirror X")
    if info.flip_plan_y:
        orient_parts.append("mirror Y")
    orient_note = f"; {', '.join(orient_parts)}" if orient_parts else ""
    fit_note = ""
    if info.n_pairs_fit > 0 and info.n_pairs_fit < info.n_pairs:
        fit_note = f", fit n={info.n_pairs_fit}/{info.n_pairs}"
    return (
        f"Detector align ({stage}): θ={info.theta_deg:.5g}° CCW, "
        f"t=({info.tx_mm:.5g}, {info.ty_mm:.5g}) mm, "
        f"RMS after={info.rms_residual_mm:.5g} mm (NN RMS before={info.rms_nn_mm:.5g} mm, "
        f"n={info.n_pairs}{fit_note}, ICP={info.icp_iterations}{orient_note})."
    )


_FINE_ALIGN_SCALE_CLIP = (0.9, 1.1)
_FINE_ALIGN_GN_MAX_ITER = 50
_FINE_ALIGN_GN_RMS_TOL_MM = 1.0e-2
_FINE_ALIGN_ICP_MAX_ITER = 5
_FINE_ALIGN_ICP_TOL_MM = 0.005


def _clamp_fine_scale(val: float) -> float:
    lo, hi = _FINE_ALIGN_SCALE_CLIP
    return float(np.clip(float(val), lo, hi))


def _rms_weighted_xy_residual(diff_xy: np.ndarray, weights: np.ndarray) -> float:
    ww = np.asarray(weights, dtype=np.float64).reshape(-1)
    d_xy = np.asarray(diff_xy, dtype=np.float64)
    if int(d_xy.shape[0]) < 1 or int(d_xy.shape[0]) != int(ww.shape[0]):
        return float("nan")
    den = float(np.sum(ww))
    if den <= 0.0:
        return float("nan")
    return float(math.sqrt(max(0.0, np.sum(ww * np.sum(d_xy * d_xy, axis=1)) / den)))


def _fine_align_row_weights(rows: list[tuple[float, ...]]) -> np.ndarray:
    """Positive weights per row from spot weight slot (index 3), default 1.0."""
    n = len(rows)
    ww = np.ones(n, dtype=np.float64)
    for i, row in enumerate(rows):
        if len(row) <= 3:
            continue
        wv = float(row[3])
        if math.isfinite(wv) and wv > 0.0:
            ww[i] = wv
    return ww


def _fine_align_qa_rms_from_dist(
    dist: np.ndarray,
    row_weights: np.ndarray,
) -> float:
    """Weighted RMS of per-row layer-NN distance (same metric as plan QA coloring)."""
    dist = np.asarray(dist, dtype=np.float64).reshape(-1)
    w = np.asarray(row_weights, dtype=np.float64).reshape(-1)
    if int(dist.shape[0]) != int(w.shape[0]):
        raise ValueError("row_weights length mismatch for fine align QA")
    finite = np.isfinite(dist) & (w > 0.0)
    if not np.any(finite):
        return float("nan")
    ww = w[finite]
    dd = dist[finite]
    return float(math.sqrt(max(0.0, float(np.sum(ww * dd * dd) / float(np.sum(ww))))))


def _fine_align_qa_rms_mm(
    planned_xyz: list[tuple[float, float, float]],
    measured_rows: list[tuple[float, ...]],
    *,
    a_is_x: bool,
    row_weights: np.ndarray,
) -> float:
    """Weighted RMS of per-row layer-NN XY distance (same metric as plan QA coloring)."""
    if not planned_xyz or not measured_rows:
        return float("nan")
    dist, _ = layer_nn_plan_xy_distances_and_expected_xyz(
        planned_xyz, measured_rows, a_is_x=a_is_x
    )
    return _fine_align_qa_rms_from_dist(dist, row_weights)


def _fine_align_pairs_from_nn(
    dist: np.ndarray,
    exp_xyz: np.ndarray,
    meas_xy: np.ndarray,
    row_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return (meas_xy, plan_xy, weights) for rows with valid layer-NN matches."""
    finite = (
        np.isfinite(dist)
        & (row_weights > 0.0)
        & np.isfinite(meas_xy).all(axis=1)
        & np.all(np.isfinite(exp_xyz), axis=1)
    )
    if not np.any(finite):
        return None
    idx = np.flatnonzero(finite)
    return (
        meas_xy[idx],
        exp_xyz[idx, 0:2],
        row_weights[idx],
    )


def _fine_align_apply_similarity_inplace(
    meas_xy: np.ndarray,
    *,
    theta_deg: float,
    sx: float,
    sy: float,
    tx_mm: float,
    ty_mm: float,
) -> None:
    """Apply ``R(θ) @ diag(sx,sy) @ m + t`` to finite plan-frame XY rows in place."""
    r_mat = _rotation_matrix_2d(theta_deg)
    sx_sy = np.array([float(sx), float(sy)], dtype=np.float64)
    t_arr = np.array([float(tx_mm), float(ty_mm)], dtype=np.float64)
    scaled = meas_xy * sx_sy
    finite = np.isfinite(scaled).all(axis=1)
    if np.any(finite):
        meas_xy[finite] = scaled[finite] @ r_mat.T + t_arr


def _fine_align_rows_from_meas_xy(
    meas_xy: np.ndarray,
    template_rows: list[tuple[float, ...]],
    *,
    a_is_x: bool,
) -> list[tuple[float, ...]]:
    """Rebuild measured row tuples from plan-frame XY and preserved row tails."""
    if a_is_x:
        return [
            (float(meas_xy[i, 0]), float(meas_xy[i, 1]), *row[2:])
            for i, row in enumerate(template_rows)
        ]
    return [
        (float(meas_xy[i, 1]), float(meas_xy[i, 0]), *row[2:])
        for i, row in enumerate(template_rows)
    ]


def _fine_align_predict_xy(
    meas_xy: np.ndarray,
    *,
    theta_deg: float,
    sx: float,
    sy: float,
    tx_mm: float,
    ty_mm: float,
) -> np.ndarray:
    r_mat = _rotation_matrix_2d(theta_deg)
    scaled = meas_xy * np.array([float(sx), float(sy)], dtype=np.float64).reshape(1, 2)
    return (r_mat @ scaled.T).T + np.array([float(tx_mm), float(ty_mm)], dtype=np.float64)


def _fine_align_gauss_newton(
    meas_xy: np.ndarray,
    plan_xy: np.ndarray,
    weights: np.ndarray,
    *,
    allow_xy: bool,
    allow_rotation: bool,
    allow_scale: bool,
) -> tuple[float, float, float, float, float, float, float]:
    """Returns (theta_deg, tx, ty, sx, sy, rms_before, rms_after)."""
    n = int(meas_xy.shape[0])
    if n < 1:
        raise ValueError("fine align needs at least one point pair")
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if int(w.shape[0]) != n:
        raise ValueError("weights length mismatch")
    sw = float(np.sum(w))
    if sw <= 0.0:
        w = np.ones(n, dtype=np.float64)
        sw = float(n)

    eff_rot = bool(allow_rotation and n >= 2)
    eff_scale = bool(allow_scale and n >= 2)
    theta_deg = 0.0
    sx = sy = 1.0

    mx_c = np.sum(w.reshape(-1, 1) * meas_xy, axis=0) / sw
    pc_c = np.sum(w.reshape(-1, 1) * plan_xy, axis=0) / sw
    if allow_xy:
        tx_mm, ty_mm = float(pc_c[0] - mx_c[0]), float(pc_c[1] - mx_c[1])
    else:
        tx_mm = ty_mm = 0.0

    diff_before = plan_xy - meas_xy
    rms_before = _rms_weighted_xy_residual(diff_before, w)

    n_cols = int(eff_rot) + int(allow_xy) * 2 + int(eff_scale) * 2

    prev_rms = float("inf")
    for _ in range(1, int(_FINE_ALIGN_GN_MAX_ITER) + 1):
        pred = _fine_align_predict_xy(
            meas_xy, theta_deg=theta_deg, sx=sx, sy=sy, tx_mm=tx_mm, ty_mm=ty_mm
        )
        res = plan_xy - pred

        if n_cols < 1:
            break

        jac_rows: list[list[float]] = []
        sqrt_w_flat: list[float] = []

        rad = math.radians(float(theta_deg))
        cos_t, sin_t = math.cos(rad), math.sin(rad)

        for i in range(n):
            mx_i, my_i = float(meas_xy[i, 0]), float(meas_xy[i, 1])
            ux = sx * mx_i
            uy = sy * my_i
            v_th = (math.pi / 180.0) * np.array(
                [-sin_t * ux - cos_t * uy, cos_t * ux - sin_t * uy],
                dtype=np.float64,
            )
            jr = np.zeros((2, n_cols), dtype=np.float64)
            cidx = 0
            if eff_rot:
                jr[:, cidx] = v_th
                cidx += 1
            if allow_xy:
                jr[:, cidx : cidx + 2] = np.eye(2, dtype=np.float64)
                cidx += 2
            if eff_scale:
                jr[:, cidx] = np.array([cos_t * mx_i, sin_t * mx_i], dtype=np.float64)
                jr[:, cidx + 1] = np.array([-sin_t * my_i, cos_t * my_i], dtype=np.float64)
                cidx += 2

            swq = math.sqrt(max(0.0, float(w[i])))
            jac_rows.extend([list(swq * jr[0, :]), list(swq * jr[1, :])])
            sqrt_w_flat.extend([swq * float(res[i, 0]), swq * float(res[i, 1])])

        if not jac_rows:
            break

        j_mat = np.asarray(jac_rows, dtype=np.float64)
        rhs = np.asarray(sqrt_w_flat, dtype=np.float64)
        delta, _, rank, _ignored = np.linalg.lstsq(j_mat, rhs, rcond=None)
        if rank < 1 or not np.all(np.isfinite(delta)):
            break

        col = 0
        if eff_rot:
            theta_deg += float(delta[col])
            col += 1
        if allow_xy:
            tx_mm += float(delta[col])
            ty_mm += float(delta[col + 1])
            col += 2
        if eff_scale:
            sx += float(delta[col])
            sy += float(delta[col + 1])

        sx = _clamp_fine_scale(sx)
        sy = _clamp_fine_scale(sy)

        pred2 = _fine_align_predict_xy(
            meas_xy, theta_deg=theta_deg, sx=sx, sy=sy, tx_mm=tx_mm, ty_mm=ty_mm
        )
        rms_now = _rms_weighted_xy_residual(plan_xy - pred2, w)
        if math.isfinite(prev_rms) and abs(prev_rms - rms_now) < float(_FINE_ALIGN_GN_RMS_TOL_MM):
            prev_rms = rms_now
            break
        prev_rms = rms_now

    if eff_rot:
        theta_use = theta_deg
    else:
        theta_use = 0.0
    sx_use = sx if eff_scale else 1.0
    sy_use = sy if eff_scale else 1.0

    pred_fin = _fine_align_predict_xy(
        meas_xy, theta_deg=theta_use, sx=sx_use, sy=sy_use, tx_mm=tx_mm, ty_mm=ty_mm
    )
    rms_after = _rms_weighted_xy_residual(plan_xy - pred_fin, w)

    theta_deg_fin = theta_use if eff_rot else 0.0
    sx_fin, sy_fin = (sx_use, sy_use) if eff_scale else (1.0, 1.0)

    rms_before = float(rms_before) if math.isfinite(float(rms_before)) else float("nan")
    return (
        theta_deg_fin,
        float(tx_mm),
        float(ty_mm),
        float(sx_fin),
        float(sy_fin),
        float(rms_before),
        float(rms_after),
    )


def _fine_align_fit_similarity_between_xy(
    orig_xy: np.ndarray,
    final_xy: np.ndarray,
    row_weights: np.ndarray,
    *,
    allow_xy: bool,
    allow_rotation: bool,
    allow_scale: bool,
    max_fit_samples: int = DETECTOR_ALIGN_MAX_FIT_SAMPLES,
) -> tuple[float, float, float, float, float] | None:
    """Single similarity map from source XY to target XY (same row indices)."""
    if int(orig_xy.shape[0]) != int(final_xy.shape[0]):
        raise ValueError("fine align XY pairing length mismatch")
    finite = (
        np.isfinite(orig_xy).all(axis=1)
        & np.isfinite(final_xy).all(axis=1)
        & (row_weights > 0.0)
    )
    idx_all = np.flatnonzero(finite)
    if int(idx_all.size) < 1:
        return None
    o_xy = orig_xy[idx_all]
    f_xy = final_xy[idx_all]
    w_acc = row_weights[idx_all]
    fit_idx = _subsample_align_indices(int(o_xy.shape[0]), int(max_fit_samples))
    theta, tx, ty, sx, sy, _, _ = _fine_align_gauss_newton(
        o_xy[fit_idx],
        f_xy[fit_idx],
        w_acc[fit_idx],
        allow_xy=allow_xy,
        allow_rotation=allow_rotation,
        allow_scale=allow_scale,
    )
    return theta, tx, ty, sx, sy


def _fine_align_similarity_icp(
    planned_xyz: list[tuple[float, float, float]],
    measured_rows: list[tuple[float, ...]],
    *,
    a_is_x: bool,
    row_weights: np.ndarray,
    allow_xy: bool,
    allow_rotation: bool,
    allow_scale: bool,
    max_fit_samples: int = DETECTOR_ALIGN_MAX_FIT_SAMPLES,
    max_icp_iter: int = _FINE_ALIGN_ICP_MAX_ITER,
    icp_tol_mm: float = _FINE_ALIGN_ICP_TOL_MM,
) -> (
    tuple[list[tuple[float, ...]], float, float, int, tuple[float, float, float, float, float]]
    | None
):
    """Layer-NN ICP + subsampled GN; returns aligned rows, QA RMS, pair count, transform."""
    rows = list(measured_rows)
    rw = np.asarray(row_weights, dtype=np.float64).reshape(-1)
    ctx = build_layer_nn_plan_context(planned_xyz)
    li_raw, orig_xy = meas_layer_indices_and_xy_from_rows(rows, ctx.hi, a_is_x=a_is_x)
    work_xy = np.array(orig_xy, dtype=np.float64, copy=True)

    dist, exp_xyz = layer_nn_dist_and_expected_from_xy(ctx, work_xy, li_raw)
    qa_before = _fine_align_qa_rms_from_dist(dist, rw)
    if not math.isfinite(qa_before):
        return None

    prev_qa = qa_before
    qa_now = qa_before
    n_pairs = 0
    for icp_i in range(int(max_icp_iter)):
        pair = _fine_align_pairs_from_nn(dist, exp_xyz, work_xy, rw)
        if pair is None:
            break
        meas_xy, plan_xy, wpair = pair
        n_pairs = int(meas_xy.shape[0])
        if n_pairs < 1:
            break
        fit_idx = _subsample_align_indices(n_pairs, int(max_fit_samples))
        theta, tx, ty, sx, sy, _, _ = _fine_align_gauss_newton(
            meas_xy[fit_idx],
            plan_xy[fit_idx],
            wpair[fit_idx],
            allow_xy=allow_xy,
            allow_rotation=allow_rotation,
            allow_scale=allow_scale,
        )
        _fine_align_apply_similarity_inplace(
            work_xy,
            theta_deg=theta,
            sx=sx,
            sy=sy,
            tx_mm=tx,
            ty_mm=ty,
        )
        dist, exp_xyz = layer_nn_dist_and_expected_from_xy(ctx, work_xy, li_raw)
        qa_now = _fine_align_qa_rms_from_dist(dist, rw)
        if not math.isfinite(qa_now):
            break
        if icp_i > 0 and (prev_qa - qa_now) < float(icp_tol_mm):
            break
        prev_qa = qa_now

    qa_after = qa_now
    if not math.isfinite(qa_after) or qa_after >= qa_before - 1e-6:
        return None

    xform = _fine_align_fit_similarity_between_xy(
        orig_xy,
        work_xy,
        rw,
        allow_xy=allow_xy,
        allow_rotation=allow_rotation,
        allow_scale=allow_scale,
        max_fit_samples=max_fit_samples,
    )
    if xform is None:
        return None
    out_rows = _fine_align_rows_from_meas_xy(work_xy, rows, a_is_x=a_is_x)
    return out_rows, qa_before, qa_after, n_pairs, xform


def fine_align_measured_to_plan(
    planned_xyz: list[tuple[float, float, float]],
    measured_rows: list[tuple[float, ...]],
    *,
    a_is_x: bool = False,
    allow_xy: bool = True,
    allow_rotation: bool = True,
    allow_scale: bool = True,
) -> tuple[list[tuple[float, ...]], DetectorFineAlign2D | None]:
    """Fine similarity map on whatever rows are passed in (post-aggregate or per-row).

    Pairing always uses per-layer nearest plan spot (plan QA). An outer ICP loop
    re-pairs after each GN step; the fit subsamples large acquisitions but applies
    the transform to every row. Reported RMS values are layer-NN QA RMS on all rows.
    """
    rows = list(measured_rows)
    if not (allow_xy or allow_rotation or allow_scale):
        return rows, None
    if not planned_xyz or not rows:
        return rows, None

    rw = _fine_align_row_weights(rows)
    icp_out = _fine_align_similarity_icp(
        planned_xyz,
        rows,
        a_is_x=a_is_x,
        row_weights=rw,
        allow_xy=allow_xy,
        allow_rotation=allow_rotation,
        allow_scale=allow_scale,
    )
    if icp_out is None:
        return rows, None

    out_rows, qa_before, qa_after, n_pairs, xform = icp_out
    theta_deg, tx_mm, ty_mm, sx, sy = xform
    info = DetectorFineAlign2D(
        theta_deg=float(theta_deg),
        tx_mm=float(tx_mm),
        ty_mm=float(ty_mm),
        sx=float(sx),
        sy=float(sy),
        rms_before_mm=float(qa_before),
        rms_after_mm=float(qa_after),
        n_pairs=int(n_pairs),
        allow_xy=bool(allow_xy),
        allow_rotation=bool(allow_rotation),
        allow_scale=bool(allow_scale),
    )
    return out_rows, info


def format_fine_align_caption(info: DetectorFineAlign2D) -> str:
    dof_parts = [
        *[name for ok, name in ((info.allow_xy, "XY"), (info.allow_rotation, "rotation")) if ok],
        *(["scale sx,sy"] if info.allow_scale else []),
    ]
    dof_label = ", ".join(dof_parts) if dof_parts else "none"
    return (
        f"Fine detector align ({dof_label}): θ={info.theta_deg:.5g}° CCW, "
        f"t=({info.tx_mm:.5g}, {info.ty_mm:.5g}) mm, "
        f"sx={info.sx:.6g}, sy={info.sy:.6g}, QA RMS={info.rms_after_mm:.5g} mm "
        f"(before={info.rms_before_mm:.5g} mm, n={info.n_pairs})."
    )


def _plan_xy_matrix_from_measured_rows(
    measured_rows: list[tuple[float, ...]],
    *,
    a_is_x: bool,
) -> np.ndarray:
    n = len(measured_rows)
    if n == 0:
        return np.zeros((0, 2), dtype=np.float64)
    a_col = np.fromiter((float(r[0]) for r in measured_rows), dtype=np.float64, count=n)
    b_col = np.fromiter((float(r[1]) for r in measured_rows), dtype=np.float64, count=n)
    return _measured_ab_to_plan_xy_array(a_col, b_col, a_is_x=a_is_x, swap_ab_axes=False)


def _invert_coarse_rigid_plan_xy(
    plan_xy: np.ndarray,
    coarse: DetectorRigidAlign2D,
) -> np.ndarray:
    """Inverse of coarse orientation + rigid map on plan-frame XY rows."""
    r_mat = _rotation_matrix_2d(coarse.theta_deg)
    t_arr = np.asarray([coarse.tx_mm, coarse.ty_mm], dtype=np.float64).reshape(2)
    signs = _plan_xy_signs(
        flip_plan_x=bool(coarse.flip_plan_x),
        flip_plan_y=bool(coarse.flip_plan_y),
    )
    out = np.asarray(plan_xy, dtype=np.float64).reshape(-1, 2).copy()
    finite = np.isfinite(out).all(axis=1)
    if np.any(finite):
        oriented = (out[finite] - t_arr) @ r_mat
        out[finite] = oriented * signs.reshape(1, 2)
    return out


def _fit_total_align_params_from_row_sets(
    source_rows: list[tuple[float, ...]],
    target_rows: list[tuple[float, ...]],
    *,
    a_is_x: bool,
    allow_xy: bool,
    allow_rotation: bool,
    allow_scale: bool,
) -> tuple[float, float, float, float, float] | None:
    if len(source_rows) != len(target_rows) or not source_rows:
        return None
    orig_xy = _plan_xy_matrix_from_measured_rows(source_rows, a_is_x=a_is_x)
    final_xy = _plan_xy_matrix_from_measured_rows(target_rows, a_is_x=a_is_x)
    rw = _fine_align_row_weights(source_rows)
    return _fine_align_fit_similarity_between_xy(
        orig_xy,
        final_xy,
        rw,
        allow_xy=allow_xy,
        allow_rotation=allow_rotation,
        allow_scale=allow_scale,
    )


def format_total_detector_align_caption(
    *,
    coarse: DetectorRigidAlign2D | None = None,
    fine: DetectorFineAlign2D | None = None,
    measured_base: list[tuple[float, ...]] | None = None,
    measured_final: list[tuple[float, ...]] | None = None,
    a_is_x: bool = False,
) -> str | None:
    """Single user-facing summary of net detector→plan alignment (all phases combined)."""
    if coarse is None and fine is None:
        return None

    allow_xy = bool(fine.allow_xy) if fine is not None else True
    allow_rotation = bool(fine.allow_rotation) if fine is not None else True
    allow_scale = bool(fine.allow_scale) if fine is not None else False

    if fine is not None and math.isfinite(float(fine.rms_after_mm)):
        qa_rms = float(fine.rms_after_mm)
    elif coarse is not None and math.isfinite(float(coarse.rms_residual_mm)):
        qa_rms = float(coarse.rms_residual_mm)
    else:
        qa_rms = float("nan")

    params: tuple[float, float, float, float, float] | None = None
    if (
        fine is not None
        and measured_base
        and measured_final
        and len(measured_base) == len(measured_final)
    ):
        src_rows = list(measured_base)
        if coarse is not None:
            base_xy = _plan_xy_matrix_from_measured_rows(src_rows, a_is_x=a_is_x)
            pre_xy = _invert_coarse_rigid_plan_xy(base_xy, coarse)
            finite = np.isfinite(pre_xy).all(axis=1)
            if np.any(finite):
                src_rows = _fine_align_rows_from_meas_xy(pre_xy, src_rows, a_is_x=a_is_x)
        params = _fit_total_align_params_from_row_sets(
            src_rows,
            list(measured_final),
            a_is_x=a_is_x,
            allow_xy=allow_xy,
            allow_rotation=allow_rotation,
            allow_scale=allow_scale,
        )
    if params is None and fine is not None:
        params = (
            float(fine.theta_deg),
            float(fine.tx_mm),
            float(fine.ty_mm),
            float(fine.sx),
            float(fine.sy),
        )
    if params is None and coarse is not None:
        params = (
            float(coarse.theta_deg),
            float(coarse.tx_mm),
            float(coarse.ty_mm),
            1.0,
            1.0,
        )
    if params is None:
        return None

    theta_deg, tx_mm, ty_mm, sx, sy = params
    parts = [
        f"Alignment: θ={theta_deg:.3g}° CCW",
        f"t=({tx_mm:.3g}, {ty_mm:.3g}) mm",
    ]
    if allow_scale and (abs(sx - 1.0) > 1e-3 or abs(sy - 1.0) > 1e-3):
        parts.append(f"sx={sx:.4g}, sy={sy:.4g}")
    if math.isfinite(qa_rms):
        parts.append(f"QA RMS={qa_rms:.3g} mm")
    return ", ".join(parts) + "."
