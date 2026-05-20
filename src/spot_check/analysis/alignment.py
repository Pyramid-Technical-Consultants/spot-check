"""Alignment."""

from __future__ import annotations

from spot_check.analysis._imports import *  # noqa: F403
from spot_check.analysis.auto_columns import AutoFitColumns, position_fit_deadtime_mask
from spot_check.analysis.spatial import (
    _cKDTree,
    _kdtree_query_k1,
    _layer_nn_plan_targets,
    _layer_nn_rms_mm,
    _layer_xy_kdtrees_2d,
    _plan_xy_by_energy_layer,
    nominal_layer_energies_mev,
)

_last_detector_align: DetectorRigidAlign2D | None = None

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
) -> tuple[np.ndarray, np.ndarray, float, float, int, int]:
    """Coarse rotation seeds + flat-plan ICP."""
    if int(meas_xy.shape[0]) < 1:
        raise ValueError("detector alignment needs at least one finite measured row with plan")
    best_rms = float("inf")
    best: tuple[np.ndarray, np.ndarray, float, float, int] | None = None
    centroid = np.nanmean(meas_xy, axis=0)
    if not (math.isfinite(float(centroid[0])) and math.isfinite(float(centroid[1]))):
        raise ValueError("detector alignment needs finite measured XY")
    angle_seeds = _detector_align_coarse_angles_deg(int(meas_xy.shape[0]))
    for init_deg in angle_seeds:
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
        holdout = _flat_nn_rms_mm(meas_xy, plan_xy, r_acc, t_acc, plan_tree=plan_tree)
        if holdout < best_rms:
            best_rms = holdout
            best = (r_acc, t_acc, rms_nn, holdout, n_iter)
    if best is None:
        raise ValueError("detector ICP alignment failed for all rotation seeds")
    r_acc, t_acc, rms_nn, rms_res, n_iter = best
    return r_acc, t_acc, rms_nn, rms_res, n_iter, int(meas_xy.shape[0])


def _build_align_samples_from_auto_cols(
    cols: AutoFitColumns,
    *,
    use_on_spot_only: bool,
    swap_ab_axes: bool,
) -> np.ndarray:
    """Plan-frame XY from columnar auto rows (position-fit on-spot by default)."""
    n = len(cols)
    if n == 0:
        return np.zeros((0, 2), dtype=np.float64)
    live = np.ones(n, dtype=bool)
    if use_on_spot_only:
        live = ~position_fit_deadtime_mask(cols)
    m_acc: list[list[float]] = []
    for i in range(n):
        if not live[i]:
            continue
        if swap_ab_axes:
            mx, my = float(cols.a[i]), float(cols.b[i])
        else:
            mx, my = float(cols.mx[i]), float(cols.my[i])
        if not (math.isfinite(mx) and math.isfinite(my)):
            continue
        m_acc.append([mx, my])
    if not m_acc:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(m_acc, dtype=np.float64)


def _apply_rigid_to_auto_fit_columns(
    cols: AutoFitColumns,
    r_mat: np.ndarray,
    tvec: np.ndarray,
    *,
    a_is_x: bool,
) -> AutoFitColumns:
    """Apply rigid map to plan-frame mx/my (and partial mx_p/my_p); refresh stored A/B."""
    from dataclasses import replace

    n = len(cols)
    if n == 0:
        return cols
    r_arr = np.asarray(r_mat, dtype=np.float64).reshape(2, 2)
    t_arr = np.asarray(tvec, dtype=np.float64).reshape(2)
    mx = np.array(cols.mx, dtype=np.float64, copy=True)
    my = np.array(cols.my, dtype=np.float64, copy=True)
    mx_p = np.array(cols.mx_p, dtype=np.float64, copy=True)
    my_p = np.array(cols.my_p, dtype=np.float64, copy=True)
    m = np.column_stack([mx, my])
    finite = np.isfinite(m).all(axis=1)
    if np.any(finite):
        m[finite] = (r_arr @ m[finite].T).T + t_arr
    mx[:] = m[:, 0]
    my[:] = m[:, 1]
    mp = np.column_stack([mx_p, my_p])
    finite_p = np.isfinite(mp).all(axis=1)
    if np.any(finite_p):
        mp[finite_p] = (r_arr @ mp[finite_p].T).T + t_arr
    mx_p[:] = mp[:, 0]
    my_p[:] = mp[:, 1]
    if a_is_x:
        a, b = mx, my
    else:
        a, b = my, mx
    return replace(cols, mx=mx, my=my, mx_p=mx_p, my_p=my_p, a=a, b=b)


def align_auto_fit_columns_to_plan_xy(
    cols: AutoFitColumns,
    planned_xyz: list[tuple[float, float, float]],
    *,
    a_is_x: bool = False,
    max_fit_samples: int = DETECTOR_ALIGN_MAX_FIT_SAMPLES,
    use_on_spot_only: bool = True,
) -> tuple[AutoFitColumns, DetectorRigidAlign2D]:
    """Rigid 2D align on flattened plan XY **before** spot/layer assignment.

    All plan spots share one Z plane; each on-spot measured row (finite fit XY) is matched
    to the nearest plan spot in 2D via multi-start ICP + Kabsch. The fitted transform is
    applied to every row's plan-frame positions (mx/my and partial mx_p/my_p).
    """
    if not planned_xyz or len(cols) == 0:
        raise ValueError("plan and auto columns are required for pre-assignment alignment")
    plan_xy = np.asarray(
        [(float(px), float(py)) for px, py, _ in planned_xyz],
        dtype=np.float64,
    )
    if plan_xy.shape[0] < 1:
        raise ValueError("plan has no scan spots for pre-assignment alignment")
    plan_tree = _cKDTree(plan_xy) if _cKDTree is not None else None

    best: (
        tuple[np.ndarray, np.ndarray, float, float, bool, int, int, int] | None
    ) = None
    best_rms = float("inf")
    for swap_ab in (False, True):
        meas_xy = _build_align_samples_from_auto_cols(
            cols,
            use_on_spot_only=use_on_spot_only,
            swap_ab_axes=swap_ab,
        )
        n_all = int(meas_xy.shape[0])
        if n_all < 1:
            continue
        fit_idx = _subsample_align_indices(n_all, int(max_fit_samples))
        meas_fit = meas_xy[fit_idx]
        try:
            r_mat, tvec, rms_nn, rms_res, n_iter, _n_fit = _detector_align_multistart_icp_flat(
                meas_fit,
                plan_xy,
                plan_tree=plan_tree,
            )
        except ValueError:
            continue
        if rms_res < best_rms:
            best_rms = rms_res
            best = (r_mat, tvec, rms_nn, rms_res, swap_ab, n_iter, n_all, int(fit_idx.size))

    if best is None:
        raise ValueError(
            "pre-assignment detector alignment needs at least one on-spot row with finite "
            "fit XY and a matching plan spot in 2D"
        )

    r_mat, tvec, rms_nn, rms_res, swap_ab, n_iter, n_all, n_fit = best
    out_cols = _apply_rigid_to_auto_fit_columns(cols, r_mat, tvec, a_is_x=a_is_x)
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
        pre_assignment=True,
    )
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
    r_arr = np.asarray(r_mat, dtype=np.float64)
    t_arr = np.asarray(tvec, dtype=np.float64).reshape(2)
    finite = np.isfinite(m).all(axis=1)
    w = m.copy()
    if np.any(finite):
        w[finite] = (r_arr @ m[finite].T).T + t_arr
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

    n_meas = len(measured_rows)
    n_plan = len(planned_xyz)
    if n_meas == n_plan and n_meas > 0:
        for swap_ab in (False, True):
            meas_xy, plan_xy = _build_align_samples_delivery_order(
                measured_rows,
                planned_xyz,
                a_is_x=a_is_x,
                swap_ab_axes=swap_ab,
            )
            n_pairs = int(meas_xy.shape[0])
            if n_pairs < 1:
                continue
            try:
                r_mat, tvec, rms_nn, rms_res, n_iter, _n_fit = _detector_align_multistart_delivery(
                    meas_xy,
                    plan_xy,
                )
            except ValueError:
                continue
            out_rows = _apply_rigid_xy_to_measured_rows(
                measured_rows,
                r_mat,
                tvec,
                a_is_x=a_is_x,
                swap_ab_axes=swap_ab,
            )
            theta = float(math.degrees(math.atan2(float(r_mat[1, 0]), float(r_mat[0, 0]))))
            info = DetectorRigidAlign2D(
                theta_deg=theta,
                tx_mm=float(tvec[0]),
                ty_mm=float(tvec[1]),
                rms_nn_mm=rms_nn,
                rms_residual_mm=rms_res,
                n_pairs=n_pairs,
                ab_axes_swapped=bool(swap_ab),
                icp_iterations=int(n_iter),
                n_pairs_fit=n_pairs,
            )
            _set_last_detector_align(info)
            return out_rows, info

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
        if n_all < 1:
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
            "detector alignment needs at least one measured row with finite fit XY and a "
            "matching plan spot on its layer (check layer assignment, plan, and detector "
            "orientation)"
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
    _set_last_detector_align(info)
    return out_rows, info

def format_detector_align_caption(info: DetectorRigidAlign2D) -> str:
    stage = "pre-assign" if info.pre_assignment else "post-assign"
    swap_note = "; Fit A↔B swapped for search" if info.ab_axes_swapped else ""
    fit_note = ""
    if info.n_pairs_fit > 0 and info.n_pairs_fit < info.n_pairs:
        fit_note = f", fit n={info.n_pairs_fit}/{info.n_pairs}"
    return (
        f"Detector align ({stage}): θ={info.theta_deg:.5g}° CCW, "
        f"t=({info.tx_mm:.5g}, {info.ty_mm:.5g}) mm, "
        f"RMS after={info.rms_residual_mm:.5g} mm (NN RMS before={info.rms_nn_mm:.5g} mm, "
        f"n={info.n_pairs}{fit_note}, ICP={info.icp_iterations}{swap_note})."
    )
